"""常驻对象仓库：cond/latent/image 张量驻留 GPU 内存，节点间只传 UUID。

VIDEO 大对象例外：put 时即编码 mp4 落盘（video_dir），仓库内只留
{"path", "fps", "num_frames"} 元数据；对象过期/清理时同步删除落盘文件。

VIDEO 预编码直通：data 已是 mp4（带 "path" 且无 "frames"）时不重新编码，
只把文件拷进 video_dir（纳入同一 TTL 清理）——load-video 的原片入仓走这条路，
避免「解码→h264 重编码→再解码」的代际损失（见 video.load 算子）。
"""
import os
import shutil
import threading
import time
import uuid

from . import lease as _lease
from . import video


class ObjectStore:
    def __init__(self, ttl_seconds: int = 3600, video_dir: str | None = None,
                 video_encoder=None, lease_alive=None,
                 orphan_ttl_seconds: int | None = None):
        self._items = {}
        self._lock = threading.Lock()
        self._ttl = ttl_seconds
        self._video_dir = video_dir
        # 可注入假编码器供无 imageio 环境测试
        self._video_encoder = video_encoder or video.encode_mp4
        # 孤儿条目清理（B2）：条目标注了租约、租约已不活跃、且龄期超过
        # orphan_ttl_seconds → 回收（见 _sweep）。lease_alive 由 main.py 注入
        # （LeaseBook.is_active）；**未注入时本机制关闭**，保持既有语义（单测/脚本）。
        self.lease_alive = lease_alive
        self.orphan_ttl_seconds = orphan_ttl_seconds

    def put(self, kind: str, data, meta: dict | None = None) -> str:
        self._sweep()
        oid = uuid.uuid4().hex
        if kind == "VIDEO" and self._video_dir:
            path = os.path.join(self._video_dir, oid + ".mp4")
            if isinstance(data, dict) and "frames" not in data and data.get("path"):
                data = self._adopt_video_file(data, path)
            else:
                data = self._video_encoder(data["frames"], data.get("fps", 24), path)
        with self._lock:
            self._items[oid] = {
                "kind": kind,
                "data": data,
                "meta": meta or {},
                "created": time.time(),
                # 归属租约（B2）：淘汰保护与孤儿清理都按它分账。
                # 不带 X-FlowX-Lease 的调用方（脚本/验收）为 None ⇒ 旧语义不变。
                "lease": _lease.current(),
            }
        return oid

    @staticmethod
    def _adopt_video_file(data: dict, path: str) -> dict:
        """预编码 VIDEO 入库：拷贝既有 mp4 到对象目录，只留元数据（不重编码）。

        拷而非引用：对象目录是仓库自己的地盘（TTL 到点要删文件），不能指向
        调用方的临时上传目录（删了别人的原片 / 或者对象过期把上传物一起删掉）。
        """
        src = data["path"]
        if not os.path.isfile(src):
            raise ValueError(f"VIDEO source file not found: {src}")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        try:
            shutil.copyfile(src, path)
        except OSError as e:  # noqa: BLE001 - 磁盘满/权限：给出可操作报错
            raise ValueError(f"VIDEO copy {src} -> {path} failed: {e}")
        return {"path": path,
                "fps": int(data.get("fps") or 24),
                "num_frames": int(data.get("num_frames") or 0)}

    def get(self, oid: str, kind: str | None = None):
        with self._lock:
            item = self._items.get(oid)
        if item is None:
            raise KeyError(f"object '{oid}' not found (expired or never created)")
        if kind is not None and item["kind"] != kind:
            raise TypeError(f"object '{oid}' is {item['kind']}, expected {kind}")
        return item

    def contains(self, oid: str) -> bool:
        with self._lock:
            return oid in self._items

    def stats(self) -> dict:
        """仓库统计（监控页/指标历史用）：对象总数 + 按 kind 分组计数。"""
        with self._lock:
            kinds = {}
            for v in self._items.values():
                k = v.get("kind") or "?"
                kinds[k] = kinds.get(k, 0) + 1
            return {"count": len(self._items), "kinds": kinds}

    def referenced_data(self, lease_id=None) -> list:
        """仓库内条目的 data 列表（供模型淘汰保护：被仓库引用的管道
        本体视同 pin，见 ModelManager.extra_pinned / main.py 注入）。

        lease_id=None（默认 / 无租约调用方）→ 全量：历史语义逐字节不变。
        lease_id 非空 → 只返回**本租约**创建的条目。

        为什么按租约收敛（B2，exec 795）：原先仓库条目不区分租约，extra_pinned
        于是把**上一轮/另一条流水线**留下的 model/clip/vae 视图也当“不可淘汰”，
        配上 OBJECT_TTL_SECONDS=86400（24h）就能把 7.4GB 常驻条目钉住一整天 —
        实测 flux 四个条目（unet 6470MB + clip 477 + vae 381 + redux 123）
        就这样赖在 8GB 卡上跟 SDXL+CN 挤，弄出 16.4GB 超订。
        跨租约的保护不需要这一层：`lease.keys_of_others` 本来就会跳过
        其它活跃租约的条目（两层职责不重叠）。
        """
        with self._lock:
            if lease_id is None:
                return [v.get("data") for v in self._items.values()]
            return [v.get("data") for v in self._items.values()
                    if v.get("lease") == lease_id]

    def clear(self) -> int:
        with self._lock:
            n = len(self._items)
            for item in self._items.values():
                self._discard_file(item)
            self._items.clear()
        return n

    def _sweep(self):
        """过期清理：常规 TTL + **孤儿条目**（B2）。返回回收条数。

        孤儿 = 条目标注了租约、该租约已不活跃（执行结束/租约 TTL 过期）、
        且龄期超过 orphan_ttl_seconds。为什么要它：常规 TTL 是 86400s（24h，
        实测环境变量），而条目标注的租约一结束，这些对象就再也没人会消费，
        却仍钉着模型本体、占着主机内存（实测 5.3GB）—— 靠人工 model.unload
        才能松开。长任务的安全网：jobs 进度心跳会续租（jobs.Job.set_progress），
        所以“正在跑的 90 分钟采样”不会被误当孤儿。
        """
        now = time.time()
        dropped_orphans = []
        with self._lock:
            expired = []
            for k, v in self._items.items():
                age = now - v["created"]
                if age > self._ttl:
                    expired.append(k)
                    continue
                lid = v.get("lease")
                # orphan_ttl_seconds=None 表示关闭；0 表示“租约一结束就回收”
                if (lid and self.orphan_ttl_seconds is not None
                        and self.lease_alive is not None
                        and age > self.orphan_ttl_seconds):
                    try:
                        alive = self.lease_alive(lid)
                    except Exception:  # noqa: BLE001 - 回调故障不阻断清理
                        alive = True
                    if not alive:
                        expired.append(k)
                        dropped_orphans.append(k)
            for k in expired:
                self._discard_file(self._items[k])
                del self._items[k]
        if dropped_orphans:
            print(f"[object-store] 回收孤儿对象 {len(dropped_orphans)} 个"
                  f"（租约已结束，B2）", flush=True)
        return len(expired)

    def discard_where(self, pred) -> int:
        """按谓词删除条目，返回删除数。
        用于模型淘汰时联动清理：删除 data 为该管道本体的 model/clip/vae 视图
        （checkpoint.load/motion.load 输出的三视图与管道是同一对象），
        否则已淘汰模型的权重会被仓库钉住无法回收（TTL 要等 1 小时）。"""
        with self._lock:
            victims = [k for k, v in self._items.items() if pred(v.get("data"))]
            for k in victims:
                self._discard_file(self._items[k])
                del self._items[k]
        return len(victims)

    @staticmethod
    def _discard_file(item):
        """对象淘汰时删除落盘文件（VIDEO mp4）；失败静默（文件可能已被外部清理）。"""
        data = item.get("data")
        path = data.get("path") if isinstance(data, dict) else None
        if path:
            try:
                os.remove(path)
            except OSError:
                pass
