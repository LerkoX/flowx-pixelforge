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

from . import video


class ObjectStore:
    def __init__(self, ttl_seconds: int = 3600, video_dir: str | None = None,
                 video_encoder=None):
        self._items = {}
        self._lock = threading.Lock()
        self._ttl = ttl_seconds
        self._video_dir = video_dir
        # 可注入假编码器供无 imageio 环境测试
        self._video_encoder = video_encoder or video.encode_mp4

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

    def referenced_data(self) -> list:
        """仓库内所有条目的 data 列表（供模型淘汰保护：被仓库引用的管道
        本体视同 pin，见 ModelManager.extra_pinned / main.py 注入）。"""
        with self._lock:
            return [v.get("data") for v in self._items.values()]

    def clear(self) -> int:
        with self._lock:
            n = len(self._items)
            for item in self._items.values():
                self._discard_file(item)
            self._items.clear()
        return n

    def _sweep(self):
        now = time.time()
        with self._lock:
            expired = [k for k, v in self._items.items() if now - v["created"] > self._ttl]
            for k in expired:
                self._discard_file(self._items[k])
                del self._items[k]

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
