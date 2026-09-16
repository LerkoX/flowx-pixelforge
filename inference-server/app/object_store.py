"""常驻对象仓库：cond/latent/image 张量驻留 GPU 内存，节点间只传 UUID。

VIDEO 大对象例外：put 时即编码 mp4 落盘（video_dir），仓库内只留
{"path", "fps", "num_frames"} 元数据；对象过期/清理时同步删除落盘文件。
"""
import os
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
            data = self._video_encoder(data["frames"], data.get("fps", 24), path)
        with self._lock:
            self._items[oid] = {
                "kind": kind,
                "data": data,
                "meta": meta or {},
                "created": time.time(),
            }
        return oid

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
