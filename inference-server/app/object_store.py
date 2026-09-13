"""常驻对象仓库：cond/latent/image 张量驻留 GPU 内存，节点间只传 UUID。"""
import threading
import time
import uuid


class ObjectStore:
    def __init__(self, ttl_seconds: int = 3600):
        self._items = {}
        self._lock = threading.Lock()
        self._ttl = ttl_seconds

    def put(self, kind: str, data, meta: dict | None = None) -> str:
        self._sweep()
        oid = uuid.uuid4().hex
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
            self._items.clear()
        return n

    def _sweep(self):
        now = time.time()
        with self._lock:
            expired = [k for k, v in self._items.items() if now - v["created"] > self._ttl]
            for k in expired:
                del self._items[k]
