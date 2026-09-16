"""VIDEO 对象落盘测试（注入假编码器，无需 imageio/GPU）。

覆盖：VIDEO put 触发落盘、仓库内只留元数据、过期/清理时删除 mp4 文件、
registry 接受 VIDEO 端口类型。
运行：python3 -m tests.test_video_store  或  python3 tests/test_video_store.py
"""
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.object_store import ObjectStore
from app.registry import OBJ_TYPES, Registry, outputs_to_store


def fake_encoder(frames, fps, path):
    """假 mp4 编码：写个占位文件，返回与真编码器同构的元数据。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(b"FAKE-MP4:" + ",".join(frames).encode())
    return {"path": path, "fps": int(fps), "num_frames": len(frames)}


def main():
    assert "VIDEO" in OBJ_TYPES
    print("VIDEO type registered OK:", OBJ_TYPES)

    with tempfile.TemporaryDirectory() as d:
        store = ObjectStore(ttl_seconds=3600, video_dir=d, video_encoder=fake_encoder)

        # --- put VIDEO：落盘 + 仓库内为元数据 ---
        oid = store.put("VIDEO", {"frames": ["f0", "f1", "f2"], "fps": 16})
        path = os.path.join(d, oid + ".mp4")
        assert os.path.isfile(path), "mp4 file not written"
        data = store.get(oid, "VIDEO")["data"]
        assert data == {"path": path, "fps": 16, "num_frames": 3}, data
        assert open(path, "rb").read() == b"FAKE-MP4:f0,f1,f2"
        print("VIDEO put -> mp4 on disk OK:", data)

        # --- 类型校验：VIDEO 对象不能用 IMAGE 取 ---
        try:
            store.get(oid, "IMAGE")
            raise AssertionError("type check missing")
        except TypeError as e:
            print("type check OK:", e)

        # --- 未配 video_dir 时 VIDEO 原样驻留内存（兼容/测试路径）---
        mem = ObjectStore(video_encoder=fake_encoder)
        oid2 = mem.put("VIDEO", {"frames": ["x"], "fps": 8})
        assert mem.get(oid2)["data"] == {"frames": ["x"], "fps": 8}
        print("in-memory VIDEO (no video_dir) OK")

        # --- clear() 删除落盘文件 ---
        assert store.clear() == 1
        assert not os.path.exists(path), "mp4 not removed on clear"
        print("clear() removes mp4 OK")

        # --- TTL 过期 sweep 删除落盘文件 ---
        store2 = ObjectStore(ttl_seconds=0, video_dir=d, video_encoder=fake_encoder)
        oid3 = store2.put("VIDEO", {"frames": ["a"], "fps": 24})
        path3 = os.path.join(d, oid3 + ".mp4")
        assert os.path.isfile(path3)
        time.sleep(0.01)
        store2.put("IMAGE", "trigger-sweep")  # put 触发 _sweep
        assert not store2.contains(oid3)
        assert not os.path.exists(path3), "mp4 not removed on TTL sweep"
        print("TTL sweep removes mp4 OK")

        # --- registry 端到端：算子返回 VIDEO → outputs_to_store 落盘 ---
        reg = Registry()

        @reg.register("video.fake", inputs={"n": "INT"},
                      outputs={"video": "VIDEO", "fps": "INT"})
        def _(n=2):
            return {"video": {"frames": [f"f{i}" for i in range(n)], "fps": 30},
                    "fps": 30}

        op = reg.get("video.fake")
        out = op.fn(n=4)
        meta = outputs_to_store(store, op, out)
        assert meta["video"]["type"] == "VIDEO" and "id" in meta["video"], meta
        assert meta["fps"] == {"value": 30, "type": "INT"}, meta
        vdata = store.get(meta["video"]["id"], "VIDEO")["data"]
        assert vdata["fps"] == 30 and vdata["num_frames"] == 4, vdata
        assert os.path.isfile(vdata["path"])
        print("registry VIDEO round-trip OK:", meta["video"])

    print("\nALL VIDEO STORE TESTS PASSED")


if __name__ == "__main__":
    main()
