"""video_flow 算子族契约 + VIDEO 预编码直通单测（无需 GPU / torch / ffmpeg）。

覆盖：
1. 族文件可加载（app.plugins 真实加载链路），5 个算子注册名带族版本后缀、
   端口契约与声明一致；
2. 纯逻辑：抽帧步长/有效 fps（frame_step）、内存形态 VIDEO 的抽帧（_subsample）、
   等比缩放与偶数取整（fit_frames）、IMAGE 形态校验（as_frame_list）；
3. ObjectStore 的 VIDEO **预编码直通**：原字节拷进对象目录、不重编码、
   对象过期/清空时同步删除（load-video 的无损快路径）。

真机口径（解码、显存、出片）由 GPU 实测与抽帧看图覆盖，本文件只管本机可判定的部分。
运行：python3 -m tests.test_video_flow
"""
import os
import sys
import tempfile
import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# app.ops 顶层 import torch + diffusers：本机（Termux）没有，桩掉后仍走真实加载链路
sys.modules.setdefault("torch", types.ModuleType("torch"))
_fake_diffusers = types.ModuleType("diffusers")
for _n in ("DDIMScheduler", "DPMSolverMultistepScheduler",
           "EulerAncestralDiscreteScheduler", "EulerDiscreteScheduler",
           "LMSDiscreteScheduler", "UniPCMultistepScheduler"):
    setattr(_fake_diffusers, _n, type(_n, (), {}))
sys.modules.setdefault("diffusers", _fake_diffusers)

from PIL import Image  # noqa: E402

from app.object_store import ObjectStore  # noqa: E402
from app.plugins import check_plugin_source, load_plugin  # noqa: E402
from app.registry import Registry  # noqa: E402

PLUGIN = os.path.join(os.path.dirname(__file__), "..", "..", "nodes", "_common",
                      "server_ops", "video_flow.py")

EXPECTED = {
    "video.load": (
        {"path": "STRING", "start": "INT", "end": "INT", "stride": "INT",
         "max_frames": "INT", "fps": "FLOAT", "width": "INT", "height": "INT"},
        {"video": "VIDEO", "fps": "INT", "num_frames": "INT", "width": "INT",
         "height": "INT"}),
    "video.to_frames": (
        {"video": "VIDEO", "start": "INT", "end": "INT", "stride": "INT",
         "max_frames": "INT"},
        {"frames": "IMAGE", "fps": "INT", "count": "INT"}),
    "video.encode": (
        {"frames": "IMAGE", "fps": "INT"},
        {"video": "VIDEO", "num_frames": "INT"}),
    "video.repaint": (
        {"model": "MODEL", "pos": "COND", "neg": "COND", "video": "VIDEO",
         "strength": "FLOAT", "steps": "INT", "cfg": "FLOAT", "seed": "INT",
         "sampler_name": "STRING", "scheduler": "STRING",
         "start_at_step": "INT", "end_at_step": "INT", "start": "INT",
         "end": "INT", "stride": "INT", "max_frames": "INT",
         "control_net": "CONTROL_NET", "control_frames": "IMAGE",
         "control_strength": "FLOAT", "preview_every": "INT"},
        {"video": "VIDEO", "seed": "INT", "num_frames": "INT"}),
    "video.sample_v2v": (
        {"model": "MODEL", "video": "VIDEO", "prompt": "STRING",
         "neg_prompt": "STRING", "strength": "FLOAT", "steps": "INT",
         "cfg": "FLOAT", "seed": "INT", "width": "INT", "height": "INT",
         "num_frames": "INT", "fps": "INT", "decode_chunk_size": "INT",
         "preview_every": "INT", "enforce_inference_steps": "BOOL"},
        {"video": "VIDEO", "seed": "INT"}),
}


def _load_family():
    with open(PLUGIN, "rb") as f:
        content = f.read()
    err = check_plugin_source("video_flow.py", content)
    assert err is None, f"check_plugin_source: {err}"
    print("ok: check_plugin_source pass（语法/契约检查）")
    reg = Registry()
    names = load_plugin(PLUGIN, reg)
    return reg, names


def check_registration():
    reg, names = _load_family()
    # 注册名一律带族版本后缀（F1：算子身份 = 实现内容）
    for base in EXPECTED:
        hits = [n for n in names if n == base or n.startswith(base + "@")]
        assert hits, f"{base} 未注册（registered: {sorted(names)}）"
        assert len(hits) == 1, f"{base} 注册了多份：{hits}"
        assert "@" in hits[0], f"{base} 少了族版本后缀：{hits[0]}"
    assert len(names) == len(EXPECTED), f"算子数不符：{sorted(names)}"
    print(f"ok: 5 算子注册（带族版本后缀），例：{sorted(names)[0]}")

    for base, (inputs, outputs) in EXPECTED.items():
        name = [n for n in names if n.startswith(base)][0]
        op = reg.get(name)
        assert op.inputs == inputs, f"{base} inputs: {op.inputs}"
        assert op.outputs == outputs, f"{base} outputs: {op.outputs}"
    print("ok: 端口契约与声明一致")


def check_pure_logic():
    import importlib.util
    spec = importlib.util.spec_from_file_location("video_flow", PLUGIN)
    vf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vf)

    # frame_step：目标 fps 低于源 fps 才降采样；stride 与降采样取较大步长
    assert vf.frame_step(24, 0, 1) == (1, 24.0)
    assert vf.frame_step(24, 12, 1) == (2, 12.0)
    assert vf.frame_step(30, 10, 1) == (3, 10.0)
    assert vf.frame_step(24, 48, 1) == (1, 24.0), "目标 fps 高于源：不插帧，保持原速"
    assert vf.frame_step(24, 0, 4) == (4, 6.0)
    assert vf.frame_step(24, 6, 2) == (4, 6.0), "stride 与 fps 步长取较大者"
    assert vf.frame_step(0, 0, 1) == (1, 24.0), "源 fps 缺失时兜底 24"
    print("ok: frame_step（降采样/步长/兜底口径）")

    # _subsample：内存形态 VIDEO（无 path）走等价切片
    frames = [Image.new("RGB", (8, 8), (i, 0, 0)) for i in range(10)]
    mem = {"frames": frames, "fps": 10}
    got, fps = vf._subsample(mem, frames, 10.0, start=2, end=8, stride=2,
                             max_frames=2)
    assert [f.getpixel((0, 0))[0] for f in got] == [2, 4], got
    assert fps == 5.0, fps
    print("ok: _subsample（start/end/stride/max_frames 切片）")

    # video_frames_of：内存形态直取；非法形态明确报错
    got, fps = vf.video_frames_of(mem)
    assert len(got) == 10 and fps == 10.0
    for bad in (None, {}, {"fps": 24}):
        try:
            vf.video_frames_of(bad)
            raise AssertionError(f"video_frames_of 应拒绝 {bad!r}")
        except ValueError:
            pass
    print("ok: video_frames_of 内存形态 + 非法形态报错")

    # fit_frames：等比缩放 + 偶数取整
    src = [Image.new("RGB", (101, 51))]
    assert [f.size for f in vf.fit_frames(src, 0, 0)] == [(101, 51)], "不给尺寸不缩放"
    assert [f.size for f in vf.fit_frames(src, 64, 0)] == [(64, 32)], "只给宽：等比"
    assert [f.size for f in vf.fit_frames(src, 0, 32)] == [(62, 32)], "只给高：等比 + 取偶"
    assert [f.size for f in vf.fit_frames(src, 64, 33)] == [(64, 32)], "定尺：取偶"
    print("ok: fit_frames（等比/定尺/偶数取整）")

    # as_frame_list：单张 / 列表 / 非法
    one = Image.new("RGB", (4, 4))
    assert len(vf.as_frame_list(one)) == 1
    assert len(vf.as_frame_list([one, one])) == 2
    for bad in ([], [one, "x"], "nope"):
        try:
            vf.as_frame_list(bad)
            raise AssertionError(f"as_frame_list 应拒绝 {bad!r}")
        except ValueError:
            pass
    print("ok: as_frame_list 形态校验")

    # effective_steps：v2v 的 strength 是"时间轴截断"（真机事故：3步×0.6 只跑 1 步）
    assert vf.effective_steps(3, 0.6) == 1, "3 步 ×0.6 真机实测就是 1 步"
    assert vf.effective_steps(3, 0.99) == 2
    assert vf.effective_steps(3, 1.0) == 3
    assert vf.effective_steps(25, 0.7) == 17, "25×0.7=17 步（可用配比）"
    assert vf.effective_steps(25, 0.03) == 0, "配比过小 → 0 步（算子应报错）"
    assert vf.effective_steps(25, 0.7, enforce=True) == 25, "enforce=true ⇒ steps 即实际步数"
    assert vf.effective_steps(1, 0.5) == 0 and vf.effective_steps(2, 0.6) == 1
    print("ok: effective_steps（截断语义 / ComfyUI 语义 / 边界）")

    # _report_effective_steps：0 步必须给可操作错误、1 步给告警、正常步数无告警
    eff, err = vf._report_effective_steps("t", 25, 0.7)
    assert eff == 17 and err is None, (eff, err)
    eff, err = vf._report_effective_steps("t", 3, 0.6)
    assert eff == 1 and "有效去噪步数=1" in err and "enforce_inference_steps" in err, err
    eff, err = vf._report_effective_steps("t", 25, 0.03)
    assert eff == 0 and "原样返回输入" in err, err
    print("ok: 有效步数日志/告警口径（0 报错语义 / 1 告警 / 正常静默）")

    # read_video/_open_reader：路径不存在时给出可操作报错（不触解码器、不依赖 imageio）
    try:
        vf.read_video("/nonexistent/nope.mp4")
        raise AssertionError("read_video 应拒绝不存在的路径")
    except FileNotFoundError as e:
        assert "not found" in str(e)
    print("ok: read_video 缺文件报错（先查存在性，不被 ImportError 盖掉）")

    # _probe 同口径（load-video 的直通分支入口）
    try:
        vf._probe("/nonexistent/nope.mp4")
        raise AssertionError("_probe 应拒绝不存在的路径")
    except FileNotFoundError:
        print("ok: _probe 缺文件报错")

    # 解码器打开失败 → ValueError「cannot open」（桩一个只会抛异常的 imageio）
    fake_io = types.ModuleType("imageio")
    fake_v2 = types.ModuleType("imageio.v2")

    def _boom(path):
        raise OSError("ffmpeg: unknown format")

    fake_v2.get_reader = _boom
    fake_io.v2 = fake_v2
    saved = {k: sys.modules.get(k) for k in ("imageio", "imageio.v2")}
    sys.modules["imageio"], sys.modules["imageio.v2"] = fake_io, fake_v2
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            junk = f.name
        try:
            vf._open_reader(junk)
            raise AssertionError("坏文件应报 cannot open")
        except ValueError as e:
            assert "cannot open" in str(e), e
        print("ok: 解码器打开失败 → ValueError（含路径）")
    finally:
        os.unlink(junk)
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


def check_video_passthrough():
    """VIDEO 预编码直通：不重编码、拷进对象目录、随对象生命周期清理。"""
    encoded, adopted = [], []

    def encoder(frames, fps, path):
        encoded.append(path)
        with open(path, "wb") as f:
            f.write(b"ENCODED")
        return {"path": path, "fps": int(fps), "num_frames": len(frames)}

    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "source.mp4")
        with open(src, "wb") as f:
            f.write(b"ORIGINAL-BYTES")
        store = ObjectStore(ttl_seconds=3600, video_dir=os.path.join(d, "obj"),
                            video_encoder=encoder)
        oid = store.put("VIDEO", {"path": src, "fps": 30, "num_frames": 48})
        data = store.get(oid, "VIDEO")["data"]
        assert not encoded, "直通路径不应调用编码器"
        assert data["fps"] == 30 and data["num_frames"] == 48, data
        assert data["path"] != src, "必须拷进对象目录，不能指向调用方原片"
        with open(data["path"], "rb") as f:
            assert f.read() == b"ORIGINAL-BYTES", "原字节必须逐字节保留"
        assert os.path.isfile(src), "原片不能被对象仓库删除"
        adopted.append(data["path"])
        print(f"ok: VIDEO 直通（原字节保留、不重编码、原片保留）")

        # 编码路径仍然工作（frames 形态）
        oid2 = store.put("VIDEO", {"frames": ["f0", "f1"], "fps": 16})
        assert encoded, "frames 形态必须走编码器"
        assert store.get(oid2, "VIDEO")["data"]["num_frames"] == 2
        print("ok: frames 形态仍走编码器（未回归）")

        # 清空仓库时删除对象文件（直通产物同口径）
        store.clear()
        assert not os.path.exists(adopted[0]), "clear() 应删除直通对象文件"
        print("ok: clear() 删除直通对象文件（TTL 同口径）")

        # 源文件缺失时明确报错
        store2 = ObjectStore(video_encoder=encoder, video_dir=os.path.join(d, "obj2"))
        try:
            store2.put("VIDEO", {"path": os.path.join(d, "ghost.mp4")})
            raise AssertionError("缺源文件应报错")
        except ValueError as e:
            assert "not found" in str(e)
        print("ok: 直通源文件缺失时报错")


def main():
    check_registration()
    check_pure_logic()
    check_video_passthrough()
    print("ALL PASS")


if __name__ == "__main__":
    main()
