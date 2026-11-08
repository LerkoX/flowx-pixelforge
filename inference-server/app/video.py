"""VIDEO 对象编码：帧序列（+ 可选音轨）→ mp4 落盘。

VIDEO 进程内契约：{"frames": [PIL.Image, ...], "fps": int,
可选 "audio": torch.Tensor(1|2, C, N)/(C, N) 或 array、"sample_rate": int}
（video.sample 产出无声契约；minimax.h3.gen 产出带 32kHz 立体声契约）；
落盘后仓库内形态：{"path": ..., "fps": int, "num_frames": int,
有声时加 "sample_rate": int}（供 GET /videos/{id} 下载）。

imageio-ffmpeg 自带静态 ffmpeg 二进制，容器内无需 apt 安装 ffmpeg；
imageio 延迟导入——未安装时仅视频落盘不可用，其余功能不受影响。
音轨 mux：imageio 先写无声 mp4 + 标准库 wave 写 PCM16 临时 wav
（零新增依赖）+ ffmpeg 合成 AAC（copy 视频流不重编码）。
"""
import os


def _to_pcm16(audio):
    """音频张量/数组 → int16 PCM 数组（channels, samples）。

    接受 torch.Tensor / numpy（float [-1,1] 或 int16），shape 兼容
    (1,C,N)（H3 立体声契约，batch=1）与 (C,N)。返回 (pcm_int16, channels)。
    """
    import numpy as np

    t = audio
    if hasattr(t, "detach"):  # torch.Tensor
        t = t.detach().float().cpu().numpy()
    t = np.asarray(t)
    if t.ndim == 3:  # (1, C, N) → (C, N)
        t = t[0]
    if t.ndim == 1:  # 单声道一维
        t = t[None, :]
    if t.dtype != np.int16:
        t = (np.clip(t, -1.0, 1.0) * 32767).astype(np.int16)
    return t, int(t.shape[0])


def _write_wav(pcm, channels, sample_rate, path):
    """PCM16 (C, N) 写 wav（标准库 wave；wave 要 (N*C,) 交错布局）。"""
    import wave

    import numpy as np

    interleaved = np.ascontiguousarray(pcm.T)  # (N, C) 交错
    with wave.open(path, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(int(sample_rate))
        w.writeframes(interleaved.tobytes())


def _mux_audio(video_path, wav_path, out_path):
    """ffmpeg 把 wav 合进 mp4（视频流 copy 不重编码，音频 AAC 128k）。
    音频比视频短时按短视频截断（-shortest）；imageio-ffmpeg 的静态二进制。"""
    import subprocess

    import imageio_ffmpeg  # 延迟导入（与 imageio 同生命周期）

    exe = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [exe, "-y", "-i", video_path, "-i", wav_path,
           "-c:v", "copy", "-c:a", "aac", "-b:a", "128k",
           "-shortest", out_path]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg mux 失败（{proc.returncode}）："
                           f"{proc.stderr[-500:]}")


def encode_mp4(frames, fps, path, audio=None, sample_rate=None):
    """PIL 帧序列（+ 可选音轨）编码为 H.264 mp4，返回仓库内形态 dict。

    audio 非空时：先写无声 mp4 → PCM16 临时 wav → ffmpeg mux AAC 后替换原文件。"""
    import imageio.v2 as imageio  # 延迟导入
    import numpy as np

    if not frames:
        raise ValueError("VIDEO object has no frames")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    arrays = [np.asarray(f.convert("RGB")) for f in frames]
    # macro_block_size=None：保持原始分辨率（H.264 要求偶数，imageio 自动 pad 1px）
    imageio.mimsave(path, arrays, fps=fps, codec="libx264", quality=8,
                    macro_block_size=None)
    meta = {"path": path, "fps": int(fps), "num_frames": len(arrays)}

    if audio is not None and sample_rate:
        pcm, channels = _to_pcm16(audio)
        if pcm.shape[-1] > 0:
            wav_path = path + ".audio.wav"
            mux_path = path + ".mux.mp4"
            try:
                _write_wav(pcm, channels, sample_rate, wav_path)
                _mux_audio(path, wav_path, mux_path)
                os.replace(mux_path, path)
                meta["sample_rate"] = int(sample_rate)
            finally:
                for tmp in (wav_path, mux_path):
                    try:
                        os.remove(tmp)
                    except OSError:
                        pass
    return meta
