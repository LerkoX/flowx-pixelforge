"""VIDEO 对象编码：帧序列 → mp4 落盘。

VIDEO 进程内契约：{"frames": [PIL.Image, ...], "fps": int}（video.sample 等算子产出）；
落盘后仓库内形态：{"path": ..., "fps": int, "num_frames": int}（供 GET /videos/{id} 下载）。

imageio-ffmpeg 自带静态 ffmpeg 二进制，容器内无需 apt 安装 ffmpeg；
imageio 延迟导入——未安装时仅视频落盘不可用，其余功能不受影响。
"""
import os


def encode_mp4(frames, fps, path):
    """PIL 帧序列编码为 H.264 mp4，返回仓库内形态 dict。"""
    import imageio.v2 as imageio  # 延迟导入
    import numpy as np

    if not frames:
        raise ValueError("VIDEO object has no frames")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    arrays = [np.asarray(f.convert("RGB")) for f in frames]
    # macro_block_size=None：保持原始分辨率（H.264 要求偶数，imageio 自动 pad 1px）
    imageio.mimsave(path, arrays, fps=fps, codec="libx264", quality=8,
                    macro_block_size=None)
    return {"path": path, "fps": int(fps), "num_frames": len(arrays)}
