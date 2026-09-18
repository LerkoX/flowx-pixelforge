"""采样实时预览：latent → RGB 近似投影（对齐 ComfyUI Latent2RGBPreviewer）+ 帧缓冲。

预览帧由 sample 算子在采样循环中逐（N）步产生，写入内存缓冲（按 job_id  keyed，
只留最新一帧），经 GET /preview/{key} 以 HTTP 二进制提供；FlowX Studio 从节点
stdout 标记（FLOWX_PREVIEW）拿到帧地址后中转拉取给画布——媒体全程 HTTP，
不走 base64，服务端也不再向 Studio 推送回调。

投影矩阵来自 ComfyUI comfy/latent_formats.py（SD15 / SDXL）；
本模块不感知对象仓库与算子注册表。
"""
import io
import threading
import time

from PIL import Image

# torch 延迟导入：仅 latent_to_jpeg 的投影矩阵需要；
# progress_card / PreviewRecorder.push_pil 在无 GPU/torch 环境也可用。

# ComfyUI latent_rgb_factors：# R G B
SD15_FACTORS = [
    [0.3512, 0.2297, 0.3227],
    [0.3250, 0.4974, 0.2350],
    [-0.2829, 0.1762, 0.2721],
    [-0.2120, -0.2616, -0.7177],
]
SDXL_FACTORS = [
    [0.3651, 0.4232, 0.4341],
    [-0.2533, -0.0042, 0.1068],
    [0.1076, 0.1111, -0.0362],
    [-0.3165, -0.2492, -0.2188],
]
SDXL_BIAS = [0.1084, -0.0175, -0.0011]


def model_hint_of(pipe) -> str:
    """按管道类名猜测模型族（StableDiffusionXLPipeline → sdxl），决定投影矩阵。"""
    return "sdxl" if "XL" in type(pipe).__name__ else "sd15"


def latent_to_jpeg(latents, hint="sd15", max_size=256, quality=70) -> bytes:
    """latent (b,4,h,w) → JPEG 字节。取 batch 第 0 张，4 通道线性投影到 RGB。"""
    import torch  # 延迟导入
    if hint == "sdxl":
        factors = torch.tensor(SDXL_FACTORS).t()
        bias = torch.tensor(SDXL_BIAS)
    else:
        factors = torch.tensor(SD15_FACTORS).t()
        bias = None
    x0 = latents[0]
    factors = factors.to(dtype=x0.dtype, device=x0.device)
    if bias is not None:
        bias = bias.to(dtype=x0.dtype, device=x0.device)
    # (4,h,w) → (h,w,4) → (h,w,3)
    rgb = torch.nn.functional.linear(x0.movedim(0, -1), factors, bias=bias)
    rgb = ((rgb + 1.0) / 2.0).clamp(0, 1).mul(0xFF)
    img = Image.fromarray(rgb.to(device="cpu", dtype=torch.uint8).numpy())
    # latent 分辨率是图像的 1/8，放大到最长边 max_size
    w, h = img.size
    scale = max_size / max(w, h)
    if scale > 1:
        img = img.resize((round(w * scale), round(h * scale)), Image.BILINEAR)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def progress_card(step, total, width=256, height=144):
    """进度卡片帧：纯渲染的进度图（不碰模型/显存）。
    视频采样等场景的中间结果是 3D latent，无法廉价投影成图；用本卡片经既有
    预览通道给画布实时反馈（step/total + 进度条）。"""
    from PIL import ImageDraw
    img = Image.new("RGB", (width, height), (15, 17, 26))
    d = ImageDraw.Draw(img)
    pct = min(1.0, step / max(1, total))
    mx, my, bar_h = 16, height - 34, 10
    d.rectangle([mx, my, width - mx, my + bar_h], outline=(80, 90, 120))
    d.rectangle([mx, my, mx + (width - 2 * mx) * pct, my + bar_h],
                fill=(34, 211, 238))
    d.text((mx, 24), f"sampling {step}/{total}", fill=(226, 232, 240))
    d.text((mx, 44), f"{pct * 100:.0f}%", fill=(148, 163, 184))
    return img


class PreviewBuffer:
    """预览帧内存缓冲：key（=job_id，uuid 不可猜）→ 最新一帧 (jpeg, progress, ts)。

    只留每 key 最新帧；写入时顺带淘汰超 TTL 的条目。帧是采样中间产物，
    瞬态数据，不落盘。
    """

    def __init__(self, ttl_seconds=3600):
        self._ttl = ttl_seconds
        self._frames = {}  # key -> (jpeg bytes, progress float, updated_at)
        self._lock = threading.Lock()

    def put(self, key, jpeg: bytes, progress: float):
        now = time.time()
        with self._lock:
            self._frames[key] = (jpeg, progress, now)
            expired = [k for k, (_, _, ts) in self._frames.items()
                       if now - ts > self._ttl]
            for k in expired:
                del self._frames[k]

    def get(self, key):
        """返回 (jpeg, progress)；无帧返回 None。"""
        with self._lock:
            item = self._frames.get(key)
        if item is None:
            return None
        jpeg, progress, _ = item
        return jpeg, progress


BUFFER = PreviewBuffer()


class PreviewRecorder:
    """把采样中间 latent 录制为预览帧写入 BUFFER；编码失败静默忽略（不影响采样）。"""

    def __init__(self, key, every=1, hint="sd15"):
        self.key = key
        self.every = max(1, int(every))
        self.hint = hint

    def want(self, step_index, total) -> bool:
        """当前步（0 起）是否需要录预览：每 every 步 + 最后一步必录。"""
        return step_index % self.every == 0 or step_index == total - 1

    def push(self, latents, progress):
        """录制 latent 预览帧（SD 系 2D latent 的 RGB 近似投影）。"""
        try:
            jpeg = latent_to_jpeg(latents, hint=self.hint)
        except Exception as e:
            print(f"[preview] latent->jpeg failed (ignored): {e}", flush=True)
            return
        BUFFER.put(self.key, jpeg, progress)

    def push_pil(self, img, progress):
        """录制已渲染的 PIL 帧（视频进度卡片/关键帧等）。"""
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=75)
        BUFFER.put(self.key, buf.getvalue(), progress)


def recorder_for(key, every=1, hint="sd15") -> PreviewRecorder:
    """按 key（通常为当前 job_id）构造录制器；构造廉价，可按需逐步调用。"""
    return PreviewRecorder(key, every=every, hint=hint)
