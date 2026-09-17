"""模型管理：checkpoint 常驻显存，LRU 淘汰；按内容嗅探分派管道类。

加载路径（ROADMAP §1 纪律）：diffusers 目录 from_pretrained / 单文件 from_single_file，
管道类由 app.sniff 按内容嗅探（SD1.x / SDXL / 视频管道），不再写死 StableDiffusionPipeline。
"""
import os
import threading
import time

import torch
import diffusers

from . import offload, sniff

MODELS_DIR = os.environ.get("MODELS_DIR", "/models")
LORAS_DIR = os.environ.get("LORAS_DIR", "/loras")
MAX_RESIDENT = int(os.environ.get("MAX_RESIDENT_MODELS", "2"))
OFFLOAD_MODE = offload.resolve_offload_mode()   # none / model / sequential
QUANTIZATION = offload.resolve_quantization()   # fp8 接口预留，未实现时告警
# M3：图像管道的 VAE 单独转 fp32 防黑图（fp16 解码溢出），代价 ~0.2GB 显存；
# 只作用于 sniff.IMAGE_ARCHS（SD1.x/SDXL），视频管道不动。VAE_FP32=0 关闭。
VAE_FP32 = os.environ.get("VAE_FP32", "1").lower() not in ("0", "false", "no")

_EXTENSIONS = (".safetensors", ".ckpt")
_LORA_EXTENSIONS = (".safetensors", ".pt", ".bin")


class ModelManager:
    def __init__(self):
        self._pipes = {}
        self._last_used = {}
        self._archs = {}  # pipe_key -> 管道类名（嗅探结果，供诊断/分派）
        self._adapters = {}  # pipe_key -> set(已加载的 adapter 名)
        self._lock = threading.Lock()

    def resolve(self, name: str):
        """按名称在 MODELS_DIR 中定位模型：diffusers 目录 或 safetensors/ckpt 文件
        （可省略扩展名）。返回 (路径, key)。"""
        d = os.path.join(MODELS_DIR, name)
        if os.path.isdir(d):  # diffusers 目录格式（from_pretrained）
            return d, name
        candidates = [name] if name.endswith(_EXTENSIONS) else [name + e for e in _EXTENSIONS]
        candidates.append(name)
        for c in candidates:
            p = os.path.join(MODELS_DIR, c)
            if os.path.isfile(p):
                return p, os.path.splitext(os.path.basename(p))[0]
        available = []
        if os.path.isdir(MODELS_DIR):
            available = sorted(
                [f for f in os.listdir(MODELS_DIR) if f.endswith(_EXTENSIONS)]
                + [f + "/" for f in os.listdir(MODELS_DIR)
                   if os.path.isdir(os.path.join(MODELS_DIR, f))])
        raise FileNotFoundError(
            f"model '{name}' not found in {MODELS_DIR}; available: {available or '(empty)'}"
        )

    def load(self, name: str):
        """加载模型到显存（幂等：已常驻则直接返回）。返回 (model_id, newly_loaded)。"""
        path, key = self.resolve(name)
        with self._lock:
            if key in self._pipes:
                self._last_used[key] = time.time()
                return key, False
            self._evict_if_needed()
            t0 = time.time()
            loader, cls_name = sniff.sniff_arch(path)
            cls = getattr(diffusers, cls_name, None)
            if cls is None:
                raise ValueError(
                    f"diffusers has no pipeline class '{cls_name}' "
                    f"(sniffed from '{path}'); 请升级 diffusers 或更换模型")
            if loader == "pretrained":
                # fp16 变体探测：组件带 *.fp16.safetensors 时按 variant 加载，
                # 避免 fp32 权重先全量读内存再转半精度（大模型内存直接翻倍）
                variant = None
                for _root, _dirs, files in os.walk(path):
                    if any(f.endswith(".fp16.safetensors") for f in files):
                        variant = "fp16"
                        break
                kwargs = {"torch_dtype": torch.float16}
                if variant:
                    kwargs["variant"] = variant
                pipe = cls.from_pretrained(path, **kwargs)
            else:
                pipe = cls.from_single_file(path, torch_dtype=torch.float16,
                                            safety_checker=None)
            if VAE_FP32 and cls_name in sniff.IMAGE_ARCHS:
                # 须在 offload 启用前转 dtype（offload 钩子只搬移不转精度）
                pipe.vae.to(torch.float32)
                print("[model-manager] vae -> fp32 (VAE_FP32=1, 防黑图)", flush=True)
            if OFFLOAD_MODE == "sequential":
                # 逐层搬移（最省显存，吞吐最低）；调用后不得再 pipe.to("cuda")
                pipe.enable_sequential_cpu_offload()
            elif OFFLOAD_MODE == "model":
                pipe.enable_model_cpu_offload()
            else:
                pipe = pipe.to("cuda")
            if QUANTIZATION != "none":
                print(f"[model-manager] WARNING: QUANTIZATION={QUANTIZATION} "
                      f"接口已预留，当前版本未实现，按原 dtype 加载", flush=True)
            pipe.set_progress_bar_config(disable=True)
            self._pipes[key] = pipe
            self._archs[key] = cls_name
            self._last_used[key] = time.time()
            print(f"[model-manager] loaded '{key}' ({cls_name}) in {time.time()-t0:.1f}s",
                  flush=True)
            return key, True

    def get(self, name: str):
        key, _ = self.load(name)
        self._last_used[key] = time.time()
        return self._pipes[key]

    def key_of(self, pipe) -> str:
        """按对象身份反查模型 key。"""
        for k, p in self._pipes.items():
            if p is pipe:
                return k
        raise KeyError("pipe is not managed by ModelManager")

    def arch_of(self, key: str) -> str:
        """模型的管道类名（嗅探分派结果），如 StableDiffusionPipeline / WanPipeline。"""
        return self._archs[key]

    def resolve_lora(self, name: str):
        """按名称在 LORAS_DIR 中定位 LoRA 文件，可省略扩展名。"""
        candidates = [name] if name.endswith(_LORA_EXTENSIONS) else [name + e for e in _LORA_EXTENSIONS]
        candidates.append(name)
        for c in candidates:
            p = os.path.join(LORAS_DIR, c)
            if os.path.isfile(p):
                return p, os.path.splitext(os.path.basename(p))[0]
        available = [f for f in os.listdir(LORAS_DIR) if f.endswith(_LORA_EXTENSIONS)] if os.path.isdir(LORAS_DIR) else []
        raise FileNotFoundError(
            f"lora '{name}' not found in {LORAS_DIR}; available: {available or '(empty)'}"
        )

    def load_lora(self, pipe_key: str, name: str) -> str:
        """把 LoRA 权重加载进指定管道（幂等，按 adapter 名缓存），返回 adapter 名。"""
        path, lora_name = self.resolve_lora(name)
        adapter = lora_name.replace(" ", "_")
        with self._lock:
            loaded = self._adapters.setdefault(pipe_key, set())
            if adapter in loaded:
                return adapter
            pipe = self._pipes[pipe_key]
            t0 = time.time()
            pipe.load_lora_weights(os.path.dirname(path) or ".",
                                   weight_name=os.path.basename(path),
                                   adapter_name=adapter)
            loaded.add(adapter)
            print(f"[model-manager] lora '{adapter}' loaded into '{pipe_key}' "
                  f"in {time.time()-t0:.1f}s", flush=True)
            return adapter

    def _evict_if_needed(self):
        while len(self._pipes) >= MAX_RESIDENT:
            victim = min(self._last_used, key=self._last_used.get)
            print(f"[model-manager] evicting '{victim}' (LRU)", flush=True)
            del self._pipes[victim]
            del self._last_used[victim]
            self._archs.pop(victim, None)
            self._adapters.pop(victim, None)
            torch.cuda.empty_cache()

    def resident(self):
        return sorted(self._pipes.keys())
