"""模型管理：checkpoint 常驻显存，LRU 淘汰。"""
import os
import threading
import time

import torch
from diffusers import StableDiffusionPipeline

MODELS_DIR = os.environ.get("MODELS_DIR", "/models")
LORAS_DIR = os.environ.get("LORAS_DIR", "/loras")
MAX_RESIDENT = int(os.environ.get("MAX_RESIDENT_MODELS", "2"))
CPU_OFFLOAD = os.environ.get("ENABLE_CPU_OFFLOAD", "0") == "1"

_EXTENSIONS = (".safetensors", ".ckpt")
_LORA_EXTENSIONS = (".safetensors", ".pt", ".bin")


class ModelManager:
    def __init__(self):
        self._pipes = {}
        self._last_used = {}
        self._adapters = {}  # pipe_key -> set(已加载的 adapter 名)
        self._lock = threading.Lock()

    def resolve(self, name: str):
        """按名称在 MODELS_DIR 中定位模型文件，可省略扩展名。"""
        candidates = [name] if name.endswith(_EXTENSIONS) else [name + e for e in _EXTENSIONS]
        candidates.append(name)
        for c in candidates:
            p = os.path.join(MODELS_DIR, c)
            if os.path.isfile(p):
                return p, os.path.splitext(os.path.basename(p))[0]
        available = [f for f in os.listdir(MODELS_DIR) if f.endswith(_EXTENSIONS)] if os.path.isdir(MODELS_DIR) else []
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
            pipe = StableDiffusionPipeline.from_single_file(
                path, torch_dtype=torch.float16, safety_checker=None
            )
            if CPU_OFFLOAD:
                pipe.enable_model_cpu_offload()
            else:
                pipe = pipe.to("cuda")
            pipe.set_progress_bar_config(disable=True)
            self._pipes[key] = pipe
            self._last_used[key] = time.time()
            print(f"[model-manager] loaded '{key}' in {time.time()-t0:.1f}s", flush=True)
            return key, True

    def get(self, name: str) -> StableDiffusionPipeline:
        key, _ = self.load(name)
        self._last_used[key] = time.time()
        return self._pipes[key]

    def key_of(self, pipe) -> str:
        """按对象身份反查模型 key。"""
        for k, p in self._pipes.items():
            if p is pipe:
                return k
        raise KeyError("pipe is not managed by ModelManager")

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
            self._adapters.pop(victim, None)
            torch.cuda.empty_cache()

    def resident(self):
        return sorted(self._pipes.keys())
