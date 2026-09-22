"""模型管理：checkpoint 常驻显存，**显存预算式** LRU 淘汰；按内容嗅探分派管道类。

淘汰策略（dev-plan §21.3 任务 1）：加载前先按 `torch.cuda.mem_get_info()` 的实时余量
与**待加载模型体积估算**（app.vram，读 safetensors header 精确算，不是文件大小）决定
淘汰谁，`MAX_RESIDENT_MODELS` 降为"条目数兜底上限"；执行中的条目 pin 后跳过淘汰
（引擎在算子执行期间 pin 在用模型，见 app.engine）。

加载路径（ROADMAP §1 纪律）：diffusers 目录 from_pretrained / 单文件 from_single_file，
管道类由 app.sniff 按内容嗅探（SD1.x / SDXL / 视频管道），不再写死 StableDiffusionPipeline。
"""
import os
import gc
import threading
import time

import torch
import diffusers

from . import offload, ops, sniff, vram

MODELS_DIR = os.environ.get("MODELS_DIR", "/models")
LORAS_DIR = os.environ.get("LORAS_DIR", "/loras")
# 条目数兜底上限（<=0 表示不限条目数，只看显存预算）
MAX_RESIDENT = int(os.environ.get("MAX_RESIDENT_MODELS", "2"))
# 显存预算式淘汰（VRAM_BUDGET=0 关闭，退回纯条目数 LRU）
VRAM_BUDGET = os.environ.get("VRAM_BUDGET", "1").lower() not in ("0", "false", "no")
# 预留余量：留给采样激活/中间张量/碎片（8GB 卡上 1024MB 约等于一个 SD1.5 半边）
VRAM_RESERVE_MB = int(os.environ.get("VRAM_RESERVE_MB", "1024"))
# 体积估算放大系数：显存占用 = 权重 + CUDA 上下文/缓冲/碎片
VRAM_LOAD_FACTOR = float(os.environ.get("VRAM_LOAD_FACTOR", "1.25"))
OFFLOAD_MODE = offload.resolve_offload_mode()   # none / model / sequential
QUANTIZATION = offload.resolve_quantization()   # fp8 接口预留，未实现时告警
# M3：图像管道的 VAE 单独转 fp32 防黑图（fp16 解码溢出），代价 ~0.2GB 显存；
# 只作用于 sniff.IMAGE_ARCHS（SD1.x/SDXL），视频管道不动。VAE_FP32=0 关闭。
VAE_FP32 = os.environ.get("VAE_FP32", "1").lower() not in ("0", "false", "no")

# SD3.5（MMDiT 新架构，dev-plan 9.6）：T5-XXL fp16 ~9.5GB，超小内存环境预算，
# 默认弃用（CLIP-L/G 双编码器保底，prompt 理解力打折但管线完整）。
# SD3_USE_T5=1 且内存足够（≥12GB WSL + 错峰/fp8 手段）时再开；
# 节点/算子级可用 use_t5 参数逐次覆盖本默认值（旋钮节点化纪律 #3）。
SD3_CLASS = "StableDiffusion3Pipeline"
SD3_USE_T5 = os.environ.get("SD3_USE_T5", "0").lower() in ("1", "true", "yes")

# 性能旋钮可选值（旋钮节点化：环境变量给默认，算子参数可逐次覆盖）
_DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
_OFFLOAD_MODES = ("none", "model", "sequential")


def _resolve_dtype(dtype):
    """dtype 旋钮：auto/空 → fp16（Pascal 无 bf16 硬件，fp16 是唯一实用默认）。"""
    if dtype in (None, "", "auto"):
        return "fp16"
    if dtype not in _DTYPES:
        raise ValueError(f"unknown dtype '{dtype}' (expect auto/fp16/bf16/fp32)")
    return dtype


def _resolve_offload(mode):
    """offload 旋钮：auto/空 → 进程级 OFFLOAD_MODE 环境变量。"""
    if mode in (None, "", "auto"):
        return OFFLOAD_MODE
    if mode not in _OFFLOAD_MODES:
        raise ValueError(
            f"unknown offload mode '{mode}' (expect auto/none/model/sequential)")
    return mode


def _resolve_use_t5(use_t5):
    """use_t5 旋钮：auto/空 → 进程级 SD3_USE_T5 环境变量（默认弃用）。"""
    if use_t5 in (None, "", "auto"):
        return SD3_USE_T5
    if isinstance(use_t5, bool):
        return use_t5
    s = str(use_t5).lower()
    if s in ("1", "true", "yes", "on"):
        return True
    if s in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"unknown use_t5 '{use_t5}' (expect auto/on/off)")

_EXTENSIONS = (".safetensors", ".ckpt")
_LORA_EXTENSIONS = (".safetensors", ".pt", ".bin")


class ModelManager:
    def __init__(self):
        self._pipes = {}
        self._last_used = {}
        self._archs = {}  # pipe_key -> 管道类名（嗅探结果，供诊断/分派）
        self._adapters = {}  # pipe_key -> set(已加载的 adapter 名)
        self._sizes = {}     # pipe_key -> 估算显存占用（字节，用于预算式淘汰与诊断）
        self._estimate_cache = {}  # (path, dtype) -> 估算字节数
        self._pins = set()   # 执行中在用的 key（淘汰跳过；引擎算子执行期间 pin）
        self._lock = threading.Lock()
        # 淘汰回调 fn(pipe)：装配层注入，用于断开外部引用
        # （如 ObjectStore 中 model/clip/vae 视图持有的管道本体）。
        # 顺序关键：必须先断外部引用，再 gc.collect 才能回收循环引用。
        self.on_evict = None

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

    def _instantiate(self, path, loader, cls_name, dtype=torch.float16,
                     use_t5=False):
        """按嗅探结果实例化管道（CPU 常驻；不做 offload/上卡，由调用方决定）。
        dtype 为加载精度旋钮；use_t5 仅作用于 SD3（False 时弃用 T5-XXL）。"""
        cls = getattr(diffusers, cls_name, None)
        if cls is None:
            raise ValueError(
                f"diffusers has no pipeline class '{cls_name}' "
                f"(sniffed from '{path}'); 请升级 diffusers 或更换模型")
        if loader == "pretrained":
            # fp16 变体探测：组件带 *.fp16.safetensors 时按 variant 加载，
            # 避免 fp32 权重先全量读内存再转半精度（大模型内存直接翻倍）。
            # 仅 fp16 有社区 variant 惯例，其他精度不按 variant 加载。
            variant = None
            if dtype == torch.float16:
                for _root, _dirs, files in os.walk(path):
                    if any(f.endswith(".fp16.safetensors") for f in files):
                        variant = "fp16"
                        break
            kwargs = {"torch_dtype": dtype}
            if variant:
                kwargs["variant"] = variant
            if cls_name == SD3_CLASS and not use_t5:
                # 弃用 T5-XXL（~9.5GB）：从_pretrained 显式置 None 跳过加载，
                # 管道编码时 T5 支路自动输出空 embed（diffusers 官方支持的用法）
                kwargs["text_encoder_3"] = None
                kwargs["tokenizer_3"] = None
                print("[model-manager] SD3: T5-XXL dropped (use_t5=off, "
                      "CLIP-L/G only)", flush=True)
            return cls.from_pretrained(path, **kwargs)
        return cls.from_single_file(path, torch_dtype=dtype,
                                    safety_checker=None, **self._single_file_kwargs(cls_name))

    @staticmethod
    def _single_file_kwargs(cls_name):
        """from_single_file 的额外参数：传 config=本地 diffusers 目录可完全避开
        hub 配置下载（宿主机 huggingface.co 不可达时的离线兜底——不传时 diffusers
        会去 hub 拉 Lykon/dreamshaper-8 之类的默认配置，网络被断即 500）。
        在 MODELS_DIR 下找 _class_name 匹配的 diffusers 目录（model_index.json）；
        找不到则返回空 dict，保持原 hub 行为。"""
        import json
        try:
            entries = sorted(os.listdir(MODELS_DIR))
        except OSError:
            return {}
        for entry in entries:
            d = os.path.join(MODELS_DIR, entry)
            idx = os.path.join(d, "model_index.json")
            if not os.path.isdir(d) or not os.path.isfile(idx):
                continue
            try:
                with open(idx, encoding="utf-8") as f:
                    meta = json.load(f)
            except (OSError, ValueError):
                continue
            if meta.get("_class_name") == cls_name:
                print(f"[model-manager] single-file config <- {d}"
                      f"（离线兜底，免 hub 配置下载）", flush=True)
                return {"config": d}
        return {}

    def _apply_offload(self, pipe, mode=None):
        """按 offload 模式应用显存治理；mode=None 时用进程级 OFFLOAD_MODE；
        none 时直接上卡。"""
        mode = mode or OFFLOAD_MODE
        if mode == "sequential":
            # 逐层搬移（最省显存，吞吐最低）；调用后不得再 pipe.to("cuda")
            pipe.enable_sequential_cpu_offload()
        elif mode == "model":
            pipe.enable_model_cpu_offload()
        else:
            pipe = pipe.to("cuda")
        return pipe

    def load(self, name: str, dtype="auto", offload="auto", use_t5="auto"):
        """加载模型到显存（幂等：已常驻则直接返回）。返回 (model_id, newly_loaded)。
        dtype/offload/use_t5 为性能旋钮（auto 继承进程级环境变量）；全默认时缓存键
        即模型名（历史行为），任一显式指定则缓存键扩展为 '名字|dtype=..|offload=..|t5=..'，
        同模型不同旋钮 = 独立常驻条目（同一 LRU 管理）。"""
        path, key = self.resolve(name)
        dt = _resolve_dtype(dtype)
        om = _resolve_offload(offload)
        t5 = _resolve_use_t5(use_t5)
        if dtype not in (None, "", "auto") or offload not in (None, "", "auto") \
                or use_t5 not in (None, "", "auto"):
            key = f"{key}|dtype={dt}|offload={om}|t5={int(t5)}"
        with self._lock:
            if key in self._pipes:
                self._last_used[key] = time.time()
                return key, False
            est = self.estimate(path, dt)
            self._evict_if_needed(est)
            t0 = time.time()
            loader, cls_name = sniff.sniff_arch(path)
            pipe = self._instantiate(path, loader, cls_name,
                                     dtype=_DTYPES[dt], use_t5=t5)
            if VAE_FP32 and cls_name in sniff.IMAGE_ARCHS:
                # 须在 offload 启用前转 dtype（offload 钩子只搬移不转精度）
                pipe.vae.to(torch.float32)
                print("[model-manager] vae -> fp32 (VAE_FP32=1, 防黑图)", flush=True)
            pipe = self._apply_offload(pipe, mode=om)
            if QUANTIZATION != "none":
                print(f"[model-manager] WARNING: QUANTIZATION={QUANTIZATION} "
                      f"接口已预留，当前版本未实现，按原 dtype 加载", flush=True)
            pipe.set_progress_bar_config(disable=True)
            self._pipes[key] = pipe
            self._archs[key] = cls_name
            self._sizes[key] = est
            self._last_used[key] = time.time()
            print(f"[model-manager] loaded '{key}' ({cls_name}) in {time.time()-t0:.1f}s",
                  flush=True)
            return key, True

    def load_vae(self, name: str, dtype="auto"):
        """单独加载 VAE 组件（Load VAE，对标 ComfyUI VAELoader）。
        外挂 VAE（vae-ft-mse 等）的意义是解码质量，故 auto 默认 fp32（非 fp16）；
        dtype 旋钮可显式覆盖（缓存键随 dtype 区分）。
        返回 (key, newly_loaded)；VAE 对象本体经 get(key) 取（VAEShim 管道视图，
        可直接喂 ops.vae_decode / vae_encode）。与管道进同一 LRU 常驻管理。"""
        path, key = self.resolve(name)
        dt = "fp32" if dtype in (None, "", "auto") else _resolve_dtype(dtype)
        key = f"vae:{key}|dtype={dt}"
        with self._lock:
            if key in self._pipes:
                self._last_used[key] = time.time()
                return key, False
            kind, loader = sniff.sniff_component(path)
            if kind != "vae":
                raise ValueError(f"'{name}' 嗅探为 {kind} 组件，vae.load 只支持 vae")
            est = self.estimate(path, dt)
            self._evict_if_needed(est)
            t0 = time.time()
            from diffusers import AutoencoderKL
            if loader == "pretrained":
                vae = AutoencoderKL.from_pretrained(path, torch_dtype=_DTYPES[dt])
            else:
                vae = AutoencoderKL.from_single_file(path, torch_dtype=_DTYPES[dt])
            if torch.cuda.is_available():
                vae = vae.to("cuda")  # 组件不做 offload（fp32 也只 ~670MB）
            shim = ops.VAEShim(vae)
            self._pipes[key] = shim
            self._archs[key] = f"AutoencoderKL({loader})"
            self._sizes[key] = est
            self._last_used[key] = time.time()
            print(f"[model-manager] loaded vae '{key}' in {time.time()-t0:.1f}s",
                  flush=True)
            return key, True

    def load_controlnet(self, name: str, dtype="auto"):
        """单独加载 ControlNet 组件（ControlNet Loader，对标 ComfyUI ControlNetLoader）。
        auto 默认 fp16（与底模计算精度对齐，残差注人 unet 前向）；
        组件不做 offload（fp16 ~0.72GB），常驻进同一 LRU（key 前缀 cn:）。
        返回 (key, newly_loaded)；ControlNetModel 本体经 get(key) 取，
        交 ops.controlnet_apply 捆绑 hint 图后在 sample 循环内逐步前向。"""
        path, key = self.resolve(name)
        dt = "fp16" if dtype in (None, "", "auto") else _resolve_dtype(dtype)
        key = f"cn:{key}|dtype={dt}"
        with self._lock:
            if key in self._pipes:
                self._last_used[key] = time.time()
                return key, False
            kind, loader = sniff.sniff_component(path)
            if kind != "controlnet":
                raise ValueError(
                    f"'{name}' 嗅探为 {kind} 组件，controlnet.load 只支持 controlnet")
            est = self.estimate(path, dt)
            self._evict_if_needed(est)
            t0 = time.time()
            from diffusers import ControlNetModel
            if loader == "pretrained":
                cn = ControlNetModel.from_pretrained(path, torch_dtype=_DTYPES[dt])
            else:
                # from_single_file 内置 lllyasviel 原版（control_v11*）转换
                cn = ControlNetModel.from_single_file(path, torch_dtype=_DTYPES[dt])
            if torch.cuda.is_available():
                cn = cn.to("cuda")
            self._pipes[key] = cn
            self._archs[key] = f"ControlNetModel({loader})"
            self._sizes[key] = est
            self._last_used[key] = time.time()
            print(f"[model-manager] loaded controlnet '{key}' "
                  f"in {time.time()-t0:.1f}s", flush=True)
            return key, True

    def get(self, name: str):
        if name in self._pipes:  # 组合键（base+motion:x）不是文件路径，命中缓存直接返回
            self._last_used[name] = time.time()
            return self._pipes[name]
        key, _ = self.load(name)
        self._last_used[key] = time.time()
        return self._pipes[key]

    def key_of(self, pipe) -> str:
        """按对象身份反查模型 key。"""
        for k, p in self._pipes.items():
            if p is pipe:
                return k
        raise KeyError("pipe is not managed by ModelManager")

    def resolve_motion(self, name: str):
        """在 MODELS_DIR/motion/ 下定位 MotionAdapter（diffusers 目录或 safetensors 单文件）。"""
        mdir = os.path.join(MODELS_DIR, "motion")
        d = os.path.join(mdir, name)
        if os.path.isdir(d):
            return d
        for ext in (".safetensors", ".ckpt"):
            candidates = [name] if name.endswith(ext) else [name + ext]
            for c in candidates:
                p = os.path.join(mdir, c)
                if os.path.isfile(p):
                    return p
        available = sorted(os.listdir(mdir)) if os.path.isdir(mdir) else []
        raise FileNotFoundError(
            f"motion adapter '{name}' not found in {mdir}; available: {available or '(empty)'}")

    def load_motion(self, base_key: str, motion: str):
        """SD1.x checkpoint + MotionAdapter 组合 → AnimateDiffPipeline。
        组合管以独立缓存键 '{base}+motion:{motion}' 常驻，参与同一 LRU。
        返回 (key, newly_loaded)。

        注意：不复用常驻基础管的组件——sequential offload 下基础管权重在 meta
        设备上，UNetMotionModel.from_unet2d 复制会报 'Cannot copy out of meta
        tensor'；且两个管道的 offload 钩子不能挂同一组件。故从磁盘全新实例化
        CPU 管道再组合（代价 ~1-2 分钟加载）。"""
        key = f"{base_key}+motion:{motion}"
        with self._lock:
            if key in self._pipes:
                self._last_used[key] = time.time()
                return key, False
            path, _ = self.resolve(base_key)
            loader, cls_name = sniff.sniff_arch(path)
            if cls_name not in sniff.IMAGE_ARCHS:
                raise ValueError(
                    f"motion.load 只支持 SD1.x 图像底模（嗅探为 {cls_name}）；"
                    f"AnimateDiff v1.5 系运动模块不兼容其他架构")
            mpath = self.resolve_motion(motion)
            # 组合管体积 ≈ 底模 + 运动模块（连同旧底模一起算，避免"底模+组合管"双份）
            est = self.estimate(path, "fp16") + self.estimate(mpath, "fp16")
            self._evict_if_needed(est)  # 组合前腾地方（显存预算 + 条目上限）
            gc.collect()
            torch.cuda.empty_cache()
            t0 = time.time()
            from diffusers import AnimateDiffPipeline, MotionAdapter
            base = self._instantiate(path, loader, cls_name)  # CPU 实例
            if os.path.isdir(mpath):
                try:
                    adapter = MotionAdapter.from_pretrained(
                        mpath, variant="fp16", torch_dtype=torch.float16)
                except Exception as e:
                    print(f"[model-manager] motion fp16 variant 不可用（{e}），回退默认权重", flush=True)
                    adapter = MotionAdapter.from_pretrained(mpath, torch_dtype=torch.float16)
            else:
                adapter = MotionAdapter.from_single_file(mpath, torch_dtype=torch.float16)
            # AnimateDiffPipeline 无 from_single_file（0.30.3），但构造函数接受普通
            # UNet2DConditionModel 并自动 UNetMotionModel.from_unet2d 转换（复制 UNet 权重 ~0.9GB）
            pipe = AnimateDiffPipeline(
                vae=base.vae, text_encoder=base.text_encoder, tokenizer=base.tokenizer,
                unet=base.unet, motion_adapter=adapter, scheduler=base.scheduler,
                feature_extractor=getattr(base, "feature_extractor", None),
                image_encoder=None)
            del base  # 组件已移交组合管，底模外壳不再需要
            gc.collect()
            # AnimateDiff 社区标准调度器：DDIM + clip_sample=False + linspace 排布；
            # 沿用底模自带的 PNDM 默认配置会出"模糊闪动无内容"的废片（实测复现）
            from diffusers import DDIMScheduler
            sched_cfg = dict(pipe.scheduler.config)
            sched_cfg.update(clip_sample=False, timestep_spacing="linspace",
                             beta_schedule="linear")
            pipe.scheduler = DDIMScheduler.from_config(sched_cfg)
            if VAE_FP32:
                # 组合管不走 load()，漏掉了图像管道的 VAE fp32 防黑图处理——
                # Pascal 上 fp16 VAE 解码出泥浆噪声废片（exec 191/195 实测）。
                # 须在 offload 启用前转 dtype（钩子只搬移不转精度）。
                # 注意 decode_latents 直接把 fp16 latent 传给 vae.decode，dtype
                # 不匹配会 RuntimeError，故包一层统一转 fp32（ops.vae_decode 已
                # 按 vae.dtype 自适应，无需改动；preview 走 latent 投影不经过 VAE）。
                pipe.vae.to(torch.float32)
                _orig_decode = pipe.vae.decode

                def _decode_fp32(z, *args, **kwargs):
                    return _orig_decode(z.to(torch.float32), *args, **kwargs)

                pipe.vae.decode = _decode_fp32
                print("[model-manager] composed vae -> fp32 (VAE_FP32=1, 防黑图)",
                      flush=True)
            pipe = self._apply_offload(pipe)
            pipe.set_progress_bar_config(disable=True)
            self._pipes[key] = pipe
            self._archs[key] = "AnimateDiffPipeline"
            self._sizes[key] = est
            self._last_used[key] = time.time()
            print(f"[model-manager] composed '{key}' (AnimateDiffPipeline) "
                  f"in {time.time()-t0:.1f}s", flush=True)
            return key, True

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

    def estimate(self, key_or_path, dtype="fp16"):
        """估算路径的显存占用（字节，带缓存）；估不出来返回 0（只靠条目上限兜底）。"""
        cache_key = (key_or_path, dtype)
        if cache_key not in self._estimate_cache:
            got = vram.estimate_bytes(key_or_path, dtype)
            if got is None:
                print(f"[model-manager] WARNING: 无法估算 '{key_or_path}' 体积"
                      f"（按 0 计，只靠 MAX_RESIDENT 兜底）", flush=True)
            self._estimate_cache[cache_key] = int(got or 0)
        return self._estimate_cache[cache_key]

    @staticmethod
    def _free_bytes():
        """可用显存（字节）：驱动余量 + 本进程缓存分配器中**可回收**的部分。

        为什么不能直接用 `mem_get_info`（实测教训）：PyTorch 缓存分配器采样后会把
        几 GB 留在池里复用，`mem_get_info` 只报驱动余量 ⇒ 采样一轮后 free=0，于是
        "下一个模型"被判定没空间，白白踢掉刚加载的模型（重载一次 ~2 分钟）。
        可回收量 = reserved - allocated（缓存池里没被张量占用的部分），淘汰路径里的
        `empty_cache()` 会把它还给驱动。无 CUDA 返回 None。
        """
        if not torch.cuda.is_available():
            return None
        try:
            free, _total = torch.cuda.mem_get_info()
        except Exception:  # pragma: no cover - 驱动异常时退化为纯条目淘汰
            return None
        cached = 0
        try:
            cached = max(0, torch.cuda.memory_reserved(0) - torch.cuda.memory_allocated(0))
        except Exception:
            pass
        return int(free + cached)

    def resident_weights(self) -> int:
        """当前常驻模型的权重总量估算（字节）。"""
        return sum(int(self._sizes.get(k, 0)) for k in self._pipes)

    def pin(self, *keys):
        """标记 key 为"执行中在用"：淘汰跳过（引擎在算子执行期间调用）。"""
        for k in keys:
            if k:
                self._pins.add(k)

    def unpin(self, *keys):
        for k in keys:
            self._pins.discard(k)

    def _do_evict(self, victim):
        """淘汰单个条目（含断外部引用 → gc → empty_cache 的既有顺序纪律）。"""
        pipe = self._pipes.pop(victim, None)
        if pipe is None:
            return False
        size = self._sizes.pop(victim, 0)
        self._last_used.pop(victim, None)
        self._archs.pop(victim, None)
        self._adapters.pop(victim, None)
        self._pins.discard(victim)
        print(f"[model-manager] evicting '{victim}' (LRU, ~{size / 1024**2:.0f}MB)", flush=True)
        # 先断开外部引用（ObjectStore 中的 model/clip/vae 视图等）：
        # 不断引用直接 GC 收不掉；引用计数为 0 但组件/钩子互相引用形成
        # 循环时也必须 gc.collect（sequential offload 下 CPU 权重可达数 GB）。
        if self.on_evict is not None:
            try:
                self.on_evict(pipe)
            except Exception as e:
                print(f"[model-manager] on_evict failed (ignored): {e}", flush=True)
        del pipe
        gc.collect()
        torch.cuda.empty_cache()
        return True

    def _evict_if_needed(self, need_bytes=0):
        """加载前腾地方：条目数上限 + 显存预算（app.vram.plan_eviction 决策）。

        need_bytes=0 时只看条目数（如 idle 场景）；>0 时按"余量 + 淘汰体积 ≥ 需求
        + reserve"判定，并把估算按 VRAM_LOAD_FACTOR 放大。
        """
        if not self._pipes:
            return
        need = int(need_bytes * VRAM_LOAD_FACTOR) if need_bytes else 0
        free = self._free_bytes() if (VRAM_BUDGET and need) else None
        plan = vram.plan_eviction(
            [(k, self._sizes.get(k, 0), self._last_used.get(k, 0)) for k in self._pipes],
            free_bytes=free or 0,
            need_bytes=need if free is not None else 0,
            reserve_bytes=VRAM_RESERVE_MB * 1024**2 if free is not None else 0,
            max_resident=MAX_RESIDENT,
            pinned=self._pins)
        for victim in plan:
            self._do_evict(victim)
        if free is not None:
            after = self._free_bytes()
            need_total = need + VRAM_RESERVE_MB * 1024**2
            if after is not None and after < need_total:
                print(f"[model-manager] WARNING: 余量仍不足（free={after / 1024**2:.0f}MB "
                      f"< need+reserve={need_total / 1024**2:.0f}MB）："
                      f"可淘汰条目已用尽或体积未知（pin={sorted(self._pins)}）", flush=True)

    # ---------- 显式卸载（dev-plan §21.3 任务 2） ----------

    def find_keys(self, target):
        """按名字/缓存键定位常驻条目。精确键 > 解析到同一文件 > 名字后缀（vae:/cn:）
        > 模糊包含（大小写不敏感）。返回匹配的 key 列表（可能为空）。"""
        target = str(target or "").strip()
        if not target:
            return []
        if target in self._pipes:
            return [target]
        want_path = None
        try:
            want_path, _ = self.resolve(target)
        except Exception:
            want_path = None
        hits = []
        for key in self._pipes:
            stem = key.split("|", 1)[0]
            if stem == target:
                hits.append(key)
                continue
            if stem in (f"vae:{target}", f"cn:{target}"):
                hits.append(key)
                continue
            if want_path is not None:
                bare = stem.split(":", 1)[1] if stem.startswith(("vae:", "cn:")) else stem
                bare = bare.split("+motion:", 1)[0]
                try:
                    if self.resolve(bare)[0] == want_path:
                        hits.append(key)
                        continue
                except Exception:
                    pass
            if target.lower() in key.lower():
                hits.append(key)
        return hits

    def unload(self, target) -> list:
        """显式卸载：target 为空/all/* = 全部；否则按 find_keys 匹配。
        走与其他加载完全一致的淘汰路径（断对象仓库视图 → gc → empty_cache）。
        返回实际卸载的 key 列表。"""
        with self._lock:
            if str(target or "").strip().lower() in ("", "all", "*"):
                keys = list(self._pipes.keys())
            else:
                keys = self.find_keys(target)
            # 执行中在用（pin）的条目不动：卸载正在被别的 job 使用的模型＝抽走它脚下的地板
            busy = [k for k in keys if k in self._pins]
            if busy:
                print(f"[model-manager] skip in-use (pinned): {busy}", flush=True)
            done = [k for k in keys if k not in self._pins and self._do_evict(k)]
            if done:
                print(f"[model-manager] unloaded {done}；resident={self.resident()}",
                      flush=True)
            return done

    def evict_idle(self, limit=1) -> int:
        """淘汰至多 limit 个**非在用**（未 pin）的最久未用条目；返回实际个数。
        OOM 自愈用（dev-plan §21.3 任务 3）：绝不动 pin 住的模型。"""
        with self._lock:
            idle = sorted((k for k in self._pipes if k not in self._pins),
                          key=lambda k: self._last_used.get(k, 0))
            done = 0
            for key in idle[:max(0, int(limit))]:
                if self._do_evict(key):
                    done += 1
            return done

    def resident(self):
        return sorted(self._pipes.keys())

    def details(self):
        """常驻条目明细（/health 诊断：体积估算/最近使用/是否在用）。"""
        now = time.time()
        out = []
        for key in sorted(self._pipes):
            out.append({
                "key": key,
                "arch": self._archs.get(key),
                "est_vram_mb": round(self._sizes.get(key, 0) / 1024**2),
                "idle_seconds": round(now - self._last_used.get(key, now), 1),
                "pinned": key in self._pins,
            })
        return out


class VramGuard:
    """运行期显存护栏（引擎注入）：算子执行期间 pin 在用模型，OOM 时淘汰非在用条目。

    引擎不直接依赖 ModelManager：只要对象有 pin_objects/unpin/evict_idle 三个方法
    （鸭子类型），便于单测注入假护栏。
    """

    def __init__(self, manager):
        self.manager = manager

    def pin_objects(self, objects) -> list:
        """把输入对象里能认出是常驻模型的都 pin 上，返回 keys（供 unpin）。"""
        keys = []
        for obj in objects or ():
            pipe = getattr(obj, "pipe", obj)  # ModelRef → 底层管道
            try:
                key = self.manager.key_of(pipe)
            except KeyError:
                continue
            if key not in keys:
                keys.append(key)
        self.manager.pin(*keys)
        return keys

    def unpin(self, keys):
        self.manager.unpin(*(keys or ()))

    def evict_for_oom(self, limit=1) -> int:
        return self.manager.evict_idle(limit)


class OomError(RuntimeError):
    """显存不足且已无可淘汰条目：信息里带可执行的自救指引。"""


def oom_help(op_name, detail=""):
    return (f"CUDA out of memory in op '{op_name}'：已淘汰全部可淘汰的非在用模型仍不足。"
            f"建议：① 降低分辨率/步数或 batch；② 加载时用 offload=model|sequential；"
            f"③ 调小 VRAM_RESERVE_MB 或提高 VRAM_BUDGET；"
            f"④ 卸载其它常驻模型（POST /model/unload 或 model-unload 节点）后重试。{detail}")
