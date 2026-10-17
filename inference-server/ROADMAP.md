# 演进路线：对标 ComfyUI 的能力补齐计划

> 定位：本服务是 ComfyUI 执行模型的**远程复刻**（类型化端口 + 对象仓库 + 拓扑调度 + 内容寻址缓存，
> `/graph` 对齐 `/prompt` 语义）。内核已与 ComfyUI 同构，剩余差距是**算子覆盖面**与少量引擎语义。
> 本文档给出差距清单与分阶段演进计划。判断依据：ComfyUI 生态流行度（ControlNet 系、SDXL 系、
> 高清放大链为流量大头）与对本服务现有代码的侵入程度。


### 阶段 10 / B3：分割与分块超分（2026-10-11 实施完成）

- **G2 SAM/GroundingDINO**：`seg.py` 族（5 算子）——`SamModel` 点/框提示（image embedding 缓存 ⇒ 复选 0.08s）、
  自动全图分割（mask-generation pipeline + IoU 去重 + keep 选择）、GroundingDINO 文本检测（`_norm_prompt` 补尾点）、
  文本→SAM 像素级蒙版。模型走 `model_manager` LRU（`sam:<variant>` / `gd:<variant>`）。
- **G4 分块重绘超分**：`tiled.py` 族 —— `tiled.plan`（纯计算切片网格，零 GPU）与 `tiled.apply`
  （逐块 img2img + **权重归一化融合** `out=Σtile·W/ΣW` + 可选接缝带二次重绘 + job 逐块进度），
  复用 `app.ops` 原语（`resolve_pipe`/`vae_encode`/`sample`/`vae_decode`）。
- **服务端 `app/` 零改动**；依赖零新增（不做镜像重建）；8GB Pascal 实测 512×384→1024×768（tile 512/overlap 64，6 块×8 步）通过。
- 未做：SAM2（`transformers==4.44.2` 无 `Sam2Model`，升版需重验 Pascal 全链）、Tiled Diffusion 跨块注意力、
  fp16 SAM（Pascal fp16 吞吐 1/64）。



### 阶段 10 / B4-S1：DDColor 老照片上色（2026-10-12 实施完成）

- 算子族 `colorize.py`（`@b77d44da027e`）：`colorize.load`（4 变体，进 model_manager LRU，键 `ddcolor:<variant>`）
  + `colorize.apply`（保留原图 L 通道只预测 ab，`saturation` 在 Lab 空间缩放色度）。
- **服务端 `app/` 零改动、零新增依赖、不重建推理镜像**；上游 DDColor 实现 945 行逐行内联（惰性工厂，Termux 可测）。
- 真机：保真对照与官方实现**逐字节一致**；L 227.9M/装载 13s/推理 512² 0.23~0.46s/峰值 3.7GB；T 55.0M/3.8s/0.13s/3.0GB。
- 未做：上色结果的自动色彩校正（用既有 `image-curves`/`image-hsl` 串联即可）、批量处理（见 B4/S2 的 D7）。



### 阶段 10 / B4-S2：电商模板 + 水印节点（2026-10-12 实施完成）

- 新节点 `image-watermark`（纯节点侧 PIL）：九宫格定位/不透明度/描边/旋转/平铺/Logo；输出水印 `box` 供复核。
- 3 条模板流水线：白底主图（1:1）/ 场景换底 / 模特图规格化（GFPGAN + RealESRGAN 4x）。
- 复核 `_tools/verify_ecom.py` **12/12**；关键事实：`matting-composite` 的 `feather=8` ≈ 24px 过渡带；
  水印"非水印区逐字节不变"可证。
- 未做（按 D7）：目录批量节点（`image-batch-load/save`）⇒ B5 候选；中文水印需外部 TTF 路径。



### 阶段 10 / B4-S3：IC-Light 重打光（2026-10-12 实施完成）

- 新算子族 `iclight.py`（`iclight.load` / `iclight.relight`）：SD1.5 基座 + IC-Light delta 权重，
  UNet conv_in 扩到 8/12 通道 + 猴子补丁 forward 注入 `concat_conds`（官方 gradio demo 逐行等价）。
  在 **diffusers 0.35.2**（比官方 pin 的 0.27.2 新）下实测完全兼容 ⇒ **零核心改动**。
- 新节点 `iclight-load`(227) / `iclight`(228)；e2e wf 150（local）/ 151（docker）。
- 验收 `_tools/verify_iclight.py` **17/17**：光方向物理翻转（Δ−19.7 → +50.2 / −100.6）、
  同种子逐字节（local↔docker 六份一致）、主体保留（对输出再抠图 IoU 0.9823）。
- 运维：常驻 4.07GB；与 flux 常驻互斥（8GB 卡）；fc/fbc 不要同时装载。
- 未做：G10 PuLID（= B5，需改部署 + 重启服务）。



### 部署记录 / Part A 落地（2026-10-13）

- `flowx-inference-server:pascal` 已重建为含**插件归属簿修复**（§52.1/§52.3）与
  **插件热更新 `.pyc` 失效修复**（§52.7）的镜像；保留回滚标签 `:pascal-prePartA`、`:pascal-partA`。
- 重启后：108 算子 / 46 插件文件 / `missing_file` 可观测 / 无幽灵条目；活体验证 **12/12**。
- 运行时配置（重建时必须照抄）：env `INFERENCE_TOKEN / HF_HOME=/hf-cache / MAX_RESIDENT_MODELS=3 /
  OFFLOAD_MODE=none / PORT=8100 / OBJECT_TTL_SECONDS=14400 / MODELS_DIR=/models`；
  挂载 `D:/flowx-data/{input,loras,models,videos,hf-cache}`；`-p 8100:8100`；`--gpus all`；`restart=always`。
- **2026-10-16 新增（exec 795 事故修复，重建时必须一并照抄）**：
  env `VRAM_STRICT=1`（物理装不下时拒绝加载，而不是掉进 WSL2 的 host memory 兜底
  变成分钟级/步）—— 本项也能热改：`POST /model/policy {"vram_strict":1}`，
  但**容器重启会丢**，所以重建命令里要显式带上；
  可选 `OFFLOAD_VRAM_FLOOR_MB=1024`、`OBJECT_TTL_ORPHAN_SECONDS=1800`。
  诊断看 `/health`（`vram_oversubscribed` / `vram_headroom_mb` / `vram_budget_mb`，
  `?modules=1` 出每个常驻条目的 device 明细）。
  本次部署方式：`docker cp` 8 个 `app/*.py` 进容器 + `restart`
  （容器内旧版备份 `/app/app.bak-20261016T222727Z`；镜像仍为 `pascal-partA4`，未重建镜像）。
- ⚠️ `plugins.d` 里 `vton_3d94ed9961db.py` 声明的算子版本是 `@3bb90bb5788d`（**文件名 hash ≠ 内容 hash**）
  ⇒ `purge-plugin-versions.py` 会把它判成"旧版本"而删除，**别在 vton 家族工作未收尾时跑该工具**。


## 0. 现状盘点（已对齐的部分）

| ComfyUI 概念 | 本服务对应 | 状态 |
| --- | --- | --- |
| 节点 INPUT_TYPES/RETURN_TYPES | `registry.py` 声明式端口类型 | ✅ 同构 |
| 图执行 + 拓扑排序 + 缓存 | `engine.py`（内容寻址缓存） | ✅ 同构 |
| 对象按引用传递（张量不出 GPU） | `object_store.py` UUID 仓库 | ✅ 同构 |
| `/prompt` `/object_info` `/view` | `/graph` `/ops` `/images/{id}` | ✅ 对应 |
| ModelPatcher.clone + add_patches | `ops.ModelRef`（LoRA 补丁视图，可串联） | ✅ 同构 |
| ckpt/safetensors 加载 + key 映射 | diffusers `from_single_file()` | ✅ 免费获得 |
| 采样器 euler/dpmpp/uni_pc 等 7 种 × sigma 排布 4 种自由组合 | `samplers.py`（M2 解耦：sampler_name × scheduler 双参数，旧一体名兼容） | ✅ 已解耦 |
| 模型显存搬移 | `OFFLOAD_MODE=none/model/sequential`（旧 `ENABLE_CPU_OFFLOAD=1` 兼容映射 model） | ✅ 三档可用 |
| dtype 探测/fallback | 硬编码 fp16 | ❌ 未做 |
| SDXL/SD3/Flux 架构 | SD1.x/SDXL 采样/编码路径已通（M5，2026-09-22）；SD3.5 走独立插件链；FLUX 未做 | ⚠️ FLUX 未做（阶段 8） |
| MASK/CONTROL_NET/UPSCALE_MODEL 等类型 | 已加入（共 11 种对象类型：含 CONTROL_NET/CONTROL/UPSCALE_MODEL/CLIP_VISION 等） | ✅ 已做（M4/M6/插件批次） |
| 进度推送 / interrupt | 异步任务体系（/jobs 轮询进度 + /interrupt，协作式取消） | ⚠️ WS 推送未做 |

## 1. 扩展方法论（贯穿所有阶段）

**⚡ 插件优先（2026-09-20 起生效）：完全新的算子默认走节点自注册（插件化），
不动服务端核心代码**：节点包携带 `server_op.py`（模块级 `register(registry)`），
节点运行时经 `flowx_client.ensure_plugin()` 比对 hash 自动上传热加载，零部署、
可独立迭代。参考实现：`nodes/detail-refine/`、`nodes/sd3-txt2img/`。
只有以下情况才动服务端核心（`main.py`/`ops.py`/`model_manager.py`）：
- 扩展核心原语/加载层（如 dtype/offload/use_t5 旋钮、新架构的模型装载路径）
- 新对象类型（`registry.py` 的 `OBJ_TYPES`）
- 插件机制本身无法满足的性能/生命周期需求

核心扩展的固定路径（仅在上述例外时使用），全程无引擎改动：

1. `app/ops.py` 加一个纯函数（张量/PIL 进出，不碰网络与 ID）
2. `app/main.py` 用 `@registry.register(...)` 注册一行
3. 新对象类型时往 `registry.py` 的 `OBJ_TYPES` 加一个名字
4. FlowX 侧用通用节点 `inference-op` 立即可用；高频能力再补专属瘦节点

**原则：只支持 safetensors 单文件 + diffusers 目录两条加载路径**（`from_single_file` /
`from_pretrained` 分派），不自己实现 state_dict key 映射；架构逐个加，不提前抽象。

## 2. 阶段规划

### 阶段 1：图生图链路（✅ 已完成）

- 新算子 `vae.encode`（IMAGE→LATENT，乘 scaling_factor 与 decode 互逆）、
  `image.load`（INPUT_DIR 服务端本地图）；`POST /images` 上传端点（客户端本地图 → IMAGE 对象）
- 采样侧 `denoise<1` 跳步 + add_noise 原本已支持，无需改动
- FlowX 侧新节点 `load-image` / `vae-encode`，示例 `pipeline/i2i.yaml`
- 验收：txt2img 出图 → vae.encode → denoise=0.6 重采样 → 构图保留、风格可变

### 阶段 2：sigma 排布与采样算法解耦（越早越便宜的重构）✅ 已交付（2026-09-21）

ComfyUI 把 `sampler_name`（更新公式）与 `scheduler`（sigma 曲线）拆成两个参数自由组合。
当前 `samplers.py` 把两者揉在一个名字里（如 `dpmpp_2m_karras`），组合一多会爆炸。

- ✅ `sample` 算子签名增加 `scheduler` 参数（normal/karras/exponential/beta），
  旧名 `dpmpp_2m_karras` 兼容映射（scheduler 缺省/normal 时生效，显式值优先）
- ✅ 组合支持性按 scheduler 类 `__init__` 签名动态判定（不写死矩阵）：
  euler/lms/dpmpp_2m/dpmpp_2m_sde/uni_pc 支持全 4 种，euler_a/ddim 仅 normal，
  非法组合明确报错（diffusers v0.35.2 源码核实）
- ✅ ksampler 节点加 scheduler 下拉（1.3.0，bundle v1.4.0）；detail-refine 透传（1.2.0）
- 未纳入：sgm_uniform/simple/ddim_uniform（diffusers 无原生开关，需自注 sigmas）
- ✅ 验收：真机 `scripts/accept-m2-samplers.py` 全过——同 seed 下旧名与新写法
  逐字节一致；4 个新组合出图内容正常；euler_a+karras 明确报错；euler 回归正常

### 阶段 3：dtype fallback 小修（配合 VAE 类节点）✅ 已交付（2026-09-17）

SD1.5 的 fp16 VAE 解码会偶发纯黑图，社区标准修法是 VAE 单独 fp32。

- ✅ `model_manager.py` 加载后把图像管道（sniff.IMAGE_ARCHS）的 `pipe.vae` 转 fp32，
  `VAE_FP32` 环境变量可控（默认开）；视频管道不动（VAE 解码在 pipeline 内部，dtype 混用会崩）
- ✅ `ops.vae_decode` latent 按 `pipe.vae.dtype` 自适应转换（采样链 latent 保持 fp16）
- ✅ 验收：GTX1080 真机 50 轮 sample+decode 无黑图（worst mean=104/255），
  脚本 `scripts/accept-vae-fp32.py`

### 阶段 4：ControlNet（流行度最高的扩展类，工程量最大）

- 新类型 `CONTROL_NET`、`MASK`
- 新算子 `controlnet.load`（ControlNetModel.from_pretrained）、
  `controlnet.apply`（COND + 控制图 + 强度 → 打包条件）、
  预处理器算子先做 1~2 个（canny 边缘、OpenPose 走外部库或客户端预处理）
- **核心改动在 `ops.sample` 采样循环**：UNet 调用注入
  `down_block_additional_residuals` / `mid_block_additional_residual`
  （按 diffusers `ControlNetModel` 输出格式对接）
- 显存：`ENABLE_CPU_OFFLOAD=1` 成为推荐配置（三件套同时在场 ~9GB）
- 验收：线稿 → 同构图出图；强度 0.5/1.0 效果区分明显

### 阶段 5：SDXL 支持（主流模型门槛）**✅ 已交付（2026-09-22，实现落地，待真机验收）**

- 加载：✅ 分派机制已交付（sniff.py 按 safetensors 头部 key / model_index.json 嗅探，
  SDXL checkpoint 已路由到 StableDiffusionXLPipeline.from_single_file）
- `clip.encode` 出 SDXL 变体：✅ 双 text encoder + pooled embedding —— 返回
  `SDXLCond(embeds, pooled, size)`；不手写编码数学，直接调管道 `encode_prompt`
  （与 sd3-clip-encode 同法），`clip_skip` 可透传
- `sample` 补 `added_cond_kwargs`：✅ 采样循环内按 COND 是否带 pooled 分派
  （text_embeds + time_ids，time_ids 由 CLIP 节点的 width/height 或 latent 推导；
  区域条件下每段带自己的 pooled）；SDXL ControlNet 前向也收 added_cond_kwargs。
  无 pooled 时路径与历史逐字节一致（SD1.x 不变性）
- 离线 config：`MODELS_DIR/_configs/` 可放 diffusers 骨架（model_index.json + 组件
  config/tokenizer）作为 `from_single_file` 的离线 config 源（免 hub，宿主机无外网）
- 覆盖范围：txt2img / i2i（vae.encode + denoise）/ hires（latent.upscale + 低 denoise）/
  ControlNet / LoRA（走既有 load_lora_weights）/ cond.set_area / combine / average
- **本期不做**：SDXL refiner 双模型管线、aesthetic_score 参数、
  SDXL-inpainting（9 通道 UNet 与现有 4 通道 latent 不兼容）
- 交付与验收记录：dev-plan 二十五（验收口径：脚本 accept-m5.py + Studio 流水线
  sdxl-txt2img / sdxl-hires，含与官方 diffusers 管道逐像素一致性可选步骤）

### 阶段 6：高清放大链

- 新类型 `UPSCALE_MODEL`；新算子 `upscale.load` / `upscale.image`
  （ESRGAN/SwinIR 系，纯 IMAGE→IMAGE，不碰采样循环）
- hires fix 链 = `latent.upscale` + 低 denoise 二次采样（阶段 1 已备好底座）
- 验收：512→2048 放大细节自然；hires fix 与一次性放大出图质量对比达标

### 阶段 7：执行体验补齐（部分完成）

- ✅ 异步任务体系：`POST /jobs` 提交 → job_id 轮询进度 → `POST /interrupt` 取消
  （采样循环内检查取消标志，对齐 ComfyUI）；与阶段 9 的异步前置合并交付
- WebSocket `/ws`：进度推送改 WS（当前轮询已可用，WS 为体验优化，待定）
- 引擎补 lazy evaluation：只执行通向输出节点的分支
- 验收：长采样可中途取消（已达成）；FlowX 画布实时显示进度（轮询通道已具备）

### 阶段 8：Flux / SD3（flow matching 范式，独立工程量）

- 采样数学不同：预测速度场而非噪声；Flux 无 neg prompt（distilled guidance 标量）
- 三文本编码器（SD3: 双 CLIP + T5；Flux: CLIP + T5）——COND 类型需要组合结构
- 大显存前提（Flux dev fp16 ~24GB，或走 fp8/量化路径），依赖阶段 7 的显存治理
- 验收：Flux schnell 4 步出图

### 阶段 9：视频模态（图生视频 / 前后帧视频）

从「不做清单」移入正式规划。前提与内容：

- **异步任务体系**（视频生成分钟级，同步 HTTP 必超时）：✅ 已交付（阶段 7 合并实施）——
  `POST /jobs` 提交 → job_id 轮询进度 → `/interrupt` 取消
- **模型管理器分派**：✅ 已交付——按模型文件/目录嗅探分派管道类
  （app/sniff.py；safetensors 头部 key 嗅探 / diffusers 目录 model_index.json），
  同时是 SDXL / AuraFlow / 视频所有线的公共前置
- 新类型 `VIDEO`；✅ 已交付——视频结果 put 时即编码 mp4 落盘（imageio-ffmpeg，
  自带静态 ffmpeg 二进制）+ `GET /videos/{id}` 下载端点（对标 `/images/{id}`），
  对象过期/清理时同步删除落盘文件
- 显存治理升级：✅ sequential offload 已交付（`OFFLOAD_MODE=sequential`），
  fp8 量化接口已预留（`QUANTIZATION=fp8` 识别配置、告警未实现，实现后置）（视频模型 5B~14B 级）
- 首个视频模型 Wan2.2-TI2V-5B（文/图生视频一体，消费级显卡可跑）：✅ `video.sample`
  算子已交付（prompt/可选首帧/尺寸/帧数/fps/steps/cfg/seed，经 callback_on_step_end
  上报进度与响应取消）；前后帧走 diffusers 现成 `WanFirstLastFrameToVideoPipeline`（待接）
  - **小显存（8GB / Pascal）替代**：Wan/CogVideoX 的 LLM 级文本编码器（T5-XXL fp16 ~9GB）
    超 8GB 卡无解；SVD-XT 1.1（纯图生视频，无文本编码器，fp16 ~4.5GB 峰值）可行——
    `video.sample` 已按 `__call__` 签名自适应（SVD：image 必填、cfg 映射
    min/max_guidance_scale、fps 进采样条件）；AnimateDiff（SD1.5+运动模块 ~3.5GB）
    需组合加载（base+adapter），后置
  - 显存优化：`decode_chunk_size` 分块解码参数已暴露；fp32→fp16 variant 探测加载
    （避免 fp32 全量读内存）
- FlowX 侧：✅ 专属瘦节点 `video-gen`（异步 job 提交/轮询/超时自动取消）+
  `save-video`（mp4 下载落盘）；widget：video-gen 画布进度卡片（服务端纯渲染帧推送）、
  save-video 内嵌 mp4 播放器（小体积 base64）——外壳零改动
- 验收（待 GPU 实测）：图生视频出片；首帧+尾帧插帧出片；长任务可取消、进度可见

### 阶段 9.5：视频到视频（v2v / 换人转绘）✅ 已交付

节点族（`video_flow`，5 节点共享一个族文件）：
- `load-video`（本机原片 → `POST /videos` 上传 → `video.load` 登记 VIDEO；**无归一化参数时原字节直通**，
  不重编码）、`video-to-frames` / `frames-to-video`（VIDEO ↔ IMAGE 批量帧序列）、
  `video-repaint`（**逐帧重绘**：每帧 vae.encode → sample(denoise=strength) → decode；
  `model` 接 ipadapter-apply 的 ModelRef 即逐帧注入参考人物 ⇒ 换人；可选 control_net 按帧注入结构条件）、
  `video-v2v`（整段 v2v：把 motion-loader 的 t2v 组合管**按组件改视**成
  `AnimateDiffVideoToVideoPipeline`，共享权重不额外占显存，**无需改核心加载层**）
- 服务端：`POST /videos`（字节入站 + 返回服务端路径）、`object_store` 支持 VIDEO 预编码直通
  （`_adopt_video_file`：拷贝既有 mp4 入库，避免"解码→h264 重编码→再解码"代际损失）
- 真机验收（2026-10-13，GTX 1080 8GB + diffusers 0.35.2）：VIDEO 编码→解码→再编码往返
  （帧内容逐帧对上）、`video.load` 归一化（抽帧/缩放）、`video.repaint` 逐帧重绘
  （无 IPA 基线 / IPA weight=0.8 A/B 可复现差异）、`video.sample_v2v` 整段出片
  （8 帧 64² 3 步，出片帧尺寸跟随源片）
- ⚠️ 真机定位到的既有缺陷（本批次修）：`load_motion` 组合管出现「`vae.decode` 已被
  `_decode_fp32` 包装、VAE 权重却是 fp16」的不一致态 ⇒ decode 时 conv2d 报
  `Input type (float) and bias type (c10::Half)`，**既有 t2v（video.sample）同样复现**。
  处置分两层（覆盖范围不同，别混）：
  ① **核心** `model_manager.load_motion` 在 offload 之后二次断言 fp32——覆盖全部组合管
     （含 t2v 的 `video.sample`），**需重启/重建推理服务生效**；
  ② **算子层** `align_vae_dtype()`（按 `inspect.ismethod` 判定"解码入口被包装过"，不按名字）
     + `_VAEDtypeShim`（`config.force_upcast` 对外恒 False、encode/decode 出入参按真实 VAE
     精度转换，拦住管道"按计算精度把 VAE 转 fp16"那一步）——覆盖本族 repaint / sample_v2v，
     **插件热更新即生效**（无需重启）
- 验收口径：视频管线**必须抽帧看图**（状态 success 不代表内容正确，见 dev-plan 视频线教训）
- ⚠️ **strength 口径坑（真机复现）**：`AnimateDiffVideoToVideoPipeline.strength` 是"时间轴截断比例"，
  `init_timestep = int(num_inference_steps * strength)` ⇒ `steps=3, strength=0.6` 只跑 **1 步**
  （DDIM linspace 3 步 = [999,666,333]，截断后只剩 t=333）；实测 ρ(源,出)=0.80~0.86，等于没重绘。
  要出风格需 `steps × strength ≥ ~12`（例 25×0.7=17 步）。
  **已处置（2026-10-14）**：算子新增纯函数 `effective_steps(steps, strength, enforce)` 并在
  `video.sample_v2v`/`video.repaint` 开头打印"有效去噪步数"：**=0 直接报错**（原样返回输入属误用）、
  **=1 告警**并给出配比建议（不再静默）；`video-v2v` 节点新增 `enforce_inference_steps` 参数
  （ComfyUI 语义：steps 即实际执行步数，strength 只决定从 sigma 排布哪一段开始）；两个模板
  参数配比已改为 25×0.7 / 20×0.55。单测覆盖见 `tests/test_video_flow.py`。
- ⚠️ **"每帧不同猫"= 源片特性，不是 v2v 引入**：定量口径（32² 尺度相邻帧 ρ / 首末帧 ρ / 亮度极差）
  ——既有 AnimateDiff t2v 出片 0.78 / 0.01 / 33~49；同机 SVD 出片 0.99 / 0.91~0.97 / 1~3；
  v2v 出片 0.85 / 0.59 / 33.8（继承）。**v2v/repaint 只能保运动、不能创造时间一致性**——
  源片不连贯时成片必然抖。演示时把 16 帧截成 8 帧（运动模块训练窗口为 16）会进一步削弱一致性。
- 建议的验收指标（视频线）：相邻帧区域 ρ（角/中/32² 粗尺度）+ 亮度极差 + ρ(源,出)，
  单看"抽了 2~3 帧像猫"会漏掉身份漂移（exec 199 当时据此判"连贯猫咪"通过）。

### 阶段 10：图像后处理与电商能力补齐（2026-10-10 立项，清单见 dev-plan §四十四）

对标 ComfyUI 六大类图像处理能力盘点后的缺口补齐（已覆盖部分见 dev-plan §44.1，勿重复投入）：
- **B1（已交付）**：传统滤镜算子（`image.adjust/curves/blur/sharpen/noise/channel`，PIL 零 GPU）
  【✅ G5 滤镜 8 节点 + G3 扩图：镜 v1.10.0，exec 600/601/602 全绿，见 dev-plan §四十六】
  Outpainting 扩图（`image.pad` + 既有 inpaint 链）【✅ 判决 10.9x】
  + 预处理器补齐【✅ G7 已交付 7 算子：anyline / lineart(三档) / softedge(pidi|hed) / scribble /
  mlsd / normal / shuffle，镜 v1.11.2，exec 605 全绿】；**teed**（`TheMistoAI/TEED` 仓库已私有，401）
  与 **seg**（controlnet_aux 0.0.10 无语义分割检测器）未纳入，seg 语义并入 G2（SAM）批次。
  预处理器权重清单见 `nodes/README.md`（宿主机 `/models/preprocessors/`，零网络加载）
- **B2（核心价值，2026-10-11 实施中）**：抠图算子 `matting.birefnet`（**BiRefNet**，用户拍板；
  `RMBG-2.0` 为 gated 401、`RMBG-1.4` 备选）+ 人脸修复算子 `face.restore`
  （spandrel **GFPGAN / RestoreFormer**；**CodeFormer 不在 spandrel 0.4.2 发布版里**，不做）。
  新增依赖 `timm==0.9.16` / `kornia==0.7.4` / `einops==0.8.0`
  （kornia 0.8+ 要求 py≥3.11，Pascal 基础镜像 py3.10 ⇒ 排除）。权重清单见 `nodes/README.md`
  （`/models/matting/{lite,main,portrait}` 各 4 件、`/models/facerestore/*.pth`）。
  真机探针（GTX 1080 fp16）：lite@1024 **930MB / ~1.5s**、main@1024 1810MB、portrait@1024 2251MB、
  lite@2048 4315MB（reserved 6.2G，8GB 卡余量紧）；两次调用逐字节一致。
  详见 dev-plan §四十八
- **B3**：SAM / GroundingDINO 交互与文本分割（`seg.sam` / `seg.grounding`）+ 分块重绘超分
  （`image.tiled_upscale`，对标 Ultimate SD Upscale）
- **B4（重工程，各自立项）**：DDColor 上色 → IC-Light 重打光 → PuLID → 电商流水线模板

**已拍板决策（2026-10-10）**：① 一律**细粒度节点**拆分（加载/处理/合成分离，中间产物对象引用接线）；
② **所有新节点必须带自定义 UI**（`ui/node-widget.js`，基于 `_common/widget-base.js`），
不允许裸参数面板。节点清单共 34 个（dev-plan §44.6.1）。

由此新增的基建改动：
- `nodes/_common/widget-base.js` 新增控件 spec：`curves` / `directionPicker` / `pointSelect` /
  `compareAB` / `checkerPreview` / `channelToggles` / `tileGridOverlay`（改后 `_tools/build-widget.py`
  **全量重建所有节点 bundle**，`check-bundle.py` 作发布门槛）
- `inference-server/app/registry.py` 的 `OBJ_TYPES` 新增模型句柄类型（`MATTING_MODEL` / `SEG_MODEL` /
  `GROUNDING_MODEL` / `RESTORE_MODEL` / `COLORIZE_MODEL` / `ICLIGHT_MODEL` / `PULID_MODEL`，
  或收敛为单个 `AUX_MODEL` + 子类型）——按 §1 例外条款，这是本阶段**唯一**需要动服务端核心的地方

约束：新增依赖须同改 `requirements.txt` + `requirements-pascal.txt` 并重建镜像；
Pascal 钉版（transformers==4.44.2 / diffusers==0.35.2 / numpy<2）不得随意放开。

**⚠️ 真机验收环境纪律（2026-10-11 实测，修正前一版结论）**：
- **手机经局域网可直连推理服务** `http://192.168.5.60:8100`（鉴权头完好，`/ops` 200 全量算子）
  ⇒ **local 执行器完全可用**（service_url 填宿主机局域网地址）；docker 执行器填容器内网
  `http://172.17.0.1:8100`。
- cpolar **HTTPS 公网隧道会剥离 `Authorization` 头**（`/health` 免鉴权才通）⇒ 经隧道的鉴权调用一律 401，
  不可用于验收；需要外网侧访问时用端口隧道（TCP）而非 HTTPS 隧道。
- 插件上传：文件名 = `op_name.replace(".", "_") + ".py"`，**跨文件重名算子一律 409 全量回滚** ⇒
  新算子族必须独立文件（`nodes/_tools/check-ops.py` 已作为 gate J 静态拦截）。

## 3. 架构守护：`sample()` 的分派纪律

唯一「加法做多了会欠债」的地方。规则：

- 采样循环骨架（timesteps 迭代、CFG 结构、scheduler.step）保持一份
- 每架构一个适配函数：`predict_noise(pipe_arch, latents, t, cond_bundle) -> noise_pred`
- SDXL/ControlNet/Flux 的差异**只允许进适配函数**，不允许在骨架里加 if-else
- 第三个架构接入之前不做统一抽象（两个实例时直接分派，三个时再提炼接口）

### 3.1 架构全覆盖（硬约束，2026-10-16）

**本系统对 GPU 架构的支持是横向全覆盖，不是“新卡优先、旧卡将就”。**
在役的 Pascal（GTX 1080，算力 6.1）与后续每一代（二代 / 三代 / 四代 / 五代…）
**都在支持范围内且同等对待**；任何优化、依赖、预编译内核与镜像都不得以“放弃某一代”为代价。
完整条款见 `docs/dev-plan.md`（一、开发纪律 第 11 条）。

落地要点：

- **速度可分档，功能与正确性不分档**：禁止把算力/架构门槛硬编码进主路径。
  反例（2026-10-16 瓶颈定位）：diffusers `can_use_cuda_kernels` 的 `capability >= 7`
  ⇒ Pascal 永远落 `forward_native`，每步全量反量化 GGUF 权重（实测 ≈6min/步）。
- 任何加速分支都必须**保留正确性等价的慢路径**，且慢路径在旧档上能跑通。
- 验收覆盖**两端**（最旧在役档 + 最新一档）；确实不支持的档位必须显式记录
  （原因 / 代价 / 重开条件），不许沉默降级。
- 选型 / 依赖 / 镜像（CUDA 版本、预编译内核可获得性等）**按档评估**，
  不得让某一代直接无法启动。

## 4. 显存治理演进

| 阶段 | 机制 | 粒度 |
| --- | --- | --- |
| 现在 | `MAX_RESIDENT_MODELS` LRU 整模型淘汰 | pipe 级 |
| 三件套起 | `OFFLOAD_MODE=model`（diffusers 按子模块搬移） | 子模块级 |
| 视频大模型 | `OFFLOAD_MODE=sequential`（逐层搬移，吞吐最低） | 层级（diffusers 自带） |
| **显存治理 2.0（已交付，见 §4.1）** | 显存预算式淘汰 + `model.unload` 显式卸载 + OOM 自愈 + 流水线显存闸门 | 条目级 + 调用级 |
| 远期（如有需要） | 参考 comfy/model_management 做张量级搬移 | 张量级 |

原则：diffusers 自带的 offload 够用就不自研；整模型淘汰保留作为兜底。

### 4.1 显存治理 2.0（2026-09-22 立项 → 当日交付；详见 dev-plan 二十一）

> **交付状态（2026-09-22）**：四项任务全部落地并真机验证 —— 预算式淘汰
> （`app/vram.py` + `ModelManager._evict_if_needed`）、显式卸载（`model.unload` 算子 +
> `POST /model/unload` + `model-unload` 节点）、OOM 自愈（`engine` 捕获 OOM → 淘汰非在用
> → 重试一次 → 失败给可执行指引）、流水线显存闸门（`inference-ensure` 的 `min_vram_mb`
> + emit 显存字段）。验收：同一张 8GB 卡载入 SVD-XT 后再跑 wf58（19 节点图+精修）
> **166.9s 成功**（旧行为同场景 exec 365 为 3939s）；服务端日志可见"SVD-XT 被预算淘汰"
> 而非堆到 `vram_free_mb: 0`；跑完后 `/health` 不再误报退化（区分驱动余量与可回收缓存）。

**起因（实测事故）**：容器 `MAX_RESIDENT_MODELS=3` + `OFFLOAD_MODE=none` + 8GB GTX 1080，
池子里同时常驻 `majicmixRealistic_v7` + `stable-video-diffusion-img2vid-xt-1-1` +
`v1-5-pruned-emaonly-fp16` → `/health` 报 `vram_free_mb: 0` → CUDA 走 host memory 兜底，
KSampler 从 0.4~0.6s/步 劣化到 17s/步、精修 1024 稳定 127s/步（exec 356/357/359 对比
exec 361/362）。这与遗留清单 #6 的次生教训（decode OOM 后 22s/步 → 7min/步，重启容器恢复）
是同一现象，当前处置仍靠人工重启容器。

**根因**：淘汰只看**条目个数**（`while len(self._pipes) >= MAX_RESIDENT`），不看模型体积与
实际余量；`MAX_RESIDENT_MODELS=3` 是为 ControlNet 三件套（checkpoint+cn+外挂 vae）设的，
叠加 SVD-XT / AnimateDiff 组合管必然溢出；且 `POST /gc` 不卸常驻模型（只清对象仓库+图缓存
+`empty_cache`），没有任何显式卸载通道。

**四项任务（本轮范围）**

| # | 任务 | 位置 | 验收口径 |
| --- | --- | --- | --- |
| 1 | 显存**预算式淘汰**：加载前按 `mem_get_info()` 余量 + 待加载模型体积估算淘汰；`MAX_RESIDENT_MODELS` 降为兜底上限；执行中的条目 pin 后跳过 | `app/model_manager.py` `_evict_if_needed()` | 8GB 卡上顺序加载 majicmix → SVD-XT → v1-5 不再出现 `vram_free_mb: 0`；采样维持基线（512×30 步 ~15-20s） |
| 2 | `checkpoint.unload` 算子（`models.unload(key)` / 全部）+ 瘦节点 `model-unload` | `app/model_manager.py` / `ops.py` / `main.py` + `nodes/model-unload/` | 跑完 SVD 段后 unload → resident 清空、余量回升 ~7GB；随后采样回到 exec 356 基线 |
| 3 | OOM **自愈**：捕获 `torch.cuda.OutOfMemoryError` → 淘汰非在用 LRU → 重试一次；`/health` 加 `degraded` 标记 | `app/engine.py` / `jobs.py` / `main.py` | 人为超预算加载不崩、自愈后成功；连续失败给出明确指引（降档 offload / 重启容器） |
| 4 | 流水线**显存闸门**：`inference-ensure` 增 `min_vram_mb` 参数，并 emit `vram_free_mb` / `resident_models` / `offload_mode`（现仅打日志） | `nodes/inference-ensure/` | 余量为 0 时 Ensure 直接失败并提示 unload/重启，不让流水线退化到 127s/步 |

**边界（不在本轮）**：offload 三档与「env 默认 + 节点参数覆盖」两级旋钮结构不动；
张量级搬移仍留远期；`QUANTIZATION=fp8` 未实现，不在本轮。

**排期建议**：任务 1+2 先行（直接救图/视频混跑），4 成本低随手做，3 视需要。

## 5. 不做清单（明确排除，避免 scope 蔓延)

- ❌ 自研 ckpt/safetensors key 映射（diffusers 已覆盖）
- ❌ 复刻 ComfyUI 前端/画布（FlowX Studio 已承担）
- ❌ 兼容 monkey-patch 类自定义节点生态（隐式约定，承接成本无上限）
- ❌ 音频模态，待视频链路完整后再评估
- ⚠️ MiniMax H3 等新开源视频模型：权重确认可用后按阶段 9 的基础设施接入，
  架构适配遵循 §3 分派纪律（当前明确排除，不做预投入）
- ⚠️ ComfyUI V3 节点 schema（comfy_api）仅保持关注，不提前对标

## 6. 里程碑速查

| 里程碑 | 阶段 | 解锁能力 | 预估侵入面 |
| --- | --- | --- | --- |
| M1 图生图 | 1 ✅ | i2i / 变体生成 | 2 个算子（已交付） |
| M2 采样器完整 | 2 ✅ | 全采样器×sigma 组合 | samplers.py 重构（已交付） |
| M3 稳定 VAE | 3 ✅ | 无黑图 | model_manager 几行（已交付） |
| M4 ControlNet | 4 | 构图控制 | 采样循环改造（✅ 已交付，见 dev-plan 十二） |
| M5 SDXL | 5 | 主流社区模型 | ✅ 已交付（2026-09-23 真机验收全绿：t1~t9 + 官方管道逐像素一致 + Studio 端到端） |
| M6 高清链 | 6 | 2K/4K 出图 | 2~3 个算子（✅ 已交付，见 dev-plan 十四） |
| M7 体验 | 7 | 进度/取消/懒执行 | 引擎 + WS（进度/取消 ✅；WS 推送与懒执行未做） |
| M8 Flux | 8 | 最新架构 | 独立采样路径（SD3.5 ✅ 已点亮；FLUX.1-schnell ✅ 已交付，GGUF 量化路线） |
| M9 视频 | 9 | i2v / 首尾帧插帧 | 异步任务 + 显存治理（i2v ✅；FLF2V 搁置） |
| M10 图像后处理与电商 | 10 | 滤镜/扩图/抠图/分割/分块超分/修复/上色/重打光 | 算子插件为主（34 个细粒度节点，全带 UI），清单见 dev-plan §四十四：**B1~B5 已全部交付**（滤镜/扩图/预处理器、抠图+人脸修复、分割+分块超分、上色+电商+重打光、批量与工程收尾）；仅 G10 PuLID 经评估明确不做、G11 的环境投影节点待做 |

### 算子名可能带版本后缀（F1，2026-10-11）

节点侧交付的插件族文件现在把注册名写成 **`<基名>@<族内容 hash 前 12 位>`**
（由 `nodes/_tools/sync-common.py` 打戳，族内注册调用统一经 `_VersionedRegistry` 包装），
上传落盘名相应为 `<OP_FAMILY>_<版本>.py`。服务端**无需改动**（算子名对注册表是不透明
字符串），但维护时注意：

- `/ops` 里同名基名可能有多条（不同版本并存，各自 `plugin_hash`/`plugin_file` 不同）；
  未版本化的历史算子（`sd.sample`、`preprocess.canny` 等）继续以基名存在，两种命名共存。
- 内容寻址之后，"同文件名不同内容"的静默覆盖在版本化族里不再发生（不同内容 ⇒ 不同名
  ⇒ 不同文件）；`detect_conflicts` 的 409 兜底只对**未版本化**的历史文件仍有意义。
- 清理旧版本：`nodes/_tools/purge-plugin-versions.py`（默认 dry-run），按本地族版本戳
  删除被取代的版本文件；删除会注销该版本的算子，旧节点再跑会自行重传（自愈）。
- 已知服务端侧遗留：409 回滚后 `plugin_ops` 归属记账会串到失败文件上（删一个文件可能连带
  注销 legacy 算子）——F1 之后该路径基本不再触发，该路径已在 2026-10-13 服务端插件归属簿修复中闭环（见 dev-log.md §52.1/§52.3/§52.7）。
