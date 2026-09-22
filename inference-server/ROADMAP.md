# 演进路线：对标 ComfyUI 的能力补齐计划

> 定位：本服务是 ComfyUI 执行模型的**远程复刻**（类型化端口 + 对象仓库 + 拓扑调度 + 内容寻址缓存，
> `/graph` 对齐 `/prompt` 语义）。内核已与 ComfyUI 同构，剩余差距是**算子覆盖面**与少量引擎语义。
> 本文档给出差距清单与分阶段演进计划。判断依据：ComfyUI 生态流行度（ControlNet 系、SDXL 系、
> 高清放大链为流量大头）与对本服务现有代码的侵入程度。

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
| SDXL/SD3/Flux 架构 | 仅 SD1.x 采样/编码路径；管道类已按内容嗅探分派（sniff.py） | ⚠️ 分派机制已备，架构适配未做 |
| MASK/CONTROL_NET/UPSCALE_MODEL 等类型 | VIDEO 已加入（共 7 种对象类型） | ⚠️ 控制/放大类未做 |
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

### 阶段 5：SDXL 支持（主流模型门槛）

- 加载：✅ 分派机制已交付（sniff.py 按 safetensors 头部 key / model_index.json 嗅探，
  SDXL checkpoint 已路由到 StableDiffusionXLPipeline.from_single_file）
- `clip.encode` 出 SDXL 变体：双 text encoder + pooled embedding（未做）
- `sample` 补 `added_cond_kwargs`（time_ids/pooled）——**此时引入架构分派**，
  用「采样循环骨架 + 每架构适配函数」而非 if-else 堆积（见 §3）
- 验收：Pony/Illustrious 系社区模型出图正常

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

## 3. 架构守护：`sample()` 的分派纪律

唯一「加法做多了会欠债」的地方。规则：

- 采样循环骨架（timesteps 迭代、CFG 结构、scheduler.step）保持一份
- 每架构一个适配函数：`predict_noise(pipe_arch, latents, t, cond_bundle) -> noise_pred`
- SDXL/ControlNet/Flux 的差异**只允许进适配函数**，不允许在骨架里加 if-else
- 第三个架构接入之前不做统一抽象（两个实例时直接分派，三个时再提炼接口）

## 4. 显存治理演进

| 阶段 | 机制 | 粒度 |
| --- | --- | --- |
| 现在 | `MAX_RESIDENT_MODELS` LRU 整模型淘汰 | pipe 级 |
| 三件套起 | `OFFLOAD_MODE=model`（diffusers 按子模块搬移） | 子模块级 |
| 视频大模型 | `OFFLOAD_MODE=sequential`（逐层搬移，吞吐最低） | 层级（diffusers 自带） |
| **显存治理 2.0（规划，见 §4.1）** | 显存预算式淘汰 + `model.unload` 显式卸载 + OOM 自愈 + 流水线显存闸门 | 条目级 + 调用级 |
| 远期（如有需要） | 参考 comfy/model_management 做张量级搬移 | 张量级 |

原则：diffusers 自带的 offload 够用就不自研；整模型淘汰保留作为兜底。

### 4.1 显存治理 2.0（2026-09-22 立项，待实施；详见 dev-plan 二十一）

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
| M4 ControlNet | 4 | 构图控制 | 采样循环改造 |
| M5 SDXL | 5 | 主流社区模型 | 架构分派落地 |
| M6 高清链 | 6 | 2K/4K 出图 | 2~3 个算子 |
| M7 体验 | 7 | 进度/取消/懒执行 | 引擎 + WS |
| M8 Flux | 8 | 最新架构 | 独立采样路径 |
| M9 视频 | 9 | i2v / 首尾帧插帧 | 异步任务 + 显存治理 |
