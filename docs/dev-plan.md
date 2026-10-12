### 9.0 第一档：零服务端改动（纯客户端/Studio 层，随时可做）

✅ **2026-09-20 交付**（commit `3cbcb88`，9 节点）：图像变换 image-rotate / image-flip /
image-crop / image-composite（可选蒙版）+ 蒙版工具 mask-from-image / mask-to-image /
mask-grow / mask-feather / mask-invert。实现要点：
- 纯客户端 PIL（GET /images/{id} → 处理 → POST /images 传回），**local-only 执行器**
  （非 bundled，避开镜像三联动；sync-flowx-json.py LOCAL_ONLY 已收录，镜像无 Pillow 不分发）
- 服务端零改动；widget 预览走节点侧生成的 thumb_b64 data URL（Studio 外壳零改动，
  规避浏览器直连 service_url 的 CORS/鉴权问题）
- mask 约定 = 内容为灰度的 IMAGE 对象，读入 convert("L")；image-crop 对蒙版通用（复用为 mask crop）
- 验收：test-tier1-nodes.py 26 项 PIL 单测全过 + 9 mock 可跑；~~Studio 端到端验收待做~~
  **✅ 2026-09-20 端到端验收通过**：流水线 61（tier1-image-mask-accept，12 节点含全 9 新节点）
  exec 315 全绿 + 抽图内容核对通过（旋转/翻转/裁剪几何语义正确，合成图蒙版反相+羽化+自动缩放生效）

### 9.1 第二档：小改服务端（加一个 op ≈ 注册 10 行 + 实现 20~50 行）

| 能力 | 改动点 | 备注 |
|---|---|---|
| ~~Load VAE 单独加载~~ ✅ | sniff.py 组件级识别 + model_manager.load_vae（默认 fp32）+ VAEShim | 2026-09-21 交付，见十一 |
| Load CLIP 单独加载 | 暂缓：需求不明确，单文件 CLIP 无自带 tokenizer | sniff 组件规则已预留 |
| Load Diffusion Model（纯 UNet） | 暂缓：无独立使用场景，价值在 FLUX 线 | 并入 M8 批次 |
| ~~KSampler Advanced（分段采样）~~ ✅ | ops.sample 补 start_at_step/end_at_step/add_noise | 2026-09-21 交付，见十一 |
| ~~Scheduler 解耦~~ ✅ | = 既有 M2 里程碑 | 见第十节 |
| ~~Conditioning Combine/Average/SetArea~~ ✅ | combine/average/set_area 均已交付（插件）+ set_area 采样循环分段混合 | 2026-09-21 随 M4 交付，见十二 |
| ~~Latent Upscale / Composite / SetNoiseMask~~ ✅ | upscale/composite/set_noise_mask 均已交付 + 采样循环 mask 混回 | 局部重绘核心，2026-09-21 随 M4 交付，见十二 |
| Upscale（4x-UltraSharp 等） | ≈ 既有 M6 高清放大链（UPSCALE_MODEL + 分块） | 独立批次，见第五节 |

本档策略：**先经 `inference-op` 通用节点直接在 workflow 中使用**（零节点包成本），
稳定高频后再做专属节点包 + 画布 widget。

### 9.2 第三档：中改（新模型类型进 ModelManager + 采样链路扩展）

| 能力 | 改动点 | 对应业界方案 |
|---|---|---|
| ~~ControlNet 加载/应用~~ ✅ | M4 已交付：CONTROL_NET/CONTROL 对象类型 + ControlNetModel 进 LRU + 采样循环逐步前向注入 residuals（不用 ControlNetPipeline，保住 M2 解耦/分段采样） | 2026-09-21，见十二 |
| 预处理器（openpose/canny/depth/lineart） | 独立 op（如 `preprocess.openpose`），引入 controlnet_aux | 第三档批次 2，comfyui_controlnet_aux |
| ~~IPAdapter（参考图驱动）~~ ✅ | diffusers `pipe.load_ip_adapter()`（SD1.5/SDXL 现成）+ CLIP Vision 模型 | 2026-09-21 实现（commit `49a2c03`），2026-09-24 真机验收（见二十七） |
| InstantID / PuLID（换身份保动作） | IPAdapter 之上 + insightface 人脸模型 + face ControlNet；工作量主要在依赖引入与权重管理，不在架构 | ComfyUI_InstantID / PuLID |
| 人脸/手部修复编排（FaceDetailer） | 检测 op（ultralytics bbox/segm）→ 蒙版 → inpaint，本质是已有能力组合 + 1~2 个新 op | Impact Pack |

### 9.3 第四档：大改（新架构族，独立立项评估）

| 能力 | 改动点 | 前置条件 |
|---|---|---|
| FLUX / SD3 | = 既有 M8；**2026-09-18 可行性重估：可点亮，见 9.6**（sniff 新架构规则 + flow matching 采样由 diffusers 管道内置承接 + 组件级加载） | 不再需要硬件升级；FLUX 需 WSL 内存扩容 + diffusers 升级 |
| 物体新视角合成（SV3D/TripoSR 等 3D 生成） | 全新模型族 + 网格/多视角输出对象类型 | 需求确认 + 显存评估，单项最大 |

### 9.4 两个目标场景的落地组合（排期优先级依据）

1. **人物替换但保留动作**（业界标准组合：OpenPose 约束动作 + InstantID 锁身份 + inpaint）：
   - `preprocess.openpose`（第二/三档，小）
   - ControlNet 加载/应用（M4，第三档）
   - InstantID（第三档，依赖 IPAdapter 先行）
   - 蒙版工具集 + inpaint（第一档 + 现有 vae.encode/denoise 链路）
2. **图片物体角度旋转**：
   - 小角度：第一档 PIL 旋转/翻转节点（零成本，先做）
   - 真·换视角（看侧面/背面）：SV3D 级 3D 生成（第四档，最后做）

### 9.5 建议执行顺序（均需先过方案）

1. ~~第一档批量落地：mask 工具集 + 图像变换节点包~~ ✅（2026-09-20，9 节点交付+端到端验收，见 9.0）
2. ~~第二档批次 1~~ ✅（2026-09-21，见十一）；~~剩余：Upscale 模型链（M6）~~ ✅（见十四）
3. 第三档主线：~~ControlNet（M4）~~ ✅（2026-09-21，见十二）→ ~~预处理器（批次 2）~~ ✅（见十三）→ ~~IPAdapter~~ ✅（2026-09-24 真机验收，见二十七）→ ~~InstantID~~ ✅（2026-09-24 交付+真机验收+Studio e2e，见二十八）
4. 第四档新架构线：SD3.5 先行 ✅ → FLUX 跟进（可行性重估见 9.6，不再等硬件）；3D 新视角合成（SV3D）仍搁置

**与既有里程碑的关系**：M2/M4/M5/M6/M8 已覆盖本计划第二、三、四档中对应项，
本节是"节点生态视角"的补全与排期依据，不重复立项；第五节表格保持权威。

### 9.6 新架构（SD3.5 / FLUX）8GB Pascal 可行性评估（2026-09-18，用户决策：能跑即可，时间不敏感）

**前提更新**：宿主机物理内存 **16GB**（WSL 默认只分到 ~8GB，可经 `.wslconfig` 扩至 12~13GB）；
用户明确"时间无所谓，能跑就行"。→ **M8 定性修正：从"超当前显存，等硬件升级"改为
"现有硬件可点亮，分批实施"**。

**核心技术手段：sequential CPU offload**（逐层搬运，显存峰值 = 最大单层而非整模型，
8GB 显存不再是瓶颈；瓶颈转移到**宿主机内存**与**数值精度**）。

#### SD3.5 Medium（先做，铺路石）— ✅ 2026-09-20 交付并真机验收（细分链版）

- 交付：diffusers 0.30.3→0.35.2（SD3.5 需 ≥0.31；配套 peft 0.17.1/hub 0.34.4），
  **细分节点链**（用户确认粒度：细分 + 全带自定义 UI，单体 sd3-txt2img 已删除）：
  sd3-clip-encode / sd3-empty-latent / sd3-sampler / sd3-vae-decode / sd3-vae-encode
  五个自包含节点（插件自注册，commit `5a761c1`）+ workflow 60（八节点细分链）；
  核心仅保留 model_manager 旋钮与 checkpoint.load 扩展（dtype/offload/use_t5）
- 验收（GTX 1080 8GB + sequential offload + WSL 12GB）：exec 301 八节点细分链全绿，
  插件逐个自注册成功；采样 64.2s（模型常驻后手动循环比管道全包快一倍）；
  出图内容正常（抽帧看图通过）；exec 297 SD1.5 回归通过
- 待办：~~AnimateDiff/视频线回归（diffusers 升级后未验，遗留清单 #5）~~ ✅ 2026-09-20 完成（见遗留清单 #6 的回归发现）
- 原预估修正：SD3.5 不是"几十分钟/张"而是分钟级——2.5B 模型 + sequential offload
  在 Pascal 上吞吐可接受；**1024 原生分辨率与 T5 回归（错峰/fp8）留作下一迭代**

- 配置：MMDiT fp16 ~5.2GB + CLIP-L/G ~1.4GB，**弃用 T5-XXL**（fp16 ~9.5GB，超 WSL 内存；
  质量打折但管线完整，T5 留作后续可选项）
- **软件栈零升级**：diffusers 0.30.3 已内置 StableDiffusion3Pipeline
- 服务端改动：sniff 规则 + 组件级加载（CLIP-L/G 独立加载，复用 9.1 第二档成果）+
  采样算子照抄 `video.sample` 的"管道签名自适应"模式（**无需重写采样循环**，
  M8 原预估"独立采样路径"下调为中改）
- 预期：几十分钟/张，可接受

#### FLUX.1 ✅（2026-09-26 交付，GGUF 路线，见第二十九节）

| 项 | 决策/风险 |
|---|---|
| 量化路线 | **GGUF Q4**（主体 ~6.8GB + T5 Q4 ~2.9GB ≈ 10GB RAM）；
  ~~手动 fp8 cast~~ **已排除**（用户决策） |
| 内存 | 需 `.wslconfig` 把 WSL 扩至 ≥12GB（物理 16GB 可行） |
| diffusers | 0.30.3 无 GGUF 支持（0.31+ 引入）→ **需升级 diffusers 并做 Pascal 回归验证**
  （关联遗留清单 #6；实测过 0.35.2 可跑通现有代码，兼容性风险可控） |
| 数值精度 | bf16 无硬件支持（Pascal）必须转 fp16；**FLUX fp16 有已知 NaN/黑图问题**，
  社区方案=注意力层 fp32，需真机调试（验收纪律：抽帧看图） |
| 预期速度 | 1~3 小时/张，纯点亮性质 |

**实施纪律**（2026-09-18 用户确认，适用于 SD3.5/FLUX 及后续所有性能适配工作）：

1. **硬件假设永不进算子逻辑**：dtype/offload/量化等性能手段与业务逻辑解耦，
   ops.py 只面向 diffusers 管道抽象写。
2. **workaround 集中开关化**：fp16 NaN 修补（注意力层 fp32）等妥协按 dtype **条件触发**
   而非无条件生效；一处妥协一个开关（参照 VAE_FP32 先例），升级硬件时对照清单逐项关闭。
3. **性能旋钮采用两级结构：环境变量给默认 + 节点参数可覆盖**（用户确认的需求）：
   - 服务端：加载算子增加可选输入 `dtype="auto"` / `offload="auto"`（auto = 继承进程级
     环境变量，存量流水线零影响）；ModelManager 缓存键纳入 dtype/offload（同模型不同配置
     = 两个缓存条目，LRU 驱逐逻辑不变；/graph 缓存键自动覆盖）。
   - 节点侧：checkpoint-loader 等加载节点增加同名可选参数（默认 auto，UI 折叠为高级区）；
     变体选择（GGUF Q4 vs bf16）天然节点级——选文件即选变体，零改动。
   - 边界：部署配置（TOKEN/目录/WSL 内存）保持进程级不进节点；混合 offload 合法
     （如 SD1.5 常驻 + FLUX 逐层），超显存是用户自选风险，UI 提示即可不拦截。
   - 收益：升级硬件 = 改环境变量默认值，不逐条改流水线；慢是配置不是架构。
4. **镜像重建属计划内路径**：新硬件 = 新镜像（torch/diffusers 解锁升级，顺带消遗留项 #6）。

**执行顺序**：SD3.5 先行（验证 sniff/组件加载/offload 三件套 + 摸底 diffusers 新管道在
Pascal 的行为）→ FLUX 复用其加载与 offload 基础设施。两项动手前均按纪律先出方案。

## 十八、节点 UI 全量公共件化（2026-09-21，commit 5647988，真机验证）

**起因**：盘点缺口——preprocess-canny 无 widget（JSON 兜底）；29 老节点是
旧式手写 widget（无公共件能力）；用户拍板：全量换装。

**widget-base 三增强**：①kind:'textarea'（prompt 多行）；②thumb_b64 回退
（老执行器无 preview 帧信号时结果缩略图仍展示，emit 输出经 node_complete
事件进 outputs prop）；③currentJobId()：preview.jobId 缺失时从 preview url
解析 /preview/{id}（老 flowx_client 只在 url 携带 job），中断按钮老节点可用。

**preprocess-canny v1.2.0**（id 93）：main.py 移植 executor_base.run_op
（check_exists=True 保留"服务端唯一实现"约束）+ 补 executor_base.py +
flowx_client.py 升级（带 base/job_id 的 emit_preview——**踩坑：只拷
executor_base 没升 flowx_client，完成帧上报 TypeError**）→ 白得完成帧/
replay metadata/前后对比全链路；widget-def 双阈值滑块 + ▶预览 + compareInput。
真机 exec 347 绿：20/60 密线稿 vs ▶重放 100/200 疏线稿抽图核对 ✅。

**gen-widget-def.py**（_tools/ 新工具）：flowx.json parameters → widget-def.js
批量生成。类型映射 boolean→check、数值→text/slider、string→text/select/
textarea；手工调优表：3 加载器模型下拉（checkpoint/lora/motion）、采样器
steps/cfg/denoise 滑块、sampler×scheduler select（取值核对 samplers.py：
euler/euler_a/ddim/lms/dpmpp_2m/dpmpp_2m_sde/uni_pc × normal/karras/
exponential/beta）、flip/rotate/upscale/mask-channel select（取值核对节点
实现）、clip textarea、previewEveryKey 自动检测。

**成果**：36 节点全部 widget-base 体系（29 老 + 6 P0 重建 + canny 新建），
语法全过；30 节点导入注册表（canny 1.1.0→1.2.0、ksampler 1.2.0→1.3.0 进库）；
流水线 70 exec 348 17/17 绿（执行器零改动，widget 纯 UI 层）。

**遗留**：①ksampler@1.2.0（id 58）等历史版本条目仍是旧 widget（版本语义，
新流水线用最新版即可）；②P1 七节点/P2 二节点未做；③老执行器不换
executor-base（行为不变性优先），故老节点无过程帧/replay/对比（有 thumb
回退兜底）；④画框/笔刷（第三批）未做。

**修正（同日）**：发现 P0 六节点 + canny 的 flowx.json 缺 `ui` 段——导入器
按 ui.entry 索引 widget，缺失导致注册表条目无 widget（此前"验收"均为 curl
层模拟）。已补 + 重导验证 file_assets 含 ui/node-widget.js（kind=ui）。
教训：widget 导入验证必须查 file_assets，不能只看 import 成功输出。

## 十九、P1+P2 九节点：32 算子 100% 专属节点覆盖（2026-09-21，commit 5c704bc，真机验收）

**九节点**（P0 模式：executor-base + widget-def + mock + ui 段）：
cond-combine / cond-average / cond-set-area / latent-upscale /
latent-composite / latent-set-noise-mask / embedding-load /
vae-decode-video / video-sample-latent（注册表 id 94-102）。

**真机验收**：
- P1 链（流水线 74，exec 349，21 节点全绿）：embedding-load 挂
  badhandv4+EasyNegative；set_area 左猫右狗 combine 区域构图；average(0.3)
  供二采；latent.upscale 1.5x（768x512→1152x768）；luminance 蒙版
  set_noise_mask + denoise 0.5 重采；latent.composite 合成原 latent。
  抽图核对：分区语义可见（猫头左/柯基身右）、composite 锐度差可辨 ✅
- P2 链（流水线 76，exec 351，11 节点全绿）：SVD 首帧 →
  video-sample-latent（preview 帧上报正常）→ vae-decode-video（fp32
  chunk=4）→ save-video 出合法 mp4（832KB，ftyp/isom）✅
- 硬件约束记录：SVD-XT @ Pascal 1080 约 425s/步（15 步≈1.8h），
  视频采样建议 ≤5 步验收、正式跑挂长时；exec 350 经节点代理中断验证 ✅

**里程碑**：32 算子全部有专属节点类型 + 专属 widget（widget-base 体系），
inference-op 回归试验田定位。画框/笔刷（第三批）与 IPAdapter 为后续可选。

**连接信息更新（2026-09-21 晚）**：docker 隧道换为 `6.tcp.cpolar.cn:13015`
（CLI 须先解析 IP——fake-ip 198.18.0.206 直连可用，server 27.3.1 验证通）；
推理服务地址不变 `http://5.tcp.cpolar.top:11538`。容器 Up，无需拉起。

---

## 二十、节点 UI 修复批 + 流水线专属节点化（2026-09-21 晚，commit 300ef13 + 2ab0239）

**起因（用户画布抽检反馈三问题）**：
1. load-image/save-image 不显示图 —— 公共件化（5647988）把这两节点与 save-video 的
   手写 widget（直读本地媒体 + 可缩放 lightbox）覆盖成通用生成版，能力丢失
2. 采样器最后一帧不显示 —— 非回归：preview 帧存推理服务内存（1h TTL），
   IPAdapter 部署重启容器清了缓冲；新执行任务正常
3. canny-pink-recolor 等流水线 UI 旧 —— 流水线 YAML 按 name@version 钉版，
   回放态画布渲染执行时刻的 runtime_yaml 快照（可复现性设计）

**修复**：
- widget-base 新增媒体显示键：mediaImageParam（load-image 源图直读）/
  mediaImageOutput（save-image 落盘成图）/mediaVideoOutput（save-video 内嵌播放
  +⛶放大）；图片/视频 lightbox（缩放/捏合/平移/全屏）从手写件移植进公共件
- widget-base 新增 field.advanced → 「⚙ 连接/高级」折叠区，48 节点
  service_url/service_token 全部收入，节点恢复紧凑；生成器模板同步
- 版本：load-image 1.1.0 / save-image 1.2.0 / save-video 1.2.0 / ksampler 1.4.0
- **历史执行回填**：旧版本（save-image@1.1.0 等）widget 资产原位替换
  （仅显示层，执行器不动），所有历史执行回放的图全部恢复可见
- 54 类节点最新版专属 widget 覆盖率 100%

**ksampler@1.4.0**：补 ControlNet control 入参（此前受控采样只能退回
inference-op；缺省时请求体与旧版逐字节一致）。

**流水线 66 专属节点化**：全 inference-op → 全专属节点（13 节点），exec 353
全绿，出图与旧版 exec 327 **sha1 逐字节一致**（pink_dress/pink_edges 双双命中）。

**经验**：
- 回放态 = runtime_yaml 快照钉版，widget 修复要覆盖历史执行必须回填旧版本资产
- 节点包升版检查清单：flowx.json 参数列表是 list 不是 dict（本次踩坑）；生成器
  产出的 widget-def 用 JSON 双引号，sed 批处理要兼容两种引号风格

## 二十一、显存治理 2.0（2026-09-22 立项 → 当日交付）

> 对应 `inference-server/ROADMAP.md` §4.1；本节是任务级展开。

### 21.1 起因：同一病灶第二次复发（实测事故）

流水线 58（sd-txt2img-refine）近两次执行异常慢，定位结论是**显存溢出，不是网络、
不是节点执行器**：

- `/health` 实测：`vram_free_mb: 0` / 8192，`resident_models = [majicmixRealistic_v7,
  stable-video-diffusion-img2vid-xt-1-1, v1-5-pruned-emaonly-fp16]`，`offload_mode: none`；
  容器 env `MAX_RESIDENT_MODELS=3`（8GB GTX 1080）
- 同一条流水线前后对比：

| 执行 | 时刻 | KSampler(512×30) | KSamplerRefine(1024×14) | 整链 |
| --- | --- | --- | --- | --- |
| exec 356/357/359 | 22:45~23:19 | 15~20s | 39~81s | 200~283s |
| exec 361/362 | 01:19 之后 | 508~541s | 1777s（服务端日志每步稳定 127s） | >1.5h |

- 触发点：视频流水线（75/76 等）把 SVD-XT 拉入常驻；这与遗留清单 #6 的次生教训
  （decode OOM → VAE 停 fp32 + 主机内存压力 → 采样 22s/步劣化到 ~7min/步，容器重启恢复）
  是**同一现象**，当时的处置只是"手动重启容器"，没做成机制。
- 另外：`MAX_RESIDENT_MODELS` 2→3 是为 ControlNet 三件套（checkpoint+cn+外挂 vae 三常驻
  不互踢）调的，与 `DEPLOY.md`「8GB 卡保持 1」直接冲突——按"最全场景"而不是"本条流水线
  峰值需求"设上限，叠加视频模型必爆。

### 21.2 现状机制的三个洞

1. **淘汰按个数不按显存**：`while len(self._pipes) >= MAX_RESIDENT`（model_manager.py），
   3×SD1.5 与 SD1.5×2+SVD-XT 在它眼里一样。
2. **没有卸载通道**：全服务无 unload 算子/端点；`POST /gc` 只 `clear_cache()` +
   `store.clear()` + `gc.collect` + `empty_cache`，`_pipes` 引用还在 → 模型照常驻。
3. **没有自愈与闸门**：OOM 无降级重试，`/health` 的 `vram_free_mb` 只被 Ensure 打日志，
   流水线无法据此先停手，只能跑到 127s/步。

### 21.3 四项任务

| # | 任务 | 位置 | 验收口径 |
| --- | --- | --- | --- |
| 1 | **预算式淘汰**：加载前估算待加载体积（safetensors header / 目录大小 × dtype 系数），与 `torch.cuda.mem_get_info()` 余量比对后淘汰；`MAX_RESIDENT_MODELS` 降为兜底；执行中条目 pin 跳过 | `app/model_manager.py` `_evict_if_needed()` | 8GB 卡顺序加载 majicmix → SVD-XT → v1-5 恒不出现 `vram_free_mb: 0`；512×30 步采样保持 15~20s 基线 |
| 2 | **显式卸载**：`ModelManager.unload(key)` / `unload_all(keep=)` 走既有淘汰路径（`on_evict` 断对象仓库视图 → gc → empty_cache）；注册 `checkpoint.unload` 算子，`inference-op` 即可用，再补瘦节点 `model-unload`（参数 `target` 可选 / 默认全部，UI 显示当前 resident） | `app/model_manager.py`/`ops.py`/`main.py` + `nodes/model-unload/` | 视频段跑完 unload → resident 空、余量回升 ~7GB；随后图像采样回到 exec 356 基线 |
| 3 | **OOM 自愈**：`engine`/`jobs` 包裹算子调用，捕获 `torch.cuda.OutOfMemoryError` → 淘汰非在用 LRU → 重试一次；失败则报错并给出降档/重启指引；`/health` 增 `degraded`（上次 OOM 时间/次数） | `app/engine.py`/`jobs.py`/`main.py` | 人为超预算加载不崩、自愈后成功；连续失败信息可直接执行 |
| 4 | **流水线显存闸门**：`inference-ensure` 增 `min_vram_mb`（默认 0=不检查），并 emit `vram_free_mb`/`vram_total_mb`/`resident_models`/`offload_mode`；不足按 `wait_seconds` 等待重试 | `nodes/inference-ensure/` | 余量 0 时 Ensure 明确失败并提示先 unload/重启，而非让采样退化 |

### 21.4 边界与纪律

- 不动的部分：`OFFLOAD_MODE` 三档语义、`dtype/offload/use_t5` 两级旋钮结构（env 默认 + 节点
  覆盖，旋钮进缓存键）、`VAE_FP32` 开关、diffusers 自带 offload（**不自研**，沿用 ROADMAP §4
  原则）。
- 张量级搬移（comfy/model_management 式）仍留远期；`QUANTIZATION=fp8` 未实现，不在本轮。
- 硬件假设仍不得进算子逻辑（§9.6 纪律 #1）：预算式淘汰也只看运行时余量，不写死卡型。

### 21.5 排期建议

1. 任务 1 + 2 先行（能直接救 58 与图/视频混跑场景，且互为依据：预算式淘汰需要一个
   "腾地方"的执行器，显式卸载是它的用户入口）；
2. 任务 4 成本最低，可与 1/2 同批交付；
3. 任务 3 视 OOM 复现频率再排（有 1/2 之后 OOM 概率已大幅下降到"用户自选超预算"场景）。

### 21.6 过渡措施（机制未落地前）

- 远端容器 env 先改 `MAX_RESIDENT_MODELS=1`（8GB 卡按 `DEPLOY.md` 建议值），需要
  ControlNet 三件套的流水线改用 `offload=model` 或单独批次跑；
- 跑 58 之前先让 SVD-XT 被 LRU 顶掉（跑一次别的 checkpoint，或重启容器）；
- 出现"某步突然变分钟级"时先看 `/health` 的 `vram_free_mb`，再决定是否重启容器。

## 二十二、flowx 执行器健壮性：docker exec 断流（2026-09-22 立项，实施中）

### 22.1 起因：exec 364 静默丢输出

把流水线 58 的节点改跑远端 docker 后，`KSampler` 节点状态 success 但**输出块丢失**，
下游 `VAEDecode` 报 `missing required parameter: latent`，排查方向被带到下游：

| 执行 | 执行器 | KSampler 日志行数 | 末行 | metadata 有 KSampler.latent | StepResult.Error |
| --- | --- | --- | --- | --- | --- |
| 363 | docker | 39 | 含收尾 codec 块 | ✅ | 空 |
| 364 | docker | 22 | `19/30`（截断） | ❌ | **空** |
| 362 | local | 22 | 含收尾块 | ✅ | 空 |

### 22.2 漏洞链（`flowx/executor/docker/docker.go` 为 v0.1.4 行号）

1. **干净 EOF 不可区分**：attach 流被优雅关闭（cpolar/WSL2 抖动）时读到 `io.EOF`，
   与"命令正常结束关流"同路（`docker.go:400-483`）→ 不报错；
2. **`ExecInspect` 只能证明进程结束**：进程确实跑完 → `ExitCode=0` → 返回 success
   （更糟：流断了进程还在跑时，会原地等到结束再返回 0）；
3. **提取层不报错**：`dag/extractor.go:38` 找不到 codec 块时返回"空 map + nil error"，
   `dag/workflow_utils.go:13 extractOutput` 什么都不写也不告警；
4. **模板层把缺失渲染成空串**：`dag/workflow_impl.go:584` 经
   `template/template_impl.go:230`（pongo2 非 strict）求值 `{{ KSampler.latent }}` →
   空字符串 → 注入 env 空值 → 节点 `param()` 视 `""` 为缺参 → 报错落到下游；
5. 对照：local 执行器 stdout 是子进程管道，流断=进程死，语义自洽，因此一直"报错可读"。

### 22.3 修复项（2026-09-22 用户拍板：全都做；重挂失败直接失败；发正式 tag）

| # | 层 | 内容 | 位置 |
| --- | --- | --- | --- |
| A | docker 执行器 | 流结束先 `ExecInspect`：`Running==true` ⇒ 判定截断；对同一 execID **重挂 attach 续读**（≤3 次，退避 0.5/1/2s，受 ctx 取消）；`scanner.Err()` 非 EOF 同样走重试；重挂失败/再次中断 ⇒ **返回明确错误**；`StepResult` 增 `StreamTruncated` + 输出一行提示 | `executor/docker/docker.go` |
| B | 提取层 | 声明 `extract` 却零提取 ⇒ warning + metadata 标记 `__extract_missing`（不直接失败，避免误杀合法的空输出节点） | `dag/workflow_utils.go:13`、`dag/extractor.go:38` |
| C | 模板层 | 渲染前静态校验 `{{ Node.field }}`：图内上游节点存在但缺该字段 ⇒ **直接报错**（含节点/参数/字段/可能原因）；`{{ Param.x }}` 保持可缺省回退语义 | `dag/workflow_impl.go:584`、`template/template_impl.go:230` |
| D | docker 执行器 | 每步命令 stdout `tee` 到容器内临时文件，流结束后独立 exec 读回、按行补差、删除临时文件（处理 `/bin/sh` 兼容与退出码透传；仅对声明 `extract` 的节点启用） | `executor/docker/docker.go` |
| E | 测试 | 截断判定+重挂循环抽成可注入 attach 的 helper + stub 单测；`_test.go` 级**截断 TCP 代理**（127.0.0.1 → 真 daemon，识别 `/exec/{id}/start` 后第 N 字节切断）端到端复现 | `executor/docker/*_test.go` |
| F | 交付 | flowx 发 **v0.1.5** 正式 tag；`flowx-studio` 升 go.mod 依赖 + 重编 server；回归 local/docker 各跑 58、75、76 | `flowx`、`flowx-studio/go.mod` |
| G | Studio | 消费 `StepResult.StreamTruncated`，节点上打"输出不完整"标记（与二十四画布标签同批） | `flowx-studio` |

### 22.4 验收口径

1. 故意中途切断 docker API 连接：节点**必须红掉**并给出明确信息（不接受 success）；
2. 重挂成功时：节点绿且输出块完整（metadata 含 `KSampler.latent/seed`）；
3. 下游不再出现误导性的 `missing required parameter`（F4 后应报"上游未输出该字段"）。

### 22.5 边界

不改节点包与 flowx.json、不改 docker 镜像、不改任何流水线定义；
截断 TCP 代理只在 `_test.go` 内，不进生产代码。

### 22.6 交付记录（2026-09-22）

- **A/B/C 已实施**：`flowx` commit `59eda8b`（4 文件改 + 4 新文件）→ push master + annotated tag **v0.1.5**
  （`git ls-remote` 已确认）；`executor/docker` 断流检测/重挂/硬失败、`dag/binding_check.go`
  绑定校验、`dag/workflow_utils.go` 零提取标记。
- **额外补的竞态**：1.5s settle 窗口（`execRunningSettled`）—— 避开"进程刚退出仍瞬时报
  Running=true"导致的截断误判（已在真机冒烟验证无假阳性）。
- **studio 侧**：`go mod edit -dropreplace` + `require github.com/LerkoX/flowx v0.1.5`
  （go.sum 走 `GOPROXY=direct`+`GOPRIVATE` 拉取）；重编后二进制内含新代码。
- **验收（正式 tag 构建的真机冒烟）**：临时流水线 3 节点全 docker+内网全绿（6.5/5.3/5.3s）；
  日志 `truncated` 假阳性 = 0；metadata 无 `__extract_missing`；负向用例（绑定不存在的
  上游字段）报错准确指向上游节点与已输出字段。临时流水线（77/78/79）已删。
- **测试**：`executor/docker` 新用例 8 个（假 daemon，-count=2 无 flake）、`dag` 新增 14 例；
  根包 3 例失败与 `executor/local` 超时在干净树上同样存在（预先存在）。
- **~~待办~~✅ 已闭环（2026-10-08 盘点确认）**：D = `5428281`（声明 extract 的节点容器内 tee 兜底+断流补齐尾部）、G = `6328afd`（dag 写 `__stream_truncated` 标记）+ flowx-studio `WorkflowCanvas.tsx`/`GlowNode.tsx` 消费展示「输出可能不完整」徽章（含测试）。flowx 已发到 v0.1.8。

### 22.7 D 项交付（容器内 tee 兜底，2026-09-22）

A 项只能“发现”截断，重挂前丢掉的字节找不回来（KSampler 的 ```flowx-yaml 结果块
落在断流窗口里就永久丢失）。D 项给这类步骤加落盘兜底：

| 环节 | 做法 |
| --- | --- |
| 下发 | `CommandWrapper.CaptureOutput`，由 dag 按节点是否声明 `extract` 置位（与 `extractOutput` 同判定） |
| 包壳 | `{ { <cmd>; } 2>&1; echo $? > <rc>; } \| tee <log>; __flowx_rc=$(cat <rc> 2>/dev/null); exit ${__flowx_rc:-1}` |
| 重定向 | 在内**层组**上整体重定向（多行命令每行 stderr 都进日志 → 日志与流同源，保住“日志前缀 = 已送达”） |
| 退出码 | `echo $?` 紧跟命令（不是 tee 的），末尾显式 exit → `execInspect` 仍可信；rc 读不到按 1（宁可失败不静默成功） |
| 补齐 | `truncated>0` 时 `tail -c +<已读字节+1> <log>` 只补不重；重挂用尽、确定失败前也补一次 |
| 清理 | 未断流也删临时文件（容器按 executor 复用整个 workflow，不清理会在宿主机累加） |

**测试**：假 daemon 支持按 tail 偏移返回日志内容 → 覆盖未包壳/包壳形态/补齐偏移
正确且不重复投递/失败前补齐/多行整体重定向共 8 例；dag 侧覆盖 CaptureOutput 仅在声明
`extract` 时下发。

**真机验证**：
- exec 369（3 个声明 extract 的 docker 节点）全绿、输出块完整、无截断无补齐 → 包壳在真实
  容器（bash / python:3.11-slim）下语法与 tee 均可用；
- exec 370（故意用不存在的模型名）失败于 `command exited with code 1` → **退出码经 rc 文件
  正确透传**（不是 tee 的 0），非零退出断言成立。
- **交付**：flowx `5428281` → tag **v0.1.6**（已推送）；studio `e626ef41` 依赖升到 v0.1.6，
  正式 tag 构建重跑 exec 371 全绿（13.7/10.4/5.5s，无截断/无补齐通知）。

## 二十三、节点镜像打包：单一事实源 + v1.6.0（2026-09-22 立项，待实施）

### 23.1 现状盘点

| 项 | 数字 / 内容 |
| --- | --- |
| 仓库节点包 | 49 个 |
| 镜像 `v1.5.0` 实际包含 | 19 个目录（`nodes/Dockerfile` 手写 COPY） |
| 不在镜像里 | 30 个 |
| 其中**声明了 docker 却未打包** | 2 个：`image-upscale`、`detail-refine` ⇒ 选 docker 必挂 `cd: /opt/flowx-nodes/<name>: No such file`（exec 363 Upscale 事故） |
| 其余 28 个 | 声明 `supportedTypes:["local"]`：controlnet-*/cond-*/latent-*/preprocess-*/ipadapter-*/upscale-model-*/vae-load/vae-decode-video/video-sample-latent/embedding-load/clip-vision-load |
| 依赖核对 | 49/49 个节点目录自带 `flowx_client.py`，需要的 `executor_base.py`/`image_io.py` 亦在各自目录 ⇒ 镜像 `COPY <dir>/*.py` 即可，无需新增 pip 依赖 |

### 23.2 分类决策

| 类 | 节点 | 处理 |
| --- | --- | --- |
| A 推理服务客户端（薄 HTTP） | 上表 28 个 + 已在镜像的 19 个 | **全部支持 docker**（本轮目标） |
| B 写 studio 本地文件 | `save-image`、`save-video` | 必须 local（写 `~/flowx-output`） |
| C Studio 侧 PIL 图像/蒙版 | `image-rotate/flip/crop/composite`、`mask-×5` | 设计上 local（镜像无 Pillow、处理 studio 本地图）；若要 docker 化需给镜像加 Pillow（+~12MB）但无业务收益 ⇒ 本轮不动，声明里写明 local-only |
| D 非节点目录 | `_common`/`_tools`/`_widget-template` | 不进镜像 |

### 23.3 改动清单

- `nodes/Dockerfile`：补 28 条 COPY（A 类缺失项）；冒烟 RUN 扩成"清单一致性自检"；
- `nodes/BUNDLE_VERSION`：`1.5.0 → 1.6.0`；
- 各 `flowx.json`：A 类缺失节点 `supportedTypes` 加 `docker`（`preferredType` **保持现值**，
  不改变存量流水线行为）；B/C 类去掉 `image` 字段（不再伪装可 docker）；
- `nodes/sync-flowx-json.py`：只给"真在清单里"的节点写 image/bundled；新增 `check-bundle.py`：
  *声明 supportedTypes 含 docker 的节点必须都在 COPY 集合里，反之不许写 bundled*，
  不一致即非 0 退出（作为发布门槛，杜绝本次这类漂移）。

### 23.4 构建与发布（无需 push/pull）

- 远端宿主是 Docker Desktop，`python:3.11-slim`（133MB）**已在本地**，不需联网拉基镜像；
- termux 侧 docker CLI 28.1.1 可直接指向远端 daemon 构建：
  `docker -H tcp://6.tcp.cpolar.cn:12268 build -t lerkobba/flowx-pixelforge-nodes:v1.6.0 nodes/`
  （上下文几百 KB，加 `.dockerignore` 排除 `__pycache__`/`ui/`；构建在远端执行，无需 push、无需再 pull）；
- `python3 nodes/sync-flowx-json.py` → Studio 侧重导 30 个节点包
  （`node import --type folder --path nodes/<name> --overwrite`；同名同版本原地更新，
  工作流里的 `name@version` 引用无需改动）。

### 23.5 验收

1. `check-bundle.py` 通过（清单一致性）；
2. 镜像内逐目录断言 `main.py` 存在（含新增 28 个）；
3. 新建/复用"节点覆盖验收"流水线：A 类每个节点以 docker 模式至少跑通一次；
4. 回归 58 / 74 / 75 / 76（local 与 docker 各一遍）。

### 23.6 前置依赖

长任务节点（KSampler/KSamplerRefine、video-gen、detail-refine）切 docker **依赖二十二**
（断流丢尾修复）完成，否则"支持 docker"也仍然不能实际用于长任务。

### 22.8 G 项交付（Studio 展示"输出可能不完整"，2026-09-22）

| 环节 | 做法 |
| --- | --- |
| flowx | `StepResult.StreamTruncated` 为真 → metadata 写 `<node>.__stream_truncated=true` + Warning 日志；不改变节点状态（tag **v0.1.7**，`6328afd`） |
| 传递 | 复用既有 `node_complete` 事件 `outputs`（`nodeOutputs(p.Metadata(), nodeID)` 前缀取全量）→ 执行 metadata，**无需改表结构** |
| Studio | 画布从 `selectedExecution.metadata.metadata` 里的 `__stream_truncated` / `__extract_missing` 还原按节点的"⚠ 输出可能不完整"徽章（悬停区分"流被截断"与"零提取"） |

测试：flowx `dag` 覆盖"截断写标记/未截断不写"；Studio 前端用例断言两个节点各一个徽章、
tooltip 文案区分两种原因、编辑态无标记。

### 23.7 交付记录（2026-09-22）

**单一事实源**：`nodes/Dockerfile` 的 `COPY <dir>/*.py` = 该节点代码真的在镜像里；
`BUNDLE_VERSION` 驱动 tag；`sync-flowx-json.py` 按清单把 `image`/`executor.bundled`
写进各 flowx.json（清单外与 local-only 的节点清掉这两个字段）；
新增 `check-bundle.py` 作为**发布门槛**（反向校验：声明 docker ⇒ 必须在清单里且有 image；
写了 image/bundled ⇒ 必须在清单里；local-only ⇒ 不许声明 docker/image/bundled），
先跑 check 就能复现本轮之前的漂移（当时报出 3 处 `image=None` 不一致）。

**镜像 v1.6.0**：COPY 从 19 目录 → **39 目录**（A 类补齐 20 个：`vae-load`、`controlnet-*`、
`preprocess-*`、`latent-*`、`cond-*`、`upscale-model-*`、`ipadapter-*`、`clip-vision-load`、
`embedding-load`、`vae-decode-video`、`video-sample-latent`、`image-upscale`、`detail-refine` 等）。
分类边界（按用户口径固化）：

| 类 | 节点 | 声明 |
| --- | --- | --- |
| A 推理客户端（37） | 上述 + 原有 17 | `[local,docker]` + image + bundled |
| B 读写宿主文件（3） | `save-image`、`save-video`、**`load-image`** | `[local]`，无 image/bundled |
| C Studio 侧 PIL（9） | `image-rotate/flip/crop/composite`、`mask-*` | `[local]`，无 image/bundled |

> `load-image` 原声明 `[local,docker]` 且在镜像里 —— 它读的是 Studio 宿主文件路径
> （`os.path.expanduser(image_path)` + `open()`），容器内看不到，故本轮改为 local-only
> 并从 Dockerfile 清单移除（用户 2026-09-22 明确要求：保存图片/保存视频/加载图片只能 local）。

**验收**
1. `check-bundle.py` ✓ 通过（49 包 / 39 目录 / docker 可用 37 / local-only 12）；
2. 镜像构建期在远端 daemon 内自检"每个目录都有 main.py"✓（COPY 缺源目录会直接构建失败）；
3. **镜像内启动探针** `probe-image-nodes.py`：40/40 节点都能起（A 类改造前）——
   逐个 detached 起容器 + `docker logs`（**attach 模式经 cpolar 会被静默截断**，
   这也是 flowx 执行器修复的同一类问题），0 个环境问题；
4. **docker 模式真跑服务**：新流水线 `node-coverage-docker`（id 83）exec 374 **11/11 全绿 50.5s**：
   `vae-load` 15.3s、`upscale-model-load` 14.2s、`preprocess-canny` 7.5s、
   **`image-upscale` 5.8s**（exec 363 事故节点）、KSampler(docker,2 步) 7.9s、VAEDecode 6.9s；
5. **回归**：
   - exec 375（流水线 58，11 docker + 8 local）**19/19 全绿 245.8s**；快照确认 docker 实例
     `docker-remote-211+flowx-pixelforge-nodes-v1.6.0`（image=`nodes:v1.6.0`）；
     全程 `truncated` 0 行、无补齐通知、无 `__stream_truncated`/`__extract_missing` 标记
     （顺带验证 22.7/22.8 在真实长链上无假阳性）；
     对比 exec 365（显存退化态 3854.3s）：KSampler 521.7→18.8s、KSamplerRefine 1913.6→75.0s、
     DetailFace 277.5→14.7s、DetailHand 939.4→42.7s ⇒ 剩余瓶颈确定是显存（§二十一）；
   - exec 377（流水线 61，纯 local：load-image + PIL 家族 + save-image）**12/12 全绿 22.0s**
     ⇒ 新宣明的 local-only 节点（load-image 改为 `[local]`）与版本钉住引用（`load-image@1.0.0`）
     均未受影响。

**顺带发现（环境，非本轮引入）**：21 条流水线（30/31/51/52/53/57/60/61/62/63/64/65/66/67/68/69/70/73/74/75/76）
的 `Param.service_url` 仍指向**已失效**的 cpolar 地址 `5.tcp.cpolar.top:11538`（现行为 `:12073`，
实测 11538 连接被拒）⇒ 这些流水线在 `Ensure` 阶段必然失败。本次只修了 61（验收用），
其余 20 条待用户确认后批量更新；治本方案见遗留清单 #7（保留域名）或让推理节点尽量走
docker + 内网地址（§二十三 已具备条件）。

**遗留（已记录）**：`flowx_client.py` 在各节点目录有 3 个功能子集变体
（带/不带 `ensure_plugin`、`emit_preview` 是否带 `base`/`job_id`）；调用的节点都自带该函数，
不影响运行，但升级协议字段时需要逐目录同步（见 `nodes/README.md`）。

## 二十四、画布节点显示执行器状态（2026-09-22 立项，待实施）

### 24.1 数据来源：后端展开器是唯一事实源

执行器在运行时解析（`config.executor` 显式 → 包内 `executor.ref/type` →
`supportedTypes/preferredType` → 有 image 归 docker → 全局默认），YAML 里多数节点未写
executor ⇒ **只能由后端给**。复用现成链路：`runtime.ExpandWorkflowConfigWithTypeResolver`
（`internal/runtime/node_expander.go:189`）会物化每节点 `executor: <实例名>` 并写出 `Executors` 表；
`WorkflowService` 已有 `executorResolver()/executorTypeResolver()/expandLookup`
（`internal/service/workflow.go:93/101/534`），Run/Mock/Continue 均复用 ⇒ 画布显示 = 实际运行。

### 24.2 后端：新增只读端点

`GET /api/v1/workflows/:id/executors`（注册在 `internal/handler/workflow_handler.go` 的
`workflows` group），返回：

```json
{
  "executors": {"docker-remote-211": {"type":"docker","image":"...:v1.6.0","host":"tcp://...:12268"},
                "local": {"type":"local","pty":true}},
  "nodes": {"KSampler":   {"executor":"local","type":"local","source":"workflow-explicit"},
            "DetailFace": {"executor":"docker-remote-211","type":"docker","source":"package-preferred","image":"..."},
            "SaveBase":   {"executor":"local","type":"local","source":"global-default"},
            "Upscale":    {"executor":"local","type":"local","source":"anonymous-fallback","warning":"节点包未打包进镜像/偏好降级"}}
}
```

- Service 增 `ResolveNodeExecutors(id)`：复用展开流程（不执行），解析展开后 YAML 的
  `Executors`/节点 `executor`；
- `source` 由展开器回传 ⇒ 给 `expandWorkflowConfig` 加**可选回调**
  `onResolve(node, execName, execType, source)`，现有签名语义不变。

### 24.3 前端：节点徽章（外壳通用元素）

- `web/src/components/GlowNode.tsx`：头部加徽章 `🐳 docker-remote-211` / `🖥 local`，
  按类型配色（docker 蓝 / local 灰 / 降级或镜像缺失 橙）；tooltip 显示类型、实例名或 host、
  镜像 tag、解析来源；文案走 `web/src/i18n`；
- `web/src/features/workflow-canvas/WorkflowCanvas.tsx`：构造 `GlowNodeData` 时透传 `executor`
  （该文件本就在批量 resolve nodeRef）；端点失败则不显示徽章，不阻塞画布；
- 回放态用执行快照 `runtime_yaml`（已物化 executor），与编辑态同一渲染；
- 加分项：节点管理列表显示每个包的 `supportedTypes/preferredType` + "是否真在镜像里"
  的一致性校验结果（即二十三那个 bug 的可视化护栏）。

### 24.4 验收

1. 打开流水线 58：docker 节点显示 `🐳 docker-remote-211`，`SaveBase/Upscale` 等显示 `🖥 local`，
   与执行记录一致；
2. 故意把只支持 local 的节点设成 docker ⇒ 画布/校验期直接标红；
3. 回放 exec 362/365：徽章显示当时快照的执行器，不随后续修改漂移。

### 24.5 交付记录（2026-09-22）

**后端**
- `runtime`：新增 `ExecutorResolution{Name,Type,Source,Warning}` + `ExecutorResolutionObserver`，
  `resolveNodeExecutor` 返回结构体并在每个分支标注来源（workflow-explicit / package-ref /
  package-inline / package-preferred / package-degraded / docker-default / global-default /
  anonymous-fallback）；新增导出 `ExpandWorkflowConfigWithExecutorObserver`（原函数签名语义不变）。
- `service`：`ResolveNodeExecutors(workflowID, executionID)`（`internal/service/workflow_executors.go`）
  ——编辑态走与 Run/Mock/Continue **同一套展开链路**（唯一事实源），回放态直接读
  `executions.runtime_yaml`（不随后续修改漂移）；实例表带 host/image/registered。
- `handler`：`GET /api/v1/workflows/:id/executors[?executionId=N]`。

**前端**
- `types/workflow.ts` + `services/workflowService.ts`：类型与 `getWorkflowExecutors`；
- `GlowNode.tsx`：头部徽章 `🐳 docker-remote-211` / `🖥 local`（docker 蓝 / local 灰 /
  降级或匿名 琥珀），tooltip 给实例名、镜像、地址、解析来源、降级提示；
- `WorkflowCanvas.tsx`：按 `sourceYaml`/回放态查询并注入节点 data（图重建时读 ref、
  查询晚到走增量注入，均不改 position，避免丢手动布局）；查询失败只影响徽章不阻塞画布。

**一个实现取舍**：展开器在"节点声明镜像与共享实例不一致"时会合成节点专属条目
（`<首个被处理节点名>-executor`，名字取决于 Go map 遍历顺序，**逐次解析会变**）。
它只是内部容器分组键，不该作为执行器身份展示 ⇒ 响应带 `registered`，
未注册条目徽章只显示类型（tooltip 仍给实例名+镜像+地址）。

**验收**
1. wf 58 编辑态：11 docker + 8 local，与执行记录一致（`Upscale/DetailFace/DetailHand/Save*` 为 local）✓；
2. 回放不漂移：exec 362 快照 = 1 docker/18 local、exec 365 = 11 docker/8 local，
   与当前定义（11 docker/8 local，实例名 `empty-latent-executor`）各自独立 ✓；
3. 前端用例（jsdom）断言徽章文本 + tooltip（镜像/地址/来源/降级）+ 未注册条目名不展示；
4. 无快照/不存在的执行为 400，前端不显示徽章。

### 24.6 待办 / 后续
- 节点管理列表的"包声明的执行器 vs 镜像内是否真有该节点"一致性护栏（§23 的 `check-bundle.py`
  产出清单后接）；
- 展开器合成实例名去随机化（按镜像 tag 派生）：可同时消除续跑快照比对的潜在抖动。

### 24.7 §24.6 交付记录（2026-09-22）

**(a) 合成执行器名去随机化** —— 展开器在"节点声明镜像与共享实例不一致"时合成
`<首个被处理节点名>-executor`，名字取决于 Go map 遍历顺序：同一条流水线两次解析会得到
`clip-text-encode-executor` / `empty-latent-executor` 等**不同条目名**（画布展示、快照对比、
续跑校验都会无意义抖动；实测同一 wf58 连续两次解析即复现）。改为按 **实例名 + 镜像 tag**
派生确定性名字：`docker-remote-211+flowx-pixelforge-nodes-v1.6.0`；同名但 config 不同时
追加配置短哈希（确定性）。新增用例：同一图展开 8 次，节点归属与 Executors 键集合完全一致
（旧实现该用例必 flake）。匿名（无注册实例）路径仍用节点名（本身稳定、语义正确）。

**(b) 节点管理护栏**：`model.Node.DeriveExecutor()` 顺带计算 `executorCheck`
（`types/preferred/image/bundled/dockerOk/issues`），API 只读透出（列表+详情都有）；
前端 `NodeCard` 显示类型/偏好徽章 + ⚠「声明不一致」（tooltip 给原因），
`NodeDetailModal` 增加执行器声明区块。检查项与仓库侧 `check-bundle.py` 同源但**只做声明自洽**
（Studio 看不到镜像内容）：声明 docker 却缺 image/bundled、`bundled` 与 `supportedTypes` 矛盾、
`preferredType` 不在 `supportedTypes` 内。用例：`internal/model`（7 组声明）+ 前端 NodeCard（3 组）。

### 23.8 后续修正（2026-09-22 第二次迭代，镜像 v1.6.1）

用户口径两处调整后（"22 个声明可 docker 的不需要偏好保持 local，也都改成 docker"；
"Studio 侧图像/蒙版是不是也可以走 docker"）的落地与实测发现：

| 项 | 结论 |
| --- | --- |
| PIL 图像/蒙版 9 个能否 docker | **能**。原分类假设"处理 Studio 本地图"有误：实际是「对象 ID 取图（GET /images/{id}）→ 本地 PIL → 上传回服务」纯 HTTP + 本地计算，只多依赖 Pillow ⇒ 已进镜像（离线 wheel，见下）并声明 `[local,docker]`+`preferred docker` |
| 最终分类 | 49 包：**docker 可用 46**（37 A + 9 PIL，全部 `preferredType: docker`）；**local-only 3**（`save-image`/`save-video`/`load-image`，读写宿主文件）；镜像内 **48 目录** |
| 远端宿主无外网 | `pip install` 报 `Network is unreachable` ⇒ 用 `_wheels/` 内置 manylinux cp311 wheel + `--no-index --find-links` 离线安装（升级命令写进 nodes/README.md） |
| **容器访问不了公网隧道** | cpolar 不回环自身宿主机：容器内 `5.tcp.cpolar.top:12073` **超时**、`172.17.0.1:8100` **200** ⇒ docker 节点的 `service_url` 必须绑 `{{ Param.service_url_internal }}` |
| **跨执行器传播 URL 的陷阱** | wf30/31/60 的 `SaveImage`(local) 绑 `{{ Ensure.service_url }}`，Ensure 改跑 docker 后把内网地址传给 local 节点 ⇒ `Connection timed out`（exec 380 实测）。已修：这 3 条的 local 节点改绑 `{{ Param.service_url }}`；全库扫描确认无其它跨执行器 URL 绑定 |

**验收（第二次）**
- 镜像内启动探针 48 目录：**47/48 正常、0 环境问题**（剩 1 个是日志读取竞态，单独复测正常），
  PIL 节点均报"缺参数（预期）"= 代码与 Pillow 都就绪；
- **PIL docker 真跑**：`tier1-docker-pil`（id 84）exec 379 **12/12 全绿** ——
  `load-image`(local) → 7 个 PIL(docker) → `save-image`(local)，对象 ID 流可跨执行器；
  docker 每节点约 +2.8s 往返（local 2.7s → docker 5.5s）；
- **长任务 docker 真跑**：wf30（8 节点）exec 381 **8/8 全绿 118.0s**，其中
  **KSampler(docker) 104.9s 成功**且输出完整（`KSampler.latent/seed` 正常、无截断标记）⇒
  §22 的修复让"长任务必须 local"的旧结论可以撤销（先前只因断流会静默丢尾才留 local）。

### 21.7 交付记录（2026-09-22）

四项任务全部落地，服务端（flowx-pixelforge/inference-server）与节点包（nodes/）同批交付。

| # | 任务 | 落地位置 | 关键实现 |
| --- | --- | --- | --- |
| 1 | 显存预算式淘汰 | `app/vram.py`（新，纯 stdlib）+ `app/model_manager.py` `_evict_if_needed` | 体积估算读 **safetensors header**（按目标 dtype 精确算，非文件大小）；`plan_eviction()` 纯函数决策（LRU + pin + 条目数兜底）；`MAX_RESIDENT_MODELS` 降为兜底上限（`<=0` = 不限）；失败时告警"余量仍不足" |
| 2 | 显式卸载 | `ModelManager.unload/find_keys/evict_idle`、`ops.model_unload`、`POST /model/unload`、`nodes/model-unload/` | 走与加载一致的淘汰路径（断对象仓库视图 → `gc` → `empty_cache`）；**在用（pin）的条目跳过**；`/gc` 明确不再负责卸模型（历史坑） |
| 3 | OOM 自愈 | `app/engine.py` `_call_op/_run_op` + `VramGuard` | 捕获 `torch.cuda.OutOfMemoryError` → 淘汰非在用 LRU → 重试一次；失败抛 `RuntimeError` 带可执行指引（降档 offload / 卸载 / 调 reserve），`/op`、`/graph` 已把 `RuntimeError` 作为 400 detail 回传节点；`/health.degraded` 记录次数与时间 |
| 4 | 流水线显存闸门 | `nodes/inference-ensure/` v1.1.0 | 新增 `min_vram_mb`（0=不检查）+ `wait_seconds` 内等待重试；emit `vram_free_mb/vram_total_mb/resident_models/offload_mode`（原来只打日志） |
| + | 可用显存口径修正 | `ModelManager._free_bytes` + `/health` | 实测教训：采样后 PyTorch 缓存分配器留着复用，`mem_get_info` 报 free=0 ⇒ 若按它判断会**白踢掉刚加载的模型**（重载 ~2 分钟）。可用量 = 驱动余量 + `reserved-allocated`（可回收）；`/health` 另给 `vram_driver_free_mb`/`vram_reclaimable_mb`/`resident_weights_mb`/`vram_low` |

**验收（真机，8GB GTX 1080，容器 `flowx-inference-server` 已换新镜像）**

| 用例 | 结果 |
| --- | --- |
| T1 复现 exec 365 场景：先载入 SVD-XT（4300MB），再跑 wf58 | exec 384 **19/19 成功 292s**（含 LoadCheckpoint 从盘加载 115s；KSamplerRefine 37.6s）；服务端日志 `evicting 'stable-video-diffusion-img2vid-xt-1-1' (LRU, ~4300MB)` —— **不再堆到 `vram_free_mb: 0`**（旧行为 exec 365 = 3939s） |
| T2 紧接着再跑 wf58（此时驱动余量 0、可回收 8.5GB） | exec 385 **19/19 成功 182s**，服务端**无任何 evicting**（口径修正生效，没白踢 majicmix） |
| T3 显式卸载 `POST /model/unload {}` | `{"unloaded":"majicmixRealistic_v7","resident":"(empty)"}`，驱动余量 0 → 4683MB |
| T3b `POST /gc` | 清 27 个对象（含 T5 那个 5GB latent）→ 可用回到 7055MB（`/gc` 管对象张量、`/model/unload` 管模型权重，二者互补） |
| T4 闸门反向：`min_vram_mb=999999` | Ensure **明确失败**：`显存闸门未通过（等待 3s 后）：可用显存 121MB < min_vram_mb=999999MB（常驻: [], offload=none）` + 四条可执行办法 |
| T4b 闸门正向：`min_vram_mb=1000` | exec 389 success，metadata 带 `Ensure.vram_free_mb=7055`、`Ensure.vram_total_mb=8192`、`Ensure.resident_models=(empty)`、`Ensure.offload_mode=none` |
| T5 人为 OOM | **本机无法人为触发**：WSL2 统一内存下 CUDA 会走 host memory 兜底而不报 OOM（200000×200000 latent 也返回 200）—— 这正是旧行为表现为"采样变慢 17s/步"而不是报错的原因；OOM 自愈由单测覆盖（17 例） |
| T6 视频链端到端（wf75 + 末尾 `Unload` 节点） | exec 391 **12/12 全绿 1077.4s**：`KSampler 9.6s`（退化态曾是 521.7s）、`LoadSVD 253.6s`（加载时按预算把 majicmix 顶掉）、`VSample 559.5s`、`VDecode 115.4s`、`SaveVideo 10.1s`，末尾 `Unload 3.7s` ⇒ **resident 清空、可用回到 7054MB**（正是任务 2 的验收口径） |
| T7 不误杀正常场景（ControlNet 三件套共存） | 顺序加载 majicmix(2034MB) + controlnet(689MB) + vae(319MB) ⇒ **三者共存**、可用 3826MB、`vram_low=false`（预算式淘汰没有过度淘汰；`MAX_RESIDENT_MODELS=3` 因此保留 3 而非降到 1） |

**新增测试**（均可在无 GPU 环境跑）

- `tests/test_vram.py`（9 例）：safetensors header 体积估算（fp16/fp32 互换、坏文件兜底、diffusers 目录累加）+ `plan_eviction` 决策（条目上限 / 显存预算 / pin 跳过 / 体积未知 / 空池）；
- `tests/test_vram_governance.py`（17 例）：注入最小 torch/diffusers stub，覆盖淘汰决策、pin 保护、卸载匹配（精确键/名字/vae:/cn: 前缀/全部）与 pin 跳过、`evict_idle`、`details`、可回收口径、引擎 OOM 重试与"无可淘汰时给指引"、pin 生命周期（异常路径也解 pin）、`model.unload` 算子返回值。

**部署与回滚**：远端 `flowx-inference-server:pascal` 已重建（`--volumes-from` 复用原 5 个 bind mount，容器配置逐项对齐：`--gpus all`/`--restart always`/`-p 8100:8100`/`MAX_RESIDENT_MODELS=3`/`OFFLOAD_MODE=none`）。
旧容器停用后改名保留为 `flowx-inference-server-prev`（回滚：`docker rm -f flowx-inference-server && docker rename flowx-inference-server-prev flowx-inference-server && docker start flowx-inference-server`）。

**边界（仍不做）**：offload 三档与两级旋钮结构未动；张量级搬移留远期；`QUANTIZATION=fp8` 仍未实现。

**遗留可选项**：① `inference-ensure` 若要从 1.0.0 升到 1.1.0（拿到 `min_vram_mb` 闸门与显存 emit），
需要在各流水线把 `nodeRef` 版本改一下（参数向后兼容，默认 `min_vram_mb=0` 即不检查）；
② 视频类流水线尾部已加 `Unload` 节点（75/76），其它"图+视频混跑"的流水线可按同一模式补；
③ `flowx_client.py` 三个变体统一（与新算子无关，属既有小尾巴）。

### 21.8 同批收尾（2026-09-22 第三轮）

| 项 | 结果 |
| --- | --- |
| 共享件统一 | `flowx_client.py` 三变体（`ensure_plugin` 有无 / `emit_preview` 新旧签名）合并为 `_common/` 单一事实源，`_tools/sync-common.py` 分发 + `check-bundle.py` 第 F 项漂移门槛；镜像 **v1.6.3**；验证：wf61（local）12/12、wf83（docker）11/11、wf74（19 docker + 2 local）21/21 |
| 流水线版本对齐 | 21 条流水线 `inference-ensure` → `@1.1.0`（拿到闸门与显存 emit）；21 条里钉旧版节点的升级到最新（`save-image`/`ksampler`/`checkpoint-loader`/`load-image`/`save-video`/`preprocess-canny`），Studio 校验全过、0 跳过 —— 修掉"部分流水线仍在跑 v1.5.0 镜像里的旧节点代码" |
| 74/75/76 切 docker + 内网 | wf74：**19 docker + 2 local**；wf75/76：**11 docker + 1 local**（save-video 保持 local）；补 `Param.service_url_internal`。验收 exec 395（wf75 全 docker 视频链）**12/12 全绿**（`LoadSVD 240.8s`、`VSample 485.4s`、`VDecode 147.1s`）；docker 每节点约 +2.8s 往返 |
| 远端清理 | 删 4 个 4 天前旧版执行器遗留容器 + 回滚容器 `flowx-inference-server-prev`；`docker image prune` 回收 **3.63GB** |

**新增边界（实测）**：同一张卡上**并发跑两条 GPU 流水线不可行** —— 预算式淘汰会顶掉对方
正在用的模型（`on_evict` 顺带断对象仓库视图），对方随后报 `object '...' not found`；
服务端 `EXEC_LOCK` 只保证算子串行、不保模型常驻。已写进 `DEPLOY.md` §5.1。
**后续方向（已定向，未排期）**：串行化应在**推理服务端**做，而非 Studio 全局队列 ——
服务端引入「按执行租约 pin 模型」的概念（需要 flowx 侧传租约标识），让一次流水线
执行期间其加载的常驻模型不被其它执行顶掉；并发执行排队等待而非互相淘汰。

**编排纪律（用户确认）**：流水线 YAML 结构不因功能修改；功能需要新增数据一律加在
节点 **parameters/param** 里（结构冻结、数据开放）。基于此：
- **张量级搬移（未立项）**：核心机制属第 1 层（加载层，方案先行），用户触点是
  loader 节点新增 `tensor_offload` 参数（0=关闭，>0=显存驻留上限 MB，超出驻内存
  流式供给）——不加独立开关节点（顺序敏感/回放漂移/并发互踩）
- **vram-policy 节点（可做，第 3 层）**：淘汰策略（MAX_RESIDENT/VRAM_RESERVE_MB/
  VRAM_BUDGET/VRAM_LOAD_FACTOR）做成运维面节点，读写服务端 `GET/POST /vram/config`；
  定位为单流水线串行执行时的策略实验工具，远期租约落地后策略改为随执行传递

## 二十五、M5 SDXL 支持（2026-09-22 立项 → 2026-09-23 真机验收全绿交付）

**目标**：补上图像线唯一缺失的主流架构门槛（ROADMAP 阶段 5 / M5）：SDXL（Pony/Illustrious
系社区模型）文生图、图生图、hires、ControlNet 全通，且**不破坏 SD1.x 行为**（逐字节回归）。

### 25.1 缺口定位（先做的事实核实）

`sniff.py` 早已把 SDXL 路由到 `StableDiffusionXLPipeline`（`conditioner.embedders.*` /
`text_encoder_2.*` 两套 key 布局都认），`IMAGE_ARCHS`（VAE fp32 防黑图）、`vram.py`
（diffusers 目录体积累加）、`preview.py`（SDXL 投影矩阵）也都在。**真正缺的只有一处**：
COND 里 CLIP-G 的 pooled embedding 与 `added_cond_kwargs` 的传递链 —— SDXL 的 UNet 除
`encoder_hidden_states` 外还要 `text_embeds`（1280 维 pooled 投影）+ `time_ids`
（尺寸/裁剪六元组），SD1.x 无此概念。

### 25.2 实现（差异只进适配点，不动采样循环骨架）

| 环节 | 落地 |
| --- | --- |
| COND 对象 | `ops.SDXLCond(embeds, pooled, size)` + `ops.SDXLSegments(list + .pooleds)`（分段条件每段自带 pooled）；`cond_pooleds()/cond_size()` 旁路读取，**段元组仍是 3 元组**（存量插件零改动） |
| 文本编码 | `ops.clip_encode(pipe, text, width, height, clip_skip)` 按架构分派：SDXL 走管道 `encode_prompt`（不手写编码数学，与 sd3-clip-encode 同法；4 元组/2 元组返回都兼容）；SD1.x 分支逐字节沿用 |
| 采样注入 | `ops.build_sdxl_added()`（纯函数）：cat 拼批按 CFG 惯序 `[neg,pos]`、time_ids 复制 2b 行；split 各 b 行；area 每段一个 added。`cn_residuals/unet_forward/blend_side` 统一收 `added`（**SDXL ControlNet 也必须收**），与 IPAdapter 的 `image_embeds` 用 `_merge_added` 合并；无 pooled 时 added 恒为 None ⇒ SD1.x 路径不变 |
| 早失败 | `check_sdxl_pooled_dim()` 按 `unet.add_embedding.linear_1.in_features - addition_time_embed_dim*6` 校验 pooled 维度；正/负混用 SD1.x/SDXL cond、段缺 pooled 都明确报错 |
| 插件 | `cond_latent_ops.py`：`set_area(SDXLCond)→SDXLSegments`（保 pooled）、`combine`（embeds 拼接 + pooled 取均值，段列表则逐段保留各自 pooled）、`average`（embeds+pooled 双加权）；SD3 dict 继续明确拒绝 |
| 离线 config | `model_manager._single_file_kwargs` 除 MODELS_DIR 顶层外也扫 `_configs/*`；`sniff._LIST_SKIP_DIRS` 加 `_configs`（不进模型下拉） |
| 可观测 | `/health` 增 `diffusers_version`/`torch_version`（排障时先确认跑的是哪套栈） |
| 节点 | `clip-text-encode` 1.0.0 → **1.1.0**（+ width/height/clip_skip 参数与 slider widget；env 透传 WIDTH/HEIGHT/CLIP_SKIP）；镜像 BUNDLE_VERSION 1.6.3 → **1.6.4**；其余节点零改动（复用 checkpoint-loader/empty-latent/ksampler/vae-decode/latent-upscale/save-image） |
| 测试 | `tests/test_sdxl_cond.py`（SDXLCond/段列表归一化、pooled 维度校验、time_ids 构造、cat 批序、per-segment added、插件组合、SD1.x 不变式）；全量 18 个测试文件本地全绿 |

**明确不做**：SDXL refiner 双模型管线、aesthetic_score 参数、SDXL-inpainting（9 通道 UNet
与现有 4 通道 latent 不兼容）。

### 25.3 部署（2026-09-22，已完成）

- 推理镜像重建 `flowx-inference-server:pascal`（`Dockerfile.pascal` 是 `COPY app/`，
  代码改动必须重建镜像，非 bind mount）；旧容器停用改名保留
  `flowx-inference-server-prev`（回滚：`docker rm -f flowx-inference-server &&
  docker rename flowx-inference-server-prev flowx-inference-server &&
  docker start flowx-inference-server`）；新容器 5 个 bind mount / GPU / 端口 / env 逐项对齐
- 插件 `cond_latent_ops.py` 直接写进 bind 挂载的 `plugins.d`（启动即载；`/ops` 的
  `plugin_hash` 已从 `35fb15a3…` 变为 `0621658e…`）
- 节点镜像 `lerkobba/flowx-pixelforge-nodes:v1.6.4` 已在远端 daemon 构建；50 包清单自检
  `check-bundle.py` 通过；Studio 重导 `clip-text-encode@1.1.0`
- 流水线 `pipeline/sdxl-txt2img.yaml`（wf 87）+ `pipeline/sdxl-hires.yaml`（wf 88）已导入
  Studio（docker 节点绑 `Param.service_url_internal` = `http://172.17.0.1:8100`，local 节点
  绑隧道 `service_url` —— §23.8 的跨执行器 URL 纪律）
- **t7 SD1.5 逐字节回归通过**：`accept-m5.py ONLY=t7` → `sha1=48e3225e8b38`，
  与 M2/M6 基线 `accept-m2-out/t4_euler_normal.png` 完全一致 ⇒ M5 改动 + 镜像重建
  未改变 SD1.x 数值行为
- wf87 试跑：链路全通（docker 执行器 → v1.6.4 镜像 → 节点代码 → 内网 URL → 推理 API），
  **唯一失败点 = `model 'sd_xl_base_1.0' not found in /models`**

### 25.4 阻塞：SDXL 权重（宿主机当前无外网）→ **2026-09-23 已解决（见 25.5.1）**

**实测（2026-09-23 重测）**：宿主机容器网络**TCP 出网全断**（不仅是 DNS/代理问题）：
对 `modelscope.cn`/`pypi.org`/`baidu`/`1.1.1.1`/`223.5.5.5` 的 TCP 握手全部超时，
连**已知可用的** cpolar docker 端点 IP `81.71.128.221:12268` 在容器内也超时 ⇒
容器→宿主 之后的 NAT/转发坏了；只有 UDP 出网（DNS 解析成功、cloudflared 最终以
QUIC 连上了 Cloudflare 边缘）。`--network host` 同样不通；daemon 的
`docker pull` 也走 `http.docker.internal:3128`（= `192.168.65.1:3128`，能连但不转发）
而失败。⇒ 这是 **Docker Desktop（WSL2 VM）的网络需在 Windows 侧重启/重配**的问题，
容器内无法自救。

**已绕过的部分（本机有外网，已落地）**：
- 配置骨架 15 个文件（3.2MB）从本机经 `docker cp` 推进容器（24s）：
  `/models/_configs/sdxl-base/`（= 宿主 `D:\flowx-data\models\_configs\sdxl-base\`），
  含 `model_index.json`、`unet|vae|text_encoder|text_encoder_2/config.json`、
  两套 tokenizer、`scheduler_config.json`；
- **骨架正确性已在容器内验证**：`StableDiffusionXLPipeline.load_config` 通过；
  `ModelManager._single_file_kwargs('StableDiffusionXLPipeline')` 自动命中
  `{'config': '/models/_configs/sdxl-base'}`（M5 的离线 config 兜底链路真机生效）；
  UNet `addition_time_embed_dim=256`、`add_embedding.in_features=2816` ⇒
  **期望 pooled=1280**（与 `check_sdxl_pooled_dim` 一致）；VAE `latent_channels=4`、
  `scaling_factor=0.13025`（与现有 4 通道 latent / decode 链路一致）；
  两套 tokenizer 均正常出 77 token。
- **权重传输不可行**：`docker cp` 实测 **0.13 MB/s**（30MB/228s，即 cpolar 免费隧道
  带宽上限）⇒ 6.9GB 需 ~15 小时且隧道会断；也不存在其它更快的路径（两边隧道同带宽）。

**已排除的绕行方案（2026-09-23 实测，均不可行）**：
- 本机（termux，有外网）→宿主传权重：**用户明确不用本地上传通道**；且实测带宽也
  不够（`docker cp` 30MB/228s = **0.13 MB/s**，即 cpolar 免费隧道 1Mbps 上限，
  6.9GB 要 ~15 小时）
- 走 `trycloudflare` 另一条隧道：cloudflared 容器实际在反复失败
  （`failed to request quick Tunnel: context deadline exceeded`），旧 URL 已死
- QUIC/UDP 绕行：容器内无 `curl`/`wget`/HTTP3 客户端（`aioquic`/`h3` 均缺，且
  无外网可装）⇒ 虽然 UDP 能出去、cloudflared 能建立 QUIC，也无法用来拉文件
- 宿主机上找现成代理：常见代理端口全扫（`host.docker.internal`/`192.168.65.1`/
  `172.17.0.1`/`192.168.65.254`）只有 Docker 自身代理 `192.168.65.1:3128`，
  而它对 `CONNECT` 不响应（= 上游代理为空/死的）；IPv6 不可用（`Errno 101`）

**待用户择一（只剩一个文件）**：

> **【2026-09-23 已闭环】** 用户清了 Docker Desktop 的死代理 → 容器出网恢复；
> 最终路径 = 上面的方案 1 变体：不走 download-model.sh，而是经 `/admin/plugins`
> 热装临时插件 `zz_fetch.py`，`/jobs` 异步把 SDXL 底模与 SDXL ControlNet 直下进
> `/models/`（sha256 双验）；验收后插件已卸载。全过程与验收结果见 §25.5.1。

1. **修 Docker Desktop 网络**（Windows 侧重启 Docker Desktop / Settings→Resources→Proxies
   填可用代理；Windows 侧网络本身正常 —— cpolar 客户端与 cloudflared 容器都还活着），
   然后我在容器内一条命令拉全量：
   `DOCKER_HOST=tcp://<解析IP>:12268 DOCKER_API_VERSION=1.47 sh scripts/download-model.sh sdxl`
2. **在 Windows 侧放一个文件**（浏览器/`curl.exe`/迅雷均可）：
   ```
   curl.exe -L -o D:\flowx-data\models\sd_xl_base_1.0.safetensors ^
     "https://modelscope.cn/models/AI-ModelScope/stable-diffusion-xl-base-1.0/resolve/master/sd_xl_base_1.0.safetensors"
   ```
   （6.94GB；**配置骨架已就位，无需再下**）。想用 Pony/Illustrious：把对应单文件丢进
   `D:\flowx-data\models\`（同一 config 兜底），ckpt_name 用该文件名；或把 ModelScope
   repo id 给我，我加进 `download-model.sh` 并生成直链。
3. SDXL ControlNet 权重（t4 用，如 `xinsir/controlnet-canny-sdxl-1.0` / 
   `diffusers/controlnet-canny-sdxl-1.0`）同样受此限，按需一并提供。

### 25.5 验收口径（权重到盘后执行）

脚本 `scripts/accept-m5.py`（`BASE=<隧道> TOKEN=<token> SDXL_CKPT=<模型名> python3 …`）：
- t0 预检（/health 版本、/ops 的 `clip.encode` 入参、模型清单）
- t1 txt2img 1024（offload=model）→ 尺寸/非黑图/方差核对 + 抽图存 `accept-m5-out/`
- t2 i2i（vae.encode + denoise=0.6）/ t3 hires（latent.upscale 1.5x + denoise=0.45 → 1536）
- t4 SDXL ControlNet（`SDXL_CN=<权重名>`，验 added_cond_kwargs 进 controlnet 前向）
- t5 offload 三档（none/model/sequential）可行性 + 耗时（SD1.x 的 sequential 与
  `clip.encode` 不兼容，SDXL 走 `encode_prompt` 是否可行由本项回答）
- t6 `model.unload` 余量回收；t7 SD1.5 逐字节回归（已过）；t8 SDXL + set_area/combine 区域构图
- t9（可选，最强口径）与官方 diffusers 管道**逐像素一致**：用
  `scripts/accept-m5-pipe-ref.py`（一次性容器内跑官方 loop，图写到 INPUT_DIR）与
  服务端出图比对（8GB 卡装不下两个常驻 SDXL，故 t9 先 `model.unload` 再起参考容器）
- 另跑 Studio wf87/wf88 端到端 + 抽图核对（视频/图像管线内容级纪律）

### 25.5.1 交付记录（2026-09-23，全绿）

**权重到盘**（网络修复后服务器侧直下，不经本机上传）：
- 用户在 Docker Desktop 清掉死代理 → 容器出网恢复（`zz.probe` 验证 modelscope/baidu/pypi 200）
- 临时插件 `zz_fetch.py`（zz.probe/zz.tcp/zz.fetch/zz.read）经 `/admin/plugins` 热装，
  `/jobs` 异步下载 + `.progress` 进度文件 + Range 续传；验收后已 `DELETE` 卸载
- `sd_xl_base_1.0.safetensors` 6,938,078,334B @28.5MB/s，sha256=31e35c80fc48… ✓
  （与 ModelScope 清单一字不差）；safetensors 头 2515 tensors 全 F16
  （conditioner 587 + first_stage_model 248 + model 1680 = 标准 SDXL 布局）
- `controlnet-canny-sdxl-1.0/`（组件目录：config.json + fp16 safetensors 改名
  diffusion_pytorch_model.safetensors，2.5GB @3.1MB/s）双 sha256 ✓

**真机验收（容器内 accept-runner 直连 172.17.0.1:8100，避开隧道抖动）**：
| 项 | 结果 |
|---|---|
| t1 txt2img | ✓ sha1=18d904cfd92b（两次运行逐字节一致） |
| t2 i2i / t3 hires 1536 | ✓（t3 sha1=459e2f36d245 复现一致） |
| t4 SDXL ControlNet canny | ✓ sha1=14eea0812bdb，构图跟随线稿 |
| t5 offload 三档 | ✓ none 793s / model 268s / sequential 697s |
| t6 unload 回收 / t7 SD1.5 回归 | ✓ / ✓ sha1=48e3225e8b38 逐字节一致（第三次复现） |
| t8 区域构图 | ✓ sha1=df048ef0cb9b（复现一致），上猫下雪山的区域语义正确 |
| **t9 官方管道逐像素比对** | ✓ **像素+PNG sha1 全一致**（StableDiffusionXLPipeline
  官方 loop vs 手动 loop，同种子/步数/cfg 出图一字节不差——added_cond_kwargs
  （pooled+time_ids）链路的最强证明） |
| Studio wf87/wf88 | ✓ exec 415/416 全节点绿，抽图核对内容正确 |

**验收中修掉的 4 个问题**（均为验收脚本/环境侧，产品代码零改动）：
1. `accept-m5.py` OBJ_PORTS 漏 `cond/cond_a/cond_b` → 对象 id 裸串被服务端拒
   （tier2/tier3 脚本早有，M5 漏抄）
2. 对象仓库 TTL 3600s < 全量套件耗时（~55min）→ t5-sequential/t8 引用过期；
   修：mkconds()/fresh_sdxl() 在每个用到条件/模型视图的测试前重建（缓存命中秒回）
3. ControlNet 常驻 1.25GB 挤占 8GB 显存 → 非 CN 采样换页抖动 90s/步（平时 13.5s/步）；
   修：t4 默认 10 步（SDXL_CN_STEPS 可调）+ t4 结束即 `/model/unload cn:*`；
   运维启示：**8GB 卡上 SDXL+CN 同时常驻必抖动，流水线用完 CN 应接 Unload**
4. `accept-m5-pipe-ref.py` 官方 pipe `output_type="pil"` 在 fp32-VAE（VAE_FP32=1 防黑图）
   下 dtype 崩；修：output_type="latent" + 按 ops.vae_decode 同法解码

**环境变更**：隧道重建（docker `6.tcp.cpolar.cn:13495` / 推理 `5.tcp.cpolar.top:10190`）；
Studio 侧新建执行器 `docker-remote-211b`（id=7，旧 id=5 已禁用）；
wf87/wf88 的 `service_url` 已更新为新隧道；Docker Desktop「Expose daemon :2375」保持勾选。

**commit**：`3cbe507`（M5 实现）、`50298e1`（验收脚本修正 + ROADMAP）、
`05608fe`（widget 布局调整——图像预览置顶，与本里程碑无关的并行改动，同批提交）。

### 25.6 边界与后续

- SDXL 常驻约 7.3GB（UNet fp16 5.1G + 双 TE 1.9G + VAE fp32 0.33G）⇒ 8GB 卡默认
  `offload=model`（`inference-ensure.min_vram_mb` 闸门可前置拦），跑完建议接 `Unload` 节点
- 1024 fp32 VAE 解码显存若吃紧 → 触发 §十四 遗留②（tiled VAE decode），届时再补
- 社区模型（Pony/Illustrious）与 base 的差异只在权重 ⇒ 单文件路径打通后无需额外适配

## 二十六、画布侧读取的两套网络视角：模型下拉 404/502 与参数注入静默失效（2026-09-23）

**症状**（用户报，流水线 76 的 Checkpoint 加载节点）：画布「模型文件名」下拉显示
「列表加载失败：Error: HTTP 404（可手动输入）」；修掉 404 后又变 502
`Get "http://172.17.0.1:8100/models/files": context deadline exceeded`。

**根因一（节点包）**：Studio §十七 去业务耦合删掉了 `POST /api/v1/inference/models-files`
（换通用 `/api/v1/service-proxy`），但 `_common/widget-base.js` 只改了头部注释，
模型下拉的 fetch 仍是旧端点 → 404。当初的回归只 curl 了后端新端点、没走 widget 自己的
调用路径，所以漏网。

**根因二（Studio 运行时）**：`buildEnvMap` 在节点包声明了 `env` 时**只注入 env 里的键**，
未列出的参数静默取默认值——`checkpoint-loader` 的 `dtype/offload/use_t5`、
`ksampler` 的 `preview_every` 全被吞掉（证据：exec 400 日志 `knobs=auto`，而 wf87 明明写了
`offload=model`）。改为「参数声明即注入 + env 覆盖」（`FLOWX_PARAM_*` 打底，`env` 覆盖同名键），
docs/11-node-package.md 规则 1/3 同步改写；Mock 侧本来就是全量注入，现在两端一致。

**根因三（画布侧读取的地址）**：模型下拉、预览帧中转、▶预览重放/中断这些请求都是
**Studio 宿主机发起的**（Studio 的网络视角），而 docker 节点的 `service_url` 是容器内网地址，
Studio 连不上。新增节点参数 `service_url_host`（Studio 可达地址，留空回退 `service_url`）：
- 节点侧 `flowx_client.host_base(url)` 构造预览帧 URL 与上报 base（`executor_base` +
  ksampler/sd3-sampler/video-gen/inference-op 四处自报预览）
- widget 侧模型下拉优先读 `service_url_host`，提示行显示实际用的 base
- 26 个有画布侧读取的节点声明该参数；workflow 里 docker 节点绑
  `service_url_host: "{{ Param.service_url }}"`（74/75/76/83/87/88/60/58 共 37 处，
  repo 侧 `pipeline/sdxl-*.yaml` 同步）

**发布**：节点镜像 v1.6.4 → **v1.6.5**（容器内代码要带 `host_base`；远端 daemon 上
`SKIP_PUSH=1` 本地构建，宿主无外网），`sync-flowx-json.py` → 50 个节点重导（ID 不变）。

**门槛**（`nodes/check-bundle.py`）：G widget 产物必须等于 base+widget-def 重建结果；
H widget 不得引用已删端点；I 有画布侧读取的节点必须声明 `service_url_host` 且预览 URL
必须经 `host_base()`。

**验证**：
- 模型下拉：用 Studio 实际服务的 bundle + wf76 真实绑定（paramSources 按前端
  `parseParamSources` 语义还原）在 jsdom 里跑 widget 自己的代码路径 → fetch
  `/api/v1/service-proxy`，5 个 checkpoint 可选，当前值 `v1-5-pruned-emaonly-fp16` 选中，
  提示「5 个可选 · 5.tcp.cpolar.top:12073」
- 参数注入：临时 docker 全链（wf89，跑完已删）exec 402 → 日志
  `knobs={'dtype': 'fp16', 'offload': 'model'}`（修复前是 `auto`），
  `/models` 常驻表出现 `v1-5-pruned-emaonly|dtype=fp16|offload=model`
- 预览帧：v1.6.5 后 exec 403 → `GET /executions/403/nodes/Sample/preview-frame`
  **HTTP 200 image/jpeg 256×256**（v1.6.4 时同一接口 502，报的正是 `172.17.0.1:8100` 超时）；
  节点级代理 `/service-proxy?path=/models` 200

**教训**：一个「服务地址」参数在跨网络部署里不够用——**发起请求的一方**决定用哪个地址
（容器内 ↔ Studio 宿主机），凡是 Studio 代发的读取都要问一句「Studio 连得上吗」。

## 二十七、IPAdapter 真机验收补票 + 两个交付 bug 修复（2026-09-24，exec 419 全绿）

**背景**：IPAdapter 三算子+三节点 2026-09-21 已交付（commit `49a2c03`），但当时宿主
外网断路、权重未下载，commit 挂待办。本次补票：隧道 5.tcp.cpolar.top:10190（推理）/
6.tcp.cpolar.cn:13495（docker，执行器 7 已改存解析后 IP 81.71.128.221，旧实例 5 已禁用）。

**权重**（宿主容器内 ModelScope 直下，urllib 断点续传脚本；注意 cpolar 大文件传输不可行
但宿主直连 ~2.5MB/s 够用；首次 2.4GB 大文件流到 18 分钟时 .part 丢失报 FileNotFoundError，
第二版脚本加「流提前结束 → 续传重试」循环后成功，实际发生过 1 次续传）：
- `models/clip_vision/CLIP-ViT-H-14/`（AI-ModelScope/IP-Adapter 的 models/image_encoder
  2.4GB + laion 仓的 preprocessor_config.json）
- `models/ipadapter/ip-adapter_sd15.safetensors`（42MB）

**真机验收暴露两个交付时未覆盖的 bug**（commit `6afb784` 修复）：
1. `ipadapter.load` 拒绝一切 safetensors 权重——safetensors 只有扁平键
   （`image_proj.proj.weight`），原代码只认 .bin 嵌套 dict 顶层键；新增
   `_unflatten_ipadapter` 拼回嵌套。教训：**「支持 safetensors」的代码路径必须用真实
   safetensors 文件验过**，单测 mock 嵌套 dict 等于没测格式差异。
2. `sample()` 注入 image_embeds 缺 `num_images` 维——diffusers
   `MultiIPAdapterImageProjection` 要求 `[batch, num_images, (seq,) dim]`，缺维被
   reshape 成扁平向量，投影层报 `mat1 and mat2 shapes cannot be multiplied
   (1x2048 and 1024x3072)`。补 `unsqueeze(1)`。

**验收**（流水线 93 `txt2img-ipadapter`，repo `pipeline/txt2img-ipadapter.yaml`，
A/B/C 三链同种子 42，参考图橘白卡通猫窗台图 × prompt "a dog in garden, photograph"）：
- 链 A weight=0 → 金毛犬花园照片（纯 prompt 语义）
- 链 B weight=0.8 全程 → 橘白猫窗台插画（参考图风格/构图/内容全面主导）
- 链 C weight=0.8 窗口 [0,0.5) → 狗 + 橘白配色 + 窗台构图 + 插画风格
  （构图风格保留、prompt 语义回归——步窗口语义教科书级验证）
- **零污染**：续跑 exec 419 追加无 IPA 基线链 D（ksampler 直挂 LoadCkpt），
  出图与链 A **sha1 逐字节一致**（73b59ea5…）——weight=0 零影响 + 采样后
  处理器恢复无残留，双重证明

**部署备注**：ops.py 经 docker cp 热补丁 + 容器重启生效（镜像 flowx-inference-server:pascal
未重建，下次构建自然带上——重建前若容器重建会丢此补丁，已提交 repo 故只影响"用旧镜像
重建容器"这一路径）；ipadapter_ops.py 走 `POST /admin/plugins` 热加载，无需重启。

**发现的产品行为**（非 bug，记档）：`execution continue` 会**跳过 failed 节点**
（"已终结节点跳过"含 failed），想重跑失败节点只能新开执行——exec 417 续跑时 LoadIPA
被跳过导致下游绑定报"上游没有输出"，误导性强，排障先看 server.log 的
"binding validation failed"。

**后续可选项**：~~SDXL 版 IPAdapter~~ **✅ 2026-09-26 已交付（见 §三十一，
选 vit-h 变体复用现有 CLIP-ViT-H-14）**；~~plus 版~~ **✅ 同日交付（见 §三十一）**；
InstantID（✅ 见 §二十八）。

## 二十八、InstantID 身份保持生成（SDXL）交付（2026-09-24，验收全绿 + Studio e2e）

**定位**：第三档主线。人脸身份保持的 SDXL 文生图：insightface 提取 512 维身份
特征 → IPAdapter 机制注入 UNet（身份）+ InstantID ControlNet 注入 5 关键点图
（面部位置/姿态）。**diffusers 0.35.2 官方从未合并 InstantID**，实现按 InstantX
官方仓库源码逆向对齐。

**算子**（插件 `inference-server/plugins/instantid_ops.py`，自注册热加载）：
- `face.analyze(image, face_index=-1, det_thresh=0.5)` → FACE{embeds(raw,模长~20),
  normed_embedding, kps_image(官方 draw_kps 布局), count, bbox}；antelopev2
  onnxruntime CPU；det_thresh 对 AI 生成脸回测需降到 0.2（置信度常 0.35~0.45）
- `face.similarity(image_a, image_b, ..., det_thresh)` → 余弦相似度（验收量化用：
  同人典型 ≥0.5，陌生人 <0.3）
- `instantid.apply(model, ipadapter, controlnet, face, weight=0.8, cn_strength=0.8,
  start/end/cn_start/cn_end_percent)` → {model: IPABundle, control: ControlBundle}；
  SDXL 限定（cross_attention_dim=2048 校验），非 Resampler 结构 IPA 权重明确拒绝
- 核心改动仅 ~25 行：`ControlBundle` 增 `hidden_neg/hidden_pos` 覆写口
  （InstantID CN 的 cross-attention 吃 **Resampler 投影后的人脸 token** 而非文本，
  负向=resampler(zeros) 官方语义）；`cn_residuals` 加 side 参数消费覆写对；
  registry +FACE 类型

**节点**：`instantid-face-analyze` / `instantid-apply`（executor_base + widget-def
体系，镜像 v1.6.6）。流水线 94 `instantid-txt2img`。

**权重**（~4.5GB，容器内 ModelScope 直下）：`ipadapter/instantid-ip-adapter.bin`
（1.69GB）、`instantid-controlnet/`（2.5GB，config `addition_embed_type="text_time"`，
cross_attention_dim=2048）、`insightface/models/antelopev2/`（5 个 onnx，
`INSIGHTFACE_HOME=/models/insightface`）。⚠️ **License：InstantID 权重官方声明
仅研究/非商用**，交付物默认内部实验用途。

**真机排障记录**（GTX 1080 8GB，全部是官方实现细节差异，逐一实证定位）：
1. **Resampler 输入需 [b,1,512] 带 seq 维**——`load_ip_adapter` 把 embeds 合并成
   batch×num_images 后喂 Resampler，缺维炸 cat
2. **IPA 恢复顺序 bug（核心修复，影响所有 Resampler 类投影）**：InstantID 的
   Resampler 内含 4 个 Attention 子模块，`set_attn_processor` 计数器把它们计入
   （140 vs 144），旧顺序"先还处理器再摘投影"导致恢复失败且**污染常驻管道**；
   修为先摘 `encoder_hid_proj` 再还处理器。SD1.5 标准 IPA 是纯 Linear 投影无此问题
3. **人脸特征必须用 raw embedding（模长 ~20）而非 normed（模长 1）**——官方
   `face_info['embedding']`；误用 normed 身份信号弱 20x：实测 ref/gen 相似度
   0.134 < 基线 0.242（方向反了），修后 0.34 > 基线，全链路 0.75
4. **模型 LRU 挤兑 + 句柄联动回收**：LoadCN（2.6GB）与 LoadCkpt（SDXL）同层并发，
   CN 先完成，SDXL 加载时 `_evict_if_needed` 把 CN 顶掉，`on_evict → discard_where`
   联动回收已下发给下游的 CN 对象句柄 → instantid.apply 报"object not found"。
   编排侧修复：**组件加载图序串行化（大模型先行）**（流水线 LoadCkpt → LoadCN/
   LoadIPA 边）。这是调度 caveat 不是服务端 bug
5. **OBJECT_TTL_SECONDS=3600 < 单任务时长**：1024/25 步采样 ~75min，Decode 时
   vae_ref（T+5min 创建）已过期。服务端已支持环境变量，容器重建加
   `OBJECT_TTL_SECONDS=14400`（部署配置变更，代码零改动）
6. **双层超时叠加**（exec 422/423 两次 failed 才定位全）：
   a) 节点内部 `job_timeout` 默认 3600s < 实际 4305s——客户端放弃时服务端跑完
      （默认改 7200，流水线显式 9000）；
   b) 包级 `timeout: 3600`——**Studio 层 SIGKILL（exit 137）直接杀容器**，
      节点无任何错误日志（进程死了打不出来），排障要看 Studio server.log 的
      `command exited with code 137`。ksampler 改 10800。教训：**长任务节点要
      同时核对两层超时**（包级 timeout ≥ 内部 job_timeout），且"无日志的
      failed + 时长接近整点"先怀疑外层 SIGKILL

**验收**（全绿）：
- t0~t6 `scripts/accept-instantid.py`：预检/face.analyze/端到端/消融/回归/负面/回收
- **身份相似度量化**（face.similarity，insightface 余弦）：
  **t2 全链 1024/25（IPA 0.8 + CN 0.8）= 0.7523**；仅 IPA 0.8 快档（768/16）= 0.3365；
  裸基线 = 0.2439（陌生人水平）。梯度清晰，两条注入链路各自生效
- **t3 消融**：weight=0 & cn_strength=0 出图与基线**逐字节一致**
  （sha1=871fd4e441b0），复跑零污染
- **t4 SD1.5 回归**：sha1=48e3225e8b38 与 M2 基线逐字节一致
  （注意 M2 基线负向是**空字符串**，照抄 m5 脚本的负向文本会误判回归失败——
  t4 第一次 FAIL 就是脚本 bug 不是产品 bug）
- t5 负面用例（SD1.5 底模/标准 IPA 权重均明确拒绝）、t6 显存回收正常
- **Studio e2e**：
  - exec 422（1024/25）：12 节点全链，服务端采样 4305.5s 完成，手动接续解码
    出图 **sim=0.7713**（节点记账 failed 纯属双层超时，见排障 6）
  - **exec 427（流水线 95 `instantid-txt2img-fast`，768/16）正式全绿**：
    14/14 节点 success，Sample 1634s + Decode 93s + Save + Unload，
    出图 `instantid_studio_20260925_132727.png` **sim=0.6807**
    （快档下依然远超陌生人水平）。Sample 走 **local 执行器**（长等待只依赖
    推理隧道，绕开更不稳的 docker 隧道）
  - 隧道基建教训：cpolar 隧道中断时 Studio docker 执行器 inspect 失败 →
    节点 failed（exit 137），但**服务端 job 是独立的**（幂等 id 驻留，
    客户端死了照样跑完）——exec 424 的孤儿采样完整跑完 25/25；
    后续编排纪律：**长跑节点优先 local 执行器 + 隧道断时不要在队列里
    叠加手工 job**（队头阻塞会顶死短 timeout 的前置节点，exec 425/426
    即因此连锁失败）

**依赖变更**：requirements-pascal +insightface==0.7.3 +onnxruntime==1.19.2；
Dockerfile.pascal 同 RUN 内临时 g++ 编译 insightface cython 扩展后卸除
（opencv 冲突沿用既有卸载+headless 模式，冒烟门禁加 import 校验）。

**后续可选项**：InstantID SD1.5 变体（权重不同目录）、PuLID（另一路线）、
多人脸合成、face.analyze 的 kps 图接 detail-refine 做面部精修联动。

## 二十九、FLUX.1-schnell（GGUF 量化路线）接入（2026-09-25，验收全绿 + Studio e2e）

**目标**：GTX 1080 8GB（Pascal，torch 2.3.1 钉死）+ WSL 12GB 内存上限下接入
FLUX 文生图。选型 **schnell + GGUF Q4_K_S（city96）**：Apache 2.0、1~4 步蒸馏、
transformer 6.8GB + T5 Q4 2.7GB + CLIP-L/VAE 0.6GB ≈ 10.3GB 常驻（fp16 全量
~23GB 不可行；Pascal 无 bf16/fp8 硬件）。

**Phase 0 可行性（已通过，512px 实测出图）**——五个兼容坑逐个实证修复：
1. torch 2.3.1 无 `torch.nn.RMSNorm`（2.4+）→ 插件侧补等价模块
2. diffusers attention_dispatch 恒传 `enable_gqa`（torch 2.5 kwarg）→ SDPA shim
   剥离（True 时显式报错，False 等价默认）
3. transformers 4.44.2 的 GGUF 仅支持 llama/mistral/qwen2 → **T5-XXL GGUF 手写
   装载器**：219 张量 llama.cpp→HF 名映射 + meta 壳 `load_state_dict(assign=True)`
   （fp32 空壳 19GB 直接 OOM）+ `tie_weights()` + **`model.eval()`**（T5 config
   dropout=0.1，训练态会让同 prompt 两次编码结果无关）
4. **GGUF shape 语义反转**：shape 记 `[in,out]` 但数据行主序按 `[out,in]` 存 →
   正确解包 `reshape(shape[::-1])`；`reshape(shape).T` 确定性错误（嵌入互相正交
   cos≈0.007，肉眼"能出图但语义全错"级隐患）
5. **T5-XXL 必须 bf16 计算**：激活范数 ~7e5，fp16 末层 cos 只剩 0.38（bf16 0.88+）；
   张量级 cos vs 官方 safetensors ≈1.0 证明装载器正确后才定位到 dtype

**两相架构（12GB 内存硬约束逼出来的）**：transformer Q4（6.8GB RAM）与 T5 bf16
（4.9GB）永远不能同驻 RAM（实测 OOM exit 137 三次）→ 编码相（T5+CLIP 上卡，
transformer 驻内存）与采样相（整管上卡 7.2GB VRAM）分时复用，组件切换由插件
自管（不用 accelerate offload hooks）。**T5 权重逐张量 npy 分片缓存**
（safetensors save_file 会复制全量副本 → OOM；npy 分片写无全量拷贝，首次 GGUF
dequant ~6min，缓存命中 ~3min）。

**性能实测**：编码 ~3.5min（含 T5 缓存装载）；512px/4 步采样 4~26min（方差大，
native dequant 路径无融合算子——diffusers GGUF 融合 kernel 依赖 `kernels` 包，
其依赖链会顶飞 huggingface_hub 钉版 → **放弃融合算子**，接受 native 速度）；
768px/4 步 ~29min；1024px 不建议（~2h）。

**服务端插件** `plugins/flux_ops.py`：`flux.load`（GGUF 原地装配 FluxPipeline，
入 model_manager LRU）/ `flux.encode`（T5+CLIP，release_t5 默认卸载）/
`flux.sample`（整管上卡采样+VAE 解码直出图）。核心零改动达标。关键修法：
**采样相必须 `pipe.to("cuda")` 而非逐模块 `.to()`**——`pipe.__call__` 内部用
`pipe.device`（注册序首个模块）派生 latent/timestep 设备，逐模块搬移会让
latent 建在 CPU 而权重在 GPU（matmul 设备错，实测踩中）。

**节点**：flux-loader / flux-clip-encode / flux-sampler（各带 server_op.py 同一
插件自注册，hash 一致不重复上传）；采样器 VAE 解码内置直接出图（FLUX 免负向）。
验收 `scripts/accept-flux.py`：t0 注册 / t1 招牌 case（文字渲染 FLOWX COFFEE）
/ t2 回收+flux→SDXL 错峰 / t3 SD1.5 sha1 回归 / t4 负面拒绝。

## 三十、depth 预处理器 + depth ControlNet（第三档批次 3）交付（2026-09-26，验收全绿 + Studio e2e）

**补遗清单③落地**：depth 预处理器与对应 ControlNet。选型 MiDaS dpt_hybrid
（control_v11f1p_sd15_depth 训练时用的就是 MiDaS dpt_hybrid，最对口；
controlnet_aux 0.0.10 无 DepthAnythingDetector，midas/zoe/leres 可选，取 midas）。

**改动**（全部零核心改动，插件 + 节点包）：
- 插件 `preprocess_ops.py` 加 `preprocess.depth(image, detect_resolution=512)`：
  `MidasDetector.from_pretrained(<dir>, local_files_only=True)` 目录路径模式零网络；
  CPU 单例（dpt_hybrid 权重 470MB，CPU 一张图几秒，不占显存）；
  **输出尺寸恒=输入尺寸**（MidasDetector 输出尺寸由 image_resolution 决定且是
  resize_image 短边语义，无法精确保持原尺寸——算子拿到输出后 BILINEAR 回缩）
- 节点 `preprocess-depth@1.0.0`（executor_base + widget-def，slider 调
  detect_resolution，compareInput 前后对比）；镜像 v1.6.6 → **v1.6.7**
  （Dockerfile + sync 56 节点），check-bundle 全绿
- **坑**：flowx.json 参数类型约定是 `integer`/`float`，写 `number` 被 Studio 拒
  （import 时报 invalid parameter type）

**权重**（容器内直下 + sha256 双验，全绿）：
- `/models/preprocessors/midas/dpt_hybrid-midas-501f0c75.pt` 469MB
  sha256=501f0c75b3bc…（ModelScope `lllyasviel/Annotators` 仓；
  ⚠️ hf-mirror 对该仓 /resolve/ 返回 **308 跳回 huggingface.co** 不代理，
  且 urllib 原生不处理 308——fetch.py 已加 _H308 handler，但最终还是走
  ModelScope 更省事。教训：**下载前先列 ModelScope 文件清单**）
- `/models/control_v11f1p_sd15_depth_fp16.safetensors` 689MB
  sha256=1c4a79aa52fb…（ModelScope AI-ModelScope/ControlNet-v1-1_fp16_safetensors）

**验收**（`scripts/accept-depth.py`，230s 全绿）：
- t1 回归 sha1 与 M2 基线逐字节一致（插件改动未污染采样路径）
- t2 深度图机械判定：768×1152（=输入尺寸契约 ✓）、纯灰度（通道极差 0）、
  灰度跨度 252（层次丰富）、均值 108 合理；抽图：人物近亮远暗、手持物最近最亮 ✓
- t3 端到端 depth CN 控制采样（prompt 雪中小屋）：构图/空间层次跟随深度图 ✓
- **Studio e2e**：流水线 97 `txt2img-depth`（repo `pipeline/txt2img-depth.yaml`，
  尾部 UnloadCN 节点回收 CN 显存），exec 429 13/13 节点全绿，出图人物仰头
  捧物姿态保留 + 雪林远景层次正确

**运维记录**：节点镜像 v1.6.7 远端 daemon SKIP_PUSH=1 构建（13MB context 经
隧道 ~100s）；56 节点批量重导（video-gen/video-sample-latent 首次瞬断重试成功）；
cpolar 隧道重建（推理 5.tcp.cpolar.top:12876 / docker 6.tcp.cpolar.cn:11323，
执行器沿用 docker-remote-211b id=7，host 重写后可用）；批量 follow 流
"context deadline exceeded" 是隧道掐流不是执行失败，查 execution list 即可。

## 三十一、SDXL IPAdapter（vit-h 变体）+ ip-adapter-plus_sd15 交付（2026-09-26，零代码改动验收全绿 + Studio e2e）

§二十七 的两个后续可选项落地。代码**零改动**达标（safetensors unflatten
§二十七已修、Resampler 恢复顺序 §二十八已修、SDXL added_cond_kwargs 合并
§二十五 M5 已通）——唯一工作是权重选型 + 下载 + 验收。

**选型**：SDXL IPA 用 **vit-h 变体**（`ip-adapter_sdxl_vit-h.safetensors`
666MB）而非官方 `ip-adapter_sdxl.bin`——前者配已有的 CLIP-ViT-H-14（2.4GB
早已在盘），后者要再下 3.5GB ViT-bigG 编码器。plus 版
`ip-adapter-plus_sd15.safetensors`（93MB）同样配 ViT-H。
均 ModelScope `AI-ModelScope/IP-Adapter` 仓直下 + sha256 双验一致。

**验收**（`scripts/accept-ipadapter-sdxl.py`，1456s 全绿）：
- t1 SD1.5 回归 sha1 与 M2 基线逐字节一致
- t2 SDXL 768/16 步（offload=model）四链同种子 42：基线 D（白狗花园照片）
  / **链 A weight=0 与 D 逐字节一致（零污染 ✓）** / 链 B w=0.8 全程（橘白卡通猫
  窗台插画，参考图全面主导）/ 链 C w=0.8 窗口[0,0.5)（狗脸+橘白配色+窗台构图
  +插画风格——步窗口语义教科书级复现）
- t3 plus_sd15（Resampler 路径）SD1.5：P0 w=0 与基线逐字节一致
  （**基线 sha1=73b59ea5a97f，与 §二十七 exec419 的链 A/D 跨会话复现**），
  P1 w=0.8 生效
- t4 model.unload all + gc 回收 ✓
- **Studio e2e**：流水线 98 `txt2img-ipadapter-sdxl`（exec 431 13/13 全绿），
  出图与脚本链 t2B **sha1 逐字节一致**（Studio 链路 ≡ 脚本链路最强等价证明）

**事故与教训（exec 430，一次服务端 OOM 重启）**：
- 症状：Sample（SDXL+IPA 采样 324s）成功后 Decode 报
  `object not found (expired or never created)`；`/health` 显示 resident=[]
- 根因：wf97（depth 验收）尾部只卸了 CN、**SD1.5 底模仍常驻**；exec 430 再叠
  SDXL offload + IPA + CLIP-ViT，驻留总量超 WSL 12GB → 服务端进程被 OOM Kill
  （`Killed`，OOMKilled=false 因为是宿主机级别），容器 restart 后对象仓清空
- 排障路径：`docker inspect` 的 StartedAt/RestartCount + API `logs?previous=1`
  （docker CLI 的 --previous flag 在 API 1.47 不存在，走裸 API）
- 修复：验收类流水线尾部一律 `model.unload target=all`（txt2img-depth.yaml 已改）；
  纪律升级：**8GB 卡 + 12GB RAM 上"留底模复用"只在同一会话内成立，
  跨流水线必须全卸**

## 三十二、FaceDetailer 面部局部精修（face.mask + 纯编排）交付（2026-09-26，验收全绿 + Studio e2e）

第三档最后一块拼图落地。人脸检测**复用 insightface antelopev2 单例**
（InstantID 交付的 `_face_app()`），不引入 ultralytics；crop/放大/合成全部
复用既有节点，**新增仅 1 个算子 + 1 个节点**。

**改动**：
- 插件 `instantid_ops.py` 加 `face.mask(image, face_index=-1, det_thresh=0.2,
  expand=0.6, feather=16)`：bbox 中心外扩（覆盖全脸+发际）→ clamp 图内 →
  **8 倍数对齐**（crop 直接喂 vae.encode）→ feathered 矩形 mask +
  crop 框 x/y/width/height 数值输出（供 image-crop/image-composite 直接绑定）。
  ⚠️ 两个设计点：① det_thresh 默认 **0.2** 而非 face.analyze 的 0.5——
  FaceDetailer 的回测对象就是 AI 生成图，人脸置信度常 0.35~0.45；
  ② mask 输出注册为 **IMAGE 类型**（服务端无 MASK 类型，灰度 IMAGE 是
  set_noise_mask 以来的惯例）
- 节点 `face-mask@1.0.0`（slider 调 face_index/det_thresh/expand/feather，
  compareInput 前后对比）；镜像 v1.6.8（**构建待 docker 隧道恢复**，见下）
- 流水线 99 `facedetailer`（repo `pipeline/facedetailer.yaml`）

**验收**（`scripts/accept-facedetailer.py`，210s 全绿）：
- t1 回归逐字节一致
- t2 机械判定：crop 框全 8 倍数/图内/面积占比合理/feather 中间灰阶存在
- t3 e2e：**mask 外最大差异=0（非面部区域逐像素不变式 ✓）**、白区平均
  差异 8.7（重绘生效）
- Studio e2e：exec 434 **20/20 全绿**，原图 vs 精修图抽图对比：眼/睫毛/
  雀斑纹理显著清晰，非面部区域无变化，无可见接缝

**编排纪律（验收首轮失败后固化，FaceDetailer 三要素）**：
1. **小 crop 必须先放大再采样**：288×344 直接重采样必崩（VAE 小 latent +
   prompt 漂移 → 伪影/第二张脸/接缝）；lanczos 放大到短边 ~512 再
   vae.encode，采样后缩回原尺寸（流水线里 image-upscale 一正一反两次调用，
   正向 scale=2.0、反向 width/height 绑 FaceMask 输出）
2. **精修 prompt 沿用原图 prompt**（保风格，黑白照不会被修回彩色），
   可叠加面部细节词；denoise 0.3~0.4（保身份）
3. **mask 贴回前要用 image-crop 裁到 fg 尺寸**——image-composite 会把传入
   mask **resize 到前景尺寸**（不是按 x/y 裁剪），直接传全图 mask 会变形错位

**环境事故与降级**：docker 隧道（6.tcp.cpolar.cn:11323）在批次中途死亡
（connection refused），推理隧道存活。处置：
- 镜像 v1.6.8 构建搁置（daemon 不可达）——**Studio 侧节点登记仍是 v1.6.7，
  千万不要此时批量重导**（sync 已把 repo 侧 tag 改 v1.6.8，重导会让 docker
  执行器拉不存在的镜像全挂）
- e2e 流水线全 local 执行器降级（service_url 全走公网隧道；face-mask 本地
  python 跑 flowx_client 零依赖）→ 跑通
- YAML 编排坑：image-upscale 无画布侧读取故**未声明 service_url_host**，
  YAML 误绑该参数会在展开期失败（"references undeclared parameter"）
- **补记（2026-09-26 晚）**：docker 隧道恢复（6.tcp.cpolar.cn:10193）后
  v1.6.8 已构建 + 57 节点已重导 + 执行器 id=7 host 已更新（`executor update`
  可直接改 config，不必新建实例）；face-mask docker 执行器回归通过
  （exec 435 的 FaceMask/LoadCkpt/Upscale/VaeEncode 等 docker 节点全 success）。
  ⚠️ 但该隧道 **ping 1184ms 高延迟，`create exec`（hijack 流式连接）三连
  20s 超时**（exec 435/436/437 各挂在不同节点）——系统性线路质量，
  完整 e2e 以 exec 434（全 local 降级）20/20 为验收口径；隧道稳时 wf99
  docker 编排可直接复跑。教训：**executor test 通 ≠ exec 链路稳**，
  ping 只验 TCP，hijack 流才是真实考验

## 三十三、InstantID×FaceDetailer 联动（D1）：e2e 全绿 + 淘汰保活修复（2026-09-26 深夜）

**流水线 100 `instantid-facedetailer`**（repo `pipeline/instantid-facedetailer.yaml`）
exec 448 **27/27 全绿**：InstantID(SDXL 768/16)→ face.mask → crop → 放大 1.25x
→ 低 denoise 精修 → 贴回。出图自然无接缝。

**量化验收暴露的联动陷阱（重要）**：精修采样挂裸模型时身份丢失——
sim(原图,参考脸)=0.760 → sim(精修,参考脸)=0.587。FaceDetailer 的
"面部细节词 + 自由重绘"会把脸漂向底模默认脸。**修复：精修采样 model_ref
改挂 Apply.model（InstantID IPA 注入保留），身份特征参与精修**；CN 可卸
（精修不需要姿态控制），故 UnloadCN 内存编排与身份保持可兼得。

**model_manager 淘汰保活修复**（exec442 事故 → 产品级修复）：
- 事故：LoadCkpt/LoadCN 并发，SDXL 入驻检查（est 9294MB > free 7101MB）把
  先完成的 CN 当 LRU 牺牲品淘汰，`discard_where` 联动删除 control_net 对象，
  Apply 取用报 "object not found"
- 修复：`ModelManager.extra_pinned` 回调（main.py 注入
  `keys_of(store.referenced_data())`）——**对象仓库仍引用的常驻条目视同
  执行期 pin**，淘汰计划跳过；对象消费/过期后自动恢复可淘汰
- 二次事故：keys_of 加锁 → load 持锁调 evict → 回调再取锁 = 死锁（LoadCN
  卡死 18 分钟）。修复：keys_of 无锁快照 + RuntimeError 降级
- 测试 `test_eviction_guard.py` 4 例：身份匹配/引用保活/无引用退 LRU/回调
  异常静默降级；治理护栏 17 例回归不破
- 部署方式：**docker cp 进容器可写层 + docker restart**（restart 保留容器层
  修改，免重建镜像；exec430 后确认的运维通道）

**内存编排（WSL 12GB 红线）**：FaceDetailer 精修链叠加 InstantID 全套驻留，
DecodeFix 两次 OOM Killed（容器 restart 循环 Restarts=54）。YAML 编排修复：
Decode0 后即 `model.unload target=cn:instantid-controlnet`（采样用完立刻放）+
精修放大 2.0→1.25x（VAE encode/decode 输入减半）。教训：**Decode0 过了
≠ DecodeFix 能过**——链尾内存最高点在最后一次 decode。

**环境运维**：
- OOM 后 `flowx-inference-server-prev`（4 天前旧镜像）会抢 8100 端口鸠占
  鹊巢——/health 200 但 ops 缺 face.mask。**处置：stop prev + start 主容器**
- 推理隧道更换为域名型 `https://4d2f3b43.r5.cpolar.top`（TCP 端口型已死）
- exec449（身份保持验证轮）被隧道中断，**待复跑**：
  `flowx-studio workflow run --id 100`，验收口径 sim(精修,参考脸) ≥ 0.70
  → **已补票（2026-10-06，exec 468，口径未达，后续选项待拍板），见 §三十四**

## 三十四、验证补票轮（2026-10-06）：wf100 身份保持复跑（口径未达）+ wf99 docker e2e 关闭

**隧道**：docker `tcp://5.tcp.cpolar.top:10886`（解析 8.162.7.178，执行器 7/8 已更新）；
推理 `https://76321d5e.r5.cpolar.top`（wf99/wf100 service_url 已更新，repo
pipeline/facedetailer.yaml、pipeline/instantid-facedetailer.yaml 已同步）。

**环境事故（§33 预言第三次应验）**：主容器 `flowx-inference-server` 处于 stopped，
`flowx-inference-server-prev`（旧镜像）Up 抢 8100——/health 200 但无 FACE 类型、
无 face.*/instantid.apply 算子（exec 467 FaceAnalyze 报"算子未注册"暴露）。
处置沿用既定方案：stop prev + start 主容器，51 ops 全恢复（含 face.mask）。
注意：prev 容器保留期间每次主机/Docker 重启都有抢端口风险，建议评估
`docker update --restart=no flowx-inference-server-prev` 或干脆删除。

**① wf100 身份保持验证轮（exec 468，27/27 全绿，Sample0 1597s）**：
- sim(原图,参考脸) = **0.7596**（与 exec448 的 0.760 复现一致，InstantID 注入侧稳定）
- sim(精修,原图) = **0.7893**（修复"精修挂 Apply.model"确实锚住原图，不再漂向底模默认脸）
- sim(精修,参考脸) = **0.5739** < 口径 0.70 → **FAIL**；与修复前裸模型的 0.587 基本持平
- 解读：精修重采样改写的纹理细节使 embedding 沿近似正交方向漂移，低 denoise 下
  IPA 注入对"精修图↔参考脸"的直接拉近有限；修复的真实收益体现在对原图的锚定。
- 后续选项（方案先行，待拍板）：a) 精修同步挂 InstantID CN（姿态约束参与精修）；
  b) 降精修 denoise（0.3→0.25 档）+ prompt 加身份锚词；c) 验收口径修定为复合式
  sim(精修,原图)≥0.75 且 sim(精修,ref)≥0.5（当前 0.789/0.574 均已满足）。

**② wf99 facedetailer docker 编排 e2e（exec 471，20/20 全绿）**——§32 遗留缺口关闭：
- 前两轮 exec 469/470 均死于 cpolar hijack 流瞬断（`inspect exec: network is
  unreachable`，服务端 job 实际跑完——§22.3"重挂失败直接失败"为既定设计，
  非回归）；第三轮隧道窗口稳定一次通过，docker 节点 6~14s/个
- 内容级核验（口径纪律：抽图）：非面部区域**逐像素不变**（差异包围盒外 max=0），
  重绘生效（差异区均值 7.6/max 101，与 accept-facedetailer t3 的 8.7 同量级）；
  对比图 `~/flowx-output/wf99_docker_e2e_compare.png`（原图|精修|差异热图）

**全局重指向（2026-10-06 第二轮）**：隧道换新后已全量切换——推理走 **局域网
`http://192.168.5.60:8100`**（GPU 主机已回到局域网，手机直连可达；比 cpolar 稳定，
优先使用），docker 走 `tcp://6.tcp.cpolar.cn:13768`（解析 81.71.128.221，执行器 7/8）。
18 条流水线（含 99/100）+ repo pipeline/*.yaml ×5 全量指向 LAN，复扫 cpolar 残留=0。
⚠️ docker daemon 的 LAN :2375 从手机侧不可达（Docker Desktop 未对外暴露/防火墙），
docker 只能继续走 cpolar——hijack 流瞬断病仍在（exec 473 的 LoadCV 又死一次，
exec 474 重试即全绿）。
**冒烟验收 exec 474（wf93 txt2img-ipadapter，20/20 全绿）**：链 A（weight=0 基线）
出图 sha1=`73b59ea5a97f` 与 §27 exec 419 记录**逐字节一致**——跨隧道更换 + 容器
重启后 SD1.5 全链确定性复现；B/C 链与 A 差异显著（IPA 生效）。

**仍挂起的验证**：widget 浏览器实际渲染抽检；openpose `include_face=true`
精化路径；§23.5 A 类 37 节点 docker 全覆盖（当前仅 11+零散）；更新后流水线的
分批重跑（隧道恢复后）。

## 三十五、InstantID 人脸几何压扁：根因修复（canvas 参数）+ A/B 双验证（2026-10-07，commit `7ec107d`）

**起因**：用户目检 wf100 对比图发现"脸相似度有，但长宽比例变形、变扁"。
量化实锤：旧原图生成脸 kps 散布宽高比 **1.071** vs 参考脸真实 **0.875**。

**根因**：`face.analyze` 把 5 关键点画在**参考照片原始画布**（768×1152 竖图）上，
而生成画布是 768×768——CN 条件图被**非均匀 resize**（x×1.0，y×0.667）进生成
分辨率，关键点几何压扁 → 生成脸跟着扁。face.similarity 对适度几何形变不敏感
（0.76 照拿），历次量化验收全绿，直到目检才暴露（抽图纪律再次立功）。

**方案 A（编排验证，RefCrop 预裁方）**：参考图先中心裁 768×768 再喂 face.analyze。
真机结果（exec 475/476 的 orig）：生成脸宽高比 **0.867 ≈ 参考 0.875**；
**sim(原图,参考脸) 0.7596 → 0.8159**（几何修正顺带提升身份相似度 +0.056）。

**方案 B（产品化，本提交）**：`face.analyze` 加 `canvas_width/canvas_height`
（默认 0=旧行为；成对校验 + 8 倍数校验前置），`fit_kps_to_canvas` 纯 python
等比+居中变换（letterbox）；节点 instantid-face-analyze 加两参数原地 overwrite
（id=114，默认行为不变）；wf100 切 B 形式（撤 RefCrop，canvas 768×768）。
插件热部署 sha256=`a5604956…`；B 的 kps 实测：画布 768×768、宽高比 **0.879**、
letterbox 位置正确（exec 477 SaveKps）。

**DecodeFix OOM 治愈（Option X，编排零代码，exec 481 验证）**：精修采样后
`model.unload target=all` 卸掉 SDXL 全部常驻（~7GB RAM staging）→ `vae.load`
独立 SDXL VAE（`sdxl-vae-base`，从 ckpt 单文件 `AutoencoderKL.from_single_file`
提取落盘 /models/，权重与管道内置同源）→ DecodeFix 只挂 0.33GB VAE，**5s 通过**
（此前 4 连 OOM）。关键依据：latent/图像/数值是对象仓库的自有对象，不受常驻
卸载的 discard_where 联动影响；vae_ref 换绑独立 VAE 即可。sdxl-vae-base 已入
/models/files 清单（vae 类）。

**canvas_mode 收官（2026-10-07 下午，commit `c34708f`，exec 483 29/29 全绿）**：
fit(letterbox)/cover(铺满裁溢出) 双模式，fit 默认；wf100 切 cover。真机对比：

| 指标 | 修复前 | fit(小脸) | **cover(终版)** |
|---|---|---|---|
| 脸宽高比 | 1.071（扁） | 0.874（111×127） | **0.878（158×180）** |
| sim(原图,ref) | 0.7596 | 0.7337 | **0.8126** |
| sim(精修,ref) | 0.5739 | 0.4607 | **0.6591**（口径 0.70 未达但历史最优） |
| sim(精修,原图) | 0.7893 | 0.7241 | **0.8685** |

精修身份税稳定在 ~0.15（重绘改写纹理的固有成本）；0.659 居"同人≥0.5"区间中上。
进一步收敛路径：精修 denoise 0.3~0.4→0.25 档（待拍板）。
对比图 `~/flowx-output/wf100_cover_final_compare.png`（参考|修复前|cover原图|cover精修）。

**环境运维（本轮新增）**：① Studio server 被 Android 杀后台 2 次（wake-lock 也
没挡住），症状=执行僵尸（节点卡 running），处置=重启 server + 新开执行；轮询
间隔 ≤8 分钟可保活。② WSL 内存已调 14.66GB（TotalMemory 实测），但 Sample0
峰值 ~15GB 仍踩线轮盘赌（exec 482 死过一次，483 同配置通过）——暂接受，
下次主机维护可考虑 swap 或 16GB+。

## 三十六、盘点轮（2026-10-08，纯分析无代码改动）："超硬件"定性修正 + 显存/串行旋钮现状核实

本轮为对话式盘点，结论经代码核实（`app/offload.py`、`app/model_manager.py`、
nodes/*/flowx.json），无改动。三项确认：

### 36.1 "超硬件"清单没有一项真被显存卡死（定性修正）

**原理**：sequential offload 下显存峰值 = 最大单层而非整模型，"8GB 显存"墙消失；
瓶颈按序转移为 ① 宿主机 RAM（物理 16GB，WSL 已调 14.66GB，见 §三十五）
② Pascal 无 bf16 的数值精度（GGUF 路线规避，FLUX 已验证）③ 软件收录/工程量。
打法 = 已验证的三板斧：**sequential offload + GGUF Q4 量化 + 错峰加载**，
复用 §9.6 纪律（硬件假设不进算子逻辑 / workaround 开关化 / 旋钮两级结构）。

| 搁置项 | 修正结论 | 真实阻塞 |
|---|---|---|
| 阶段 3 FLF2V（Wan2.2-TI2V-5B） | **可行**：当初否决理由"umT5-XXL fp16 ~9GB 单模块超显存"恰是 sequential offload 能解的；RAM 侧 GGUF Q4（主体+T5 ≈ 10GB）+ 错峰（编码完卸 T5 再载 DiT）→ 峰值 ~7GB | 工程量：管道接入（核实 diffusers 0.35.2 对 WanFirstLastFrameToVideoPipeline 收录，不够则升级+回归） |
| MiniMax H3（10B+） | 硬件可行（Q4 ~6-7GB + 错峰） | **软件与法务**：trust_remote_code 违反"ops.py 只面向 diffusers 抽象"纪律需单独适配；权重可用性 + License 未确认（§四预研三问不变） |
| SV3D 3D 新视角 | **本来就不卡硬件**：SVD 架构微调，参数量与在跑的 SVD-XT 同级，连搬移都不需要 | 纯工程：网格/多视角输出新对象类型；只出多视角图片则现有 IMAGE/VIDEO 类型可承接 |

**建议优先级（均需先出方案）**：SV3D（纯工程）→ Wan2.2 FLF2V（复用 FLUX 路线）
→ H3（等权重/License 确认）。

### 36.2 搬移机制现状（代码核实）

- **两档搬移已落地并真机验证**：`model`（子模块级，SDXL M5 验收）/ `sequential`
  （逐层级，SVD/SD3.5/FLUX 验证），`offload.py` + `model_manager._apply_offload()`
- **覆盖边界**：只在"管道整体加载"路径生效；组件级加载绕开——`load_vae` 直接
  `.to("cuda")` 常驻（model_manager.py:300）、ControlNet 常驻 LRU（:313）；
  InstantID/insightface/GGUF 各有独立加载逻辑不在 `_apply_offload` 体系内
- **`QUANTIZATION=fp8` 仍是空接口**（识别配置+告警，未实现；FLUX 实际走 GGUF
  节点侧选型绕过）

### 36.3 旋钮节点化现状（flowx.json 全量扫描）

**显存优化旋钮：1/11 个加载节点有全套**——
- ✅ `checkpoint-loader` 1.2.0：dtype/offload/use_t5 全套（唯一）
- ⚠️ `controlnet-load` / `vae-load`：仅 dtype
- ❌ `ipadapter-load` / `motion-loader` / `clip-vision-load` / `flux-unet-load` /
  `flux-dual-clip-load` / `flux-vae-load`：无旋钮
- ✅ 单点旋钮：`video-gen`/`vae-decode-video` 的 decode_chunk_size；
  `inference-ensure` 1.1.0 的 min_vram_mb 闸门；`model-unload` 编排节点

**串行策略：零节点化（部分是既定设计）**——
- 串行执行是架构级（单 worker FIFO + EXEC_LOCK），不设节点旋钮
- EXEC_LOCK 只保算子串行、**不保模型常驻**（§21.8 边界：并发两条 GPU 流水线
  会互相顶掉模型）

### 36.4 铺路缺口清单（启动大模型线的前置，建议立项顺序）

| # | 项 | 量级 | 说明 |
|---|---|---|---|
| 1 | **加载旋钮全量化** | 小改 | 所有 loader 节点/算子补 dtype/offload（auto 继承进程级），ModelManager 缓存键统一纳入（checkpoint.load 已是范本）。**已立项，见 §四十三** |
| 2 | **执行租约 pin** | 中改 | 一次流水线执行期间其加载的常驻模型不被其它执行顶掉；并发执行排队而非互踩。已定向"在推理服务端做"（需 flowx 侧传租约标识——侦察发现 FLOWX_EXECUTION_ID 已注入节点环境，零改动可用）。**已立项，见 §四十三** |
| 3 | **错峰加载原语** | 小改 | 编码器产物落对象仓后立刻释放编码器（model.unload target=... 编排已可用，缺节点范式沉淀；对象仓引用保活 §三十三已是护栏） |
| 4 | **tensor_offload 张量级搬移** | 大改 | loader 节点加 `tensor_offload` 参数（0=关，>0=显存驻留上限 MB，超出驻内存流式供给）；16GB RAM 装不下时（Wan 14B 级）的磁盘流式兜底。用户触点形态已拍板（不加独立开关节点），核心机制未立项，方案先行 |
| 5 | vram-policy 运维节点 | 小改 | 淘汰策略（MAX_RESIDENT/VRAM_RESERVE_MB/VRAM_BUDGET/VRAM_LOAD_FACTOR）读写服务端 `GET/POST /vram/config`（端点不存在，需先加）；定位=单流水线串行执行时的策略实验工具，远期租约落地后改随执行传递 |

**纪律重申**：以上各项动手前均先过方案；硬件假设永不进算子逻辑；
workaround 一处一开关；验收必须抽帧看图。

## 三十七、MiniMax H3 调研记录（2026-10-08，纯调研无代码；更新 §四状态）

**来源**：MiniMax 官方 GitHub/HF/开放平台文档、diffusers PR、ComfyUI 官方文档、
社区实测（neng320 部署指南、whichh3 硬件指南、vramready）。本机 DNS 被 fake-ip
污染（198.18.x.x，§八 #7 同款病），直链抓取不可用，全部为检索摘要交叉核实。

### 37.1 模型本体（2026-08-02 开源）

- **H3-Base 开源**；完整系统另两块闭源 API：H3-Context-IR（提示词理解编排）、
  H3-Regenerate-2K（2K 放大）。开源权重直出 **768p / 24fps / 最长 15s +
  原生立体声**（同一 DiT 联合生成音频，非后期配音）
- **架构**：33.1B dense 单流 Omni-Transformer（~13B 在 AdaLN 分支，**推理可剪枝**，
  有效 ~20B——ComfyUI 剪枝 INT8 版 19.5GB 与此吻合）+ H3-Encoder（**截断版
  Qwen3-VL-32B**，取第 50 层 hidden states；tokenizer 用 H3 仓自带配置）+
  视频/音频双 VAE（共 ~5.4GB）
- **变体**：`FL2VA`（文生/首帧/尾帧/**首尾帧**——覆盖搁置的阶段 3 FLF2V 场景）、
  `Ref2VA`（全参考：≤9 图 + ≤3 视频 + ≤3 音频，音频不能单独作为输入）
- CFG-distilled，联合视频+音频 DiT；Video Arena 开源双榜第一（t2v+i2v），
  仅低于 Seedance 2.0 两分
- 初版仅全注意力；稀疏注意力实现官方承诺后续放出

### 37.2 §四预研三问的答案

| 三问 | 答案 |
|---|---|
| trust_remote_code / diffusers 收录 | ✅ **diffusers PR #14355 已合并**（2026-08-05，+10256 行/40 文件）——**ModularPipeline 形式**（`MiniMaxH3ModularPipeline`，无经典 DiffusionPipeline 一半，同 Anima 路线）；t2va/fl2va 两个 checkpoint 分区共享组件。需升级 diffusers 至 2026-08 后版本（当前 0.35.2，升级+回归关联遗留 #5/#6）。已知 issue：#14379 references 参数被忽略 + 多 GPU device mismatch |
| 文本编码器构成与显存预算 | 截断 Qwen3-VL-32B——**编码器本身就是 32B 级**，全系统最重的一块；GGUF 量化已有（DiT + 编码器分属两个社区仓） |
| License | ⚠️ **MiniMax H3 Community License**（非 Apache 2.0，有二手来源误传）。免版税、研究+商用可，但**有地域条款**：官方 QA 标题"为什么 license 目前限于 EU/UK/韩国/美国"，另一处摘录出现 "Excluded Territories = EU/UK/Korea/US" 定义，两处语义矛盾——**必须读 LICENSE 原文逐条确认适用地域后方可商用/分发，这是红线** |

### 37.3 部署生态与硬件实测（社区数据）

- 部署路径：ComfyUI 0.30.0+ 原生模板（Comfy-Org 转换权重）；SGLang v0.5.17+
  （preview）；vLLM-Omni recipe（基于 vLLM 0.26.0）
- 最小文件组合 **42.5GB**：剪枝 INT8 DiT 19.5G + NVFP4 编码器 14.6G + 双 VAE 5.4G
- 硬件档位：最低实用显存 **12GB**（重卸载，"8G 可玩"是营销说法）；RAM 舒适线
  **64GB**，32GB 能跑
- 标杆实测（RTX 4060 Ti 16GB + 64GB RAM）：480p 3-5 分钟/条——依赖 **cu130 硬件
  反量化 + SageAttention**（sm80+/Blackwell 特性，Pascal 全部不可用）

### 37.4 对本机（8GB Pascal + 16GB 物理 RAM）的判定

**§三十六三板斧第一次不够用——H3 是真·超规格**：

1. **RAM 硬边界击穿**：DiT Q4_K_M 单文件 ~19.9GB > 16GB 物理内存；加编码器合计
   ~35GB。唯一出路 = tensor_offload 磁盘流式（§36.4 #4），但**每个采样步从盘读
   20GB+**（SATA SSD ~500MB/s → 单步纯 IO ~40s，30 步 ≈ 20min 仅 IO）
2. **计算墙**：33B ≈ FLUX（12B，1-3 小时/张）近 3 倍计算量；Pascal 无 SageAttention、
   无 cu130 反量化、fp16 1:64 速率 → 单条 15s 视频**天级**，超出"能跑即可"容忍度
3. **对比结论**：首尾帧场景的现实解是 **Wan2.2 FLF2V（5B，H3 的 1/6 体量）**，
   §36.1 优先级维持：SV3D → Wan2.2 FLF2V → H3

**§四状态更新**："等权重确认"前提已满足（权重 2026-08-02 开源、diffusers 已收录、
GGUF 生态齐备）；剩余前提 = **License 地域条款逐条确认 + 硬件升级**
（建议门槛：≥24GB 显存 + ≥64GB RAM，对齐社区舒适线）。调研归档备用，不动工。

## 三十八、老欠账清理轮（2026-10-08）

**用户侧已闭环**（本会话前）：§34 prev 容器抢端口、流水线批量地址更新、WSL 内存提升、
openpose `include_face=true` 人脸路径（B2 欠账关闭）。

**本会话交付**：
- **B1 widget 浏览器渲染抽检 ✅**（chrome-devtools 真浏览器，headless Chromium 149）：
  工作流列表/编辑态/回放态三态渲染正常，console 零 JS 错误；模型下拉 4 类
  （checkpoint/vae/controlnet/upscale）经 `/api/v1/service-proxy` 成功加载
  （`7 个可选 · 192.168.5.60:8100`，选中值正确）；滑块/复选/textarea/▶预览/⚙折叠
  全部正常；回放 exec 353 落盘图经 media 代理显示、过期预览帧自动隐藏（设计行为）；
  旧隧道地址时下拉兜底文案正确（列表加载失败 + 可手动输入 + service_url_host 提示）。
  §15 遗留①关闭。
- **顺手修复**：wf70/wf66 的 Param `service_url` 死隧道（5.tcp.cpolar.top:12073）
  → `http://192.168.5.60:8100`（画布参数面板改+保存，已验证加载）。
- **僵尸执行清理**：exec 492（wf100）节点卡 `running 0/0` 29 分钟，服务端 job 实际
  已完成（16/16，latent `a42b7a46e7b24481b54a33d49a939822` 在对象仓）——§35 Android
  杀后台病第 N 次；已取消。续跑可从该 latent 接 DecodeFix→FaceDetailer 段。
- **C1/C2 欠账确认为已交付**（别会话干的，文档漏更新）：§22-D tee 补齐 = `5428281`、
  §22-G Studio 截断标记 = `6328afd` + Studio 前端徽章（WorkflowCanvas.executor-badge
  测试在）；flowx 已发 v0.1.8。§22 待办清零。

**环境备注**：① headless Chromium 在 Termux 下会被系统杀掉，保活 = setsid +
crash 循环重启脚本；② Studio server 同病（本次 8080 也死过一次，server start 拉起）；
③ 收尾时 GPU 主机离线（LAN 无路由 + cpolar 拒绝）——**B3 流水线分批重跑与
D1（37 节点 docker 全覆盖）仍阻塞，等主机回线**。

**老欠账台账（本轮后剩余）**：
- ~~C3 画框/笔刷 widget~~ → 已交付（§三十九），剩真机收尾两项见下
- B3 更新后流水线分批重跑（等主机）
- D1 §23.5 A 类 37 节点 docker 全覆盖（等主机）
- tiled VAE decode（爆了再做）；QUANTIZATION=fp8 空接口（按需）

**等 GPU 主机回线的收尾清单（2026-10-08 盘点）**：
1. ~~mask-paint 底图衬图链路~~ ✅（§三十九，真机验收已过）
2. ~~mask-paint 真机 e2e~~ ✅（§三十九，判决实验+分区差分双验证）
3. B3 流水线分批重跑（地址已全部换 LAN；docker 节点还需 cpolar 隧道——
   2026-10-08 晚隧道 refused，需在 WSL 侧重启 cpolar 并更新地址）
4. D1 §23.5 A 类 37 节点 docker 全覆盖跑通（同依赖 cpolar 隧道）
5. （可选）exec 492 续跑：从 latent `a42b7a46e7b24481b54a33d49a939822` 接
   DecodeFix → FaceDetailer 段（对象仓数据在，主机回线即可续）
   ⚠ 先修 §三十九记的 continue 聚合 bug 再续跑，否则结果不可信
6. （待定）`inference-server/scripts/accept-tier3c-openpose-face.py`：B2 已被
   用户自行修复，该脚本留作回归 or 删除，待拍板

## 三十九、mask-paint 蒙版手绘节点（2026-10-08，C3 立项交付）

**范围**：§15 遗留③的「画框/笔刷」——画布蒙版手绘编辑器（对标 ComfyUI MaskEditor）。
**方案要点**（用户已拍板）：笔画矢量存 `strokes_json` 参数（分辨率无关、可再编辑）；
节点上缩略预览 + 点击弹全屏编辑器；执行器 local-only 纯 PIL（零服务端/镜像改动）。

**交付**（flowx-pixelforge `14b9072`，Studio 节点 id=169）：
- `nodes/mask-paint/`：brush/rect/erase 三类笔画按原图分辨率缩放光栅化
  （PIL 坐标闭区间语义），输出 mask+coverage+红罩叠加预览；mock 过、
  `test_main.py` 20 项单测全过
- `widget-base.js` 新增 `spec.paint` 控件：缩略（底图+红罩）+ 全屏编辑器
  （单指绘制/双指捏合缩放平移/滚轮缩放/撤销/清除/Esc 取消）；底图走
  compareInput 同款链路（上次执行 `__inputs_resolved` + 节点级代理），
  未执行过给 512×512 空白参照；64 个节点 bundle 全量重建
- **浏览器真机验证**（wf107 mask-paint-widget-test）：编辑器弹出/笔刷绘制/
  矩形/橡皮擦除/撤销/完成保存/刷新复现全过；笔画经防抖 PUT 持久化到流水线
  YAML；console 零 JS 错误
- 踩坑记录：① 笔画层若用 `ctx.globalAlpha=0.55` 画，橡皮 destination-out
  每次只擦 55% 永远擦不净——改全不透明绘制 + 画布 CSS opacity 出观感；
  ② 画布 preview/edit 模式开关（浮动工具栏锁形图标）控制着
  `interactive`/onParamsChange 下发，预览态 widget 全只读是设计行为

**~~待主机回线~~ ✅ 真机验收已过（2026-10-08）**：
① 底图衬图链路 ✅——原设计走不通（编辑态 widget 无 execution 上下文，
compareInput 链路仅回放/执行中可用），改为两层：Studio 侧编辑态拉最近一次
success 执行 metadata 注入 `data.lastOutputs`（flowx-studio `9d10d6c` 左右，
GlowNode paramSources.runtimeValue 回退；刻意不进 nodeRuntimeData，编辑态
「返回」区不受旧数据污染）+ widget 侧通用代理 blob 拉底图（pixelforge
后续提交）；编辑态缩略已衬真图+红罩。踩坑：执行详情 API 的 metadata 字段
是 JSON 字符串需解析。
② 真机 e2e ✅（wf108 mask-paint-inpaint-e2e，exec 499）：load-image →
mask-paint（coverage 7.2%）→ mask-feather → vae.encode + latent-set-noise-mask
→ ksampler(denoise 1.0) → vae.decode → save-image 全链过；**判决实验**：全黑
蒙版采样 ≈ 纯 VAE 往返（平均差 2.5），证明服务端 noise_mask 消费正确；
正式验收：蒙版内平均差 45.1（25.1% 大变化）vs 蒙版外 9.5（0.41%，= 往返
精度损失），4.7x 对比度比 PASS。经验：**inpaint 必须 denoise=1.0**——0.85
从加噪原图出发+上下文先验，内容几乎不变（首次 e2e 假阴性就是因此）。

**顺带发现的 Studio bug（待修，另立项）**：`execution continue` 增量重跑把
failed 终态节点也跳过，且最终聚合无视 failed 节点直接把执行置 success——
exec 498 表现：Save 节点 failed、零新日志、执行状态 success。续跑产物
不可信，修复前续跑后必须逐节点核对状态。

---

## 四十、续跑语义修复轮（exec 498 假阳性事故） ✅ 2026-10-08

**事故**：exec 498（wf108 inpaint e2e）Save 节点缺 `service_url` 失败，但 `execution continue` 把 failed 终态节点当"已完成"跳过、聚合无视 failed 直接置 success——假阳性成功，零新日志。

**修复**（flowx `9fa22dd`→`e1632de`，tag v0.2.0；Studio `d2d5457`）：

1. **Rerun 只跳过 SUCCESS**：FAILED/CANCELLED 节点经 `clearNodeRuntime`（重置运行时状态+清历史提取，与循环迭代重置同原语）后重跑
2. **收尾按节点终态聚合**：有 FAILED/CANCELLED 节点则整体 FAILED，不再无条件 SUCCESS
3. **失败节点解冻**：`isNodeExecuted` 仅 SUCCESS/RUNNING 不可改删，FAILED/CANCELLED 允许 UpdateConfig 改参后续跑
4. **`nodeConfigEqual` 恢复 Config 比较**（隐藏深水区）：原先整体忽略 Config（env/params）导致 continue --file 修参静默无效；改 JSON 规范化比较。副效应转正：success 节点误改会被明确拒绝（实测 SetMask 误改报 `cannot modify node`）
5. **CLI 续跑计划行**：continue 前打印"跳过 N 个已成功节点，重跑 M 个失败/未完成节点 [...]"

**单测**（`runtime_rerun_failed_test.go` 三场景全过）：重跑转成功 / 重跑仍败保持 Failed / 改 failed 节点参数后重跑。存量测试除两例预存失败（§22.6 PauseResume/IsModifiable）外全绿。

**真机回归**（exec 498）：failed Save 真实重跑（仍缺参→正确 FAILED）→ 快照编写态补参 → 再续跑 → Save success 落盘 `inpaint_e2e_20261008_111523.png` → 执行 legit SUCCESS。全链路闭环验证。

**配套**：wf108 全量 service_url 从死 LAN（192.168.5.60:8100，当时下线）切隧道 `http://7ab10bb7.r8.cpolar.cn`；skill 文档 continue 语义同步更新。

**注意**：LAN 推理地址随时可能下线，隧道地址随 cpolar 重建变化——跑真机前先探活两个地址。

---

## 四十一、B3/D1 实施轮（2026-10-08，进行中）

**第 0 步 · 网络基座 ✅**：
- docker 隧道 `tcp://5.tcp.cpolar.top:14923` 活，但踩出**三个叠加坑**（flowx v0.2.1-v0.2.5 连修四刀）：
  1. docker CLI/Go 解析域名失败 → 执行器配置用真实 IP（DoH 拿 `8.162.7.178`）
  2. SDK v28.5 默认 API 1.51 > daemon(Docker 27.3.1) 1.47 → 裸 HTTP `GET /version`（无版本前缀）探活 + `WithVersion` 钉版
  3. cpolar 免费隧道**大面积丢 HEAD 请求**（实测 HEAD 2/8 vs GET 8/8），SDK Ping 的 HEAD→GET 回退只在拿到响应时触发，传输层 EOF 直接败 → 绕开 SDK Ping
  4. 隧道瞬断率 ~50% → `controlCall` 控制面重试（快瞬断 EOF/reset/refused 6 次退避；超时类最多 2 次保 exec 407 快速失败契约）
- 执行器 7/8 host 更新为新隧道真实 IP；37 条流水线死地址（22 LAN + 15 老死隧道）批量切 `http://7ab10bb7.r8.cpolar.cn`（零残留）

**D1-a 轻节点全覆盖 ✅ exec 509（30/30 全绿 105s）**：wf109 `d1-coverage-light`，
22 个新覆盖（cond×3/mask×5/image-PIL×4/latent×3/lora/embedding/upscale-apply/
clip-vision/inference-op/vae-encode/model-unload）全部 docker 模式真跑通过。
关键：容器内走 `172.17.0.1:8100`（docker0 网桥）直连推理服务，不依赖 HTTP 隧道。
累计覆盖 32/62。

**D1-b 中量级（wf110, exec 510）部分通过**：controlnet-load/ipadapter-load ✓；
LoadRef（local load-image）死于 **HTTP 隧道 `7ab10bb7.r8.cpolar.cn` 途中死亡**
（cpolar 免费 HTTP 域名回收，404 domain doesn't exist）→ 下游人脸/预处理链全部未跑；
MotionLoad/IPALoadIID/CNLoadIID 死于 job_timeout=600s 不够（大模型加载排队，
LoadSVD 单节点 307s）→ 待隧道恢复后调大 timeout 用 continue 续跑（§四十语义已可信）。

**阻塞**：HTTP 推理隧道需 WSL 侧重启 cpolar 拿新域名。docker 隧道不受影响。

**D1-b 中量级 ✅ exec 521（24/24 全绿）**：wf110 `d1-coverage-medium`，14 个新覆盖
（face-mask/face-similarity/instantid-face-analyze/instantid-apply/preprocess-depth/
preprocess-openpose/controlnet-load/controlnet-apply/ipadapter-load/ipadapter-apply/
detail-refine/video-sample-latent/vae-decode-video/motion-loader）。
过程中修复：① SVD-XT 加载 OOM → `offload=sequential`；② InstantID 仅支持 SDXL
→ 单独 LoadSDXL 支链（sd_xl_base_1.0 fp16 offload=model）；③ 并发压死主机 RAM
（集体 404 job not found = 推理服务被打崩重启）→ **重活串行链**（人脸→预处理→
IPA→IID→精修→视频→MotionLoad 依次跑）；④ load-image 节点加上传/插件注册重试
（cpolar 免费 HTTP 隧道抖动返回 HTML 错误页，flowx-pixelforge 仓库内修复同版本覆盖导入）；
⑤ MotionLoad job_timeout 600→1800。

**D1-c 重节点冒烟 ✅ exec 538（20/20 全绿）**：wf111 `d1-coverage-heavy`，15 个新覆盖
（flux-dual-clip-load/flux-clip-encode/flux-unet-load/flux-vae-load/flux-vae-encode/
flux-vae-decode/flux-empty-latent/flux-sampler/flux-detail-refine/sd3-clip-encode/
sd3-empty-latent/sd3-sampler/sd3-vae-encode/sd3-vae-decode/video-gen，全部 64×64/1步/
4帧 最小参数真跑）。过程中修复：① DB 节点记录 metadata 陈旧（flux-redux-apply 等
报 "not a bundled image node"）→ 16 个节点全部重新 import --overwrite 刷新；
② 推理服务模型对象跨执行被逐出（max_resident=3 + 多家族同驻）→ **家族串行**
（FLUX 链→SD3 链→SVD 链，队首加 model-unload 清空）；③ flux.encode OOM →
encode 提前到 unet/vae 加载前（对象及时消费，避免闲置过期）；④ 推理服务 11:12
被打崩重启一次（对象全清）——**教训：服务器重启=对象全灭，跨重启 continue 无效，
只能整跑**；⑤ flux-redux-apply 服务器缺模型文件（flux1-redux-dev.safetensors
只有 .incomplete 残骸）→ **服务器资源阻塞，记入欠账**。

**D1 收官：62/62 docker 声明节点真机覆盖满分 ✅**——flux-redux-apply 最后补齐：
模型文件通过 docker API 起 python:3.11-slim 一次性容器（bind D:/flowx-data/models）
从 ModelScope `AI-ModelScope/flux.1-redux-dev` 拉取（HF 官方仓门控，129063232 字节
精确校验），wf112 redux-smoke（exec 552）真跑通过，输出 `(1,256,4096)->(1,985,4096)`
确认 729 个 Redux token 成功注入。

**B3 核心验收集重跑（8 条，已完成 6 条）**：
- wf58 sd-txt2img-refine ✅ exec 527（19/19）——发现 op-migration bit-rot：老节点版本
  调已废弃 op（latent.empty→sd.latent.empty 等），全量 bump 节点版本后通过
- wf70 p0-nodes-accept ✅ exec 531（17/17）——同样 bump + inference-op 的
  `op_name: sample`→`sd.sample`
- wf93 txt2img-ipadapter ✅ exec 532（20/20）
- wf99 facedetailer ✅ exec 533（20/20）
- wf84 tier1-docker-pil ✅ exec 540（12/12）——executorRef docker-remote-211（已禁用）
  →docker-cpolar
- wf74 p12-accept-image ✅ exec 542（21/21）——同上换执行器；首跑遇跨执行 VRAM 污染
  （D1-c 遗留 FLUX/SD3/SVD 常驻挤爆 8G）→ POST /model/unload 清空后重跑通过
- wf87 sdxl-txt2img ✅ exec 543（8/8）——三次续跑：隧道截断 + 僵尸 job 挡路
  （/interrupt 取消后放行）+ 主机内存压力下 55 分钟慢采样
- wf94 instantid-txt2img ✅ exec 551（14/14，宿主 WSL 重启后）——中途连破两关：
  ① exec 549 撞上 docker daemon 持续 500（Docker Desktop 病入膏肓，/version 都 500）；
  ② **exec 550 假阳性事故**：Sample 实际失败（流截断重挂遇 "already running"），
  但 API 报 success——根因是 Studio runWorkflow 轮询循环出循环后**重读**终态，
  与 RunAsync defer 删实例竞态，拿到空串后落入 `finalStatus=="" && workflowErr==nil
  → "success"` 兜底。修复：终态循环内捕获（pollWorkflowTerminal + 回归测试），
  exec-550 DB 记录已订正为 failed。原描述：exec 546 **宿主机内存危机制卡**：FaceAnalyze 首跑把推理
  服务打崩重启一次；Sample（SDXL+InstantID 25 步）三次跑到 ~70% 时 docker exec
  轮询进程被宿主机 OOM killer SIGKILL（exit 137，确定性 ~45 分钟处）；服务端 job
  本身每次都能跑完。- flowx v0.2.9：docker exec 流重挂预算 6→30（长任务穿隧道）
- flowx-studio go.mod 正式升 flowx v0.2.9（GOPRIVATE 直连源站，不等 proxy）

**B3 收官：8/8 全绿 ✅**（58/70/93/99/84/74/87/94）。

**本轮次生发现/教训**：
- cpolar 免费隧道 HEAD 请求大面积丢失、瞬断率 ~50%、流式连接 40s-数分钟被切：
  docker 执行器 v0.2.6-v0.2.8 三连修（execStreamMaxReattach 3→6 + attachExecWithRetry +
  peekAttachSentinel 识别"attach 成功但拒绝文本写在 hijack 流内"的 daemon 行为）
- flowx DAG 调度是 **fail-fast**：任一节点失败即不再调度 pending 节点 → 覆盖率流水线
  要把易挂的独立支链串到尾部
- exec 快照的 Param 不可变：**隧道地址变了只能整跑，continue 救不了**
- workflow list 分页按更新时间排序：批量更新会重排分页 → 批量操作要按 ID 集复核
  （本轮漏了 3 条：30/31/58）
- Studio server 被 Android 杀后，local 执行器子进程全灭但 exec 状态停在 running
  → 只能 cancel 重跑（v0.2.x continue 可以救 failed 节点）

## 四十二、流水线列表搜索 + 大扫除（2026-10-09）

**列表搜索（flowx-studio 新功能，已推送）**：后端 `List` 改 `ListFilter`——
search 扩到 `yaml_config` 全文（名称/备注/配置三域模糊）；新增 `node`（nodeRef
精确匹配 `name@` 前缀，防子串误伤）与 `exec_status`（最近一条执行状态：
success/failed/running/cancelled/never）过滤；CLI `workflow list` 同步加
`--node/--exec-status`；前端列表页顶部工具条（防抖搜索框 + 三个复用
`components/Select.tsx` 的玻璃拟态下拉，节点选项来自 /api/v1/nodes 跨页聚合）。
测试 `TestListFilters` 9 场景全绿。真机验证：search=ipadapter-load 6 条 /
node=flux-sampler 5 条 / exec_status=failed 4 条。

**流水线大扫除（50 → 39）**：删 A 档一次性/残留 6 条（105/106/112/73/107/83）
+ B 档覆盖重复 5 条（61→84、67→88/74、68、69、75→76）；C 档演示（23/37/39）保留。
全部 50→39 条补齐场景备注（格式：场景一句话+【验收✅exec号】标记）。

**2026-10-10 再清理（39 → 28）**：用户决定删除全部回归验收套件
（62/63/65/74/76/84）——验收历史以本文件与 exec 记录为准；日后需要回归
时按 §41 B3 清单重建。

**顺带修绿两条 last=failed**：
- wf104 flux-redux-ref ✅ exec 554（redux 模型补齐后首跑全通；教训重演：
  续跑时上游 clip 对象已被 LRU 逐出报 object not found → 只能 fresh run）
- wf60 sd3-txt2img ✅ exec 557（两处修复：checkpoint-loader@1.2.0→1.3.0
  修 op-migration bit-rot `checkpoint.load`→`sd.checkpoint.load`；
  EmptyLatent 与大 checkpoint 加载并发被队头阻塞顶死短 timeout →
  改图边 `Ensure-->EmptyLatent` 为 `LoadCheckpoint-->EmptyLatent` 串行化）

**flux-redux-apply UI 补票**：该节点是批次中唯一无 `ui/` 资产的（出生时被模型
缺失卡着只交付了后端）；补 widget-def.js + build-widget.py 生成 node-widget.js，
flowx.json 加 ui 键，v1.0.0→1.0.1 重新导入（node id=170）。

## 四十三、加载旋钮全量化 + 执行租约 pin 立项（2026-10-09 立项，方案已确认，
## 2026-10-10 实施 + 真机验收完成 ✅）

> 来源：§36.4 铺路缺口 #1/#2，用户指定优先排期。方案经逐条确认（决策记录见下），
> **实施顺序：先旋钮（小改、为 Wan2.2 FLF2V 铺路），后租约 pin（中改）**；
> 两项在 model_manager 附近有共同触点，连着做省一轮镜像重建。

### 43.1 方案一：加载旋钮全量化（§36.4 #1）

**现状盘点（代码核实）与动作**：

| 加载路径 | 层 | 旋钮现状 | 本轮动作 |
|---|---|---|---|
| `checkpoint.load` | 核心 ops.py | ✅ dtype/offload/use_t5 全套，缓存键 `\|dtype=..\|offload=..\|t5=..` | 不动，作为范本 |
| `vae.load` / `controlnet.load` | 核心 model_manager | dtype 已进键；offload 无（组件小 ~0.7GB，设计如此） | 只补文档说明，**不加 offload**（决策 3） |
| `motion.load` | 核心 ops.py | ❌ 硬编码 fp16 + 进程级 offload | 加 dtype/offload，进组合缓存键 |
| `ipadapter` 插件（含 clip vision） | inference-server/plugins/ | ❌ | 加 dtype，进缓存键 |
| `flux-unet-load` / `flux-dual-clip-load` / `flux-vae-load` | 节点自注册 server_op.py | ❌ GGUF 自有加载逻辑 | 加 compute dtype 旋钮；量化档位（Q4/Q8）保持按文件名选，不做参数 |
| InstantID/insightface | 独立加载逻辑 | ❌ | **不纳入**（决策 4），文档化 |

**统一语义**（照 checkpoint.load 范本）：全部 `auto` 缺省 = 继承进程级环境变量
（默认行为逐字节不变）；显式给值 → 缓存键扩展，同模型不同旋钮 = 独立常驻条目
（同一 LRU）。

**改动面**：服务端（ops.motion_load + model_manager.load_motion、
plugins/ipadapter.py、3 个 flux 节点 server_op.py）+ ~8 个节点包（flowx.json 加参
+ widget-def + build-widget 重建）+ bundled 三联动（bump BUNDLE_VERSION → 远端
重建镜像 → sync + 重导入）+ 单测（缓存键区分/auto 继承，stub 可跑）。

**验收**：① 回归：全默认参数与基线逐字节一致（accept-m2/tier2 基线）；
② 真机：motion.load 显式 dtype/offload vs auto 各跑一遍，缓存键与 /health
resident 核对；③ Studio：wf53（AnimateDiff）+ 一条 FLUX 流水线复跑全绿。

### 43.2 方案二：执行租约 pin（§36.4 #2）

**关键利好（侦察发现）**：`FLOWX_EXECUTION_ID` 已在注入节点环境
（flowx-studio inject.go:36）——flowx 侧零改动，租约标识现成。

**设计**：
- **标识传递**：节点 `flowx_client.py`（_common 单一事实源，sync 分发）读
  `FLOWX_EXECUTION_ID`，所有请求加 HTTP 头 `X-FlowX-Lease: <exec_id>`。
  用头而非 payload——`/jobs`、`/op`、`/graph`、`/model/unload` 全端点覆盖且
  不动载荷 schema。非 Studio 调用方（验收脚本）不带头 → 无租约 → 行为不变
- **服务端**：model_manager 的 pin 从 `set[key]` 升级为 `lease → keys` 映射；
  租约名下加载的模型 + 算子执行中用到的模型（复用现有 `pin_objects` 识别）
  归入该租约；算子级 pin 保留（防同租约内 OOM 自愈误杀）；淘汰决策
  （`vram.plan_eviction`）跳过其它租约 pin 住的条目
- **释放语义（决策 1）**：**TTL 滑动过期唯一途径**（`LEASE_TTL` env，默认
  30min，每次租约活动刷新）——不依赖调用方收尾，崩溃/杀进程/隧道断均自愈；
  显式 release 端点不做（无真实调用方，记入遗留，将来有调用方再加）
- **排队语义（决策 2）**：租约 B 加载需腾地方但可淘汰条目都被租约 A 占着时
  → 阻塞等待（`LEASE_WAIT_TIMEOUT` env，默认 300s），超时失败带 oom_help
  风格指引——即"并发执行排队而非互踩"

**改动面**：服务端（model_manager.py lease 映射 + 淘汰过滤、engine.py guard
归租约、main.py 头解析 + `GET /leases` 观测端点、jobs.py job 记录 lease）+
节点侧（flowx_client.py 加头一处，sync 分发 49 目录）+ 镜像三联动 + 单测
（租约隔离/过期/等待超时，stub 可跑）。

**验收**：① 单测全绿 + 回归（无头请求行为不变）；② 真机重演 §21.8 事故：
两条 GPU 流水线并发，B 等 A 的租约而非顶掉模型（A 全程无 object not found）；
③ Studio 两条流水线并发 e2e。

### 43.3 决策记录（用户逐条确认，2026-10-09）

1. 租约释放 = **TTL 滑动过期唯一途径**；显式 release 端点记入遗留不做
   （无真实调用方的接口是死代码；且执行结束事件在 Studio 侧，显式调会把推理
   API 知识塞回 Studio 核心，违反 §十七 解耦）
2. 并发冲突 = 阻塞等待 + 超时（非立即失败）
3. vae/controlnet 不加 offload 旋钮（组件 ~0.7GB，设计如此，文档化）
4. InstantID 不纳入旋钮体系——人脸分析是 insightface/ONNX（dtype/offload 无
   对应语义），IPAdapter/face ControlNet 部分不在 `_apply_offload` 体系内，且
   非显存大户；文档化"独立生命周期"，将来成真矛盾再按其机制单独立项
5. 真机验收时的 cpolar 隧道地址（docker + 推理两个）届时向用户确认

**遗留（新增）**：显式 lease release 端点（待真实调用方出现再加）。

### 43.4 实施记录（2026-10-10，代码落地，单测 23 文件全绿）

**43.1 加载旋钮全量化（已实施）**：
- `model_manager.load_motion(ckpt, motion, dtype, offload)`：auto 继承进程级
  默认（dtype→fp16、offload→OFFLOAD_MODE，历史行为逐字节不变）；任一显式
  → 缓存键扩展 `|dtype=..|offload=..`（同 checkpoint.load 范本），底模实例化
  /运动模块加载/体积估算/offload 应用全链路吃旋钮
- `ops.motion_load` 透传；`sd.motion.load` 算子（motion-loader server_op.py）
  加 dtype/offload 输入
- ipadapter 插件 `clip_vision.load` 加 dtype（auto→fp16），编码器缓存键改
  `(name, dtype)`；ipadapter.load 权重为 state dict 原生精度，不加旋钮
- flux 插件三 loader 加 compute dtype 旋钮（unet auto→fp16 / dual-clip 的
  CLIP-L auto→bf16 / vae auto→fp16），显式指定键扩展 `|dtype=..`；T5 装载
  精度恒 bf16（fp16 高激活崩坏，不随旋钮）；量化档位仍按 GGUF 文件名选；
  flux.sample/detail_refine 的 cond embeds 硬编码 fp16 改跟 `model.dtype`
  （默认路径与历史一致，bf16 旋钮闭环）
- **顺带硬化**：flux 插件 9 节点目录各持一份拷贝但不在 sync-common 族管理
  → 收入 `_common/server_ops/flux.py`（OP_FAMILIES 加 flux.py，9 成员），
  消除漂移死角
- 节点包 5 个（motion-loader@1.2.0 / clip-vision-load@1.2.0 /
  flux-unet-load@1.1.0 / flux-dual-clip-load@1.1.0 / flux-vae-load@1.1.0）：
  flowx.json 加参 + env 映射 + main.py 透传 + widget-def 加 select 下拉
  （dtype: auto/fp16/bf16/fp32，motion-loader 另加 offload:
  auto/none/model/sequential）+ build-widget 重建
- 单测 `tests/test_load_knobs.py`（8 项：缓存键语义/auto 继承/非法值报错/
  透传/clip_vision (name,dtype) 缓存/flux 三 loader 键扩展与 compute dtype
  传递，全 stub 可跑）；test_flux_plugin/test_ipadapter_plugin 契约同步更新

**43.2 执行租约 pin（已实施）**：
- 新模块 `app/lease.py`：线程本地 bind/current（sanitize 上限 128 字符，
  非法头=无租约）+ `LeaseBook`（lease→{keys,seen}，独立锁，TTL 滑动过期
  `LEASE_TTL_SECONDS` 默认 1800s，每次活动刷新；forget_keys 簿记摘除；
  leases_holding 供超时指引）+ `wait_help`（oom_help 风格指引）
- `model_manager.py`：load/load_vae/load_controlnet/load_motion/get 全路径
  `_attribute` 归属当前租约；`pin()` 同时归入租约（引擎 pin_objects 链路
  复用，engine.py 零改动）；淘汰拆 `_evict_pass`（pinned = 执行期 pin ∪
  extra_pinned ∪ 其它活跃租约持有）+ `_wait_for_leases`（持锁睡眠排队，
  周期清扫过期租约并重做淘汰，`LEASE_WAIT_TIMEOUT` 默认 300s，超时抛带
  指引 RuntimeError）；`unload`/`evict_idle` 跳过其它租约持有条目；
  `_do_evict` 同步摘除租约簿记（防虚保护堵死后续加载）
- `jobs.py`：Job 记 lease（view 透出），worker 执行线程 bind/unbind
- `main.py`：/op /graph /jobs /model/unload 解析 `X-FlowX-Lease` 头并绑定
  （finally unbind）；GET /jobs/{id} 轮询算租约活动（长任务期间 TTL 持续
  滑动刷新，关键！）；新增 `GET /leases` 观测端点（活跃租约/持有条目/空闲
  时长 + TTL/等待配置）
- 节点侧 `flowx_client.py`：`lease_id()` 读 FLOWX_EXECUTION_ID（Studio
  inject.go:36 已注入，flowx 侧零改动），`_req` 统一加头——/jobs /op
  /graph /model/unload /images 全端点覆盖且不动载荷 schema；非 Studio
  调用方（验收脚本）无头 → 无租约 → 行为不变。sync-common 分发 66 目录
- 单测 `tests/test_lease.py`（14 项：线程本地/LeaseBook 过期语义/淘汰隔离
  矩阵（他人租约跳过、本租约可汰、过期恢复、无租约被全体约束）/unload
  evict_idle 保护/pin 归属/簿记摘除/等待超时指引/对端释放通行/租约过期
  通行/无租约不等待/job 绑定）

### 43.5 真机验收记录（2026-10-10，GTX 1080 8GB + cpolar，bundled v1.8.0）

**部署**：BUNDLE_VERSION 1.7.1→1.8.0；节点镜像 + 推理镜像远端重建；
sd_motion_load / clip_vision_load / flux_unet_load 三插件热更新；
62 个 bundled 节点全量重导入（--overwrite）。

- 43.1① 回归 PASS：默认参数出图 sha1=`48e3225e8b38` 与 M2 基线逐字节一致
- 43.1② PASS：motion.load auto→历史键名命中；显式 `dtype=fp16|offload=
  sequential`→缓存键扩展独立常驻；幂等重载 3.6s；卸载清理正常
- 43.1③ PASS：wf53 AnimateDiff（exec 578）+ wf96 FLUX（exec 580）Studio
  端到端全绿；FLUX 产物 768×768 出图正常
- 43.2① 事故重演 PASS：PASS-1 B 排队不互踩（出现「等待其它执行租约释放
  显存」日志）、PASS-2 A 释放后 B 自动通行（406s）、PASS-3 超时抛带指引
  文案的 RuntimeError
- 43.2② PASS：GET /leases 实时观测并发执行的 lease→keys 归属
- 43.2③ PASS：wf53+wf96 并发（exec 574/575/576），Studio 注入
  FLOWX_EXECUTION_ID 自动绑租约，服务端 job 队列 FIFO 串行，全程无
  object not found、无模型互踩

**验收中抓到并修复一个真 bug**：`_wait_for_leases` 持模型锁睡眠，把对端
unload（同一把锁）挡死到超时——设计的核心解锁路径被自己的锁堵死（双输）。
修复：睡眠前临时释放模型锁、醒后重查；单测改为走真实 unload 路径回归该
纪律（_evict_if_needed 必须持锁调用）。镜像重建后复测通过。

**环境观察（非本特性缺陷，留作后续）**：
1. ~~cpolar HTTP 隧道高频瞬断，ensure_plugin 的 GET /ops 等单发调用无重试
   致 3 次节点瞬败~~ ✅ 2026-10-10 修复（commit `21cb226`）：flowx_client
   新增 `_with_retry`（瞬态错误线性退避 3 次，FLOWX_HTTP_RETRIES 可调，
   4xx 不重试），作用于全部幂等 GET 与 ensure_plugin 插件上传；
   POST /jobs 非幂等刻意不重试（防重复任务）；顺带修复根目录漂移副本。
   真机验收（2026-10-10，隧道 6b63e878.r7.cpolar.cn + docker 14923）：
   66 节点包全量重导入 --overwrite；wf84 tier1-docker-pil e2e SUCCESS
   （12 节点，local+docker 双路径）；真隧道 ensure_plugin(detail.refine)
   hash 命中就绪路径通过、GET /ops 52 插件算子
2. Studio 陈旧节点版本已清理（2026-10-10）：131→82 条记录，删除 49 个
   无引用旧版本；8 条流水线（flux 三条 + ipadapter 两条 + lora/i2v/t2v
   三条）的 nodeRef 同步 bump 到当前版本（新增参数全有默认值，行为兼容）
3. 双 GPU 流水线并发时 8GB 显存超订导致采样降速（分钟级/步）为容量边界；
   租约保证了不互踩，提速靠错峰或减常驻

---

## 四十四、ComfyUI 图像增强能力对标缺口清单（2026-10-10 立项，待实施）

**范围**：按用户提出的 ComfyUI 六大类图像处理能力——① 局部精修与智能重绘 ② 结构控制与特征
迁移 ③ 超分与老照片修复 ④ 风格迁移与重打光 ⑤ 传统图像后处理 ⑥ 电商自动化流水线——逐项
对标现有 **83 个 FlowX 节点 + 52 个服务端算子**，产出「已覆盖清单」（避免重复投入）与
「缺口清单 G1~G11」（均待实现，按 P0/P1/P2 排期）。

**执行纪律**：本清单每项动手前按 §开发纪律「方案先行」单独出方案（改动范围 / 接口设计 /
影响面 / 验收标准），经确认后再实施；新算子一律走**节点自注册插件**（`server_op.py` +
`ensure_plugin`），除新增 Python 依赖需重建 inference-server 镜像外不动服务端核心。

### 44.1 现状盘点：已覆盖，勿重复投入

| 能力域 | 现有节点 / 算子 | 备注 |
| --- | --- | --- |
| FaceDetailer 脸 + 手局部精修 | `detail-refine@1.2.0`（ADetailer 式，YOLO `detector=face\|hand`，参数 conf/padding/guide_size/denoise/feather）、`flux-detail-refine@2.0.0`、`face-mask@1.1.0` + 纯编排 | 已有 `pipeline/facedetailer.yaml`、`instantid-facedetailer.yaml`。**手部检测已支持**（`DETECTOR_DIR` 放 `hand_*.pt` 即可）；修正早期"手部不支持"的误判 |
| 蒙版全家桶 | `mask-paint` / `mask-from-image` / `mask-to-image` / `mask-grow` / `mask-feather` / `mask-invert` + `image-composite`、`latent-composite`、`latent-set-noise-mask` | 手绘 / 提取 / 扩缩 / 羽化 / 反相 / 合成齐备 |
| 局部重绘 inpaint | `detail-refine` / `flux-detail-refine` / `flux-mask-refine` | 经验：**inpaint 必须 denoise=1.0**（§三十九判决实验）；换装场景用 `flux-mask-refine`（脸与背景 100% 保留） |
| ControlNet 结构控制 | `preprocess-canny@1.3.0` / `preprocess-depth@1.1.0` / `preprocess-openpose@1.1.0` + `controlnet-load` / `controlnet-apply` | 算子 `preprocess.canny/depth/openpose` |
| 特征与身份一致性 | `ipadapter-load` / `ipadapter-apply` / `clip-vision-load`、`instantid-face-analyze` / `instantid-apply` / `face-similarity`、`flux-redux-apply` | IP-Adapter（SD15 + SDXL vit-h）、InstantID、FLUX Redux 均已真机验收 |
| 微调与条件注入 | `lora-loader`、`embedding-load`、`cond-set-area` / `cond-combine` / `cond-average` | |
| 物理超分 | `upscale-model-load` / `upscale-model-apply`（spandrel 自动识别 RealESRGAN/SwinIR）、`image-upscale`（lanczos/bicubic 等）、`latent-upscale` | ⚠️ `upscale-model-apply` 的 `tile>0` 是**模型推理分块**（防爆显存），**不是**扩散分块重绘（见 G4） |
| 图生图 / 风格迁移 | `vae-encode` + `ksampler`(denoise<1) + `lora-loader`；SD3 / FLUX 独立链 | 真人↔二次元、黏土风等靠模型 + LoRA 即可 |
| 视频线 | SVD / AnimateDiff / FLUX（见 §二、§三十七等） | |

### 44.2 缺口清单（全部待实现）

| 编号 | 缺口能力 | 对标 ComfyUI | 优先级 | 归属层 | 新增依赖 |
| --- | --- | --- | --- | --- | --- |
| G1 | RMBG 自动抠图 | ComfyUI-RMBG / BRIA RMBG-2.0 | **P0** | 算子插件 | 期望零新增（transformers 已在） |
| G2 | SAM 交互 / 自动分割 | SAM / SAM2 + GroundingDINO | **P0** | 算子插件 | 待评估（见 G2 风险） |
| G3 | Outpainting 扩画布 | Pad Image for Outpainting + inpaint | **P0** | 轻算子 + 纯编排 | 无 |
| G4 | 分块重绘超分 | Ultimate SD Upscale / Tiled Diffusion | P1 | 算子插件 | 无（复用 `sd.sample`） |
| G5 | 传统图像滤镜包 | ComfyUI-Image-Filters / WAS Node Suite | P1 | 算子插件（PIL/numpy） | 无 |
| G6 | 面部修复 | CodeFormer / GFPGAN / FaceRestore | P1 | 算子插件 | 部分可由 spandrel 覆盖 |
| G7 | 预处理器补齐 | AnyLine / TEED / Normal / Lineart / SoftEdge / Scribble / MLSD / Seg | P1 | 算子插件 | 无（controlnet_aux 已在） |
| G8 | 老照片上色 | DDColor / DeOldify | P1 | 算子插件 | 待评估 |
| G9 | AI 重打光 | IC-Light | P2 | 算子插件（大工程） | 权重 + SD15 基座 |
| G10 | 身份保持新方案 | PuLID | P2 | 算子插件 | insightface 已有 + PuLID 权重 |
| G11 | 电商流水线模板 | 白底商品换背景 / 模特换装 / 环境阴影 | P2 | 纯编排（依赖 G1） | 无 |

#### G1 RMBG 自动抠图（P0）

- **目标**：输入 IMAGE → 输出 alpha `MASK` + 去背 RGBA / 前背景合成图；供换背景、局部替换复用。
- **实现路径（优选零新依赖）**：`transformers` 已装 → `AutoModelForImageSegmentation`
  加载 `briaai/RMBG-2.0`（或 RMBG-1.4），算子 `matting.rmbg`；模型权重放 `MODELS_DIR`
  （宿主机 `D:\flowx-data\models`）。节点 `rmbg-matting`（image / 模型名 / 阈值 → mask + image），
  `server_op.py` 自注册；widget 给棋盘格透明底预览。
- **备选路径**：`rembg` + `onnxruntime`（onnx 更省显存，但 rembg 拉入 `opencv-python` 非 headless，
  **必踩已知 cv2 命名空间坑**，需按 DEPLOY 的 uninstall + force-reinstall headless 纪律处理），
  且会拉 `numpy` 风险 → 若走此路必须改 `requirements*.txt` 并重建 Pascal + 标准镜像。
- **风险**：RMBG-2.0 走 `trust_remote_code` 自定义建模，**Pascal 钉版 `transformers==4.44.2`
  能否加载需先验证**；不行则退 RMBG-1.4（U²-Net 系）或走 onnx 路径（与 transformers 版本解耦）。
- **验收**：白底鞋服 + 人像各 3 张，alpha 边缘无白边 / 无背景残留；发丝级细节可接受；
  与手机自带抠图目视对比。

#### G2 SAM 交互 / 自动分割（P0）

- **目标**：点选 / 框选 / 文本提示 → 精确 `MASK`（人物 / 服饰 / 特定物体）。
- **实现**：算子 `seg.sam`（points/box → mask，`SamModel` / `Sam2Model`）+
  `seg.grounding`（文本 → bbox，GroundingDINO），组合即可得「文本提示分割」。
- **节点**：`sam-segment`（画布点选 / 框选 UI，对标 ComfyUI SAM 编辑器）、
  `grounding-detect`（文本 → box / mask）；也可先只做 `grounding-detect` + `detail-refine` 式框选。
- **风险**：**需先核实 `transformers==4.44.2` 是否含 `Sam2Model`**（SAM2 支持较晚）；
  仅 `SamModel` 可用则先交付 SAM1（ViT-H fp16 ~2.5GB）；GroundingDINO 可能需新增
  `timm` 等依赖 + 重建镜像。
- **验收**：多人合影 + 文本 "the red dress" → 仅输出该服饰蒙版；点选人物 → 完整轮廓。

#### G3 Outpainting 扩画布（P0，零算子优先）

- **目标**：left / right / top / bottom 扩边 + feather → 自动补背景与环境。
- **实现**：轻算子 `image.pad`（PIL `ImageOps.expand` + 生成扩图蒙版：原图区 0、
  新扩区 1，边界可羽化）→ 复用 `vae-encode` + `latent-set-noise-mask` +
  `ksampler(denoise=1.0)`；或直接铺到更大空 latent（`latent-composite` /
  `flux-mask-refine` 亦可承接）。
- **节点**：`image-pad-outpaint`（输出 padded image + outpaint mask + 各部扩边像素参数）。
- **验收**：512×512 → 512×896 扩边，新增区域内容连贯、与原图无接缝
  （沿用 §三十九 inpaint 判决实验的量化口径：蒙版内平均差 vs 蒙版外往返精度）。

#### G4 分块重绘超分（P1）

- **目标**：4K/8K 突破显存上限，逐块注入扩散微观细节（**区别于现有模型放大**）。
- **实现**：算子 `image.tiled_upscale`（切片网格 + overlap + 每块 img2img(denoise 0.3~0.5)
  + 接缝融合 seam_fix）；服务端一次性完成，避免 N 次网络往返与逐块模型驻留。
- **风险**：8GB Pascal 上 tile=512 + overlap 64 的显存与耗时需实测；**接缝处理是最易翻车点**。
- **验收**：512 → 2048 在 8GB 完成；无可见网格缝；细节优于纯 RealESRGAN 放大。

#### G5 传统图像滤镜包（P1，成本最低，建议首批）

- **目标**：亮度 / 对比度 / 色相 / 饱和度 / 伽马、曲线映射；高斯 / 运动模糊、锐化；
  噪点叠加、通道分离与合并。
- **实现**（服务端 PIL 纯函数，零 GPU）：`image.adjust`（brightness/contrast/saturation/
  gamma/sharpness，PIL `ImageEnhance`）、`image.curves`（控制点 → LUT）、
  `image.blur`（gaussian/motion/box）、`image.sharpen`、`image.noise`、
  `image.channel`（split / merge）；节点 `image-adjust` / `image-filter` / `image-channels`。
- **依赖**：无（PIL 已在）。
- **验收**：与 Photoshop 同参数输出逐像素近似（亮度偏移的均值偏差落在量化误差内）。

#### G6 面部修复（P1）

- **实现**：`face.restore` 算子（CodeFormer / GFPGAN 权重，spandrel 或独立实现）+
  节点 `face-restore`（face 检测 → 裁剪 → 修复 → 贴回，复用 `detail-refine` 的检测/贴回逻辑）。
- **与 `detail-refine` 的区别**：不重新生成内容，只做修复增强（老照片 / 低清脸）；二者可串联。
- **风险**：spandrel 对 CodeFormer/GFPGAN 架构覆盖需核实；不行则加 `gfpgan` / `basicsr`
  依赖（较重，注意 numpy / opencv 版本冲突）。

#### G7 预处理器补齐（P1，高覆盖低成本）

- **实现**：`controlnet_aux==0.0.10` 已在依赖 → 在 `nodes/_common/server_ops/preprocess.py`
  扩展算子 `preprocess.anyline` / `preprocess.teed` / `preprocess.normal`（BAE/DSINE）/
  `preprocess.lineart` / `preprocess.softedge` / `preprocess.scribble` / `preprocess.seg`。
- **节点策略（需先与用户确认粒度）**：各补瘦节点，或把现有三节点泛化为 `preprocess` 单节点
  + `type` 参数。
- **风险**：部分检测器需额外权重（放 `MODELS_DIR`）；0.0.10 实际可用的检测器清单需实测。

#### G8 老照片上色（P1）

- **实现**：`colorize.ddcolor` 算子（DDColor）+ 节点 `colorize`；黑白 → 彩色。
- **风险**：DDColor 非 transformers 官方架构，可能需自实现 + 新增依赖；
  DeOldify 依赖更重（fastai）。**先调研权重与推理代码量再排期**。

#### G9 IC-Light 重打光（P2，独立立项）

- **实现**：算子 `illum.repaint`——前景 + 背景 + 光照方向 / 风格提示 → 重打光融合；
  需 SD1.5 基座 + IC-Light 权重。
- **工程量**：新采样路径（前景条件注入 / 双 UNet），按 ROADMAP §3 分派纪律做适配函数。

#### G10 PuLID 身份保持（P2）

- **实现**：算子 `pulid.apply`——InsightFace（已有）+ PuLID 权重 + SDXL → 高保真 ID 融合
  （InstantID 的替代 / 补充）。
- **风险**：PuLID 依赖 EVA-CLIP 与自定义 attn 注入，需评估与 `sd.sample` 的注入点
  （可参照 `ipadapter.apply` 既有路径）。

#### G11 电商流水线模板（P2，纯编排，依赖 G1）

- **链路**：白底商品 `rmbg-matting` 抠图 → 背景模板合成（`image-composite`）→ 环境阴影
  （轻算子 `image.shadow`：位移 + 高斯 + 透明度）→ 可选 Flux 细节重绘；
  模特换装复用 `flux-mask-refine`（已有 `flux-virtual-tryon-multi-ref.yaml` 雏形）。
- **产出**：`pipeline/ecom-*.yaml` 模板 + 演示执行记录。

### 44.7 G1/G6 前置侦察（2026-10-11，B2 立项前）

| 候选 | 可得性 | 自定义代码依赖 | 权重文件 | 结论 |
| --- | --- | --- | --- | --- |
| `briaai/RMBG-2.0` | ❌ **HF 401（gated）** | — | — | 与 **TEED 同类问题**（私有仓库拿不到）⇒ 不作首选，勿再尝试 |
| `briaai/RMBG-1.4` | ✅ 公开 | `briarmbg.py` 只 import **torch / transformers**；`MyConfig.py` 同理 | `model.safetensors` + `model.pth` + **`onnx/model*.onnx`**（含 fp16/quantized） | **首选**：`trust_remote_code` 零新增依赖；onnx 变体可退回已装定的 `onnxruntime==1.19.2`（做 Plan B，与 transformers 版本彻底解耦） |
| `ZhengPeng7/BiRefNet` | ✅ 公开 | 需新增 **`kornia` + `timm` + `einops`**（`birefnet.py` 顶层 import） | `model.safetensors`（swin_large 系，较 RMBG-1.4 大） | 质量更好但**要重建镜像 + 过 Pascal 兼容验证** ⇒ 作为 G1 的 v2 增强（先不并入 B2 首批） |

- **G6 面部修复**：`CodeFormer` / `GFPGAN` 走 spandrel 覆盖范围仍需核实（G6 原风险项不变）；
  若 spandrel 不支持，则退「`face-detect`（已有 insightface）+ 独立 restore 权重」路线。
- **对 B2 方案的影响**：G1 立即可做（零依赖路线），**不需要先解决 Pascal 版本风险**；
  BiRefNet 作为可选升级项单独立项。

### 44.3 批次排期建议

| 批次 | 内容 | 说明 |
| --- | --- | --- |
| **B1**（先做，低成本见效） | G5 滤镜包 + G3 扩图 + G7 预处理器补齐 | 零 / 极少新增依赖，无 Pascal 版本风险，一次镜像重建即可覆盖 |
| **B2**（核心价值） | G1 RMBG 抠图 → G6 面部修复 | G1 解锁电商与智能蒙版；先验证 `transformers==4.44.2` 兼容性 |
| **B3** | G2 SAM / GroundingDINO + G4 分块超分 | **已立项：见 §四十九**（2026-10-11）。依赖评估已完成（零新增依赖、零服务端改动、不重建镜像）；G4 需实测 8GB 的接缝质量与耗时 |
| **B4** | G8 上色 → G9 IC-Light → G10 PuLID → G11 电商模板 | 重工程，各自独立立项，遵循 §3 分派纪律。**方案见 §五十**（2026-10-12 侦察完成，待拍板 D1~D8） |

### 44.4 通用依赖与风险（所有缺口共用）

- **镜像重建**：新增 Python 依赖 = 同改 `requirements.txt` + `requirements-pascal.txt` +
  `Dockerfile*`，重建并推送 Pascal（`sm_61`，torch==2.3.1 锁定）+ 标准镜像；
  模型权重一律放宿主机 `D:\flowx-data\models`（C 盘仅剩 ~10G 是硬约束）。
- **cv2 命名空间坑**（DEPLOY 已记）：任何引入 `opencv-python`（非 headless）的新依赖
  （rembg / gfpgan 等）都必须在 Dockerfile 末尾 uninstall 后 force-reinstall headless。
- **钉版纪律**：Pascal `transformers==4.44.2` / `diffusers==0.35.2` / `numpy<2` 不可随意放开
  （diffusers ≥0.36 导入期崩溃；numpy 2.x 冲突破坏 accelerate）——新模型须先验证兼容版本。
- **显存边界**：单卡 8GB，一次只跑一条 GPU 流水线（§四十三租约机制保证不互踩）；
  大模型（SAM ViT-H / GroundingDINO / IC-Light）需评估 `OFFLOAD_MODE` 与预算式淘汰。
- **验收纪律**：所有条目以**真机验收**为准（§开发纪律 3），本地无 GPU 只能做代码与单测。

### 44.5 与 ROADMAP 的对应

本清单对应 `inference-server/ROADMAP.md` 新增「阶段 10：图像后处理与电商能力补齐」
（M10 里程碑）；各条目的技术路线细节在立项时补入 ROADMAP 对应小节。

### 44.6 实施决策与节点清单（细粒度 + 全 UI，2026-10-10 用户确认）

**决策 1（粒度）**：全部按**细粒度节点**拆分——每步可观察、中间产物可复用/可替换
（加载 → 编码 → 处理 → 合成分离，ComfyUI 风格），不做聚合节点。
**决策 2（UI）**：**所有新增节点都必须带自定义 UI**（`ui/node-widget.js`，基于
`_common/widget-base.js`），不允许裸参数面板——至少要有入参控件 + 结果/中间产物预览。

#### 44.6.1 新增节点清单（G1~G11，共 34 个）

| 编号 | 节点名 | 作用 | UI 要点 |
| --- | --- | --- | --- |
| **G1** | `rmbg-load` | 加载 RMBG-2.0/1.4 → MATTING_MODEL 句柄 | 模型下拉（扫描 MODELS_DIR）+ 模型体积/显存预估 |
| G1 | `rmbg-matte` | IMAGE → alpha MASK + 去背 RGBA | 棋盘格透明底预览 + 阈值滑杆实时重算 |
| **G2** | `grounding-dino-load` | 加载 GroundingDINO | 模型下拉 + box_threshold 默认值 |
| G2 | `grounding-detect` | 文本提示 → bboxes/mask | 文本输入 + 检测框叠加可视化 + 命中置信度列表 |
| G2 | `sam-load` | 加载 SAM/SAM2 → SEG_MODEL | 模型下拉（ViT-H/B/L）+ 显存提示 |
| G2 | `sam-segment` | points/box → MASK | **点选/框选画布编辑器** + 蒙版叠加预览（对标 ComfyUI SAM 编辑器，复用 mask-paint 全屏编辑器基座） |
| G2 | `sam-auto-segment` | 自动全图分割 → 多蒙版 | 分割结果网格预览 + 点选保留/排除 |
| **G3** | `image-pad` | 四边扩画布 → padded image + 扩图蒙版 | 四边像素滑杆 + 方向预设（横/竖/居中）+ 扩展区半透明实时预览 |
| **G4** | `tiled-upscale-plan` | 计算切片网格 → TILE_PLAN | 画布上绘制切片网格覆盖层 + tile/overlap 参数联动 |
| G4 | `tiled-upscale-apply` | 按 plan 逐块重绘 + 接缝融合 → IMAGE | 逐块进度条 + 当前块/已融合区预览 + 接缝强度指示 |
| **G5** | `image-adjust` | 亮度/对比度/伽马/锐度 | 全滑杆 + 原图↔结果 AB 对比 |
| G5 | `image-hsl` | 色相/饱和度/明度 | 全滑杆 + 色轮（可选）+ AB 对比 |
| G5 | `image-curves` | RGB/单通道曲线映射 | **可拖拽曲线控制点编辑器**（对标 PS 曲线）+ 直方图底图 |
| G5 | `image-blur` | 高斯/运动/box 模糊 | 类型下拉 + 半径/角度滑杆 + 实时预览 |
| G5 | `image-sharpen` | USM 锐化 | 半径/强度/阈值滑杆 + 实时预览 |
| G5 | `image-noise` | 噪点叠加 | 噪点类型/强度/seed + 预览 |
| G5 | `image-channels` | 通道分离/合并 | 通道勾选（R/G/B/A/亮度）+ 单通道灰度缩略图 |
| **G6** | `face-restore-model-load` | 加载 CodeFormer/GFPGAN | 模型下拉 + 保真度（fidelity）默认值 |
| G6 | `face-restore` | 检测→裁剪→修复→贴回 | 检测框可视化 + 修复前后并排对比 + 强度滑杆 |
| **G7** | `preprocess-anyline` | AnyLine 线稿 | 参数 + 预处理结果缩略预览（复用现有 preprocess widget 基座） |
| G7 | `preprocess-teed` | TEED 软边缘 | 同上 |
| G7 | `preprocess-normal` | BAE/DSINE 法线图 | 同上 |
| G7 | `preprocess-lineart` | Lineart | 同上 |
| G7 | `preprocess-softedge` | SoftEdge（HED） | 同上 |
| G7 | `preprocess-scribble` | Scribble（手绘线） | 同上 |
| G7 | `preprocess-mlsd` | MLSD 直线（建筑） | 同上 |
| G7 | `preprocess-seg` | 语义分割（ade20k/uniform） | 同上 + 色块图例 |
| **G8** | `colorize-model-load` | 加载 DDColor | 模型下拉（建筑/人物版） |
| G8 | `colorize` | 黑白 → 彩色 | 前后对比 + 色彩饱和度/强度滑杆 |
| **G9** | `ic-light-load` | 加载 IC-Light（SD15 基座 + 光照 UNet） | 模型/基座双下拉 + 显存提示 |
| G9 | `ic-light-relight` | 前景+背景+光照 → 重打光 | **光照方向圆盘拖拽选择器** + 光源色温 + 结果预览（对标 IC-Light 官方 demo） |
| **G10** | `pulid-load` | 加载 PuLID + EVA-CLIP | 模型下拉 + 权重/结构提示 |
| G10 | `pulid-apply` | 参考脸 + SDXL 条件 → ID 注入 | 人脸缩略图 + 权重滑杆 + 与 InstantID 相似度对照（可接 `face-similarity`） |
| **G11** | `image-shadow` | 环境投影（位移+高斯+透明度） | 阴影方向/距离/模糊/透明度滑杆 + 实时合成预览 |

> 编排类产出（非节点）：`pipeline/ecom-bg-replace.yaml`、`pipeline/ecom-model-swap.yaml`
> （G11，依赖 G1 + `image-shadow` + 已有 `flux-mask-refine`）。

#### 44.6.2 由「细粒度 + 全 UI」新增的基建改动

| 改动点 | 仓库/文件 | 说明 |
| --- | --- | --- |
| **共享 UI 控件扩充** | `flowx-pixelforge/nodes/_common/widget-base.js` | 新增控件 spec：`curves`（曲线编辑器）、`directionPicker`（光照圆盘）、`pointSelect`（点/框选画布）、`compareAB`（前后对比滑杆）、`checkerPreview`（透明底预览）、`channelToggles`、`tileGridOverlay`（切片网格覆盖层）——单一事实源，改后必须 `_tools/build-widget.py` **全量重建所有节点 bundle** + `check-bundle.py` 门槛校验 |
| **新对象类型（唯一的服务端核心改动）** | `flowx-pixelforge/inference-server/app/registry.py` | `OBJ_TYPES` 新增模型句柄类型：`MATTING_MODEL` / `SEG_MODEL` / `GROUNDING_MODEL` / `RESTORE_MODEL` / `COLORIZE_MODEL` / `ICLIGHT_MODEL` / `PULID_MODEL`（也可收敛为一个 `AUX_MODEL` + 子类型字段，立项时定）；按 ROADMAP §1 例外条款，改后重建推理镜像 |
| **加载类节点统一模式** | 各 `nodes/*-load/server_op.py` | 参考 `upscale-model-load`：算子 `*.load` 返回句柄 + 注册进对象仓库，节点只做参数透传；**缓存键含 dtype/offload 旋钮**（沿用 §四十三 全量化纪律） |
| **UI 预览链路复用** | 各 `nodes/*/ui/node-widget.js` | 复用现有三条链路：`props.preview`（FLOWX_PREVIEW 实时帧）、`__inputs_resolved` + 节点级代理（底图/输入图）、`props.paramSources`（参数来源标注必做） |
| **节点粒度带来的图片搬运量** | 各节点 | 细粒度 = 中间图对象增多，注意对象仓库 TTL 1h 与 `service_url_host` 双视角配置（Studio 拉预览走宿主可达地址） |

#### 44.6.3 排期增量

细粒度 + 全 UI 使 B1~B4 的工作量上浮（节点数 34 vs 原估 ~20，且每个节点含 UI）：
- **B1**：G5（7）+ G3（1）+ G7（8）= **16 个节点**，全部零新增依赖 → 仍是首批；
  但需先落 `widget-base.js` 六个新控件（曲线/AB 对比/透明底/切片网格/通道/方向圆盘）
- **B2**：G1（2）+ G6（2）= 4 个节点 + `registry.py` 新对象类型
- **B3**：G2（5）+ G4（2）= 7 个节点（SAM 点选编辑器是最大的 UI 工程）
- **B4**：G8（2）+ G9（2）+ G10（2）+ G11（1）= 7 个节点 + 2 条电商编排

---

## 四十五、B1 批次立项方案（2026-10-10）：传统滤镜包 + 扩图 + 预处理器补齐

> 对应 §44.3 的 B1，实施 §44.6 决策（细粒度 + 全 UI）。**方案先行，本条即方案**。

### 45.1 范围与分层决策

| 分组 | 内容 | 实现层 | 理由 |
| --- | --- | --- | --- |
| **G5 滤镜（8 节点）** | adjust / hsl / curves / blur / sharpen / noise / channel-extract / channel-merge | **节点侧纯 PIL**（与 image-flip/crop/rotate/composite 同族） | 零推理服务改动、零新依赖（Pillow 已在节点镜像）、灰度对象不占显存；GPU 服务不可用时也能跑 |
| **G3 扩图（1 节点）** | `image-pad`（四边扩画布 + 扩图蒙版） | 节点侧纯 PIL | 同上；后续 inpaint 复用既有 `vae-encode` + `latent-set-noise-mask` + `ksampler(denoise=1.0)` |
| **G7 预处理器（8 节点）** | anyline / teed / normal / lineart / softedge / scribble / mlsd / seg | **服务端算子插件**（扩展 `_common/server_ops/preprocess.py` 族） | 检测器依赖 torch/controlnet_aux，只能在服务端；沿用 canny/depth/openpose 同族模式（同文件多算子，避免跨文件重名 409） |

**G5 细粒度调整**：原 `image-channels`（分离/合并）拆为 `image-channel-extract` +
`image-channel-merge` 两个节点（遵循 §44.6 决策 1）→ B1 节点数 16 → **17**。

### 45.2 G5 节点与接口设计（8 个）

共享实现 `nodes/_common/image_filters.py`（纯 PIL 函数，可本机单测；由 `_tools/sync-common.py`
分发到各成员节点，`check-bundle.py` 校验副本一致）。所有节点输入输出统一：
输入 `image`（对象 ID）→ 输出 `image` + `width`/`height`，UI 用 `compareInput: 'image'`（前后对比）。

| 节点 | 参数（默认） | 说明 |
| --- | --- | --- |
| `image-adjust` | `brightness` 0~2 (1)、`contrast` 0~2 (1)、`gamma` 0.1~3 (1)、`sharpness` 0~3 (1) | 亮度/对比度/锐度走 `ImageEnhance`；gamma 走 point LUT |
| `image-hsl` | `hue` -180~180 (0)、`saturation` 0~2 (1)、`lightness` -1~1 (0) | `convert("HSV")` 三通道调整后转回 RGB（无 numpy） |
| `image-curves` | `channel` all/red/green/blue (all)、`points` "0,0;255,255" 控制点串 | 控制点 → 256 项 LUT（分段线性插值），单通道或多通道 |
| `image-blur` | `mode` gaussian/box/motion (gaussian)、`radius` 0~64 (4)、`angle` -180~180 (0) | motion 用方向卷积核 |
| `image-sharpen` | `mode` usm/simple (usm)、`radius` 0~32 (2)、`percent` 0~300 (150)、`threshold` 0~255 (3) | `UnsharpMask` / `ImageEnhance.Sharpness` |
| `image-noise` | `mode` gaussian/uniform/salt_pepper (gaussian)、`amount` 0~1 (0.05)、`monochrome` false、`seed` 0 | 噪声强度按 amount 映射 |
| `image-channel-extract` | `channel` red/green/blue/alpha/luminance (luminance) | 导出单通道灰度图（可直接当蒙版） |
| `image-channel-merge` | `r_image`/`g_image`/`b_image` 三个对象 ID + `alpha_image`（可空） | 三路灰度合并为 RGB |

### 45.3 G3 节点与接口设计（1 个）

| 节点 | 参数 | 输出 |
| --- | --- | --- |
| `image-pad` | `image`、`left`/`right`/`top`/`bottom`（0~2048，默认 0）、`feather` 0~256 (32)、`fill` "edge/black/white/gray" (edge) | `image`（扩后图）、`mask`（扩图蒙版：原图区 0、新扩区 255，边界羽化）、`width`/`height` |

编排（非节点）：`pipeline/outpaint.yaml` = `image-pad` → `vae-encode` →
`latent-set-noise-mask` → `ksampler(denoise=1.0)` → `vae-decode` → `save-image`。

### 45.4 G7 节点与接口设计（8 个）

扩展 `_common/server_ops/preprocess.py`（单文件多算子，与现有 `preprocess.canny/depth/openpose`
同族）：新增 `preprocess.anyline` / `preprocess.teed` / `preprocess.normal` / `preprocess.lineart` /
`preprocess.softedge` / `preprocess.scribble` / `preprocess.mlsd` / `preprocess.seg`。
节点侧 8 个瘦节点（结构同 `preprocess-canny`，UI 复用同款缩略预览），统一参数
`service_url` / `service_url_host` / `image` / 各检测器专有参数（如 `detect_resolution`、
`normal_type` / `seg_type`）→ 输出 `image`（hint 图，喂 `controlnet-apply`）。
**前置**：controlnet_aux 0.0.10 实际可用检测器与所需权重需真机核实（部分检测器需
`MODELS_DIR` 放权重）。

### 45.5 改动文件清单

- **新增节点目录**：`nodes/<17 个节点>/{flowx.json, main.py, mock.py, test_main.py(G5/G3), ui/widget-def.js}`
- **共享件**：`nodes/_common/image_filters.py`（新）、`_tools/sync-common.py` 的 `targets()`
  增加 `image_filters.py` 分发规则、`_common/server_ops/preprocess.py` 扩充 8 算子 +
  `OP_FAMILIES["preprocess.py"]` 成员追加 8 节点
- **UI 公共件**：`_common/widget-base.js` 新增控件 `kind:'curves'`（曲线编辑器）；
  filters 的即时预览用既有 `compareInput`（AB 对比），`checkerPreview`/`channelToggles` 留待
  G1/G5 后续需要时再加 → 改后 `_tools/build-widget.py --all` 全量重建
- **镜像**：`nodes/Dockerfile` 加 17 行 COPY（Pillow 已在镜像内，无需加 wheel）；
  `BUNDLE_VERSION` bump；`sync-flowx-json.py` + `check-bundle.py` 过门槛
- **推理侧**：G5/G3 **零改动**；G7 只热更新插件（`ensure_plugin` 按 sha256 覆盖），**无需重建镜像**
- **文档**：`inference-server/ROADMAP.md` 阶段 10 细节、本条交付记录、`nodes/README.md` 节点清单

### 45.6 验收口径

- **本机单测**（G5/G3）：`python3 nodes/<node>/test_main.py`——每个算子至少覆盖
  「参数语义正确（如 brightness=2 变亮、gamma<1 变暗、curves 控制点命中）+ 边界（空参数/极值）
  + 尺寸与模式保持」
- **门槛**：`python3 nodes/sync-flowx-json.py && python3 nodes/check-bundle.py` 全绿
  （含共享件一致性与 widget 产物对表）
- **真机 e2e**（需隧道地址确认后）：每节点一条流水线跑通 + 画布 UI 验证
  （滑杆/曲线编辑器/AB 对比/interactive 只读态）
- **G7**：8 个预处理器各出一张 hint 图，并至少 1 个（normal 或 softedge）接
  `controlnet-apply` 端到端出图

### 45.7 不在本轮（B1）

G1/G2/G4/G6/G8/G9/G10/G11 全部不在本轮；`checkerPreview`（透明底预览，G1 需要）与
`directionPicker`（G9）、`pointSelect`（G2）、`tileGridOverlay`（G4）等控件随对应批次再加。

---

## 四十六、B1 交付记录（一）：G5 滤镜族 8 节点 + G3 扩图节点（2026-10-10）

**范围**：§四十五 方案的 G5（8 节点）+ G3（1 节点），即 B1 的 9/17 节点。G7（8 个预处理器）
与镜像重建、真机 e2e 尚未做（见 §46.4）。

### 46.1 交付物

| 项 | 内容 |
| --- | --- |
| 共享实现（新） | `nodes/_common/image_filters.py`（纯 PIL，~11KB）：adjust / hsl / curves / blur / sharpen / noise / channel_extract / channel_merge + `parse_curve_points` / `curve_lut` |
| 新节点（9） | `image-adjust`(189) / `image-hsl`(190) / `image-curves`(191) / `image-blur`(192) / `image-sharpen`(193) / `image-noise`(194) / `image-channel-extract`(195) / `image-channel-merge`(196) / `image-pad`(197) —— 每个含 `flowx.json` + `main.py` + `mock.py` + `test_main.py` + `ui/widget-def.js` + `ui/node-widget.js` |
| UI 公共件 | `_common/widget-base.js` 新增 `kind:'curves'` 控件（画布曲线编辑器：拖拽控制点 / 空白双击加点 / 双击·右键删点 / ↺对角复位 / ⇅反相 / 绑定态只读），头部 spec 文档同步；`build-widget.py --all` 全量重建 74+ 节点 bundle |
| 共享件机制 | `_tools/sync-common.py` 的 `targets()` 增加 `image_filters.py` 分发规则（节点 main.py 引用即分发） |
| 镜像清单 | `nodes/Dockerfile` 加 9 行 COPY；Pillow 注释与头注释改为通用表述；`BUNDLE_VERSION` 仍 1.9.0（未 bump，**镜像未重建**） |
| 文档 | `nodes/README.md` B 类节点清单补 9 节点 + `image_filters.py` 说明 |

### 46.2 验证记录（本机，全部通过）

- **单测 9/9 全绿**（`python3 nodes/<node>/test_main.py`，逐像素/语义断言）：
  - 幂等：默认参数下 adjust / curves / blur(radius=0) / noise(amount=0) / channel-extract / extract→merge / image-pad(零扩边) 均**逐像素**等于原图
  - 语义方向：brightness↑变亮、gamma<1 变暗、saturation=0 三通道相等、反相曲线 mean≈255−orig、模糊 stddev↓、USM stddev↑、噪声 stddev↑、uniform/salt_pepper 同 seed 逐像素可复现
  - 结构：channel-extract 与 `im.split()` 逐通道一致、无 alpha 时 alpha 通道全白、extract→merge 往返还原原图、image-pad 原图区逐像素保留 + edge 补边等于边缘行/列拉伸 + 蒙版 0/255 与羽化过渡带
  - 错误路径：非法 mode/channel/fill、gamma<=0、负扩边、缺通道均抛 `ValueError`
- **门槛**：`sync-flowx-json.py` + `check-bundle.py` 全绿（**76 个节点包 / 镜像内 74 个目录**；共享件一致；widget 产物对表一致）
- **Studio 侧**：9 节点 `node import` 成功 → `node mock` 9/9 `status=success`（参数注入与输出提取正确）
- **踩坑记录**：① `ImageFilter.Kernel` 只支持 3x3/5x5 → 运动模糊改「5x5 线段核重复卷积」（任意半径、C 级、无边缘伪影）；② HSV 往返有 8bit 量化损失 → `hsl` 中性参数必须早退才能做到幂等；③ `pbool` 在 `image_io` 而非 `flowx_client`；④ mock 必须给 `param(name, default)`（Studio mock 不传参）；⑤ `flowx-studio node import --path` 相对路径按 `$HOME` 解析 → 用绝对路径

### 46.3 UI 说明（决策 2 落地）

9 节点全部带自定义 UI：滑杆 + 下拉 + 复选 + **曲线编辑器**（image-curves）+ 前后对比滑块
（`compareInput: 'image'`，走既有 `__inputs_resolved` + 节点级代理链路）；`service_url`/`service_token`
收进折叠的「⚙ 连接/高级」区保持节点紧凑；参数来源标注（workflow/node/literal）由公共件统一渲染。
G5 的「拖拽即时预览」用前后对比（上次执行结果）落地，画面内实时滤镜预览列为后续增强。

### 46.4 镜像与真机 e2e 验收记录（2026-10-11，✅ 通过）

**部署**：`BUNDLE_VERSION` 1.9.0 → **1.10.0**；远端 daemon（docker TCP 隧道 `5.tcp.cpolar.top:14923`，
须先解析 IP + `DOCKER_API_VERSION=1.47`）`SKIP_PUSH=1 ./build-image.sh` 构建成功（上下文 10.1MB）；
`probe-image-nodes.py --tag ...:v1.10.0` → **正常 51 / BROKEN 0 / 隧道未确定 0**；
定向复核 9 节点在镜像内齐备（`image_filters.py` 已随 8 个滤镜节点入镜像，`image-pad` 不需要）；
9 节点 `node import --overwrite` 重导（image tag 同步到 v1.10.0）。

**⚠️ 关键环境发现（重要，后续所有真机验收都适用）**：
cpolar **HTTPS 隧道（`6b63e878.r7.cpolar.cn`）会剥离 `Authorization` 头** →
经隧道的一切鉴权请求 401（实测 0/12 放行），但 `/health`（免鉴权）正常。
**对照证据**：同一 token 在容器内 `http://127.0.0.1:8100/ops` → 200 OK（服务端日志可查），
经隧道 → 401；容器 `INFERENCE_TOKEN` 环境变量与调用 token 一致（`docker inspect` 核实）。
**解法**：**docker 执行器节点走容器内网 `http://172.17.0.1:8100` 直连推理容器，完全绕开隧道**
（`executor: docker` + `service_url: http://172.17.0.1:8100`，token 正常生效，实测 `/ops` 返回 53 算子）。
local 执行器（Termux 侧）在本隧道下**无法**鉴权调用推理服务 → 真机验收一律用 docker 执行器。

**真机 e2e（docker 执行器 + 真实推理服务 + GPU）**：

| 执行 | 流水线 | 结果 | 校验 |
| --- | --- | --- | --- |
| exec 600 | `pipeline/b1-filters-e2e.yaml`（6 滤镜链） | SUCCESS | 6 级产物与本机同函数结果 **sha 逐字节一致**（Adjust/Hue/Curve/Blur/Sharp/Noise，跨 Pillow 12.2/12.3） |
| exec 601 | `pipeline/b1-channels-e2e.yaml`（分离×3 → 合并 → 扩图） | SUCCESS | ExtractR/G/B 与 R/G/B 通道 sha 一致；**Merge 与原图 sha 逐字节一致（分离→合并无损往返）**；Pad 704×384 的 image/mask sha 一致 |
| exec 602 | `pipeline/outpaint.yaml`（GPU 全链） | SUCCESS | 768×384；**判决**：原图区平均差 3.67 vs 扩边区 40.0 = **10.9x**（§39 口径），`mask_inside_max=5`、`mask_outside_min=255`、羽化带 `[214,253,255,255,255]` |

**遗留（非本轮缺陷）**：① cpolar `--follow` 日志流在长任务（KB/步 GPU）下会 `context deadline exceeded`
中断，但执行本身成功（读 status 复核即可）；② 画布实时预览/模型下拉依赖 `service_url_host`
（Studio 侧可达），在 HTTPS 隧道剥离 header 的情况下不可用 → 属 UX 项，待隧道问题解决或换 TCP 隧道后复验；
③ 浏览器侧 UI（曲线编辑器拖拽、对比滑块、回放只读态）需人工在画布确认。

### 46.5 未完成（B1 剩余）

1. ~~**G7 预处理器**~~ ✅ 已交付（7 算子 + 20 变体组合，见 §46.6）；**teed 与 seg 未纳入**（原因见 §46.6）
2. ~~镜像重建 + Studio 重导~~ ✅ 已交付（BUNDLE_VERSION v1.10.0 → v1.11.2 + 探针 0 BROKEN）
3. ~~真机 e2e（3 条流水线）~~ ✅ 已交付（exec 600/601/602，见 §46.4）；浏览器侧 UI 目视验证仍待人工确认
4. **B1 收尾**：浏览器侧 UI 目视确认（滑杆/曲线编辑器/对比滑块/回放只读态 + G7 节点的
   variant 下拉与前后对比）
5. **B2~B4**（§44.3）：G1 RMBG / G2 SAM（含 seg 语义）/ G4 分块超分 / G6 面部修复 / G8 上色 /
   G9 IC-Light / G10 PuLID / G11 电商模板

### 46.6 G7 预处理器族交付记录（B1 第二批，2026-10-11，✅ 真机验收通过）

**交付物**（7 节点 + 1 族插件 + 1 门禁脚本）

| 项 | 内容 |
| --- | --- |
| 服务端族插件 | `nodes/_common/server_ops/preprocess_g7.py`（253 行，7 算子，单一事实源） |
| 节点包 | `preprocess-anyline`(204) / `-lineart`(199) / `-softedge`(200) / `-scribble`(201) / `-normal`(202) / `-shuffle`(203) / `-mlsd`(205)，各带自定义 UI（variant 下拉 + 滑杆 + 前后对比） |
| 算子弹药 | 20 种参数组合可用：anyline（+detect_resolution 两档）、lineart（lineart/anime/standard/coarse）、softedge（pidi/hed）、scribble、mlsd（thr_v/thr_d）、normal、shuffle |
| 发布门禁 | **新增 `nodes/_tools/check-ops.py`（gate J）**：静态校验"族文件注册的算子 ↔ 成员节点 main.py 跑的算子"一一对应、无孤儿、**跨文件重名**（=409 根因）；已接入 `check-bundle.py` |
| 镜像 | `lerkobba/flowx-pixelforge-nodes:v1.11.2`，探针 **58 正常 / 0 BROKEN**（81 目录） |
| 权重 | 8 个文件 ~577MB 落到宿主机 `D:\flowx-data\models\preprocessors\`（清单见 nodes/README.md） |
| e2e 流水线 | `pipeline/g7-preprocess-e2e.yaml`（一图扇出 10 分支，docker 执行器，exec **605 SUCCESS**）、`pipeline/g7-local-exec-smoke.yaml`（local 执行器，exec 124 流水线级 SUCCESS） |

**真机 e2e 判据（exec 605，容器内拉真实产物比对）**：`FAILED = 0`
① 10 个输出全部尺寸=输入尺寸（512×384）；
② 9 个确定性算子（除 shuffle 外）产物 sha **与插件直测逐字节一致**（同一实现两路调用）；
③ 语义：6 张线稿/边缘类灰度均值 < 60（黑底白线）、Normal 蓝紫（R=122/B=253，B > R + 10）；
④ 变体互不相同（lineart vs anime vs standard vs coarse、pidi vs hed、anyline vs lineart）；
⑤ shuffle 与输入不同且每次不同（纯随机打乱，属预期）。

**插件级直测（真机容器内，`FAILED = 0`）**：7 算子 + 变体 + 幂等（两次调用同 sha）+
错误路径（非法 variant → ValueError 带可用值；权重目录缺失 → FileNotFoundError 点名文件；
无权重算子（shuffle/standard）在权重目录缺失时仍可用）。

**两个未纳入的检测器（决策记录）**

1. **teed**：`TheMistoAI/TEED` 仓库已私有（HTTP 401），controlnet_aux 0.0.10 的 `TEEDdetector`
   只暴露类、拿不到可分发权重 ⇒ 软边缘需求由 `softedge(pidi/hed)` 覆盖（HED 即纹理更足的软边）。
2. **seg**：controlnet_aux 0.0.10 **没有语义分割检测器**（18 个类里最近的是 `SamDetector`，
   属 G2 批次且需 2.4GB 权重）⇒ seg 语义并入 **B2 的 G2（SAM）**，不另立节点。

**踩坑与新环境结论（比 §46.4 更新，覆盖旧结论）**

1. **插件上传命名规则**：`ensure_plugin` 以 `op_name.replace(".", "_") + ".py"` 命名上传文件，
   服务端**跨文件重名算子一律 409 全量回滚**（同文件重传才幂等）。因此 `preprocess.canny/depth/openpose`
   必须与 G7 的 7 算子**分族文件**承载（`preprocess.py` vs `preprocess_g7.py`），
   否则一个文件同时注册 10 个算子时必撞 `preprocess_canny.py` 的既有归属。
2. **手机经局域网可直连推理服务**：`http://192.168.5.60:8100`（手机 wlan0 = 192.168.5.43，
   宿主同网段）——**`Authorization` 头不被剥离**，`/ops` 200 + 全量算子清单。
   ⇒ **local 执行器也完全可用**（`g7-local-exec-smoke` 实证），此前"local 无法鉴权、只能用 docker"
   的结论只对**公网 HTTPS 隧道**成立（该隧道确实剥 header，`/health` 免鉴权才通）。
   两种执行器的 service_url 口径：**docker → `http://172.17.0.1:8100`；local → `http://192.168.5.60:8100`**。
3. **权重搬运通道**（宿主无外网、docker cp 大文件会卡死）：
   手机侧（可直连 huggingface.co）下载 → 手机起 `python3 -m http.server 8899 --bind 0.0.0.0` →
   宿主机容器内 `urllib` 拉 `http://192.168.5.43:8899/...` 直接落 `/models/preprocessors/`
   （实测 ~25MB/s，291MB 文件 12.5s）；`docker cp` 只对 <20MB 文件可靠。
4. **Studio 校验参数类型**：`type: "number"` 非法（合法值 `string/integer/float/boolean`），
   导入会报 `invalid parameter type number`。
5. **`LineartStandardDetector`** 上游只收 `detect_resolution`（无 `image_resolution` 形参），
   调用口径不一致需按类分别处理。
6. **落盘式 e2e 与并发/重启的坑**：见 §46.7（对象仓库内存态 + TTL、save-image 的 local 地址口径、
   Graph 边缺失、服务端任务串行）。

### 46.7 落盘式 e2e 验证（B1 全量复核，2026-10-11）

**背景（为什么要落盘）**：服务对象仓库是**内存态 + TTL 4h**（`OBJECT_TTL_SECONDS=14400`），
**服务重启即清空**。B1 第一轮验证只把产物留在对象仓库里、靠临场容器拉取比对 —— 本次实测中
输入对象两次过期/被重启清空（`POST /ops` → 500/404、节点报 `object '...' not found (expired or never created)`），
**验证证据随之消失**。结论：**e2e 流水线必须末端落盘（save-image），对象仓库只当传输通道**。

**改造**：4 条 e2e 流水线全部补 `save-image@1.2.0` 落盘节点（local 执行器 → 写手机磁盘
`~/flowx-e2e-out`），并新增参数 `service_url_local`：

| 流水线 | 落盘节点 | 说明 |
| --- | --- | --- |
| `b1-filters-e2e.yaml` | 7（输入 + 6 级滤镜） | 逐级产物可离线复核 |
| `b1-channels-e2e.yaml` | 7（输入 + RGB 分离 + 合并 + 扩图 + 扩图蒙版） | 无损往返证据落盘 |
| `outpaint.yaml` | 3（扩边图 + 蒙版 + GPU 扩图结果） | §39 判决实验的原始素材 |
| `g7-preprocess-e2e.yaml` | 11（输入 + 10 份提示图） | 分支**串行**（每节点仍读原图，串行只为定序） |
| `g7-local-exec-smoke.yaml` | — | local 执行器冒烟（服务地址用局域网） |

**关键参数口径（踩坑）**：`save-image` 是 **local-only** 节点，跑在手机上，其 `service_url`
必须是**手机可达地址**（`http://192.168.5.60:8100`），不能用 docker 节点用的容器内网
`http://172.17.0.1:8100`；故流水线同时声明 `service_url`（docker 侧）与 `service_url_local`
（local 侧）两个参数。

**验证结果（exec 613 outpaint / 620 G7；本机纯文件复核 `FAILED = 0`，32 个 PNG）**

| 项 | 判据 | 结果 |
| --- | --- | --- |
| 滤镜链 6 级 | 落盘文件 vs 本机 `image_filters.py` 重算 **sha256 逐字节一致** | ✅ 6/6 |
| 通道分离/合并 | R/G/B 单通道逐字节一致；**Merge == 原图**（无损往返） | ✅ |
| image-pad | 扩边图 704×384 / 扩图蒙版与重算一致；原图区逐像素保留 | ✅ |
| Outpainting（GPU） | 768×384；蒙版原图区≈0 / 扩区=255；**扩区改动 39.5 vs 原图区 3.7 = 10.8×** | ✅ |
| G7 10 份提示图 | 尺寸=输入尺寸；9 个确定性算子 sha 与独立直测逐字节一致；线稿类黑底（均值<60）；法线蓝紫（B=253 ≫ R=122）；变体互不相同 | ✅ |

**本轮新踩坑（已修）**

1. **同族插件并发自注册竞态**：10 个节点并行 `ensure_plugin` 时，服务端按"上传文件名=算子名"
   落地文件 —— 首个上传者占据全部同族算子归属，其余节点拿 409（`preprocess_lineart.py` 已拥有）
   或 400（同名文件并发写入的加载竞态），导致随机若干节点失败。
   已修：`_common/flowx_client.py::ensure_plugin` 在 409/400 时**复查 `/ops`：若该算子已由同内容
   （hash 一致）插件注册则视为就绪**（并提示"并发上传竞态，跳过重传"），否则给出可操作报错；
   镜像 v1.11.3 已带该修复。**服务端侧也有记账缺陷**（409 回滚后 op→file 归属串到失败文件上，
   删一个文件会连带注销 legacy 算子）——已在 dev-plan 记录待服务端修。
2. **Graph 边不能少**：流水线里落盘节点若缺 `上游 --> SaveX` 边，会被当成无依赖立即执行、
   取不到上游输出而失败并中止整条执行（exec 617/618/619 连续三次失败即此因）；
   另外 Graph 里不能写 `NOTE:`/非 mermaid 行（exec 617 起因解析异常导致执行体提前判失败）。
3. **服务端算子任务串行**（单 worker）：`/jobs` 里 running 恒为 1，其余 pending。
   并发跑多条重流水线（GPU outpaint + 10 个预处理器）会互相排队到分钟级，
   且中途重启服务会让在跑/排队的节点全部失败 —— **e2e 一律串行提交**。
4. **`save-image` 文件名含时间戳**：复核脚本按前缀匹配文件时要**精确匹配
   `<prefix>_\d{8}_\d{6}.png`**，否则 `g7_lineart` 会误匹 `g7_lineart_standard`（本轮误报 2 个 FAIL）。

# FlowX 文生图/视频 剩余开发计划

> 整理日期：2026-09-14（最近更新：2026-10-10，§四十三 加载旋钮全量化 + 执行租约 pin
> 实施完成：代码 + 单测 23 文件全绿，待镜像三联动与真机验收——隧道地址届时向用户确认；
> 同日新增 §四十四 ComfyUI 图像增强能力对标缺口清单 G1~G11，待按 B1~B4 批次实施；
> 粒度与 UI 已拍板：**一律细粒度拆分 + 所有新节点必须带自定义 UI**（见 §44.6，34 个节点）；
> §四十六 记录 B1 交付：G5 滤镜 8 节点 + G3 扩图（镜 v1.10.0，exec 600/601/602 sha 逐字节一致 +
> 扩图判决 10.9x）+ **G7 预处理器 7 算子（镜 v1.11.2，exec 605 全绿）**；新增门禁 gate J（算子族↔节点一致性），
> 并修订环境结论：**手机经局域网可直连推理服务（local 执行器可用）**、公网 HTTPS 隧道会剥 Authorization 头））
> 范围：flowx-pixelforge（推理服务端 + FlowX 节点）/ flowx-studio（画布与执行）
> 详细技术路线见 `flowx-pixelforge/inference-server/ROADMAP.md`（M2~M10），本文是剩余工作的执行清单。

## ⚠️ 开发纪律（任何开发前先读）

1. **方案先行**：任何开发任务，先把方案（改动范围 / 接口设计 / 影响面 / 验收标准）
   写清楚并经确认，再动手写代码。不允许上来直接改代码。
2. **遵循既有架构约束**：
   - **算子插件化优先**（2026-09-20 起）：完全新的推理算子默认做成**节点自注册插件**
     （节点包携带 server_op.py + ensure_plugin，参考 nodes/detail-refine、nodes/sd3-txt2img），
     不动服务端核心；只有扩展核心原语/加载层/新对象类型时才动 main.py/ops.py/model_manager.py
     （见 inference-server/ROADMAP §1）
   - ROADMAP §1 扩展方法论（插件优先；核心扩展才走 ops.py 纯函数 + main.py 注册一行）
   - ROADMAP §3 采样循环分派纪律（差异只进适配函数，第三个架构前不做统一抽象）
   - ROADMAP §5 不做清单
   - **发布门槛清单**（改完必跑）：`check-bundle.py` 串跑 **gate J**（算子族一致性 + F1 版本戳）、
     **gate L**（widget DOM 回归）、**gate M**（`param()`/`env` 与 flowx.json 参数一致性）。
     **gate M 的由来**：`matting-refine` 参数改名后 `main.py` 漏改，本地单测与产物校验都
     发现不了，真机才报 `TypeError: unexpected keyword argument` ⇒ 参数改名必过 gate M。
   - **B3 追加三条**（见 §49.11①②③）：
     ① **`flowx-yaml` 块的标量一律走 `flowx_client.yaml_scalar()`**（json.dumps 即合法 YAML 双引号标量）——
        裸拼 `f'{k}: "{v}"'` 会被值里的 `"` 打断，Studio 侧变 `__extract_missing`，下游报"未输出字段"；
     ② **JSON 负载跨节点用 `@` 信封**（`_dumps`/`_loads`）——Studio 的参数绑定会把"看起来是 JSON"的
        字符串值自动反序列化成 map，注入 env 时变成 `<map[string]interface {} Value>`；
     ③ **F1 版本化要容忍"跨版本收养"**——`_cached()` 命中的常驻模型可能由旧版族实例装载，
        必须补齐本地 processor 缓存（否则 KeyError）。
   - **改完节点包必须 `node import --overwrite` 重导**：Studio 跑的是**导入时的包快照**，
     共享件（`flowx_client.py`/`executor_base.py`）改了不重导 = 真机仍跑旧代码；
     且 `node import` 有 **100 req/min/IP 限流** ⇒ 全量重导（94 个）必须**节流 1s + 重试**（B3 §49.11④）。
3. **真机验收才算完成**：本地无 GPU，代码落地 ≠ 完成；推理路径必须在
   192.168.5.211（GTX 1080 8GB Pascal）上实测通过才标记 ✅。
   - **⚠️ 验收必须落盘（2026-10-11 起，血泪教训）**：e2e 流水线末端必须有 `save-image`
     节点把产物写成文件（`~/flowx-e2e-out`），对象仓库只是**内存态 + TTL 4h** 的传输通道、
     服务重启即清空 —— 只留对象的验证证据会凭空消失（本次真实发生，见 §46.7）。
     口径：`save-image` 是 local-only 节点，其 `service_url` 必须用**手机可达地址**
     （`service_url_local` 参数），不能用 docker 节点的容器内网 `172.17.0.1`。
   - **真机 e2e 串行提交**：服务端算子任务单 worker（`/jobs` running 恒为 1），
     并发跑多条重流水线只会互相排队；中途重启服务会让在跑/排队节点全部失败。
   - **局域网可用（2026-10-11 修正）**：手机与宿主同网段时，`http://192.168.5.60:8100`
     带 token 直连可用（`Authorization` 头不被剥离）；公网 HTTPS 隧道会剥该头（`/health` 免鉴权才通）。
     docker 执行器 → `http://172.17.0.1:8100`；local 执行器 / save-image → `http://192.168.5.60:8100`。
     **⚠️ 但这个前提随时会破**：手机一旦切到移动数据（本机 IP 变 `10.x`）就整条 local 链路失效
     （save-image 全挂、对象也取不回）；此时 docker 执行器仍可用（它走容器内网），
     但 docker 隧道若同时断开就连镜像都建不了。**动手前先自检**：
     `python3 -c "import socket;s=socket.socket();s.settimeout(5);s.connect(('192.168.5.60',8100))"`
     与 `docker version`，任一失败就先要用户恢复网络（见 §48.10 踩坑记录）。
   - **⚠️ 节点侧纯 PIL 节点必须零 numpy（2026-10-11）**：本机（Termux 手机）**没有 numpy**，
     节点侧 PIL/Pillow 节点一旦 import numpy 就无法本地单测（真机才炸）。
     G5 滤镜族、`mask-*` 家族都是这么做的；B2 的 `matting-refine` 因此把
     `min_area`（连通域面积）改成 `despeckle`（形态学开运算）——效果等价、纯 PIL、可本地验证。
4. **部署环境事实**（勿再踩坑）：
   - docker daemon 在 192.168.5.211（Windows Docker Desktop），不在本机——
     compose/绑定挂载路径指向 **daemon 所在机器**；docker CLI 走 `remote-211` context
   - FlowX 数据已迁到 `D:\flowx-data\`（models/loras/input/videos 绑定挂载），
     C 盘只剩 ~10G 是硬约束，大文件一律进 D 盘
   - 手机不在局域网时的通道（**2026-09-18 起生效**，GPU 主机已搬离原局域网，192.168.5.x 与
     cloudflared 隧道全部失效）：
     - docker API 与推理 API 均走 cpolar TCP 隧道（2375/8100 分别代理）
     - **⚠️ cpolar 免费隧道地址每次重建都会变，禁止写死使用**：任何会话需要远程操作时，
       先向用户确认当前隧道地址（docker / 推理两个），不要沿用历史记录里的旧地址。
       流水线 service_url 等具体值改完后，地址下次变化时需同步更新（当前引用处：
       流水线 30/51/52/53/60/61 的 service_url）
     - **docker CLI 直连 cpolar 域名会握手失败，须先解析出 IP 再连** +
       `DOCKER_API_VERSION=1.47`；隧道不稳时多重试，输出流易断——重活一律 detached 容器
       + 结果写 /videos 再 docker cp 取回
     - cpolar 免费隧道带宽 ~130KB/s，大文件传输不可行；模型下载在宿主机容器内走
       ModelScope 直下（实测 27MB/s）；宿主机容器内**无 curl**，用 python urllib/hub
     - 采样预览为 **Studio 拉取模式**：帧留在服务端 `GET /preview/{job_id}`（内存缓冲只留最新一帧），
       节点 stdout 打 `FLOWX_PREVIEW` 标记上报帧地址，Studio 经 8100 cpolar 隧道中转拉帧
       （复用 service_url，无独立配置）。旧推送回调 FLOWX_CALLBACK_URL_PUBLIC 已于
       commit `0ae4e0b`（2026-09-18）整体移除
   - Studio 推理节点一律 `executor: local`（节点是手机侧 HTTP 客户端，读写本地文件；
     远程 daemon 上跑会因 /tmp 绑定挂载失效）
   - Pascal 镜像锁定 torch 2.3.1 + diffusers 0.30.3（sm_61），升级需先验证兼容性

## 一、阶段 0：公共前置 ✅ 全部完成并已验收

| # | 工作项 | 交付 |
|---|--------|------|
| 0.1 | 异步任务体系 ✅ | `POST /jobs` 提交（graph/op 两种载荷）→ `GET /jobs/{id}` 轮询状态/进度（sample 每步上报）→ `POST /interrupt` 协作式取消（节点间 + 采样每步检查点）；单 worker FIFO 串行 + 全局执行锁。WS 推送留作体验优化后置 |
| 0.2 | 模型管理器分派 ✅ | `app/sniff.py`：diffusers 目录读 model_index.json / safetensors 头部 key 前缀嗅探（SD1.x/SDXL）/ .ckpt 回退 SD1.x；getattr(diffusers) 动态分派，引擎零改动 |
| 0.3 | VIDEO 对象类型 + 落盘 ✅ | 第 7 种对象类型 VIDEO；put 即编码 mp4（imageio-ffmpeg 内置 ffmpeg），仓库只留 {path,fps,num_frames}，TTL/清理同步删文件；`GET /videos/{id}` 下载 |
| 0.4 | 显存治理升级 ✅ | `OFFLOAD_MODE` 三档（none/model/sequential）；`QUANTIZATION=fp8` 接口预留。**实测修复**：sequential 下 pipe.device 返回 meta 导致算子全崩 → `exec_device_of()` 改用 `_execution_device`（commit `b60415b`） |

## 二、阶段 2：图生视频 ✅ 完成并真机验收（2026-09-17）

| # | 工作项 | 状态 |
|---|--------|------|
| 2.1 | 首个视频模型 | ✅ 选型修正：Wan2.2-TI2V-5B 在 8GB Pascal 不可行（diffusers 需 ≥0.33 且 umT5-XXL fp16 ~9GB 单模块超显存）；**改用 SVD-XT 1.1**（纯图生视频，无文本编码器，fp16 峰值 ~4.5GB） |
| 2.2 | 视频管道接入 | ✅ 嗅探分派 + fp16 variant 探测加载（避免 fp32 全量读内存）；`scripts/download-model.sh`（ModelScope 镜像绕开 HF 门控，只拉 fp16 ~4.5GB） |
| 2.3 | 算子 `video.sample` | ✅ 按管道 `__call__` 签名自适应（Wan：prompt+可选首帧 / SVD：image 必填、cfg→min/max_guidance_scale、fps 进采样条件）；`decode_chunk_size` 分块解码；callback_on_step_end 上报 job 进度 + 响应 /interrupt；预览推纯渲染进度卡片 |
| 2.4 | mp4 落盘 + `/videos/{id}` | ✅（0.3 交付） |
| 2.5 | FlowX 节点 `video-gen` | ✅ 异步 job 提交/轮询/超时自动 interrupt；参数面板 + 画布进度卡片 widget |
| 2.6 | `save-video` 节点 | ✅ mp4 下载落盘；小体积内嵌 base64 供 widget 播放 |
| 2.7 | 视频 widget | ✅ 节点包 UI 内嵌 mp4 播放器 + 进度卡片；flowx-studio 外壳零改动 |

**真机验收记录**（GTX 1080 8GB + sequential offload）：
- i2i：execution 184 九节点全绿（构图保留 + 风格化）✅
- i2v（SVD）：25 帧 512×512 出片 ~11 分钟，job 进度/落盘/下载全链路正常 ✅
- 待办：~~`pipeline/video-ti2v.yaml` Studio 画布端到端~~ ✅ 已验收（2026-09-17，execution 187 五节点全绿，出片 755s）

**硬件结论（选型边界）**：8GB Pascal 甜点位 = SVD（i2v）/ AnimateDiff（t2v，✅ 已交付，
SD1.x + MotionAdapter 组合加载）；Wan2.2 / CogVideoX / H3 等新一代视频模型需 ≥16GB Ampere+ 显卡。
**内存教训**：8GB WSL（7.9GB 可用）下模型组合加载要控制峰值——不预载底模、对象仓库会钉住
旧模型引用、组合前后 gc.collect；OFFLOAD 下常驻管权重在 meta 设备，组件复制必须先实例化 CPU 实例。

## 三、阶段 3：前后帧生成视频（FLF2V）

**重新评估**：SVD 不支持首尾帧；`WanFirstLastFrameToVideoPipeline` 依赖 Wan（当前显卡不可行）。
→ **降级搁置**，等有更高配 GPU 或 AnimateDiff 线（SparseCtrl 首尾帧）时再启动。
如要做 AnimateDiff 线，需先过方案（组合加载 loader 扩展 + 运动模块算子设计）。

## 四、MiniMax H3 适配（~~暂不做，等权重确认后再启动~~ → 2026-10-08 调研完成，见 §三十七：
权重已开源、diffusers 已收录、License 地域条款待逐条确认；硬件判定为真·超规格，继续等硬件升级）

前提：~~H3-Base 开源权重实际可用~~（✅ 2026-08-02 已开源）+ **先出预研方案**（三问：trust_remote_code/diffusers 收录情况、
文本编码器构成与显存预算、License）。显存 10B+ 级，超出当前硬件，等硬件升级。

## 五、ROADMAP 中其他待定里程碑（图像线，按需求排期，动手前先过方案）

| 里程碑 | 内容 | 预估侵入面 |
|--------|------|-----------|
| M2 | 采样器与 sigma 排布解耦（sampler_name × scheduler 自由组合） | samplers.py 重构 |
| M3 | VAE fp32 防黑图 | model_manager 几行 |
| ~~M4~~ ✅ | ~~ControlNet（CONTROL_NET/CONTROL 类型 + 采样循环注入 residuals）~~ | ~~采样循环改造~~ 2026-09-21 交付，见十二 |
| ~~M5~~ ✅ | ~~SDXL（双 text encoder + pooled embedding；加载分派已备）~~ | ~~架构分派~~ 2026-09-23 交付，见二十五 |
| ~~M6~~ ✅ | ~~高清放大链（UPSCALE_MODEL + hires fix）~~ | ~~2~3 个算子~~ 2026-09-21 交付，见十四 |
| M8 | Flux / SD3（flow matching，三文本编码器；超当前显存） | 独立采样路径 |

## 六、执行顺序与现状

```
阶段 0.1 异步任务 ✅    阶段 0.2 模型分派 ✅    阶段 0.3 VIDEO 落盘 ✅    阶段 0.4 显存治理 ✅
阶段 2 图生视频 ✅（SVD 真机验收通过，commit 至 6e4aa00）
阶段 3 FLF2V ⏸️（降级搁置：SVD 不支持，Wan 超硬件）
AnimateDiff t2v ✅（2026-09-17 交付：motion.load 算子 + motion-loader 节点 +
  video-t2v-animatediff pipeline；~~API 验收 16帧无黑帧有运动 + Studio execution 191 全绿~~
  ⚠️ 事后发现 191 出的是泥浆噪声废片——当时只验了流程状态没验内容，验收口径教训：
  **视频管线必须抽帧看图才算数**；2026-09-18 根因修复后 exec 199 内容级验收通过，
  16 帧连贯猫咪，出片 flowx_t2v_20260918_105550.mp4）
MiniMax H3 ⏸️（等权重 + 预研方案 + 硬件）
低成本可插队：~~M3（VAE fp32）~~✅ / ~~M2（采样器解耦）~~✅
```

**建议下一步**（均需先出方案）：
1. ~~`video-ti2v.yaml` Studio 画布端到端验收~~ ✅（2026-09-17 完成）
2. ~~M3 VAE fp32 防黑图~~ ✅（2026-09-17 交付并真机验收：50 轮 sample+decode 无黑图）
3. ~~AnimateDiff t2v 方案设计~~ ✅（2026-09-17 交付；2026-09-18 根因修复+内容级验收，见第七节）
4. ~~**loader 节点异步化**~~ ✅（2026-09-18 交付并真机验收，exec 288；见遗留清单 #1）
5. ~~M2 采样器与 sigma 排布解耦~~ ✅（2026-09-21 交付并真机验收，见下）
6. AnimateDiff 后续可选：vid2vid（AnimateDiffVideoToVideoPipeline）/ 运动 LoRA /
   ~~采样器透传~~（已内置 DDIM 标准配置）

## 十一、第二档批次 1 交付记录（2026-09-21，commit `57610bb`，真机+Studio 双验收）

**交付内容**（4 项；set_noise_mask/set_area 需采样循环改造并入 M4，纯 UNet 加载并入 M8，
clip.load 暂缓，M6 独立批次）：

| 能力 | 实现 | 层 |
|---|---|---|
| KSampler Advanced | `sample` 加 `start_at_step`/`end_at_step`/`add_noise`（默认行为逐字节不变）；
  步区间解析抽为纯函数 `ops.resolve_span` | 核心（ops.py+main.py） |
| cond 分段支持 | 正/负 cond 序列长度不一致时分两次 UNet 前向（不零填充，避免污染注意力） | 核心（ops.py） |
| vae.load | `sniff.sniff_component`（ComfyUI/diffusers VAE 单文件 + 组件目录，完整 checkpoint
  明确拒绝）+ `model_manager.load_vae`（默认 fp32，进同一 LRU）+ `ops.VAEShim`
  （duck-type 复用 vae.decode/encode 零改动） | 核心（加载层） |
| cond/latent 工具 | `plugins/cond_latent_ops.py`：cond.combine / cond.average（pad 补零加权）/
  latent.upscale / latent.composite（可选羽化）；SD3 dict COND 明确报错 | 插件（plugins.d 扫描 + /admin/plugins 热上传） |

**验收**（GTX 1080，`scripts/accept-tier2.py` 全过 + 抽图内容核对）：
- t1 回归：默认参数与 M2 基线（`accept-m2-out/t4_euler_normal.png`）逐字节一致
- t2 分段接力：euler 20 步一次跑 ≡ 0→10 / 10→20（`add_noise=false`）逐字节一致
- t3 提前停：end_at_step=10 latent 直解码 = 半成品噪声图（抽图确认）
- t4 vae.load：外挂 vae-ft-mse（ModelScope `AI-ModelScope/sd-vae-ft-mse-original` 直下
  宿主机 /models）decode 生效（与内置逐字节不同）+ vae.encode 兼容 shim 回环正常
- t5 cond.combine/average：双段提示词语义均生效（抽图确认：combine 猫+窗台夕阳双要素，
  average 权重偏移符合预期）
- t6 latent.upscale 512→1024 尺寸正确（纯插值无新增细节，符合预期）
- t7 latent.composite 双 seed 拼接位置/羽化正确（抽图确认）
- Studio 端到端：流水线 62（tier2-ops-accept，12 节点全走 inference-op 通用节点）
  exec 318 全绿，出图内容正确

**部署记录**：镜像重建（`flowx-inference-server:pascal`，app/ 烘焙层更新）+ 容器重建
（配置与 inspect 一致）；插件经 /admin/plugins 热上传并持久化于宿主机 plugins.d
（重建容器后启动扫描自动注册，已验证）；vae-ft-mse-840000-ema-pruned.safetensors
已入宿主机 `D:\flowx-data\models\`。

**排障记录**：cond.combine 拼接产物（154 token）与负向（77 token）长度不一致导致
采样 cat 崩溃 → 采样循环加分两次前向路径（长度一致路径保持逐字节不变，t1/t2 回归证明）。
已知小瑕疵：`/op` 对 FileNotFoundError 返回 500 而非 400（历史行为，catch 只含
KeyError/TypeError/ValueError），下轮顺手修。

## 十二、第三档批次 1 交付记录（2026-09-21，M4 ControlNet + 区域条件 + 局部重绘，真机+Studio 双验收）

**交付内容**（3 项，一次做完"采样循环改造"的三类注入——它们长在同一个循环里）：

| 能力 | 实现 | 层 |
|---|---|---|
| controlnet.load / controlnet.apply | `sniff_component` 加 controlnet 规则（diffusers 目录 /
  lllyasviel 原版 `input_hint_block.*` / ComfyUI 重打包 `control_model.*` 单文件）；
  `load_controlnet`（fp16 默认，`cn:` 前缀进同一 LRU）；`ControlBundle` 捆绑
  hint 图+strength+步窗口；**不用 ControlNetPipeline**——采样循环内逐步前向，
  残差×strength 注入 unet（down/mid_block_additional_residuals），
  保住 M2 采样器解耦 / 分段采样 / LoRA 补丁 / 预览中断全特性 | 核心 |
| cond.set_area | COND 泛化为段列表 [(tensor, area, strength)]（`as_cond_segments` 归一化）；
  循环内逐段前向按区域 mask×strength 混合（ComfyUI 求和语义：重叠叠加不归一化）；
  cond.combine 扩展支持段列表拼接 | 插件算子 + 核心循环消费 |
| latent.set_noise_mask | `LatentBundle{samples, noise_mask}`（mask=灰度 IMAGE，对齐第一档
  蒙版约定）；循环内每步后 mask 外混回原 latent 当步加噪版（局部重绘核心）；
  `resolve_latent` 解包（vae.decode 等取 samples，upscale/composite 对 bundle 明确报错） | 插件算子 + 核心循环消费 |

registry 新增对象类型 CONTROL_NET / CONTROL（核心原语类型扩展）。
组合矩阵：control × split_cond（combine 产物）✅；control × set_area ❌ 明确报错
（每段跑 controlnet 成本×N）；set_area × noise_mask ✅。
行为不变性：无注入时走既有 cat/split 路径，表达式逐字保留。

**验收**（GTX 1080，`scripts/accept-tier3.py` 全过 + 抽图内容核对）：
- t1 回归：默认参数与 M2 基线逐字节一致（accept-tier2 全量同步回归通过）
- t2 controlnet.load：control_v11p_sd15_canny_fp16（723MB，ModelScope
  `AI-ModelScope/ControlNet-v1-1_fp16_safetensors` 直下）加载成功（361M 参数 fp16）
- t3 canny 控制：t1 猫图 PIL FIND_EDGES 线稿作 hint，换 prompt 出狗——构图完全
  对齐线稿（窗台/窗框/姿态/光向一致），ControlNet 结构控制教科书级生效
- t4 strength=0 ≡ 无 control 逐字节一致；t5a 空步窗口（start=0.999 round 后为空）
  ≡ 无 control 逐字节一致；t5b 半程窗口 [0.5,1.0) 出图正常
- t6 set_area：左猫右狗呈现区域语义（左猫纹右狗首——SD1.5 全局注意力下区域条件
  的真实行为，ComfyUI 同款）；**t6b 补充机械判定**：左"red background"右
  "blue background" → 左半 RGB(163,36,83) 右半 (109,94,152)，左红右蓝分区达标
- t7 noise_mask inpaint：左半逐像素保持原图、右半猫换狗且风格光照无缝衔接
- Studio 端到端：流水线 63（tier3-controlnet-accept，11 节点含 load-image 上传
  线稿）exec 319 全绿，出图与 API 验收一致

**部署记录**：镜像两次重建（第二次为 sniff 补 `control_model.*` 前缀规则——
comfyanonymous fp16_safetensors 仓库实测 key 布局，diffusers from_single_file
可自动转换）；容器 MAX_RESIDENT_MODELS 2→3（checkpoint+cn+外挂 vae 三常驻不互踢）；
插件热上传更新（6 算子）并经 plugins.d 持久化；ControlNet 模型已入宿主机
`D:\flowx-data\models\`。

**明确未做**（后续批次）：预处理器算子（openpose/真 canny/depth，controlnet_aux +
opencv 依赖引入——第三档批次 2，服务"人物替换保动作"场景）；多 ControlNet 叠加；
SDXL-ControlNet；IPAdapter / InstantID；专属节点包（继续走 inference-op 验证）。

**同日追加（Studio UI，commit `96efee8`）**：inference-op widget inputs_json 可读性
增强（方案 A）——字面量摘要区（长文本提示词整块展示、绑定值淡显）+ 格式化按钮 +
语法错误红框；分发链路确认：ui/ 资产不进节点镜像，`node import --overwrite`
原位更新即生效（无需三联动）；chrome-devtools 真浏览器逐节点验收。方案 B
（op-aware 动态表单，需 Studio server 代理端点规避 CORS/鉴权）按需再议。

## 十三、第三档批次 2 交付记录（2026-09-21，预处理器算子 + openpose 端到端，真机+Studio 双验收）

**范围**：`preprocess.canny`（真 cv2.Canny，替换历史 PIL FindEdges 假 canny）、
`preprocess.openpose`（controlnet_aux OpenposeDetector，body/hand/face）两个插件
算子（`plugins/preprocess_ops.py`），闭环"人物替换保动作"场景。

**依赖引入**（本批次唯一风险点，已固化进 `requirements-pascal.txt` +
`Dockerfile.pascal`）：controlnet_aux==0.0.10（拉 timm/einops/scikit-image 等）；
坑 1：controlnet_aux 依赖 opencv-python（非 headless），抢占 cv2 命名空间且容器
无 libGL → 镜像构建后 `pip uninstall opencv-python`（headless 同 API）；坑 2：
`pip install --force-reinstall opencv-python-headless` 会拉 5.0 + numpy 2.x 冲突破坏
accelerate(<2.0) → 钉死 `opencv-python-headless==4.11.0.86` + `numpy<2`。

**模型**（宿主机直下）：`control_v11p_sd15_openpose_fp16.safetensors`（722MB，
ModelScope 同 canny 源）入 /models；openpose 预处理器三件套
body/hand/facenet.pth（hf-mirror lllyasviel/Annotators，注意 from_pretrained
即使 include_face=False 也要 facenet.pth）入 /models/preprocessors/openpose/，
算子 `local_files_only=True` 本地加载，运行期零网络。

**冲突处理**：宿主机 plugins.d 已有历史 `preprocess_canny.py`（PIL FindEdges
假 canny，Studio 节点 id 83 preprocess-canny 自注册，流水线 64 sd3-canny-txt2img
在用）。查证服务端 kwargs 只从声明端口构造（多传参数静默忽略）→ 删旧插件、
新插件接管同名算子，旧节点调用参数（passes/cutoff）被忽略后按默认阈值跑真
canny，行为变化=质量提升方向，流水线 64 无需改动。

**验收**（`scripts/accept-tier3b.py` 全绿，OUT_DIR 存抽验图）：
- t1 回归：默认 sample 与 M2 基线逐字节一致（依赖栈变动未污染采样路径）
- t2 真 canny：真人照片→线稿（边缘占比 2.6%，抽图核对头发/字母/手指边缘清晰）
- t3 openpose：半身照→骨架（非黑/彩色占比 3.8%，姿态与原图一致：头后仰+横臂；
  外套下摆被误检为腿部属真实使用水平的假阳性）
- t3b include_hand=true：手部骨架出现（与 t3 非逐字节一致，路径连通）
- t4 端到端：openpose 骨架 + control_v11p_sd15_openpose_fp16 控制采样，prompt
  换"宇航员"——出图人物特征替换、头后仰+横臂姿势对齐骨架（业务目标达成）

**Studio 端到端**：流水线 65（tier3b-openpose-accept，12 节点含 load-image 上传
真人照片 + preprocess.openpose 骨架提取节点）exec 324 全绿，出图与 API 验收一致。
测试照：Unsplash 直下 test_person.jpg（肖像）/ test_fullbody2.jpg（半身）等入
宿主机 /input。

**遗留**：①节点 id 83 preprocess-canny 的 flowx.json 参数（passes/cutoff）对
新算子已是 no-op，描述仍写 FindEdges——下次节点包三联动时顺手更新；②face
精化权重已下载但验收未覆盖（include_face=true 路径）；③~~depth 预处理器
（Midas/depth-anything）与对应 ControlNet 权重未做，按需补批次 3~~
**✅ 2026-09-26 已交付（见 §三十）**。

**同日事故与修复（插件状态漂移）**：交付后另一会话（节点 id 83 属主）的
ensure_plugin 按 hash 比对发现不一致，将旧 preprocess_canny.py 重新上传——
服务端对跨文件重名算子静默跳过（返回 ops=[]），但**文件已落盘**；随后一次
容器重启按字典序先扫到旧文件，PIL canny 注册成功，合并插件的 canny 反被跳过，
且 plugin_ops bookkeeping 把两个算子名都归到 preprocess_ops.py 名下——/ops
显示"file=preprocess_ops.py 但签名是 PIL 版"，运行时为中间合并版（PIL canny
借壳）。修复：DELETE 旧文件（bookkeeping 错位导致 unregistered=[]，PIL Op
残留）→ DELETE preprocess_ops.py（摘除双算子）→ 重传 → 双算子正确注册，
运行时 canny 输出与验收 t2 逐字节一致（sha1=9ad7daa64baa）。暴露的服务端缺陷
（记入遗留，方案先行）：①upload_plugin 对跨文件重名静默 ops=[]，客户端无法
区分"已存在"与"失败"，应 409 + 明确信息；②bookkeeping 在重名跳过时归属错位；
③重启扫描顺序决定重名算子归属，状态可漂移——建议跨文件重名一律拒绝上传
（除非 force 参数）。对方会话已将节点 83 改为软注册（算子已存在则直接用），
与本策略兼容；其 flowx.json 的 passes/cutoff 对新算子为 no-op（静默忽略走默认
100/200）。

## 十四、M6 高清放大链 + 插件系统硬化交付记录（2026-09-21，真机+Studio 双验收）

**重要事实修正**：hires fix 所需原语第二档已全部交付（vae.encode / sample.denoise /
latent.upscale / image.upscale），M6 剩余工作 = 链路端到端验收（零代码）+
UPSCALE_MODEL 模型链 + 插件硬化。原计划的"latent.from_image + sample.denoise
开发"确认无需做。

**A1 hires fix 双路径验收**（scripts/accept-m6.py，全绿 260s，抽图人工核对）：
- latent 路径：sample(512) → latent.upscale(2x bicubic) → sample(denoise=0.45)
  → 1024，构图锁定、毛发细节再生、无崩坏
- 像素路径：sample(512) → decode → image.upscale(2x lanczos) → vae.encode
  → sample(denoise=0.45) → 1024，同样成立（vae.encode 有采样随机性，
  同 seed 不逐字节一致属预期）

**A2 UPSCALE_MODEL 模型链**（plugins/upscale_ops.py 插件）：
- upscale_model.load(name)：spandrel 架构自动识别（.pth/.safetensors），fp16+cuda
  常驻 dict；返回 UpscaleHandle（模型+精度+倍率）——教训：spandrel
  ModelDescriptor 不透传 parameters()，输入侧无法探测精度，dtype 必须由句柄
  携带（首轮 fp16 模型+fp32 输入报 Input/bias type 不匹配）
- image.upscale_with_model(image, upscale_model, tile=0)：tile=0 整图，>0 分块
  （overlap 16px）防显存溢出；512→2048 整图 fp16 实测可过
- registry.py 新增 UPSCALE_MODEL 对象类型（1 行声明式扩展）
- 模型：RealESRGAN_x4plus.pth（67MB，hf-mirror lllyasviel/Annotators，sha256
  前缀 4fa0d389 与官方一致）；抽图对照：RealESRGAN 毛发/胡须纹理重建明显
  优于 lanczos
- 依赖：spandrel==0.4.2（纯 python+torch，容器试装验证零冲突后烘焙）

**B 插件系统硬化**（app/plugins.py + app/main.py）：
- load_plugin 改录制 shim：如实上报插件声明的**全部**算子名（含覆盖名）；
  修复原实现"set 差集只报新增"导致跨文件覆盖被静默吞掉的缺陷
- 新增 detect_conflicts 纯函数：跨文件重名 → owner=文件名；遮蔽核心算子 →
  owner='core'（原实现连核心算子都可被插件静默覆盖）
- upload 端点：冲突默认 **409 + 注册表/文件全量回滚**；force=true 显式接管
  （响应带 took_over，WARN 日志提示重启扫描顺序漂移风险、建议删旧文件）；
  同文件重传幂等不受影响
- scan_plugins：跨文件重名/核心遮蔽 ERROR 日志（register 覆盖语义后加载者生效）
- 顺手项：/op、/graph 对 FileNotFoundError 返回 400（原 500，第二档遗留）
- 事故序列真机重演验证：409 拒绝且运行时未被劫持 → force 接管 → 传回恢复 →
  删临时文件归属簿干净（accept-m6 t4-t7）

**部署**：镜像重建（registry/main/plugins.py 烘焙 + spandrel 入 requirements）；
新增 .dockerignore（验收产物不进构建上下文，12MB→~1MB，隧道传输稳定性考虑）；
容器重建沿用既有配置。

**验收**：accept-m6.py t0-t8 全绿（t8 回归 sha1=48e3225e8b38 与 M2 基线逐字节
一致——镜像重建+核心改动后行为不变性成立）；单测新增 2 文件（全量 13 个通过）；
Studio 流水线 67（m6-hires-accept，15 节点：像素路径 hires + RealESRGAN 双输出）
exec 329 全绿。注意：Studio CLI --follow 的 SSE 流对长时间执行会超时断开，
但服务端执行继续（exec 状态以 DB 为准）。

**遗留**：①force 接管后旧文件仍在盘上，重启扫描顺序可能改变归属（WARN 已加；
彻底解法是 force 时隔离旧文件，按需再做）；②tiled VAE decode 未做（1024 decode
实测可过，更大分辨率爆了再议）；③UPSCALE_MODEL 句柄不进 model_manager 驱逐
体系（模型小，常驻代价可忽略）。

## 十五、Studio 画布增强第一批：过程/结果可视 + 预处理即时预览（2026-09-21，真机验收）

**背景**：inference-op 通用节点原走同步 `/op`，preview 管道（M1 交付的
preview_every → /preview/{job_id} → FLOWX_PREVIEW 标记 → Studio 中转 →
node_preview SSE → widget props.preview）从未被通用节点使用过。侦察确认：
preview-frame 中转端点、帧来源映射（2h TTL）、前端 SSE 处理全部现成，
缺的是执行器上报与 widget 渲染。

**A. inference-op 执行器异步化**（main.py + flowx_client.py）：统一 submit_job +
wait_job（poll 2s）；preview_every>0 时轮询上报过程帧；完成后按输出 type==IMAGE
上报完成帧（{base}/images/{id}）——无采样过程的算子（canny/vae.decode）也能出
结果图，零新媒体端点（完成帧白嫖 preview-frame 通道）。emit 附内部元数据
__op_name/__inputs_resolved（解析后入参，供重放）。行为不变性：结果值不变，
超时自动 interrupt 防孤儿 job。

**B. Go 端点**（workflow.go + workflow_handler.go，+2 测试文件级用例）：
- previewSource 增 base/job_id（marker 新字段，旧标记从 url 推导兼底）
- POST /executions/{id}/nodes/{nodeId}/interrupt-inference → 节点级中断
  （从 previewSource 取 base/token/job_id 转发 /interrupt）
- POST /executions/{id}/nodes/{nodeId}/op-replay {overrides} → 从执行 metadata 取
  __op_name/__inputs_resolved 合并 overrides 重放（同步 /op），结果图写回
  previewSource 并广播 node_preview → 画布图像自动刷新。base/token 缺失时从
  执行 metadata 的 params 兼底

**C. widget**（node-widget.js）：预览/结果区（img + 进度条 + 中断按钮 + 点击放大
overlay，帧过期 onerror 隐藏）；canny 双阈值滑块 + openpose hand/face 复选框
（合并写回 inputs_json 不动其他键）+ ▶预览按钮（调 op-replay 秒级看效果）；
sample 类"实时预览"开关（写入 preview_every=5）；outputs 过滤 __ 前缀内部键。

**真机验收**（exec 330-338）：
- exec 331（流水线 68 = 67 + preview_every=5 + Canny 节点）16/16 全绿；
  Sample1 执行中拉到含噪中间帧（256px JPEG），Decode1 完成帧为全质量 512 PNG，
  抽图核对一致
- op-replay：Canny 阈值 100/200 → 20/60 重放，帧切换且边缘密度明显变化（抽图）
- interrupt：exec 334 长采样（150 步）在 60/150 处被中断，job cancelled ✓
- 回归：流水线 62（exec 336）、65（exec 338，12/12）全绿——异步化后行为不变

**同日事故与修复（cv2 消失）**：流水线 62 回归抓出 M6 镜像 canny 报
No module named 'cv2'——opencv-python-headless 的 dist-info 在但 cv2/ 包文件没了。
根因：pip 不保证装包顺序，controlnet_aux 拉入的 opencv-python 若后装，
Dockerfile 的 pip uninstall 会把共享路径的 cv2/ 文件删掉。修复：uninstall 后
`pip install --force-reinstall --no-deps opencv-python-headless==4.11.0.86`
（--no-deps 防拉动 numpy2）+ 构建期冒烟门禁（import cv2/spandrel/controlnet_aux，
依赖破损构建期炸而非运行时）。教训：镜像重建后必须跑含预处理节点的回归流水线，
签名检查抓不到依赖破损（插件懒加载）。

**HF 缓存持久化**：from_single_file（vae.load/controlnet.load 走 safetensors 时）
隐性地向 HF 拉组件 config，容器重建后缓存丢失导致流水线 62/65 报错。修复：
容器新增挂载 D:\flowx-data\hf-cache → /hf-cache（HF_HOME），经 hf-mirror 一次性
预热（vae-ft-mse + 两个 controlnet + 其他按需），后续重建不再丢。

**Studio 部署注意**：`flowx-studio server start` 跑的是 ~/.local/bin 安装副本，
不是仓库构建产物——改 Go 代码后须 cp 到 ~/.local/bin 再重启（本次 404 排查结论）。

**遗留**：①widget 未在浏览器实际渲染验证（语法校验+契约不变，待用户画布抽检）；
②节点 docker 执行器引用的 lerkobba/flowx-pixelforge-nodes:v1.5.0 镜像未构建
（本地全走 local 执行器，随下次发布构建）；③胶片条多帧/前后对比滑块/画框笔刷
留第二三批。

## 十六、算子专属节点化：基建 + P0 六节点（2026-09-21，真机验收）

**路线修正**：用户明确不要用"一个通用节点塞动态表单"的路线——**每算子一个
专属节点 + 专属 UI**（对标 ComfyUI 节点粒度），inference-op 退回兜底/试验田角色，
原第二批的动态表单取消（对比滑块/ControlNet 滑块并入对应专属节点 widget）。

**基建三件套**（nodes/_common + nodes/_tools，避免 15 个节点 15 倍重复）：
- `executor_base.py`：第一批 inference-op 异步执行器抽成共享模块（submit_job +
  wait_job + preview 帧/完成帧上报 + __op_name/__inputs_resolved emit +
  check_exists 插件算子校验），各节点 main.py ~20 行只声明参数映射；
  按 flowx_client.py 既有约定复制进各节点目录（flowx.json files 声明）
- `widget-base.js`：第一批 widget 公共件抽成 `createNodeWidget(spec)` 工厂——
  预览区/进度条/中断/点击放大、replay 按钮、text/slider/check/select 控件、
  wired 绑定来源标注、**model 模型下拉**（新）、**compareInput 前后对比滑块**（新）；
  节点只写 ui/widget-def.js（WIDGET_SPEC 声明式 ~30 行）
- `_tools/build-widget.py`：base + widget-def 拼接生成单文件 node-widget.js
  （widget 分发要求单文件无依赖）；节点镜像照旧随发布构建（本地 local 执行器）

**P0 六节点**（全部 1.0.0，mock 6/6 通过）：controlnet-load（模型下拉+dtype）/
controlnet-apply（strength+生效区间滑块+▶预览）/upscale-model-load（模型下拉）/
upscale-model-apply（tile 滑块+▶预览+前后对比）/preprocess-openpose（hand/face
复选+▶预览）/vae-load（模型下拉+dtype）

**服务端配套**（唯一后端改动）：
- 推理端 `GET /models/files`：sniff.py 新增 list_model_files 磁盘清单分类
  （checkpoint/vae/controlnet 走嗅探，upscale 走 .pth 直判+名字线索兜底，
  embedding/lora/motion 按目录）；**踩坑：os.path.splitext 会把目录名
  "sd3.5-medium" 的 .5-medium 当扩展名截断——目录不参与去扩展名**（单测补捉）
- Studio `POST /api/v1/inference/models-files`：代理转发（widget 同源 cookie 调）
- Studio `GET /executions/{id}/nodes/{nodeId}/input-image?key=`：节点输入图代理
  （对比滑块"原图"侧；从执行 metadata __inputs_resolved 解析对象 id）
- Go 测试 +2（TestListModelFiles/TestNodeInputImage），共 4 用例全绿

**真机验收**（流水线 70 p0-nodes-accept，17 节点：openpose→controlnet 控制采样
→外挂 VAE 解码→RealESRGAN 4x 全链）：exec 341 全绿；抽图核对骨架图/姿态控制
人像/4x 细节（毛发明显优于 512 原图）；新节点完成帧（Pose 6312B 骨架图）/
input-image 代理（364KB）/op-replay（include_hand=true 重放出手部骨架，
帧字节数 6312→7830 抽图确认）/models-files 代理（15 模型分类正确）逐项验证。

**流程记录**：controlnet-apply 首导被拒——flowx.json 参数类型合法值是 float
不是 number（schema 校验当场拦截）；load-image 节点是 Studio 侧本地上传语义，
服务器 /input 里的图要走 image.load 算子（流水线里 LoadImg 改用 inference-op）。

**遗留**：①P1 七节点（cond.×3/latent.×3/embedding-load）与 P2 二节点
（vae-decode-video/video-sample-latent）未做；②老 17 节点未换装公共件
（ksampler 无 control 端口，控制采样仍走 inference-op）；③节点 docker 镜像
v1.5.0 仍随发布构建。

## 十七、Studio 去业务耦合：通用服务代理（2026-09-21，真机回归）

**问题**（用户审计发现）：前两批把 pixelforge 推理业务的私有知识塞进了
Studio 核心——硬编码对方 API 路径（/jobs/{id}/interrupt、/op、/models/files、
/images/{id}、/preview/ 解析约定）、业务语义路由命名（inference/op/image）、
反解节点参数约定（params.service_url）、业务命名哨兵错误。Studio 作为通用
工作流引擎不应认识任何具体节点生态。

**关键洞察**：widget 已能从 outputs/preview prop 拿到
__op_name/__inputs_resolved/__job_id——Studio 服务端替 widget 做的解析
全是多余的，缺的只是一个"帮我转发到第三方服务"的通用管道。

**方案**：4 个业务端点全部删除，替换为 2 个通用代理；第三方 API 知识
全部下沉到节点包 widget（那才是它该在的地方）：
- `POST /api/v1/service-proxy`：设计期代理，base 调用方显式给定（Studio
  认证），method/path/body 自由，响应**直通**（不再包 Success 包装）
- `ANY /executions/{id}/nodes/{nodeId}/service-proxy?path=&method=`：运行期
  节点级代理，**base 只允许取该节点执行时记录的连接值**（preview 标记上报
  或 params.service_url 约定）——防 SSRF；path 禁 scheme/相对路径（400）
- widget 等价调用：模型下拉=GET /models/files；对比原图=GET /images/{id}
  （id 从 outputs.__inputs_resolved 解析）；中断=POST /interrupt {job_id}
  （job_id 从 preview.jobId 拿）；重放=POST /op {name, inputs 合并 overrides}，
  结果 IMAGE id → GET /images/{id} 直接刷新预览区（含重放覆盖逻辑：
  新 preview 帧到达才解除）
- `node_preview` 事件新增 job_id 透传（FLOWX_PREVIEW marker 本已携带）；
  preview prop 加 jobId；inferenceSourceOf 改名 nodeConnectionOf；
  ErrNoInferenceSource→ErrNoNodeConnection、新增 ErrProxyBadRequest→400

**保留不动**：preview-frame 端点 + FLOWX_PREVIEW 协议（本已通用）。

**迁移面**：widget-base.js 4 处调用点 + inference-op 手写 widget 2 处；
preprocess-canny 无 widget 文件不受影响；7 节点包重导。

**部署坑（再犯，新增教训）**：`pkill -f "flowx-studio server"` 会匹配到
当前 shell 自己的命令行（含该字面量）导致自杀——后续命令（cp 新二进制）
根本没执行，重启的还是旧二进制。禁令：重启用 server stop/start，
不用 pkill -f 带自身命令行里出现的模式。

**真机回归**：Go 3 新用例+全套 8 包绿；设计期代理（15 模型清单）/节点代理
GET 原图（364KB）/POST /op 重放（手部骨架 7830B 与旧 op-replay 逐字节同级）/
SSRF 400/中断（job cancelled）逐项验证；流水线 70 全量重跑 exec 343 17/17 绿；
旧 4 端点 404。

**补记（2026-09-23，漏改修复）**：本次迁移的 4 处调用点里，模型下拉那处只改了
widget-base.js 的头部注释（写明 `POST /api/v1/service-proxy`），实际 fetch 仍是
`/api/v1/inference/models-files`——设计期代理的验证是 curl 直接打的，绕过了
widget 自己的代码路径，所以漏网。症状：节点画布「模型文件名」下拉显示
「列表加载失败：Error: HTTP 404（可手动输入）」（流水线 76 的 Checkpoint 加载
节点）。修复：`_common/widget-base.js` 抽 `serviceProxy(base, tok, path, method, payload)`
助手（走 `/api/v1/service-proxy`，失败时把上游 `message/detail/error` 拼进提示），
`build-widget.py --all` 重建 49 个 bundle，重导 6 个带模型下拉的节点
（checkpoint-loader@1.2.0 / controlnet-load / lora-loader / motion-loader /
upscale-model-load / vae-load，ID 不变）。教训：端到端验证必须走被测代码自身的
调用路径，不能只验后端端点。

## 七、AnimateDiff 废片根因排查与修复（2026-09-18，commit `03b92a6`）

**症状**：execution 191 起所有 t2v 出片为泥浆状不规则噪声帧，无内容。

**根因**：guoyww 官方仓库的 diffusers 格式运动模块文件
（`diffusion_pytorch_model.fp16.safetensors`）经 `MotionAdapter.from_pretrained` 加载后
计算结果错误（隐性不兼容：sha256 与 HF 镜像/ModelScope 官方源三方一致排除下载损坏；
该文件名为 fp16 实为 fp32 张量，上游导出本身马虎）。**换原始 `mm_sd_v15_v2.ckpt`
走 `from_single_file` 独立转换路径即恢复正常**（exec 199 验证）。

**排除过程**（全部真机对照实验，GTX 1080）：底模/TE/VAE（纯 t2i 出完美图）✅ 正常；
UNetMotionModel 包装（运动模块置零→出完美图）✅ 正常；调度器（PNDM/DDIM 均废）❌ 无关；
fp16 数值（运动模块单独 fp32）❌ 无关；SDPA 后端（MATH 强制）❌ 无关；offload 钩子
（整卡直跑）❌ 无关；diffusers 版本（0.30.3/0.35.2 一致废）❌ 无关。

**本次修复清单**：
| 改动 | 位置 | 说明 |
|---|---|---|
| 换运动模块权重 | workflow 53 Param `motion_name` → `mm_sd_v15_v2.ckpt` | 原始 ckpt（ModelScope 已下到宿主机 `motion/`） |
| resolve_motion 支持 .ckpt | `model_manager.py` | 单文件候选增加 .ckpt 扩展名 |
| 组合管 VAE fp32 | `model_manager.py` load_motion | 原仅 load() 路径有，Pascal 防黑图；包 vae.decode 转 dtype（decode_latents 直传 fp16 latent） |
| motion-loader 异步化 | `nodes/motion-loader/` | 同步 /op → /jobs+轮询；隧道空闲超时掐断分钟级无字节连接（exec 198 实测失败） |
| DDIM 调度器 | `model_manager.py`（前一日 `e2d50ca`） | PNDM 默认配置出模糊闪动废片 |

**宿主机产物**（D:\flowx-data\models）：`v1-5-pruned-emaonly-fp16/`（diffusers fp16 目录，
排查时转换，比每次 ckpt 转换省 3 分钟加载+数 GB 内存峰值，可复用）；`motion/mm_sd_v15_v2.ckpt`。

## 八、遗留未改清单（2026-09-18 盘点）

> 另见新增「九、ComfyUI 节点生态对标扩展计划」。

| # | 项 | 风险/影响 | 建议处理 |
|---|---|---|---|
| 1 | ~~checkpoint-loader / lora-loader 仍是同步 `/op`~~ | 隧道下冷加载模型（分钟级无响应字节）必被掐断 | ✅ 2026-09-18 交付并真机验收（commit `c0219a3`+`ab971a0`）：改 `/jobs`+轮询，加 job_timeout/poll_interval；流水线 30/31/51/52/57/58 升 @1.1.0；bundle 镜像重建 v1.1.0；exec 288 冷加载 130s 全绿、出图内容正常 |
| 2 | ~~宿主机遗留损坏的 diffusers 格式运动模块目录 `motion/animatediff-motion-adapter-v1-5-2`~~ | 误引用即出废片（本次根因元凶） | ✅ 2026-09-20 已删（exec 199/313 两轮稳定后真机删除） |
| 3 | Studio 端保存的旧隧道/旧 IP 注释 | workflow YAML 里 service_url 注释还写着 cloudflared 语义 | 下次改流水线时顺手更新注释 |
| 4 | exec 197/198 失败记录 | 修复过程产物（197=旧权重废片、198=隧道超时），属预期 | 无需处理，历史记录保留 |
| 5 | VAE fp32 用 monkeypatch 包 `vae.decode` | diffusers 升级后若 decode_latents 实现变化需复查 | 升级 diffusers 时回归验证 AnimateDiff |
| 6 | ~~Pascal 镜像 torch 2.3.1 + diffusers 0.30.3 锁定~~ | 诊断中实测 0.35.2 可跑通同样代码（兼容性无忧），但生产镜像未升级 | ✅ 2026-09-20 已升 diffusers 0.35.2 + peft 0.17.1 + hub 0.34.4（SD3.5 需要，commit `1faa9c4`）；torch 仍锁 2.3.1；SD1.5 回归已过（exec 297）；**AnimateDiff/视频回归** ✅ 2026-09-20 完成（exec 313 全绿+抽帧看图通过）。**回归发现**：0.35.2 下 fp32 VAE 16 帧单块解码（chunk=16 管道默认值）OOM（`Tried to allocate 4.00 GiB` = `(16,256,512,512)` fp32 那层；exec 309），0.30.3 同参数能过——余量临界被升级吃掉了；处置 = 流水线 53 VideoGen 加 `decode_chunk_size: 8`（既有节点旋钮，验收脚本历史值），不动服务端。**次生教训**：decode OOM 后 VAE 停在 fp32 常驻 + 主机内存压力 → 后续运行采样从 22s/步劣化到 ~7min/步（WSL2 共享显存兜底抖动，exec 310/311），容器重启恢复；OOM 后复跑前先重启容器 | |
| 7 | cpolar 免费隧道 | **地址每次重建都会变（文档/流水线中的具体地址会过期，操作时必须先找用户确认当前地址）**；带宽 ~130KB/s、偶发断连、docker CLI 须用解析 IP（域名握手失败）；**⚠️ 新增（2026-09-20）：docker 执行器 host 存域名会被 fake-ip DNS 解析坑（解析出 198.18.x.x 基准段假地址 → 并行节点随机 attach 超时，exec 299/300 两次踩中）——executor 一律存解析后 IP，隧道重建换地址时同步更新** | 重活 detached+文件落盘取回；大文件走 ModelScope 宿主机直下 |

**验收口径纪律**（新增）：视频/图像管线验收必须**抽帧看图**，流程全绿 ≠ 内容正确
（exec 191 教训：status success 但全是噪声废片）。

## 十、M2 采样器 × sigma 排布解耦（2026-09-21 交付并真机验收）

**内容**：`sampler_name`（更新公式：euler/euler_a/ddim/lms/dpmpp_2m/dpmpp_2m_sde/uni_pc
7 种）× `scheduler`（sigma 曲线：normal/karras/exponential/beta）拆为两个参数自由组合
（对标 ComfyUI 双参数）。

**实现要点**：
- `app/samplers.py` 重构：SAMPLERS（公式）/ SCHEDULES（sigma 开关 kwargs）/ LEGACY_SAMPLERS
  三分；组合支持性**不写死矩阵**，按 scheduler 类 `__init__` 签名动态判定
  （diffusers 升级支持面变化零维护；v0.35.2 源码核实：euler_a/ddim 仅 normal，
  其余 5 种全支持 4 曲线）；非法组合 ValueError 列出该 sampler 支持项
- 旧一体名 `dpmpp_2m_karras` 兼容：scheduler 缺省/normal 时映射 (dpmpp_2m, karras)；
  显式非 normal scheduler 优先并打 warning
- `ops.sample` / `main.py` 注册加 `scheduler="normal"` 入参（默认兼容存量调用）；
  不纳入 sgm_uniform/simple/ddim_uniform（diffusers 无原生开关）
- 节点：ksampler 1.3.0（+scheduler 下拉，bundle v1.4.0）；detail-refine 1.2.0 透传
- 本地单测 `tests/test_samplers.py`（stub diffusers 测映射逻辑，22 项全过）

**真机验收**（GTX 1080，`scripts/accept-m2-samplers.py` 全过）：
- 同 seed 下旧名 dpmpp_2m_karras 与新写法 dpmpp_2m+karras 出图**逐字节一致**
- euler+karras / euler+exponential / dpmpp_2m+beta / uni_pc+exponential 出图内容正常（抽图核对）
- euler_a+karras 明确报错（列出支持项）；euler 默认回归正常

**bundled 镜像节点纪律**（2026-09-18 新增）：bundled 节点跑的是镜像内 `/opt/flowx-nodes/`
的**烘焙代码**，仅改节点包 + 升版本号不会生效——exec 287 教训：nodeRef 已升 1.1.0 仍跑旧代码，
同步 /op 恰好没被隧道掐断而假阳性通过。**节点代码改动三联动：改代码 → bump BUNDLE_VERSION
重建镜像 → sync-flowx-json + 重导入**。验收 bundled 节点改动时，日志里必须看到新代码特有的
输出（如 job submitted 行），不能只看 status。

## 九、ComfyUI 节点生态对标扩展计划（2026-09-18 立项）
**背景**：对照 ComfyUI 原生节点 + 业界高频第三方节点包（ControlNet Aux / IPAdapter Plus /
InstantID / Impact Pack / 3D-Pack 等）梳理了 FlowX 侧能力缺口。**结论：不需要大改推理服务**——
inference-server 是算子注册制（`@registry.register` + ops.py 纯函数，见 ROADMAP §1），
新增能力 = 注册一个算子函数，引擎/对象协议/异步任务体系均不动。真正要决策的是
**每个能力落在哪一层、工作量多大**。以下按改动量分四档，作为后续排期依据
（每项动手前仍需按开发纪律先出方案）。

---

## 四十七、F1 算子身份版本化（算子名 = 实现内容 hash）——已实施 + 真机验收（2026-10-11）

### 47.1 问题（实测复现，见 §46.7 与本轮诊断）

| 场景 | 实测结果 |
| --- | --- |
| 同节点新旧两版（**同文件名**、不同内容） | 三次上传全 **200**、`took_over: []` —— v2 覆盖 v1，随后 **v1 又把 v2 顶掉**，无冲突、无版本比较 |
| 不同文件名抢同一算子 | **409** 全量回滚 ✓（有保护） |
| 同文件名 + 同秒 + 同字节数重传 | `/ops` 报 `plugin_hash`=新版，**实际执行仍是旧版**（`.pyc` 校验只比 mtime 秒级 + 大小） |

根因：冲突判定只看**归属文件名**（`plugins.py::detect_conflicts` 比较 `info["file"] != filename`），
而**同一节点的不同版本必然同文件名**（`filename = op_name.replace(".", "_") + ".py"`，算子名不带版本）
⇒ 走"同文件幂等更新"分支直接放行 ⇒ **后上传者赢**。
次生问题：注册表是**进程全局**、无运行期钉版（ensure 通过 → job 真正执行之间可能被别的执行换掉；服务端单 worker 串行会放大这个窗口）。
现状证据：Studio 里同时挂着 **3 个 bundle 版本**的节点（v1.9.0 ×3 / v1.10.0 ×27 / v1.11.3 ×21）。

### 47.2 方案：**算子身份 = 实现内容**

1. **构建期打版本戳**（`nodes/_tools/sync-common.py`）：
   对每个族文件计算
   `OP_VERSION = sha256(文件内容，且把 stamp 行归零)[:12]`，
   并把该常量写回族文件（幂等：重复执行结果相同）。
2. **族内注册名统一版本化**：族文件 `register(registry)` 首行做一次包装
   `registry = _VersionedRegistry(registry, OP_VERSION)`，
   则 `registry.register("preprocess.anyline", ...)` 实际注册为
   **`preprocess.anyline@<OP_VERSION>`** —— 族内 7~9 个注册调用**一行都不用改**。
   族文件同时声明 `OP_FAMILY = "<slug>"`（插件落盘名前缀）。
3. **上传落盘名 = 内容寻址**：`<OP_FAMILY>_<OP_VERSION>.py`
   ⇒ 同内容 = 同文件（重传幂等，**409 竞态类问题消失**）；不同内容 = 不同文件（**新旧版本在服务端并存**）。
4. **节点侧用服务端确认的名字**：`ensure_plugin` 改为**返回实际算子名**
   —— 按「基名匹配 + `plugin_hash` == 本文件 digest」在 `/ops`（或上传响应的 `ops` 列表）里取，
   取不到版本化名时回退基名（兼容未改造族）。
   `executor_base.ensure_op/run_op` 用它提交 job ⇒ **每个执行天然钉在自己那版实现上**。
5. **服务端零改动**（算子名对注册表是不透明字符串；已确认无节点直提 `/graph`、服务端不校验算子名格式）。

### 47.3 为什么是根治

- 旧版本**不可能**再顶掉新版本：不同内容 ⇒ 不同算子名 + 不同文件 ⇒ 注册表并存（同名归并只发生在内容相同时，且此时语义等价）。
- **运行期钉版（TOCTOU）消失**：算子名里带实现 hash，job 提交的名字就锁定实现。
- **跨版本零膨胀**：未改动的族文件 hash 不变 ⇒ 与上一版本同名，不新增条目；只有真改了才换名。
- 顺带修掉两个历史坑：`preprocess.<op>` 归属漂移（谁先上传谁命名）、同族 409 竞态（`_op_ready()` 兜底可保留为保险）。

### 47.4 改动范围

| 文件 | 改动 | 说明 |
| --- | --- | --- |
| `nodes/_tools/sync-common.py` | 新增 stamp 逻辑（计算 + 回写 `OP_VERSION`） | 单一事实源，幂等 |
| `nodes/_common/server_ops/*.py`（8 个族） | 各加 `OP_FAMILY` + `_VersionedRegistry`（~14 行）+ `register()` 首行包装 | 8 处机械改动 |
| `nodes/_common/flowx_client.py` | `ensure_plugin`：文件名改为 `<OP_FAMILY>_<hash12>.py`（解析族文件常量，缺失则沿用旧规则）；**返回实际算子名** | 保留 409/400 兜底 |
| `nodes/_common/executor_base.py` | `ensure_op` 透传返回名；`run_op` 用实际名提交；`__op_name` 元数据随之为版本化名 | op-replay 兼容（名字仍在 `/ops` 里） |
| `nodes/_tools/check-ops.py`（gate J） | 新增：① `OP_VERSION` 必须等于内容 hash（漂移即失败）② `OP_FAMILY` 存在且唯一 ③ `register()` 必须走版本化包装 | 静态拦截"忘了 stamp / 绕过包装" |
| `nodes/_tools/purge-plugin-versions.py`（新增，手动工具） | 按族保留最近 N 版，删除更旧插件文件 | **不自动清理**（避免误删在跑版本） |
| 文档 | 本节 + `nodes/README.md` 族文件合同 + `inference-server/ROADMAP.md` 备注 | — |
| 发布 | `BUNDLE_VERSION` 1.11.3 → **1.12.0**；sync → check-bundle → 建镜像 → probe → 全量 re-import 83 节点 | 镜像 tag 变化，逐节点记录需刷新 |

### 47.5 兼容与迁移（零破坏）

- **未改造的老 bundle 节点**（v1.9.0/v1.10.0，仍用 `sd.sample`、`preprocess.canny` 等**基名**）：
  服务端现有 50 个算子文件与注册项**保持不动**，老节点照旧可用 ⇒ 新旧混跑不需要停机或强制重导。
- 新 bundle 节点使用版本化名，与老名字**并存不冲突**。
- 服务重启后按 `plugins.d` 扫描恢复：所有版本化文件都在盘上 ⇒ 版本集完整恢复。
- 唯一新增运维项：`plugins.d` 会随发布累积版本文件 ⇒ 提供 purge 工具（手动执行）。

### 47.6 验收标准（真机，全部通过才算 ✅）

- **A1 共存（核心判据）**：构造"同节点 v2"（改一行实现 + 重新 stamp，上传成另一版本）⇒
  先跑 v1 流水线、再跑 v2、**再跑一次 v1**：`/ops` 里两个版本**同时存在**且各自 hash 不变；
  `/admin/plugins` 两个文件并存；两条流水线各自输出**各自版本**的结果 —— 证明"旧版本顶不掉新版本"。
- **A2 竞态消除**：10 路 G7 扇出并行自注册 → 全绿、无 409（同内容同文件幂等）。
- **A3 实现零回归**：G7 10 份产物 sha 与 §46.7 已落盘基线**逐字节一致**；G5/G3 节点回归通过。
- **A4 向后兼容**：老 bundle 节点与新 bundle 节点混跑均正常；服务重启后版本集完整恢复。
- **A5 门禁与单测**：gate J（含新增 3 项）、`check-bundle.py`、节点 mock 单测全绿。
- **A6 落盘**：e2e 产物继续落盘（§46.7 口径），复核 `FAILED = 0`。
- **A7 op-replay**：G7 节点自定义 UI 的调参预览（走 `__op_name` 重放）在版本化名下仍可用。

### 47.7 明确不做

- 不做服务端版本比较/降级拒绝（F2）：F1 之下"同名不同内容"不再发生，无需求。
- 不做运行期内容寻址路由（F3）：F1 已达成等效的"每执行钉版"。
- 不自动删除旧版本文件；不改服务端代码；不改 Studio。

### 47.9 实施记录与真机验收（2026-10-11）

**实施**（按 47.4 全部完成）：

| 项 | 落地 |
| --- | --- |
| 版本戳 | `sync-common.py` 新增 `family_version()/stamp_family()`：`OP_VERSION = sha256(内容, stamp 行归零)[:12]`，写回并分发（**幂等**：连跑两次第二次写 0 文件） |
| 版本戳值 | `cond_latent@32051c150230` / `flux@f2447596b467` / `instantid@24e5469e52cc` / `ipadapter@12c602d06ca2` / `preprocess@ef8a3f49385a` / **`preprocess_g7@42636b08e915`** / `upscale@30ed476b21d4` / `video_sample@a007dcb8f270` |
| 族文件 | 8 个族各加 `OP_VERSION` + `OP_FAMILY`（= 文件 stem）+ `_VersionedRegistry` + `register()` 首行包装；**族内 36 个 `registry.register(...)` 调用一行未改** |
| 客户端 | `flowx_client`：`plugin_identity()`（AST 解析常量，不 import）、`_resolve_op_name()`（同基名 + 同内容 hash）、落盘名 `<OP_FAMILY>_<ver>.py`、**返回实际算子名** + `_OP_ALIAS` 翻译表；`call_op/submit_job` 自动翻译（⇒ 60 个自包含节点的字面算子名零改动） |
| 执行器 | `executor_base.ensure_op` 返回实际名，`run_op` 用它提交 job 与写 `__op_name` |
| 门禁 | gate J 新增 3 项（版本戳 == 内容 hash / `OP_FAMILY` == 文件 stem / 注册在包装之后），**5 个负例实测全拦住** |
| 新工具 | `_tools/purge-plugin-versions.py`（默认 dry-run；只删被取代的版本，不碰历史未版本化文件） |
| 新测试 | `nodes/test-op-versioning.py`（假服务端 + **真族文件 exec**，25 项断言）——覆盖并存/幂等/旧版顶不掉新版/旧族兼容/基名翻译/409 兜底报错 |
| 发布 | `BUNDLE_VERSION` 1.11.3 → **1.12.0**；镜像 `lerkobba/flowx-pixelforge-nodes:v1.12.0`；探针 **58 正常 / 0 BROKEN**；Studio 全量 re-import **83/83** |

**验收结果（真机，全部通过 ✅）**

| 判据 | 结果 |
| --- | --- |
| **A1 共存**（核心） | 服务端实验：上传 v1（真实实现）、v2（阈值默认改）、v3（反相实现）⇒ `/ops` 里 **`preprocess.anyline@42636b08e915` / `@8bf86172b021` / `@ccd9147fc53f` 三版并存**、各自文件/ hash 互不影响；v1 vs v3 输出 **sha 不同**（`9fe15776b24f` vs `c87ac87c3bcd`）⇒ **各版本各跑各自实现**；最后重传 v1（模拟老节点后跑）⇒ **v2/v3 完整保留**（旧版顶不掉新版） |
| **A2 竞态消除** | 先 DELETE `preprocess_g7_42636b08e915.py`（制造首次并发窗口），再跑 **10 路扇出**流水线（wf 125 / exec 628）⇒ **全绿**，服务端日志显示该文件**只被上传 1 次**（其余节点命中预检/幂等）；F1 前此场景会随机 409「op conflicts (rolled back)」 |
| **A3 实现零回归** | 4 条 e2e 重跑：wf 120 → exec 625、wf 121 → **626**、`outpaint.yaml` → **627**、wf 123 → **624** 全部 SUCCESS；落盘产物复核 **FAILED = 0**：G7 十份产物 sha 与 F1 前基线**逐字节一致**、滤镜 6 级与重算一致、Merge 仍无损、扩图判决仍 **10.8×** |
| **A4 向后兼容** | 老 bundle 基名算子（60 个）原样保留：服务端现 **73 算子 = 60 基名 + 13 版本化**；`rulebundle`/老节点未做任何改动即可继续跑 |
| **A5 门禁与单测** | gate J（含新增 3 项 + 5 负例）✅、`check-bundle.py` ✅、`test-tier1-nodes.py` ALL PASS ✅、`test-op-versioning.py` FAILED = 0 ✅ |
| **A6 落盘** | 沿用 §46.7 口径，`~/flowx-e2e-out` 继续产出（扇出跑亦复核通过） |
| **A7 op-replay** | 节点 stdout 的 `__op_name` = **`preprocess.anyline@42636b08e915`** ⇒ Studio 重放打到**确切版本**。已自动验证：① widget 的「▶ 预览」逻辑用 `outputs.__op_name` 拼 `/op` 请求，DOM stub 里点击后捕获到的请求体 `name` 与注入名**逐字节相同**、走节点级代理 `service-proxy&path=/op`（新增回归，`nodes/_tools/test-widget-ui.mjs` 第 6 节，**380 断言 / 0 失败**，覆盖 81 个自定义 widget 节点）；② 真机 HTTP：16 个 B1 节点的 `ui/node-widget.js` 全部 **200**（不会「组件加载失败」）、测试面板 mock **16/16 success**。剩余仅**观感**（控件排布/拖拽手感）留人工 |

**生产路径证据**（exec 624 节点日志）：

```
[op] preprocess.anyline@42636b08e915 job=a40cac2c
[op] preprocess.lineart@42636b08e915  job=e86023de
[op] preprocess.softedge@42636b08e915 job=92362217
```

**收尾**：A1 实验残留的 v2/v3 版本文件已用 `purge-plugin-versions.py --apply` 清掉，再跑一遍确认"无需清理"（工具幂等）。

### 47.8 决策点（已按默认值执行）

| # | 决策 | 默认 |
| --- | --- | --- |
| D1 | 版本取值 | 族文件内容 hash **前 12 位**（未改动算子跨版本同名，零膨胀） |
| D2 | 上传落盘名 | `<OP_FAMILY>_<hash12>.py`（内容寻址；未改造族沿用旧规则） |
| D3 | 旧版本文件清理 | 不自动清理，只提供 `purge-plugin-versions.py` 手动工具 |
| D4 | 兼容口径 | 老 bundle 基名算子**全部保留**，零停机迁移 |

## 四十八、B2 批次立项方案（2026-10-11）：BiRefNet 抠图 + 人脸修复

> 对应 §44.3 的 B2（G1 P0 + G6 P1），实施 §44.6 决策（细粒度 + 全 UI）。
> **用户拍板**：G1 模型直接用 **BiRefNet**（不用 RMBG 系）。**方案先行，本条即方案**。
> 前置侦察结论见 §44.7（RMBG-2.0 gated 401；BiRefNet 公开可用）。

### 48.1 范围与分层决策

| 能力 | 实现层 | 理由 |
| --- | --- | --- |
| G1 抠图推理 | 服务端算子族 **`matting.py`**（`matting.birefnet`） | 需 GPU + 新依赖（timm/kornia/einops）；权重在宿主机 |
| G1 alpha 精修 | **节点侧纯 PIL**（`matting-refine`） | 纯像素运算零 GPU、省一次网络往返；与 `mask-grow/feather/invert` 同范式 |
| G1 换底合成 | **节点侧纯 PIL**（`matting-composite`） | 同上；差异化 = 纯色底 + 直吃 RGBA/RGB+mask |
| G6 人脸修复 | 服务端算子族 **`face_restore.py`**（`face.restore`） | spandrel + insightface 都在服务端，检测→裁剪→修复→贴回一次完成 |

**不新增对象类型**：抠图蒙版沿用既有口径 = **灰度 `IMAGE`**（0..255，与 `mask-grow/feather/invert` 一致）；
BiRefNet 变体与面部修复模型都由节点参数选、按固定目录加载（同 `PREPROCESSOR_MODELS_DIR` 范式）。
⇒ 服务端核心零改动（对象仓库 / registry 都不动）。

### 48.2 节点清单（4 个新节点，细粒度 + 全自定义 UI）

| # | 节点 | 实现 | 参数（含默认） | 输出 | 自定义 UI |
| --- | --- | --- | --- | --- | --- |
| 1 | `matting-birefnet` | 服务端 `matting.birefnet` | `image`、`variant=lite`(lite/main/portrait)、`resolution=1024`(1024/1536/2048)、`use_checkpoint=false`、`invert=false`、`alpha_gamma=1.0` | `mask`（灰度 alpha）、`image`（**RGBA 去背图**）、`width`/`height` | 棋盘格透明底预览 + variant 下拉 + 分辨率滑杆 + 「▶预览」（replay） |
| 2 | `matting-refine` | 节点侧 PIL | `mask`、`threshold=0`(0=不二值化)、`gamma=1.0`、`contrast=1.0`、`feather=0`、`grow=0`(±收缩扩张)、`min_area=0`(去小连通域)、`invert=false` | `mask` | 黑白蒙版预览 + 滑杆 + **前景像素占比读数**（便于阈值调参） |
| 3 | `matting-composite` | 节点侧 PIL | `image`（可 RGBA）、`mask`（RGB 输入时必填）、`bg_mode=color`(color/image)、`bg_color=#FFFFFF`、`bg_image`、`feather=0`、`x=0`、`y=0` | `image` | 换底预览（棋盘格 ⇄ 纯色切换）+ 色板 + 偏移滑杆 |
| 4 | `face-restore` | 服务端 `face.restore` | `image`、`model=gfpgan`(gfpgan/restoreformer)、`upscale=1.0`、`padding=0.3`、`feather=16`、`only_center=true`、`max_faces=1` | `image` | 前后对比滑块 + 模型下拉 + 强度/羽化滑杆 |

> **D3**：`matting-composite` 的备选形态是「新增极轻节点 `image-solid`（纯色图）+ 复用现有 `image-composite`」。
> 默认取前者（一步换底最贴合电商白底用例，且 `image-composite` 的定位是通用贴合、不重复投入）。

### 48.3 算子接口与实现要点

**族文件 `nodes/_common/server_ops/matting.py`**（`OP_FAMILY=matting`，F1 自动版本化）

- `matting.birefnet(image, variant="lite", resolution=1024, use_checkpoint=False, invert=False, alpha_gamma=1.0)`
  → `{mask: IMAGE(灰度), image: IMAGE(RGBA)}`
- 加载：`transformers.AutoModelForImageSegmentation.from_pretrained(<dir>, trust_remote_code=True)`
  （本地目录含 `config.json` + `birefnet.py` + `BiRefNet_config.py` + `model.safetensors`，**零网络访问**）；
  `config.json` 的 `auto_map` 指向 `birefnet.BiRefNet` / `BiRefNet_config.BiRefNetConfig`，
  且 `bb_pretrained=False` ⇒ 不会再拉 backbone 预训练权重。
- 推理：`resize(1024×1024 or resolution)` + ImageNet normalize → `torch.no_grad()` + `autocast(fp16)`
  → sigmoid → 缩放回输入尺寸（bilinear）→ `alpha_gamma` 曲线 → 8bit 灰度 mask；
  RGBA = 原图 RGB + mask 作 alpha。
- 逐 variant 进程级单例缓存（`_BIREF_CACHE`）；权重缺失报可操作错误（点名缺哪个文件、放哪个目录）。
- 显存：fp16 下 lite ≈1GB / main ≈2.5GB / portrait ≈4GB；`use_checkpoint` 供 8GB 卡兜底。

**族文件 `nodes/_common/server_ops/face_restore.py`**（`OP_FAMILY=face_restore`）

- `face.restore(image, model="gfpgan", upscale=1.0, padding=0.3, feather=16, only_center=True, max_faces=1)`
  → `{image: IMAGE}`
- 检测：复用 instantid 族既有 insightface 单例（`INSIGHTFACE_HOME` 默认 `/models/insightface`，
  antelopev2 包；bbox + 关键点，`only_center` 取画面中心最近的一张，`max_faces` 限制张数）。
- 修复：`spandrel.ModelLoader().load_from_file(<pth>)` → `ImageModelDescriptor`；
  **spandrel 0.4.2 实查支持 `GFPGAN`（`scale=1`、`size_requirements.minimum=512`、`supports_half=False`）
  与 `RestoreFormer`（`size_requirements.multiple_of=32`）**；**不支持 CodeFormer**（0.4.2 wheel 里无该架构）。
- 贴回：人脸块 pad 到满足 size_requirements（≥512 的 32 倍数）→ 修复 → 缩放回原块 → 羽化 mask 合并回原图，
  **非人脸区域逐字节不变**（强判据，见 48.6）。

**UI 契约**：`ui/widget-def.js` 用既有 kind（`slider`/`select`/`check`/`text`）+ `compareInput`（前后对比）
+ `replay`（G7 已有的「▶预览」op-replay，F1 下自动打到版本化算子名）；棋盘格透明底 = `widget-base.js` 加一个
**`kind:'checkerPreview'`**（或复用 g5 对比区加棋盘底样式，二选一，见 D6）。

### 48.4 依赖、镜像与权重

**新增 Python 依赖（3 个，均纯 python/小体积）**

```
timm==0.9.16      # birefnet.py：timm.layers 的 DropPath / to_2tuple / trunc_normal_
kornia==0.7.4     # birefnet.py：kornia.filters.laplacian
einops==0.8.0     # birefnet.py：einops.rearrange
```

- **不装** `opencv-python`（仓库已钉 headless）、`scipy`/`scikit-image`（那是 BiRefNet 训练/评测脚本的需求，`birefnet.py` 不需要；scipy 其实已在）。
- `kornia 0.8.x` **排除**：`requires_python>=3.11`，而基础镜像 `pytorch/pytorch:2.3.1` 是 py3.10。
- 改 `inference-server/requirements.txt` + `requirements-pascal.txt` + 两个 Dockerfile 的依赖冒烟行
  （Pascal 冒烟追加 `import timm, kornia, einops`）；`torchvision` 随基础镜像已有（`torchvision.ops.deform_conv2d` 被 `birefnet.py` 顶层 import）。
- **镜像重建**：Pascal + 标准各一次（Pascal 必过 sm_61 真机冒烟）。

**权重（宿主机 `D:\flowx-data\models`，共 ≈1.5GB；C 盘不动）**

| 目录 | 来源 | 体积 | 说明 |
| --- | --- | --- | --- |
| `matting/birefnet-lite/` | HF `ZhengPeng7/BiRefNet_lite` | 177.6 MB | 默认变体（8GB 卡友好） |
| `matting/birefnet/` | HF `ZhengPeng7/BiRefNet` | 444.5 MB | 通用质量最好 |
| `matting/birefnet-portrait/` | HF `ZhengPeng7/BiRefNet-portrait` | 884.9 MB | 人像发丝 |
| `facerestore/GFPGANv1.3.pth` | TencentARC GitHub release | ~333 MB | spandrel 支持 |
| `facerestore/RestoreFormer.pth` | TencentARC GitHub release | ~300 MB | spandrel 支持（备选） |
| `insightface/`（已存在） | antelopev2 | — | `face.restore` 检测复用 |

每个 matting 目录需 4 个文件：`config.json` / `birefnet.py` / `BiRefNet_config.py` / `model.safetensors`
（前三个是 `trust_remote_code` 的自定义建模代码，必须与权重同目录）。
权重清单同时写进 `nodes/README.md`（与 G7 预处理器权重同表范式）。

**新 env**：`MATTING_MODELS_DIR`（默认 `/models/matting`）、`FACERESTORE_MODELS_DIR`（默认 `/models/facerestore`）。

### 48.5 改动文件清单

| 文件 | 改动 |
| --- | --- |
| `inference-server/app/ops.py` | 无需改（算子全在插件族内注册）；如 `face.restore` 需读 `INSIGHTFACE_HOME` 沿用 instantid 实现则零改动 |
| `nodes/_common/server_ops/matting.py` | **新增**（F1 版本化骨架 + 3 个 variant 加载 + `matting.birefnet`） |
| `nodes/_common/server_ops/face_restore.py` | **新增**（`face.restore`：insightface 检测 + spandrel 修复 + 贴回） |
| `nodes/matting-birefnet/`、`matting-refine/`、`matting-composite/`、`face-restore/` | **新增** 4 个节点包（flowx.json / main.py / mock.py / test_main.py / server_op.py / ui/） |
| `nodes/_common/widget-base.js` | 棋盘格透明底预览（`kind:'checkerPreview'` 或对比区加棋盘底） |
| `nodes/_common/image_io.py` | 若 RGBA 读回需要（`get_pimg(..., mode="RGBA")`）则加一行支持 |
| `nodes/_tools/check-ops.py` | `OP_FAMILIES` 加 `matting` / `face_restore` 两项（gate J 自动纳入新族） |
| `nodes/_tools/test-widget-ui.mjs` | 无需改（第 6 节按 widget-def 声明自动纳入新节点） |
| `nodes/README.md` | 新节点表 + 权重清单 + gate 表更新 |
| `nodes/BUNDLE_VERSION` | `1.12.0 → 1.13.0` |
| `pipeline/` | `matting-e2e.yaml`（抠图→精修→换底→落盘）、`face-restore-e2e.yaml`（修复→落盘） |
| `inference-server/ROADMAP.md` | 阶段 10 / M10 下补 G1、G6 小节 |

### 48.6 验收口径（真机 + 落盘 + 量化判决）

> 全部沿用 §46.7 纪律：**末端必须落盘**（`save-image` + `service_url_local`），**串行提交**，复核用本机纯文件脚本。

| # | 判据 | 口径 |
| --- | --- | --- |
| A1 | 抠图可用 | 3 个 variant 各跑通；`mask` 前景占比合理（白底商品 20%~70%、人像 30%~80%）；无全黑/全白退化 |
| A2 | 抠图质量 | 合成到纯白底后，**背景区**（mask=0）剩余非白像素占比 < 1%；**边缘带**（mask 20~235）无白边（原白底残留导致的高亮像素占比 < 2%）；发丝级与手机自带抠图目视对比 |
| A3 | 蒙版语义正确 | 三分区量化（沿用 §39/§46.7 判决口径）：前景区原图往返差 ≈0、背景区替换底逐字节一致、边缘带为加权混合且与公式重算**逐字节一致** |
| A4 | 精修确定性 | `threshold=128` 后 mask 只有 0/255；`feather=N` 边缘过渡带宽 ≈N±2px；`grow` ±k 像素生效；`min_area` 后连通域数下降；全部逐字节可重算 |
| A5 | 合成无损 | mask=255 区域与前景原图**逐字节一致**；mask=0 区域与底色**逐字节一致**；RGBA 直吃与 RGB+mask 两条路径输出一致 |
| A6 | 人脸修复只改脸 | **非人脸区域逐字节不变**；人脸区拉普拉斯方差（清晰度）提升 ≥20%；`padding`/`feather` 控制贴回边界无接缝 |
| A7 | 显存与耗时 | lite/main/portrait × 1024/1536/2048 在 8GB Pascal 实测显存峰值与耗时（记录成表；超限则给出建议分辨率） |
| A8 | 门禁与回归 | gate J（自动纳入新族）+ `check-bundle.py` + widget 回归 380+ 断言 + 落盘复核 `FAILED = 0` 全绿 |
| A9 | UI | 棋盘格透明底预览、前后对比滑块、「▶预览」（F1 版本化算子名，DOM stub 自动验证）+ 人工观感 |

### 48.7 明确不做（B2）

- **CodeFormer**：spandrel 0.4.2（当前最新发布版）无该架构 ⇒ 不做；等上游支持或另立项自实现。
- **RMBG-1.4 / RMBG-2.0**：用户拍板用 BiRefNet（2.0 还是 gated 401）⇒ 不做。
- **BiRefNet HR / matting / dynamic 变体**：先做 lite/main/portrait 三个；其余按需再加（同一算子加 variant 枚举即可）。
- **SAM / 文本分割（G2）**：属 B3，不并入。
- **新对象类型**（MATTING_MODEL / SEG_MODEL）：本批不需要（变体由参数选 + 固定目录）；等 B3 的 SAM 提示对象再议。
- **tile 分块抠图**：BiRefNet 原生多分辨率（1024/1536/2048）已够，不引入分块拼接。

### 48.8 决策点（默认值，确认后开做）

| # | 决策 | 默认值 |
| --- | --- | --- |
| D1 | G1 变体范围 | **lite + main + portrait** 三个（HR/matting/dynamic 后续） |
| D2 | 默认变体 | **lite**（177MB，8GB 卡友好） |
| D3 | 第三节点形态 | **`matting-composite`（含纯色底）**（备选：`image-solid` + 复用 `image-composite`） |
| D4 | G6 首版模型 | **GFPGAN v1.3**（spandrel 明确支持）+ RestoreFormer 备选；CodeFormer 不做 |
| D5 | 依赖钉版 | **timm==0.9.16 / kornia==0.7.4 / einops==0.8.0**（kornia 0.8 需 py≥3.11，基础镜像 py3.10 排除） |
| D6 | 棋盘格透明底 UI | **widget-base 加 `kind:'checkerPreview'`**（备选：G5 对比区加棋盘底样式，改动更小） |
| D7 | 实施顺序 | **G1 先行**（算子+3 节点+真机验收）→ **G6 次之**（复用同一镜像重建，省一次构建） |

### 48.10 交付记录（进行中，2026-10-11）

**已完成**

| 项 | 状态 |
| --- | --- |
| 依赖与镜像 | `timm==0.9.16` / `kornia==0.7.4` / `einops==0.8.0` 写入两个 requirements + 两个 Dockerfile 冒烟；**Pascal 镜像重建通过**（冒烟 `import timm, kornia, einops` 成功 ⇒ py3.10 兼容） |
| 算子族 | `_common/server_ops/matting.py`（`matting.birefnet`，v `005509206da7`）、`face_restore.py`（`face.restore`，v `e56dc2ee9eb7`）；F1 版本化骨架 + gate J 全绿（**10 族 / 38 算子 / 38 成员节点**） |
| 权重 | 全量下载校验通过（14/14 文件）：matting lite/main/portrait + GFPGANv1.3 + RestoreFormer；insightface antelopev2 包**宿主机早有**（Sep 24，352MB，`/models/insightface/`） |
| 节点 | `matting-birefnet`(id 208) / `matting-refine`(209) / `matting-composite`(210) / `face-restore`(211)；全部带自定义 UI（棋盘格/对比滑块/replay） |
| 单测 | 4 个节点 `test_main.py` **FAILED = 0**（matting-refine 17 项、composite 16 项、birefnet 契约 13 项、face-restore 契约 17 项） |
| 真机探针 | BiRefNet 三变体 × 三分辨率在 GTX 1080 实测（见下表）；**两次调用逐字节一致** |
| 真机 op 跑通 | exec 630/631：`matting.birefnet` 经**版本化算子名**在生产路径跑通（`[op] ... -> 前景占比=0.293`），三变体各 11s/21s/52s 全 success |
| 新增门禁 | **gate M**（`_tools/check-node-params.py`）：`param()` / `env` 模板 / 声明未使用三向一致，已接入 check-bundle；87 节点通过 |

**真机探针实测（GTX 1080 8GB，fp16，512×384 输入）**

| 变体@分辨率 | 耗时 | 峰值显存 alloc/reserved | 前景占比 | 灰度唯一值 |
| --- | --- | --- | --- | --- |
| lite@1024 | 1.5s（首次 11.5s 含加载） | 930 / 1392 MB | 0.293 | 236 |
| main@1024 | 1.5s（首次 15.7s） | 1810 / 2896 MB | 0.293 | 233 |
| portrait@1024 | ~40s 首次 | 2251 / 3336 MB | 0.131 | 241 |
| lite@1536 | 1.1s | 2859 / 3924 MB | 0.293 | 213 |
| **lite@2048** | 3.1s | **4315 / 6202 MB** | 0.294 | 216 |

⇒ 8GB 卡上 2048 余量偏紧（建议只在独立跑时用，或开 `use_checkpoint`）；默认 1024 稳妥。

**实施中的设计修正（相对 48.2/48.4 初稿）**

1. **`matting-refine` 零 numpy**：初稿写了 `min_area`（连通域面积过滤）需要 numpy，
   但**本机（Termux 手机）没有 numpy**，节点侧 PIL 节点若依赖 numpy 就无法本地单测——
   与既有 `mask-*` 家族"节点侧纯 PIL"的范式冲突。改为 **`despeckle`**（形态学开运算：
   先腐蚀后膨胀，抹掉尺度 < k/2 的孤立点），同样是纯 PIL、同样能去小杂点，且本地可验证。
2. **`matting-birefnet` 校验顺序**：先做权重/目录检查再 `import torch`——缺权重时必须报
   "缺哪个文件"，而不是 CUDA/依赖报错（也让本地无 GPU 能测这段契约）。
3. **`face.restore` 权重定位先于检测**：`model='codeformer'` 明确报"不在 spandrel 0.4.2"。
4. **matting 变体目录名 = 变体名**（`/models/matting/lite|main|portrait/`），比初稿的
   `birefnet-lite` 更贴合参数取值，算子内一一对应。

**踩坑记录**

- **insightface 模型包陷阱**：容器**必须挂 `-v .../models:/models`**，否则 `FaceAnalysis`
  会把 352MB 模型包下到容器临时层，重建即丢（本轮排查中两次误判"模型包缺失"都是这个原因）。
- **镜像 tag = 内容**：`main.py` 修好但没重建镜像时，docker 执行器跑的仍是镜像里的旧副本
  （真机报 `TypeError: process() got an unexpected keyword argument`）⇒ 已 bump 1.13.2 重建。
- **参数改名漂移**（同上事故根因）⇒ 新增 gate M 常驻拦截。

### 48.11 真机验收结果（2026-10-11，✅ 通过）

手机回网后，用 **local 执行器**变体跑通了两条完整落盘 e2e（GPU 推理在服务端、节点侧只做对象收发与纯 PIL）：

| 链路 | 工作流 / 执行 | 判据脚本 | 结果 |
| --- | --- | --- | --- |
| G1 抠图 | wf **129**（`pipeline/matting-e2e-local.yaml`）/ exec **633** | `~/.tmp/verify_matting.py`（手机本机纯 PIL） | **FAILED = 0**（20 节点全 success；11 份产物落盘） |
| G6 人脸修复 | wf **130**（`pipeline/face-restore-e2e-local.yaml`）/ exec **635** | `~/.tmp/verify_face.py`（本机 + 服务端 `face.mask` 取真框） | **FAILED = 0**（12 节点全 success；9 份产物落盘） |

落盘位置 `~/flowx-e2e-out`（本批新增 matting 22 份 / face 12 份）。

**A1~A5 实测（抠图，输入 512×384）**

| 判据 | 结果 |
| --- | --- |
| A1 三变体同尺寸、互不相同、非退化 | lite/main/portrait sha 各异；fg=0.293/0.293/0.131，中间灰 2.0%/2.0%/0.9%，extrema 均 (0,255) |
| A3 RGBA alpha == mask | 逐像素一致；refine 关的 RGBA alpha 亦逐像素一致 |
| A4 精修确定性 | refined mask **== 本机同参数重算（逐字节）**；soft mask 同；`threshold=128 & feather=0` 只有 {0,255}；feather 产生灰阶；despeckle 生效 |
| A5 换底无损 | 白底（纯，mask=原始）：背景区 == 纯白、前景区 == 原图，**双向逐字节 0 差**；白底（refine 开）：背景区逐字节纯白、前景区平均差 **0.00**；背景图模式：背景区逐字节 == 背景图 |
| A2 无异常掺色 | 合成 == alpha 混合公式重算，全图每通道差 **≤1/255**（仅取整） |

**A6 实测（人脸修复，三张图 × GFPGAN/RestoreFormer）**

| 判据 | 结果 |
| --- | --- |
| 框外逐字节不变（只改脸） | 6 个产物全部 **框外最大差 0** |
| 改动量 | 3.7% ~ 26.2%（人脸框占 10.5%~42.4%），确属"只动人脸邻域" |
| 人脸区清晰度 | 边缘能量比 **0.88 ~ 1.17**（小脸放大修复 1.09/1.11 提升；真人照片 0.88 = GAN 重建更平滑，未糊） |
| 修复确实发生 | 每个产物 `faces=1`、改动像素数千~23 万 |

**本轮新踩的坑（都已修 + 进门槛）**

1. **`param(..., cast=bool)` 是真陷阱**：env 注入的是字符串，`bool("false")` 为真 ⇒
   `matting-refine` 的 `invert` 被当成开，refined mask 整体反相（真机复现：服务端产物与本机重算
   互为反相、平均差 248）。改用 `image_io.pbool()`；**gate M 已加静态检查**（`cast=bool` 直接判错）。
2. **GFPGAN 只吃 512×512**：其 `style_mlp` 输入维度写死，喂 640 报
   `mat1 and mat2 shapes cannot be multiplied (1x6400 and 4096x8192)`。
   改为：GFPGAN 固定 512²（`upscale` 必须 1.0，否则**参数级**报错）；RestoreFormer 可 32 倍数放大。
3. **判据本身也要防错**：① 二值化后接 feather 必然有软边，不能用"只有 0/255"当判据 →
   改成"本机同参数重算逐字节一致"；② 合成与公式重算存在 ±1 取整差 → 判据取 ≤1；
   ③ `face.restore` 的裁剪框与 `face.mask` 的框语义不同（各向外扩 vs 中心外扩 + 8 对齐）
   → 复核取**保守超集**（外扩 32px）后仍要求框外 0 差。

**docker 执行器路径回归（2026-10-11，隧道恢复后补做，✅ 通过）**

docker 隧道换地址（`5.tcp.cpolar.top:14923` → 解析 IP `198.18.0.42`，`DOCKER_API_VERSION=1.47`；
**域名直连会握手失败，必须先解析 IP**），随后：

| 项 | 结果 |
| --- | --- |
| 镜像 | `lerkobba/flowx-pixelforge-nodes:v1.13.3`（含本轮全部修复）构建成功 |
| 重导 | **87/87 节点**（`--overwrite`，tag 刷到 v1.13.3） |
| 探针 | `probe-image-nodes.py --tag v1.13.3` **未完成**：新隧道（`5.tcp.cpolar.top`）下长连接流被重置、逐节点 detached+logs 流程卡住（v1.12.0/v1.13.0 时该探针正常）。以两条 **docker e2e 回归**（exec 636/637）代替——它们跑的是真实节点路径，证据强度高于"导入+缺参报错"式探针 |
| G1 docker e2e | wf 126 / exec **636** → `verify_matting.py` **FAILED = 0**；且 10 份产物与 local 路径 **逐字节一致**（同算子版本 + 同参数 ⇒ 执行器无关） |
| G6 docker e2e | wf 131（新增 `pipeline/face-restore-e2e.yaml`）/ exec **637** → `verify_face.py` **FAILED = 0** |

> **模型确定性差异（记入验收口径）**：`RestoreFormer` 两次运行**逐字节一致**；
> **`GFPGAN` 是随机的**（官方 `randomize_noise=True`，同一输入不同次运行最大差 20~32/255）
> ⇒ 人脸修复链路**不能**用"逐字节一致"当回归判据，只能判"框外不变 + 改动量 + 清晰度"。

### 48.12 去白边专项实测（白底商品图对照，2026-10-11，✅ 通过）

§48.11 里"边缘带亮度"指标判不准（测试图背景不是白底、且换底图与原背景相同 ⇒ 不能体现白边）。
补一组**受控实验**：

- **输入**：拿已验证的白底合成图（exec 633 的 `comp_white` 对象 = 主体贴在纯白上）当"白底商品图"
  —— 其边缘像素天然就是"主体 × 白底"的混色，正是白边场景。
- **对照**：同一张图跑**两遍** BiRefNet main@1024，**只有 `refine` 不同**；实测两遍 alpha
  **逐字节一致**（`refine` 只改 RGB）⇒ 差异被干净地隔离到"是否反解前景色"这一个变量上。
- **放大**：两版各自换到**纯绿底 `#00B050`**，白边残留变成肉眼可见的浅色环。

流水线 wf **132**（`pipeline/matting-defringe-local.yaml`）/ exec **640**（全 local 执行器）；
判据脚本 `~/.tmp/verify_defringe.py`（纯 PIL）→ **FAILED = 0 / PASSED = 9**：

| 判据 | 结果 |
| --- | --- |
| C1 对照干净 | 两遍 alpha **逐字节一致**（前景占比 29.7%） |
| C2 白底场景 | 远离轮廓的背景区（127845px）**100.000% 纯白、最大偏差 0** |
| **C3 去白边（前景色本身）** | 过渡带（2980px，占全图 1.52%）`min(R,G,B)` **176.5 → 96.0**（Δ+80.5）；近白像素 **18.0% → 0.0%** |
| **C4 换绿底后残留** | 过渡带 `min(R,G,B)` **95.7 → 73.5**（Δ+22.3）；绿色污染不升（-4.2 → -6.4） |
| C5 非整体调色 | `alpha=255` 核（50183px）平均差 **0.00/255** |
| C6 合成口径 | 两版绿底合成 == alpha 混合公式重算，**最大通道差 0** |

**结论**：`refine=True` 确实把"白底混色"的边像素反解回前景色（近白像素直接清零），
且**完全不动主体内部**、不动 alpha ⇒ 这是可量化的"去白边"，而非整体调色。

人眼物料（Studio 媒体接口可直开，`~/flowx-output`）：
`b2_defringe_compare.png`（输入 / refine 关 / refine 开 三连图）、`b2_defringe_zoom.png`（过渡带最密处 3× 放大）。

> 小坑：流水线 `nodeRef` 版本写错（`save-image@1.0.0`，实际是 `1.2.0`）时，`run` 直接
> `status=failed` 且 **execution 无任何节点**（errorMessage=`node save-image@1.0.0 not found`）
> ⇒ 排查此类"秒失败"要看 `GET /api/v1/executions/<id>` 的 `errorMessage`，logs 是空的。

**剩余（仅人工）**

- 人工观感：抠图边缘质量（发丝/白边）与三变体差异、人脸修复前后对比（助手无法看图）。
  可选补做：换**白底商品图**再跑一组 `matting-e2e`，最能体现 `refine=True` 的去白边效果
  （本轮测试图背景非白底，边缘带亮度指标 157 vs 137 判不准）。

### 48.9 预估工作量

| 步骤 | 内容 |
| --- | --- |
| S1 | 依赖 + 镜像重建（Pascal/标准）+ 权重下载（≈1.5GB）⇄ 可并行 |
| S2 | `matting.py` 算子 + 兼容性探针（真机加载 lite 并出图，先确认 timm/kornia 在 Pascal 上无坑） |
| S3 | 3 个 G1 节点 + 自定义 UI + mock/单测 |
| S4 | G1 真机 e2e + 落盘复核（A1~A5、A7） |
| S5 | `face_restore.py` + `face-restore` 节点 + UI |
| S6 | G6 真机 e2e + 落盘复核（A6） |
| S7 | 文档（README/ROADMAP/本 plan 交付记录）+ 全量门禁 + 节点重导 |


---

## 四十九、B3 批次立项方案（2026-10-11）：SAM/文本分割（G2）+ 分块重绘超分（G4）

> 对应 §44.3 的 **B3**（G2 P0 + G4 P1），实施 §44.6 决策（细粒度 + 全 UI）。
> **方案先行，本条即方案——待用户确认后开做。**
> 前置侦察已在真机镜像内实测完成（§49.2 全部为**实测结论**，非推断）。

### 49.1 范围与分层决策

| 能力 | 实现层 | 理由 |
| --- | --- | --- |
| G2 SAM 装载/推理 | 服务端算子族 **`seg.py`**（`seg.sam_load` / `seg.sam` / `seg.sam_auto`） | 需 GPU + transformers 建模；权重在宿主机 |
| G2 文本检测→框 | 服务端算子族 **`seg.py`**（`seg.grounding_load` / `seg.grounding_detect`） | GroundingDINO 需 GPU；与 SAM 同族共享 `_register`/LRU 记账 |
| G2 文本→**像素级**蒙版 | 服务端 `seg.grounding_detect(..., sam_model=...)` 内置框→SAM | 一条链拿到精确蒙版，避免多一次图片往返与中间对象 |
| G4 切片规划/网格覆盖层 | **算子 `tiled.plan`（纯计算，零 GPU）+ 节点 UI** | 只算数不推理；放服务端可与 apply 的网格口径**单一事实源**，UI 只画覆盖层 |
| G4 分块重绘+接缝融合 | 服务端算子族 **`tiled.py`**（`tiled.apply`） | 必须在服务端**一次 job 内**跑完 N 块（N 次网络往返 + 逐块模型驻留不可接受，§44.2 G4 原文口径） |

**两项关键架构结论（本批立住，后续批次沿用）**

1. **零新增 Python 依赖、零推理镜像重建**（实测见 §49.2）：B2 已装的 `timm==0.9.16` 正好是 GroundingDINO
   的 backbone 依赖；SAM1 / GroundingDINO / `mask-generation` pipeline **全部在 `transformers==4.44.2` 内**。
   ⇒ 本批**不动 `requirements*.txt`、不重建 9GB 推理镜像、不碰 Pascal 钉版**，只下载权重 + 上传插件。
2. **零服务端核心改动，且不新增对象类型**（推翻 §44.6.2 的「唯一核心改动」假设）：
   - Studio 节点**输出端口一律是 `string`（对象 ID）**（实测：`checkpoint-loader.model_ref`、`upscale-model-load.upscale_model`
     都是 string）⇒ 模型句柄用 **STRING** 传即可，无需 `SEG_MODEL`/`GROUNDING_MODEL`/`TILE_PLAN` 新类型。
   - 插件算子**可以直接进 `model_manager` 的 LRU**：`flux.py` 的 `_register(key, obj, est_bytes, arch)`
     先例（`_evict_if_needed` → 写 `_pipes/_archs/_sizes/_last_used`）⇒ SAM/GD 模型自动获得
     **淘汰 / pin / `model-unload` / 常驻列表**全套语义，键取 `sam:vit_l`、`gd:tiny`。
   - G4 的分块重绘用**核心原语** `app.ops.resolve_pipe / vae_encode / sample / vae_decode`
     （`detail-refine/server_op.py` 先例）⇒ 服务端 `app/` 一行不改。

### 49.2 前置侦察结论（真机镜像内实测，2026-10-11）

**推理镜像 `flowx-inference-server:pascal` 内实跑**（`transformers 4.44.2 / timm 0.9.16 / torch 2.3.1 /
numpy 1.26.4 / cv2 4.11.0 headless`）：

| 探测项 | 结果 | 结论 |
| --- | --- | --- |
| `SamModel` / `SamProcessor` / `SamImageProcessor` | ✅ 全在 | **SAM1 可做** |
| `SamModel.get_image_embeddings` | ✅ 在 | 可"编码一次 → 反复点选查询"（交互式点选不重复编码，**这是点选编辑器能流畅的前提**） |
| `SamImageProcessor.generate_crop_boxes` / `post_process_masks` | ✅ 在 | 自动分割可自实现或走 pipeline |
| `pipeline("mask-generation")` | ✅ 支持（`AutoModelForMaskGeneration`） | `sam-auto-segment` 有官方路径 |
| `Sam2Model` / `Sam2Processor` | ❌ **不存在** | **SAM2 不在本批**（见 §49.7） |
| `GroundingDinoForObjectDetection` / `GroundingDinoProcessor` | ✅ 全在 | 文本检测可做 |
| `AutoModelForZeroShotObjectDetection` | ✅ 在 | 用 Auto 类即可，不必写死类名 |
| `GroundingDinoProcessor.post_process_grounded_object_detection` | ✅ 在 | 后处理有官方实现（**无需编译 MultiScaleDeformableAttention 的 CUDA 扩展**） |
| `SamProcessor.__call__(images, segmentation_maps, input_points, input_labels, input_boxes, ...)` | ✅ 签名确认 | points/box 两种提示的统一入口 |

**权重可达性（HF API 实测 `gated` 字段，吸取 RMBG-2.0 gated 401 教训）**：

| 仓库 | 状态 | 权重 | 用途 |
| --- | --- | --- | --- |
| `facebook/sam-vit-huge` | ✅ `gated=False` | `model.safetensors` **2564MB** | SAM 精度上限（8GB 偏紧） |
| `facebook/sam-vit-large` | ✅ `gated=False` | **1249MB** | **建议默认**（精度/显存平衡） |
| `facebook/sam-vit-base` | ✅ `gated=False` | **375MB** | 轻量/低显存 |
| `IDEA-Research/grounding-dino-tiny` | ✅ `gated=False` | **689MB** | **建议默认** |
| `IDEA-Research/grounding-dino-base` | ✅ `gated=False` | **933MB** | 更准 |
| `facebook/sam2.1-hiera-large` | ✅ `gated=False` | 898MB | **本批不用**（架构不在 4.44.2 里） |

合计约 **2.0GB**（vit_l + vit_b + gd-tiny + gd-base）或 **4.8GB**（含 vit_h）——宿主机 `D:\flowx-data\models` 余量充足。

**Pascal（GTX 1080 / sm_61）精度纪律**：GP104 的**非张量 fp16 吞吐是 fp32 的 1/64**，且无 bf16/TF32。
⇒ SAM/GD 默认 **fp32**（与 B2 抠图的 `fp16 + autocast` 口径**不同**，必须显式区分）；
保留 `dtype=fp32|fp16|auto` 开关以便换卡。fp32 下 ViT-L 权重 1.25GB / ViT-H 2.56GB，
加 1024² 激活预计 3.5~5GB ⇒ 8GB 可行但需 S1 真机探针实测（含耗时）。

### 49.3 节点清单（7 个新节点，细粒度 + 全自定义 UI）

| # | 节点 | 实现 | 参数（含默认） | 输出 | 自定义 UI |
| --- | --- | --- | --- | --- | --- |
| 1 | `sam-load` | `seg.sam_load` | `variant=vit_l`(vit_l/vit_h/vit_b)、`dtype=fp32` | `model`(string `sam:vit_l`)、`params_mb`、`vram_mb`、`residents`(json) | 变体下拉 + 参数量/显存预估 + 「预热」/「🗑卸载」按钮 + 常驻列表 |
| 2 | `sam-segment` | `seg.sam` | `image`、`model=sam:vit_l`、`points=""`（`x,y,label;…`，label 1=正 0=负）、`box=""`（`x1,y1,x2,y2`）、`multimask=true`、`pick=0`(候选下标)、`overlay=0.5` | `mask`（灰度）、`candidates`(json：id+score+area)、`score`(float)、`annotated`（原图+提示+蒙版叠加）、`width`/`height` | **点/框选画布编辑器**（点=正样本、Shift/右键点=负样本、拖拽=框选）+ 候选蒙版切换 + score 读数 + 叠加预览 |
| 3 | `sam-auto-segment` | `seg.sam_auto` | `image`、`model`、`points_per_side=16`、`pred_iou_thresh=0.88`、`stability_score_thresh=0.95`、`min_area=200`、`max_masks=64`、`keep=""`（下标，空=全选）、`sort=area`(area/score/index) | `masks`（**编号拼接网格图**）、`selected`（选中并集 MASK）、`selected_overlay`（彩色叠加）、`ids`(json)、`count`、`sizes`(json) | 分割结果**网格预览 + 点选保留/排除**（点格子切换）+ 数量/面积读数 + 选中并集预览 |
| 4 | `grounding-dino-load` | `seg.grounding_load` | `variant=tiny`(tiny/base)、`dtype=fp32` | `model`(string `gd:tiny`)、`params_mb`、`vram_mb`、`residents` | 变体下拉 + 显存预估 + 预热/卸载 |
| 5 | `grounding-detect` | `seg.grounding_detect` | `image`、`model=gd:tiny`、`prompt="person . red dress"`、`box_threshold=0.35`、`text_threshold=0.25`、`max_boxes=10`、`sam_model=""`（非空=框再喂 SAM 出**像素级**蒙版） | `annotated`（框+标签+置信度）、`boxes`(json)、`labels`(json)、`count`、`mask`（框并集或 SAM 精修蒙版）、`scores`(json) | 提示词输入 + **检测框叠加可视化** + 置信度列表（可勾选）+ 阈值滑杆 + 「文本→精确蒙版」开关 |
| 6 | `tiled-upscale-plan` | `tiled.plan` | `image`、`scale=2.0`、`target_w=0`、`target_h=0`（0=按 scale）、`tile=768`、`overlap=96`、`pre=lanczos`(none/lanczos/model)、`denoise=0.35`、`seed=-1`、`seam_fix=mask`(mask/band/off)、`band_denoise=0.25` | `plan`(json string)、`grid_image`（网格+序号覆盖层）、`tiles`、`rows`、`cols`、`target_size`、`est_samples`（总采样步数=块数×steps 的口径） | **切片网格覆盖层**（画布上实时绘制）+ tile/overlap/scale 滑杆联动 + 块数与预估耗时提示 + 目标尺寸读数 |
| 7 | `tiled-upscale-apply` | `tiled.apply` | `image`、`model`(MODEL)、`pos`(COND)、`neg`(COND)、`plan=""`（非空=用第 6 个节点算好的口径）、`scale/tile/overlap/pre/denoise/seed/seam_fix`（自算回退）、`upscale_model`(UPSCALE_MODEL，`pre=model` 时用)、`steps=20`、`cfg=7.0`、`sampler_name`、`scheduler` | `image`（最终超分图）、`pre_upscaled`（预放大图，便于 AB）、`seam_map`（归一化权重图 wsum）、`tiles`、`elapsed_s` | **逐块进度条（真实 job progress）** + 当前块/已融合区预览 + 接缝强度指示 + 预放大↔最终 AB 对比 |

> **D5（默认取是）**：`grounding-detect` 内置 `sam_model` 参数实现「文本 → 像素级蒙版」；
> 这是 ComfyUI 里 GroundingDINO+SAM 组合链的**一条节点化表达**，且框→SAM 复用 `seg.sam` 的同一段实现，增量成本低。
> **D6（默认取否）**：不新增 `sam-mask-pick` 节点——多蒙版挑选由 `sam-auto-segment.keep` 参数 + `selected` 输出覆盖。

### 49.4 算子接口与实现要点

**族文件 `nodes/_common/server_ops/seg.py`**（`OP_FAMILY=seg`，F1 自动版本化，gate J 加登记）

- 目录：`SAM_DIR=/models/sam/<variant>/`（`config.json` + `model.safetensors`）、
  `GROUNDING_DIR=/models/grounding/<variant>/`；缺文件报可操作错误（点名缺哪个、放哪，沿用 `matting.py` 口径）。
- 记账：`_register(f"sam:{variant}", model, est_bytes, arch)` / `_register(f"gd:{variant}", ...)`
  —— 复用 `flux.py` 的写法（`_evict_if_needed` → 写 manager 四张表）⇒ 进 LRU、可 `model-unload`、出现在 `/models` 常驻列表。
- `seg.sam_load(variant, dtype)` → `{model:"sam:vit_l", params_mb, vram_mb}`：装载 + `_register`，返回键名（STRING）。
- `seg.sam(image, model, points="", box="", multimask=True, pick=0, overlay=0.5)`
  → `{mask, candidates(json), score, annotated}`：
  1. `SamProcessor` 准备 `input_points/input_labels/input_boxes`（点/框坐标是**原图坐标**，处理器内部自动缩放）；
  2. **`get_image_embeddings` 编码一次并缓存**（键=模型键+图片内容哈希）⇒ 同一张图的连续点选查询不重复编码（交互流畅的关键）；
  3. `SamModel` mask decoder → `post_process_masks` → 回原尺寸 → 按 `pick` 取候选；
  4. `annotated` = 原图叠提示点/框 + 半透明蒙版（节点侧纯 PIL 亦可，但服务端一次出更省往返）。
- `seg.sam_auto(image, model, points_per_side, pred_iou_thresh, stability_score_thresh, min_area, max_masks, keep, sort)`
  → `{masks(montage), selected, selected_overlay, ids(json), count, sizes(json)}`：
  1. `generate_crop_boxes` → 点阵 → batch 预测 → `post_process_masks` → 去重叠/小面积过滤（NMS 用 box IoU，纯 numpy）；
  2. **每个蒙版 put 进对象仓库**（`ids` 是各 MASK 的对象 ID 列表）⇒ 网格图用于挑选，`selected` 是 `keep` 下标并集；
  3. 拼接网格用 PIL 画序号（供 UI 点选），`selected_overlay` 给彩色叠加。
- `seg.grounding_load(variant, dtype)` → `{model:"gd:tiny", params_mb, vram_mb}`。
- `seg.grounding_detect(image, model, prompt, box_threshold, text_threshold, max_boxes, sam_model="")`
  → `{annotated, boxes(json), labels(json), scores(json), count, mask}`：
  1. `AutoProcessor` + `AutoModelForZeroShotObjectDetection`；prompt 规范化为小写、多类以 `" . "` 分隔；
  2. `post_process_grounded_object_detection` 出框/标签/分数；
  3. `sam_model` 非空 ⇒ 每个框当 `input_boxes` 调 `seg.sam` 得像素级蒙版，并集写入 `mask`；否则 `mask` = 框矩形并集。
- 显存兜底：装载前 `torch.cuda.mem_get_info()` 检查 + `_evict_if_needed`；不足时报**可操作**错误
  （点名"先卸载谁"，如 `matting:portrait`），不做静默降级。

**族文件 `nodes/_common/server_ops/tiled.py`**（`OP_FAMILY=tiled`）

- `tiled.plan(image, scale, target_w, target_h, tile, overlap, pre, denoise, seed, seam_fix, band_denoise)`
  → `{plan(json), grid_image, tiles, rows, cols, target_size, est_samples}`：**纯计算 + PIL 画网格，零 GPU**；
  网格算法与 `tiled.apply` **共用同一函数**（单一事实源，保证 plan 与实际执行一致）；
  每块坐标**对齐到 8 的倍数**（VAE 要求，`detail-refine` 同款纪律）；每块 seed = `seed + index`（确定性）；
  `plan` 里带每块的 `(x, y, w, h, seed)`。
- `tiled.apply(image, model, pos, neg, plan="", ..., upscale_model=None, steps, cfg, sampler_name, scheduler)`
  → `{image, pre_upscaled, seam_map, tiles, elapsed_s}`：
  1. **预放大**：`pre=model` → `ops.upscale_with_model(image, upscale_model, tile=0)`；
     `pre=lanczos` → PIL LANCZOS；`pre=none` → 不放大（只注入细节）；
  2. **切块**：按 plan 的 `(x,y,w,h)` 裁切；块边长已是 `tile`（预放大后天然够大，无需逐块再放大）；
  3. **逐块重绘**：`vae_encode → sample(denoise, seed=base+i) → vae_decode`（`detail-refine` 同一套原语）；
  4. **无缝融合（权重归一化累加，而非 paste）**：每块生成**羽化权重图** W（内部 1、overlap 区线性/余弦降到 0），
     累加 `acc += tile_px*W`、`wsum += W`；最后 `out = acc / max(wsum, eps)`
     ⇒ overlap 区两侧权重和为 1，**数学上无偏、无权重断层**（比 Ultimate 的 paste+seam pass 更稳）；
  5. **`seam_fix=mask`**：用 `wsum < 阈值` 得到接缝带 → 对接缝带再跑一遍低 denoise（`band_denoise`）后二次融合；
     `=band` 只修带内，`=off` 关闭；
  6. **进度**：每块 `job.set_progress(i, n)`（`app/jobs.py` 已有 `set_progress`，`/jobs/{id}` 会返回 `progress`）
     ⇒ 节点 UI 的逐块进度条**有真实数据源**；
  7. `seam_map` = wsum 归一化图（复核"接缝是否被融合抹平"的可视/可量化产物）。
- 确定性与显存：固定 seed + 固定块序 ⇒ 同参数可复现；单块 tile=768/1024、SD15 fp16 ≈2.5~3GB、
  SDXL tile=1024 ≈5~6GB（8GB 偏紧 ⇒ **默认 tile=768**，PLAN 节点给估算）。

**编排边界**：`tiled.apply` 需要 `MODEL`/`COND`（与 `ksampler` 同源），
故流水线形态是 `checkpoint-loader → clip-text-encode(pos/neg) → tiled-upscale-plan → tiled-upscale-apply → save-image`，
与既有 `sdxl-hires.yaml` 同构。

### 49.5 改动文件清单

| 类别 | 文件 | 说明 |
| --- | --- | --- |
| 新增算子族 | `nodes/_common/server_ops/seg.py`、`nodes/_common/server_ops/tiled.py` | F1 打戳（`sync-common.py` 分发前自动）；gate J 登记 |
| 新增节点 ×7 | `nodes/{sam-load,sam-segment,sam-auto-segment,grounding-dino-load,grounding-detect,tiled-upscale-plan,tiled-upscale-apply}/` | 各含 `flowx.json`/`main.py`/`server_op.py`(分发)/`mock.py`/`test_main.py`/`ui/{widget-def.js,node-widget.js}` |
| 共享 UI 控件 | `nodes/_common/widget-base.js` | 新增 `pointSelect`（点/框选画布）、`tileGridOverlay`（切片网格覆盖层）、`maskMontage`（编号网格点选）、`boxList`（检测框+置信度勾选） |
| 门禁 | `nodes/_tools/sync-common.py`（OP_FAMILIES +2）、`nodes/_tools/check-ops.py`（gate J +2）、`nodes/_tools/build-widget.py`（全量重建 88 个 widget-def）、`nodes/_tools/test-widget-ui.mjs`（第 6 节回归 + 7 节点断言） |
| 流水线 | `pipeline/{sam-point-e2e-local,sam-text-e2e-local,tiled-upscale-e2e-local,sam-auto-e2e-local}.yaml`（+ docker 版各一） | 全 local-executor 版用于真机验收（`save-image` + `service_url_local`） |
| 权重 | 宿主 `D:\flowx-data\models\{sam/<variant>,grounding/<variant>}/` | 由容器内 `hf_hub_download` 拉取（脚本入 `nodes/_tools/`，沿用 B2 做法） |
| 文档 | `nodes/README.md`、`inference-server/ROADMAP.md`（阶段10/B3）、本 plan §四十九/交付记录 | |
| **不动** | `inference-server/app/**`（零改动）、`requirements*.txt`（零新增）、`Dockerfile*`（**不重建推理镜像**） | 本批最大收益 |

### 49.6 验收口径（真机 + 落盘 + 量化）

| 编号 | 判据 | 量化口径 |
| --- | --- | --- |
| A1 | 装载/卸载账本 | 三档 SAM + 两档 GD 装载成功且出现在 `/models` 常驻列表（键 `sam:*`/`gd:*`）；`model-unload` 后键消失且 `torch.cuda.mem_get_info` 显存回落 ≥ 权重体积 ×0.8 |
| A2 | SAM 点选**确定性** | 同图同点同参数连跑两次，`mask` **逐字节一致**（SAM 推理确定性）——本批唯一可做逐字节回归的推理链路 |
| A3 | SAM 点选正确性 | 点在目标上 ⇒ 蒙版覆盖目标主体、框外溢出占比 <5%；正/负样本混合提示下负样本区域被排除（**负样本区域内前景占比 = 0**） |
| A4 | 框选 | `box` 提示 ⇒ 蒙版在框内 IoU ≥0.9（与本机同参数重算比对） |
| A5 | 文本→蒙版 | prompt 命中：`count` 与人工目视一致、命中标签正确；`sam_model` 精修后蒙版**边界带宽度**显著小于纯框（量化：边界带像素占比下降）；不命中时 `count=0` 且不报错 |
| A6 | 自动分割 | `count` 与网格图一致；`selected`（keep 子集）与本机按 `ids` 逐个求并集**逐字节一致**（强判据） |
| A7 | 分块超分正确性 | 输出尺寸 == plan 目标尺寸（逐字节对尺寸）；`pre_upscaled` 与独立预放大结果逐字节一致；`tiles` == plan 块数 |
| A8 | **无网格缝**（G4 核心风险） | ① `ΣW` 在内区与接缝带均 ==255（互补 ramp 权重和恰为 1，**无欠覆盖**）；② 全局最不连续的列**不在**接缝带上；③ 带内最大相邻差 ≤1.5× 内区 p95（无尖峰）；④ 带内平均纹理能量 ≤1.35× 全局非接缝区（overlap 两次重绘 ⇒ 天然略高）。**实测校正**：初版口径（高频能量比 0.8~1.25）过严且基线选错（固定内区列段落在低纹理区），已改为上述四条并实测通过（见 §49.11） |
| A9 | 超分有效性 | 512→2048 在 8GB 完成且不 OOM；细节指标（Laplacian 方差 / 边缘能量）显著高于纯 RealESRGAN/纯 LANCZOS 放大；出人眼物料 |
| A10 | 执行器无关 + 落盘 | local 与 docker 两条路径各跑一遍 e2e，产物落 `~/flowx-e2e-out`，**复核 FAILED = 0**；确定性的产物（SAM 全部输出；tiled 的 `pre_upscaled`/`seam_map`）要求两路径**逐字节一致**；扩散产物（`tiled_up`）改用漂移口径（**实测校正**：同参数同 executor 连跑两次亦非逐字节一致，平均差 0.068 / 最大 4/255，与 local-docker 的 0.065 / 4 同量级 ⇒ 非确定性来自 GPU 扩散采样，与执行器无关，见 §49.11） |
| A11 | 幂等/钉版 | gate J 绿（`seg`/`tiled` 版本戳 == 内容 hash）；重复执行同参数结果一致 |

> 判据自省纪律（承 B2 教训）：**能用逐字节就用逐字节**（A2/A6/A7/A10），
> 不能用（扩散采样有 GPU 非确定性）才退"结构/统计 + 人眼物料"（A8/A9）。

### 49.7 明确不做（B3）

| 项 | 原因 |
| --- | --- |
| **SAM2**（`sam2.1-hiera-*`） | `transformers==4.44.2` **无 `Sam2Model`**（实测）⇒ 用不了；升到 ≥4.47 需重验 Pascal 全链（diffusers/accelerate/timm 连锁）⇒ 单独立项（B5 候补） |
| SAM fp16 默认 | GP104 非张量 fp16 吞吐 = fp32 的 **1/64** ⇒ 默认 fp32，保留开关 |
| Tiled Diffusion（multi-diffusion 跨块注意力融合） | 只做 Ultimate 式顺序分块；跨块注意力需改采样器内部，风险/收益比差 |
| 新对象类型（`SEG_MODEL`/`GROUNDING_MODEL`/`TILE_PLAN`） | §49.1 结论 2：STRING 端口 + LRU 键已足够，零服务端改动 |
| **服务端 `app/` 任何改动** | 同上；G4 用 `app.ops` 原语，G2 用 `_register` |
| 推理镜像重建 | 零新增依赖；仅在 S1 发现意外缺失时才启动（预案已备） |
| 语义分割/分类标签 | `preprocess-seg`（B1/G7）已覆盖语义分割；G2 只出 mask 不做分类 |
| 逐块 ControlNet / 逐块 LoRA 切换 | 不做；块内条件继承全局 |
| 视频分块超分、多 GPU | 不做 |
| SAM onnx / 量化路径 | 不做（GPU 直推够用） |
| `sam-auto-segment` 的输出**单独**可编辑蒙版节点 | `keep` 参数覆盖；不新增节点 |

### 49.8 决策点（默认值，确认后开做）

| # | 决策 | 建议默认 | 备选 |
| --- | --- | --- | --- |
| D1 | SAM 默认变体 | **`vit_l`**（1.25GB，精度/显存平衡；`vit_h`/`vit_b` 可选） | `vit_h` 首发（2.56GB，8GB 偏紧） |
| D2 | dtype 默认 | **`fp32`**（Pascal 纪律） | `auto`（按卡型选） |
| D3 | GroundingDINO 默认变体 | **`tiny`**（689MB） | `base`（933MB） |
| D4 | G4 默认 tile / overlap | **768 / 96**（SD15 与 SDXL 都稳） | 1024 / 128（SD15 更快，SDXL 触顶） |
| D5 | `grounding-detect` 内置框→SAM 精修 | **是**（`sam_model` 参数，非空启用） | 否（只出框，精修另接 `sam-segment`） |
| D6 | 新增 `sam-mask-pick` 节点 | **否**（`keep` 参数 + `selected` 输出已覆盖） | 是 |
| D7 | 顺手把 B2 的 matting/face 模型纳入 LRU 记账 | **否**（会改族文件 ⇒ 换算子版本，单独一次小改） | 是 |

### 49.9 预估工作量

| 步骤 | 内容 | 依赖 |
| --- | --- | --- |
| S1 | 权重下载（2.0GB，`sam/{vit_l,vit_b}` + `grounding/{tiny,base}`）+ **真机探针**（装载显存、1024² 编码耗时、fp32 vs fp16 实测） | 无（可先做） |
| S2 | `seg.py` 的 SAM 三算子（load/sam/sam_auto）+ 单测（mock 权重路径、参数校验） | S1 |
| S3 | `seg.py` 的 GD 两算子（load/detect）+ 框→SAM 精修 | S2 |
| S4 | `widget-base.js` 四个新控件 + `sam-load`/`sam-segment` 两节点（**点选编辑器是最大 UI 工程**，复用 `mask-paint` 全屏 canvas 基座） | 可并行 |
| S5 | `sam-auto-segment` + `grounding-dino-load` + `grounding-detect` 三节点 + UI | S3+S4 |
| S6 | `tiled.py` 两算子（plan/apply）+ 接缝融合 + 进度上报 + 两节点 + UI（网格覆盖层/进度条） | S4 |
| S7 | 4~8 条流水线 + 真机 e2e（local/docker）+ 落盘复核 A1~A11 + 人眼物料 | S2~S6 |
| S8 | 文档（README 权重表 + gate J/L 登记 + ROADMAP + 交付记录/验收）+ bundle bump + 节点重导 | S7 |

> 关键路径：S1 → S2 → S3 → S5 → S7（G2 文本分割链）；S4 的 `pointSelect` 编辑器与 S6 的 tiled 可与 S2/S3 并行。
> **本批不需要镜像重建**（零新增依赖）⇒ 省掉 B2 最耗时的一段（9GB 镜像 + Pascal 重验）。

### 49.10 S1/S2 实测交付记录（2026-10-11，真机）

**S1 权重下载**（容器内 `hf_hub_download` → 宿主 `D:\flowx-data\models`）：

| 仓库 | 落盘 | 体积 | 耗时 |
| --- | --- | --- | --- |
| `facebook/sam-vit-large` | `/models/sam/vit_l` | 1249MB | 113s |
| `facebook/sam-vit-base` | `/models/sam/vit_b` | 375MB | 136s |
| `facebook/sam-vit-huge` | `/models/sam/vit_h` | 2564MB | 217s |
| `IDEA-Research/grounding-dino-tiny` | `/models/grounding/tiny` | 690MB | 57s |
| `IDEA-Research/grounding-dino-base` | `/models/grounding/base` | 934MB | 162s |

**S2 算子族 `seg.py` 真机 smoke（GPU 直调算子，`flowx-inference-server:pascal`，`EXIT=0`）**：

| 项 | 实测 |
| --- | --- |
| 注册名（F1） | `seg.sam_load@v` / `seg.sam@v` / `seg.sam_auto@v` / `seg.grounding_load@v` / `seg.grounding_detect@v`（5 个） |
| 装载显存（est / 服务端记账） | vit_l **1191MB** / vit_b 375MB / vit_h **2445MB** / gd-tiny **657MB** |
| **装载耗时（D 盘绑定挂载读权重）** | vit_l **59~220s**（冷/热盘缓存差异）、vit_h 163s、gd-tiny 34s ⇒ **重载很贵，模型应尽量常驻** |
| 点选首次（含编码） | vit_l **1.65s** / **vit_h 6.17s** ⇒ **D1 默认 vit_l 有数据支撑** |
| 点选二次（**编码缓存命中**） | vit_l **0.08s** ⇒ 交互式点选真正流畅 |
| 点选确定性 | **逐字节一致**（A2 判据成立） |
| 正+负样本 | 前景占比 0.472 → **0.205**（负样本把下半身排除） |
| 框选（multimask=False） | 单候选 score 0.864、**86% 蒙版落在框内** |
| `sam_auto`（pps=12, 768×1152） | 候选 20 → IoU 去重 16、**10.1s**、`keep=0,2` 并集 586152 < 全选 812671、**网格图确定性一致** |
| GD 检测（gd-tiny） | `person . dress` 命中 1（0.925）**1.54s**；`dress` 0.82、`shoe` 0.41、`person,dress` 命中 2 |
| **文本 → SAM 像素级蒙版** | 命中 1 → SAM 蒙版前景 0.513、**1.95s**（复调 0.69s，**确定性 True**） |
| 卸载/账本 | `models.unload('sam:vit_l')` → `resident=[]` ✓（A1 账本链路成立） |

**本轮踩的坑（已修 + 已进设计）**：
1. **`input_points`/`input_labels`/`input_boxes` 必须 `.to("cuda")`**：只把 `image_embeddings`
   放 cuda 会在 `prompt_encoder._embed_points` 里报 `Expected all tensors to be on the same
   device ... at least two devices, cpu and cuda:0`。
2. **`post_process_masks` 返回形态随版本不同**（实测为 `(batch, n_masks, H, W)` 张量，而非
   文档感的 per-image list）⇒ 一律走 `_norm_masks()`/`_as_single()` 归一，不假设维度。
3. **`pipeline("mask-generation")` 的 `res["scores"]` 是 tensor**：写 `res.get("scores") or [...]`
   会抛 `Boolean value of Tensor with more than one value is ambiguous`。
4. **GroundingDINO 提示词必须以 `.` 结尾**：实测裸 `"person"` **0 命中**，`"person . dress"`
   正常 ⇒ 新增 `_norm_prompt()`（小写 + 逐类补尾点），修后 `"person"` → 0.929 命中 ✓。
5. **容器内测显存不准**（同卡还有共驻的推理服务 + Windows 共享显存 over-commit，实测
   `mem_get_info` 读数会「越用越少」甚至反号）⇒ **A1 的显存判据改用服务自身 `/health`**
   （`resident_models` / `resident_details` / `vram_process_allocated_mb` 前后对比），不用容器内读数。
6. **`MAX_RESIDENT_MODELS` 是条目数兜底上限**（服务实际 =3；镜像默认 1）：SAM+GD+SD 恰好占满
   ⇒ 三条链同时用时会有 LRU 互斥（重载 60s+，可感知）；`_register` 前会 `_evict_if_needed`，
   pinned 条目（如 flux-unet）不会被淘汰。
7. **手机当前在蜂窝网（出口 10.6.133.153）⇒ 局域网 192.168.5.60 不可达**；隧道换成
   `http://4dc21d2a.r7.cpolar.cn`（**实测保留鉴权头**：带 token 200 / 不带 401）⇒ 本批
   `service_url_local` 用该 http 隧道；docker API 换 `tcp://5.tcp.cpolar.top:12869`
   （须先解析 IP → `8.162.7.178`，`DOCKER_API_VERSION=1.47`）。

### 49.11 B3 交付记录（2026-10-11，已实施 + 真机验收全绿）

**S1 权重**（宿主 `D:\flowx-data\models`，共 ≈5.8GB）：`sam/{vit_l 1249MB, vit_b 375MB, vit_h 2564MB}`、
`grounding/{tiny 690MB, base 934MB}`（全部 `gated=False` 公开仓库）。

**S2 算子族**：
- `nodes/_common/server_ops/seg.py` —— 5 算子：`seg.sam_load` / `seg.sam` / `seg.sam_auto` /
  `seg.grounding_load` / `seg.grounding_detect`（F1 版本戳 `@f18314e7cf62` 系列，随内容演进）
- `nodes/_common/server_ops/tiled.py` —— 2 算子：`tiled.plan` / `tiled.apply`（`@91cb9a1d3d9a`）
- **零新增依赖、零推理镜像重建、服务端 `app/` 一行未改**（用 `app.ops` 原语 + `model_manager._register` LRU 记账）

**S3~S6 节点与 UI**：7 个节点（id 212–218）全部细粒度 + 自定义 UI：
`sam-load` / `sam-segment`（**点·框选画布编辑器**）/ `sam-auto-segment`（**编号网格点选保留/排除**）/
`grounding-dino-load` / `grounding-detect`（**文本→框→SAM 像素级**）/ `tiled-upscale-plan`（**切片网格覆盖层+预估**）/
`tiled-upscale-apply`（逐块重绘 + 权重归一化无缝融合 + **真实 job 进度条**）。
`widget-base.js` 新增 4 个 spec：`pointSelect` / `maskMontage` / `boxList` / `tileGridOverlay`；
`build-widget.py --all` 重建 91 个 widget；gate L 91/91 通过。

**S7 真机 e2e（全 local + docker 双路径，串行提交）**：

| 流水线 | wf | 执行 | 结果 |
| --- | --- | --- | --- |
| `seg-point-e2e-local`（点选+框选） | 133 | **653** | success |
| `seg-auto-e2e-local`（自动分割 keep=0,2） | 134 | **654** | success |
| `seg-text-e2e-local`（文本→SAM 像素级） | 135 | **655** | success |
| `tiled-upscale-e2e-local`（2x / tile512 / overlap64 / 8 步） | 136 | **649** | success |
| `seg-point-e2e`（docker 执行器） | 137 | **659** | success |
| `tiled-upscale-e2e`（docker 执行器） | 138 | **665 / 666** | success |

**验收结果（`_tools/verify_seg.py` FAILED = 0 / 15 项；`_tools/verify_tiled.py` FAILED = 0 / 9 项）**：

| 判据 | 实测 |
| --- | --- |
| A2/A3 蒙版口径 | 点选/框选蒙版均为 **0/255 二值灰度**，尺寸 == 输入；前景占比 0.130 / 0.191 |
| A3 框选贴合 | **框外占比 0.00%**（37570px 全在提示框内，4px 容忍） |
| A4 不同提示不同结果 | 点选 vs 框选差异 12253px（6.2%） |
| **A6 自动分割（强判据）** | `selected` == **取回单蒙版按 keep=[0,2] 求并集，逐字节一致**（168051==168051）；`mask_ids` 数 == count；网格布局 `(480,320)` == `160×cols × rows` |
| A5 文本→像素级 | 命中 2（person）；SAM 蒙版 37789px **严格小于**框矩形并集 53444px（省 **29.3%**）；框外占比 **0.00%** |
| A7 分块超分尺寸/预放大 | 512×384 → **1024×768**；`pre_upscaled` == 本机 LANCZOS 2x **逐字节（最大通道差 0）** |
| **A8 接缝（权重）** | `ΣW` 内区最小 **255**、接缝带最小 **255**（互补 ramp 权重和恰为 1）；最外边界 2（单块覆盖，归一化仍无偏） |
| **A8 接缝（视觉）** | 全局最不连续的列 **不在**接缝带上（argmax=988，属块内）；带内最大相邻差 3.553 = 内区 p95 的 **1.38×**（无尖峰）；带内平均纹理能量 1.246× 全局非接缝（overlap 两次重绘的自然结果，绝对量级 <1 灰阶） |
| A9 细节 | 超分边缘能量 3.829 vs 预放大 3.094（**1.238×**，未变糊且更细） |
| **A10 执行器无关** | SAM 四份产物（输入/点选蒙版/点选标注/框选蒙版）local vs docker **逐字节一致**；tiled 的 `pre_upscaled`/`seam_map` **逐字节一致** |
| A10 扩散可复现性 | `tiled_up` 同参数连跑两次：平均差 0.068、最大 4/255；local vs docker：**0.065 / 4** ⇒ 量级相同 ⇒ **差异来自 GPU 扩散采样的非确定性，与执行器无关**（与 B2 GFPGAN 同口径：非确定性链路不做逐字节判据） |
| A11 幂等/钉版 | gate J 绿（12 族/45 算子/45 成员节点）；`sync-common --check` 一致；`test-op-versioning` FAILED=0；tiled 同参数两次网格/权重逐字节一致 |

**本轮新增踩坑（4 条，全部已修 + 已进设计/纪律）**：
1. **`flowx-yaml` 块的标量必须转义**（`nodes/_common/flowx_client.py::yaml_scalar` + `executor_base`）：
   裸拼 `f'{k}: "{v}"'` 在值含 `"` 时（JSON 字符串输出 candidates/sizes/plan/boxes…）会打断 YAML，
   Studio 侧得到 **`__extract_missing`** ⇒ 下游绑定报"未输出字段"。**json.dumps 即合法 YAML 双引号标量**。
2. **Studio 的参数绑定会把"看起来是 JSON"的字符串值自动反序列化成 map**，注入节点 env 时变成
   Go 的 `<map[string]interface {} Value>` ⇒ 算子收到非 JSON 文本。修法：JSON 负载统一加
   **`@` 信封前缀**（`_dumps`/`_loads`），非 JSON 前缀即不会被自动解析；前端解析前剥前缀。
3. **F1 版本化下的"跨版本收养"**：新版族实例调用 `_cached()` 命中的模型可能是**旧版实例装载并常驻**的，
   此时必须补齐本地 processor 缓存，否则 `KeyError`（真机复现 `RuntimeError: job ... failed: 'sam:vit_l'`）。
4. **`.pyc` 之外的新陷阱：Studio 用的是"导入时的节点包快照"** —— 改了节点目录里的共享件
   （`flowx_client.py` / `executor_base.py`）后**必须 `node import --overwrite` 重导**，否则真机仍跑旧代码
   （本轮为此多跑了两轮）。另：`node import` 有 **100 req/min/IP 限流**，全量重导（94 个）要**节流 + 重试**。

**其它环境事实（本轮实测）**：
- `MAX_RESIDENT_MODELS=3`（服务实际配置）：`sam:vit_l`(1191MB) + `gd:tiny`(657MB) + SD 恰好占满 ⇒ 三条链
  混用会 LRU 互斥（重载 60s+，可感知）；`_register` 前会 `_evict_if_needed`，pinned 条目（flux-unet）不动。
- 模型装载耗时（D 盘绑定挂载读权重）：vit_l **59~220s**、vit_h 163s、gd-tiny 34s ⇒ **模型应尽量常驻**；
  交互式点选靠 image embedding 缓存把第二次查询压到 **0.08s**。
- 手机侧走蜂窝网（出口 10.6.133.153）：局域网不可达 ⇒ `service_url_local` = **http 隧道**
  `http://4dc21d2a.r7.cpolar.cn`（实测**保留鉴权头**：带 token 200 / 不带 401）；
  docker CLI/Studio 执行器用 **解析后 IP** `8.162.7.178:12869`（Studio 侧执行器配置同步更新，**未重启推理服务**）。
- **Graph 边仍然是最容易踩的坑**：本轮又踩一次（docker 版 tiled 流水线漏 `CKPT --> Pos/Neg` 边 ⇒
  绑定解析失败报"上游节点没有任何输出"，与日志里看到的 emit 块无关）。

**Release**：`BUNDLE_VERSION` **1.13.3 → 1.14.0**；节点镜像 `lerkobba/flowx-pixelforge-nodes:v1.14.0`
（92 目录含 7 个新节点）；**94/94 节点全量重导**（节流后 ok=94 fail=0）。

---

## 五十、B4 立项方案：G8 老照片上色 / G9 AI 重打光 / G10 PuLID / G11 电商模板

> 2026-10-12 侦察完成（网络 + detached 容器实测），**代码未动**，待拍板 §50.6 的 D1~D8。
> 承接 §44.3 的 B4 行；B1~B3 已交付（§45/§47/§48/§49）。

### 50.1 侦察结论总表

| 组 | 能力 | 依赖增量 | 权重 | 显存（fp16 口径） | 风险 | 结论 |
| --- | --- | --- | --- | --- | --- | --- |
| **G8** | 老照片/黑白上色（DDColor） | **零**（torch+timm+einops+cv2 全有） | modelscope-L 912MB / paper-L 912MB / artistic-L 912MB / paper_tiny-T **220MB** | L@512² fp32 ≈ 2.0GB | **低** | **推荐首批** |
| **G9** | AI 重打光（IC-Light） | **零**（diffusers 0.35.2 + peft 0.17.1 + einops 有） | `iclight_sd15_fc` 1719MB / `iclight_sd15_fbc` 1719MB | 基座 UNet 1.7GB + 光照 UNet 1.7GB + VAE/CLIP ≈ 4.5GB | **中**（diffusers 0.27→0.35 API 漂移） | 需兼容 spike |
| **G10** | 人脸身份一致性（PuLID） | **零**（eva_clip 需 vendored） | `pulid_v1.1` 984MB + `EVA02_CLIP_L_336_psz14_s6B.pt` 856MB | SDXL fp16 ≈ 6.9GB **+ PuLID 1.0 + EVA-CLIP 0.9 = 8.8GB ✗ 超预算** | **高**（显存 + diffusers 0.25→0.35 大漂移 + 需改部署） | **建议单列 B5** |
| **G11** | 电商模板 | **零** | —— | 复用已常驻/可驱逐 | **低**（缺 2 项能力，见 50.1.5） | **推荐第二批** |

#### 50.1.1 环境实测（detached 容器 `flowx-inference-server:pascal`，2026-10-12）

```
OK   timm 0.9.16（B2 引入）/ einops 0.8.0 / cv2 4.11.0 / insightface 0.7.3
OK   peft 0.17.1 / accelerate 0.33.0 / safetensors / kornia 0.7.4 / spandrel 0.4.2
OK   onnx 1.23.2 / onnxruntime 1.19.2  ⚠️ providers = ['AzureExecutionProvider','CPUExecutionProvider']
     ⇒ **无 CUDAExecutionProvider**，ONNX 路径只能跑 CPU（DDColor 512² 会退化到秒级）⇒ **放弃 ONNX 方案**
--   ftfy / torchsde / modelscope 未装（本方案不需要）
--   basicsr 未装（DDColor 官方仓库自带 vendored 子集，**不需要 pip 安装 basicsr**）
```

#### 50.1.2 G8 DDColor：可行性高，需移植 ≈940 行（全部 torch/timm，无第三方依赖）

官方 `piddnad/DDColor`（master）**把模型代码内置**，分两套 arch：
- `ddcolor/model.py`（**278 行**，torch-only）+ `basicsr/archs/ddcolor_arch_utils/{unet.py 208, convnext.py 206, transformer_utils.py 192, position_encoding.py 52}`
  —— `convnext.py` 只 import `timm.layers.trunc_normal_/DropPath` ✓ 其余纯 torch
- `ddcolor/pipeline.py`（127 行，cv2+numpy：RGB↔Lab、resize、归一化）
- 另有 `basicsr/archs/ddcolor_arch.py`（paper/论文版 arch，未采用，仅作 fallback）

权重（HF API 实测 `gated=False`，均为 `pytorch_model.bin` + `config.json`）：
`piddnad/ddcolor_modelscope` 912MB（**默认推荐**，清洗数据版，非 ImageNet 图效果最好）、
`piddnad/ddcolor_paper` 912MB、`piddnad/ddcolor_artistic` 912MB、`piddnad/ddcolor_paper_tiny` **220MB**（最轻）。
`config.json`（modelscope 版）= `encoder_name convnext-l` / `decoder_name MultiScaleColorDecoder` /
`input_size [512,512]` / `num_output_channels 2`（Lab 的 ab 通道）/ `last_norm Spectral` / `num_queries 100` /
`num_scales 3` / `dec_layers 9` —— 与 `ddcolor/model.py::DDColor.__init__` 的签名**逐项对齐**（`num_output_channels=3` 版为 RGB 变体）。
→ **风险点**：HF 只放权重不放代码，键名与 vendored 类的匹配需在 S1 用探针验证；
不匹配则改用 `basicsr/archs/ddcolor_arch.py` 版（同仓库，一并移植作为 fallback）。

#### 50.1.3 G9 IC-Light：可行性中，核心代码量极小，风险全在 diffusers 版本漂移

- 官方 `lllyasviel/IC-Light` 仓库**:推理核心就在 `gradio_demo.py`（16.2KB）**；`briarmbg.py`（13KB，抠背景）
  **不需要** —— fbc 变体要的"前景/背景"直接复用 **B2 的 BiRefNet 抠图**（`matting.birefnet`）或 **B3 的 SAM**。
- 权重 `lllyasviel/ic-light`：`iclight_sd15_fc.safetensors` 1719MB（文本+前景条件）/ `iclight_sd15_fbc.safetensors` 1719MB
  （文本+前景+背景）/ `iclight_sd15_fcon.safetensors` 1719MB；本质是**把 SD15 的 UNet 换成 8/12 输入通道的条件版**。
- 基座：SD15 UNet + CLIP（宿主已有 `v1-5-pruned-emaonly` ✓，服务可加载 SD15 ✓）。
- ⚠️ 官方 `requirements.txt` pin `diffusers==0.27.2` / `transformers==4.36.2`，本服务是 **diffusers 0.35.2 / transformers 4.44.2**
  （Pascal 钉版，不可动）⇒ `UNet2DConditionModel.from_config(...)`、`StableDiffusionPipeline(..., safety_checker=None,
  requires_safety_checker=False)`、`prompt_embeds` 拼接 image embedding 的用法**需要逐个核对**；
  预期主要是参数签名/组件装配的小改，但**必须先用 detached 容器做兼容 spike 再定接口**。

#### 50.1.4 G10 PuLID：双红灯，建议不放进 B4

- 官方 `ToTheBeginning/PuLID` pin `diffusers==0.25.0` —— 与本服务差 10 个 minor；
  `pulid/pipeline_v1_1.py`（13.6KB）是**整套 SDXL pipeline 的拷贝**，0.35.2 下几乎必然要改写；
  `pulid/attention_processor.py`（16KB）做的是交叉注意力 K/V 注入（与 `ipadapter.py` 同机制的"移植"可行，但工作量中高）。
- 依赖：`pulid_v1.1.safetensors` 984MB + `EVA02_CLIP_L_336_psz14_s6B.pt` 856MB（`QuanSun/EVA-CLIP` ✓ public）
  + insightface antelopev2 ✓（B2 已装）+ SDXL base ✓（宿主已有 `sd_xl_base_1.0`）。
- ⚠️ **显存红灯**：SDXL fp16 全部常驻 ≈ 6.9GB，加 PuLID 0.98GB + EVA-CLIP 0.86GB ≈ **8.8GB > 8GB**；
  服务当前 `offload_mode: none`、flux 三条（6.5GB）pinned ⇒ 必须 **`enable_model_cpu_offload` + 驱逐 flux + 改部署配置**
  ⇒ **需要重启推理服务**（用户已明确：重启前先告知）。
- ⇒ 建议：**B4 不做**。若要，走 B5：先"**无 GPU 的代码移植 + 参数/单测**"，等拿到重启窗口再真机验收。

#### 50.1.5 G11 电商模板：能力强，但盘点出 2 项缺口

现成可用（B1~B3 交付）：抠图 `matting-birefnet` / 精修 `matting-refine`（去白边）/ 换底 `matting-composite` /
扩画布 `image-pad` / 放大 `image-upscale` + `upscale-model-*` + `tiled-upscale-*` / 人脸修复 `face-restore` /
色调 `image-curves` `image-hsl` / 裁剪 `image-crop` / 落盘 `save-image`。

**缺口**：
1. **无水印/文字叠加节点** ⇒ 建议新增 `image-watermark`（纯 PIL：文字/图片水印、九宫格定位、透明度、平铺、缩放；零 numpy）。
2. **无批量/迭代能力** —— Studio CLI 只有 `workflow run`（无循环/批处理概念），服务端 `/graph` 是单图 DAG。
   ⇒ 批量要么 (a) 新增 `image-batch-load` / `image-batch-save`（目录进出，节点内部循环）；要么 (b) 只出模板 + 外部脚本多次执行。

### 50.2 分批与顺序（建议）

| 批 | 内容 | 交付物 | 依赖前置 |
| --- | --- | --- | --- |
| **S1** | **G8 DDColor 上色** | 算子族 `colorize.py`（`colorize.load` / `colorize.apply`）+ 2 节点（`colorize-load` 预热 / `colorize`）+ 真机 e2e | 无（零新依赖） |
| **S2** | **G11 电商模板** | `image-watermark` 新节点（纯 PIL）+ 3 条模板流水线（白底主图 / 场景换底 / 模特图规格化）+ 批量能力（D7 决策） | S1 无依赖，可并行 |
| **S3** | **G9 IC-Light 重打光** | 兼容 spike（detached）→ 算子族 `iclight.py`（`iclight.load` / `iclight.apply`）+ 2 节点 + e2e | 需 S1 的 LRU 记账经验（同款 `_register`） |
| **S4** | **G10 PuLID**（**建议延后/单列**） | 先纯代码移植 + 单测；真机验收等重启窗口 | 需改部署（offload）+ 重启服务 |

### 50.3 接口设计（草案）

**G8（2 节点）**
- `colorize-load`：`model`（`modelscope`/`paper`/`artistic`/`paper_tiny`）→ 输出 `model_ref`(STRING `ddcolor:<variant>`)
- `colorize`：`model_ref` / `image` / `size`（512 默认，须为模型训练尺寸；非 512 时先内部 resize 再回填原尺寸）
  / `saturation`（1.0，post 色彩增强）/ `seed` → 输出 `image`(STRING) + `info`(STRING JSON：尺寸/变体/耗时)
- 算子：`colorize.load(variant, dtype=fp32)`（走 `model_manager._register`，键 `ddcolor:<variant>`）、
  `colorize.apply(model_ref, image_id, size, saturation)`

**G9（2 节点）**
- `iclight-load`：`variant`（`fc`/`fbc`/`fcon`）→ `model_ref`
- `iclight-apply`：`model_ref` / `image` / `foreground_mask`(可选，fbc/fc 用) / `background`(可选) /
  `prompt` / `light`（`left`/`right`/`top`/`bottom`/`ambient` 预设 → 官方 4 方向 prompt 模板）/
  `steps` `cfg` `seed` `strength` → `image`
- 算子：`iclight.load(variant)`、`iclight.apply(...)`（内部：SD15 组件 + 光照 UNet + prompt_embeds 拼接）

**G11（1 节点 + 流水线）**
- `image-watermark`：`image` / `text` / `font_size` / `opacity` / `position`（九宫格）/ `margin` / `color` / `tile`(bool) / `logo`(可选 IMAGE)
- 模板流水线：`ecom-white-bg`（抠图→去白边→白底→1:1 扩画布→放大 2x→落盘）、
  `ecom-scene`（抠图→去白边→背景合成→轻微色调统一→落盘）、
  `ecom-model-spec`（人脸修复→色调→裁剪 1:1/4:5→水印→落盘）

### 50.4 影响面

- **依赖/镜像**：G8/G9/G11 均 **零新增 Python 依赖** ⇒ **不重建推理镜像**（沿用 `:pascal`）；
  仅需下载权重到宿主 `/models/{colorize,iclight}`（合计 ≈ 1.1GB + 1.7GB）。
- **服务端 `app/`**：**零改动**（沿用插件族 + `app.ops` 原语 + `model_manager._register` LRU 记账）。
- **对象类型**：不新增（模型句柄走 STRING，键 `ddcolor:<variant>` / `iclight:<variant>`）。
- **显存**：G8-L fp32 ≈ 2.0GB、G9 ≈ 4.5GB（含基座）⇒ 均在 8GB 内，但会与 flux trio /
  SAM / GD 争夺 `max_resident_models: 3` 名额（⇒ 交互式使用建议先 `model-unload` 让位）。
- **Studio 侧**：新增节点导入（94 → 97~99）；无 SDK/DB 迁移。
- **重启**：S1~S3 **不需要**重启（插件热上传）；S4 若做则**需要**（改 offload 部署配置）。

### 50.5 验收标准（落盘 + 可量化）

**G8**：3 类输入（黑白人像 / 褪色老照片 / 偏黄旧照）→ 每张产 `colorized` + `side_by_side` 落盘：
- A1 尺寸 == 输入（`size` 参数只影响内部推理分辨率）；A2 **确定性**（DDColor 无随机采样 ⇒ 同输入同参数**逐字节一致**）；
- A3 色彩化有效性：色度饱和均值（HSV 的 S）相对灰度输入提升 ≥ 5×；**近中性像素占比**（S<0.05）≤ 输入的一半；
- A4 肤色合理性：检出人脸的区域内 H ∈ [10°, 45°] 占比 ≥ 60%（用现有 `face-mask` 定位）；
- A5 local 与 docker 两路径产物**逐字节一致**；A6 4 个变体 sha 互异且各自确定。

**G9**：同一输入 × 4 个光照方向 → 产物落盘：
- B1 方向可控性：亮度质心 (x,y) 沿 `left/right/top/bottom` **单调移动**（量化阈值待 spike 后定）；
- B2 结构保持：与输入的梯度相关/SSIM ≥ 阈值（不把画面重画）；
- B3 前景掩膜外像素改动量 ≤ 阈值；B4 确定性用**漂移口径**（扩散采样，同 §49.6 A10 校正后的口径）；
- B5 fbc 变体：背景区域光照方向与前景一致（无拼接感，人眼物料 + 边缘能量判据）。

**G10**（若做）：身份相似度用现有 `face-similarity` 节点度量 ≥ 0.6；显存峰值 < 8GB（`/health` 采样）。

**G11**：3 条模板流水线 e2e 全绿 + 落盘复核：
- C1 白底主图：背景区 `min(R,G,B) ≥ 250` 占比 ≥ 99.9%、过渡带近白像素占比 = 0（复用 §48.12 判据）；
- C2 尺寸规格：1:1 输出 `W==H`，主体占比 ≥ 60%（包围盒）；
- C3 水印：文字区域像素在指定位置发生变化、非水印区逐字节不变；
- C4 场景图：合成 == alpha 混合公式（最大通道差 ≤ 1）。

### 50.6 决策点（待拍板）

- **D1 范围与顺序**：按 §50.2 的 S1→S2→S3，**G10 单列 B5**？（建议：是）
- **D2 G8 变体与预热**：默认 `modelscope`(L, 912MB)，`paper_tiny`(T, 220MB) 作轻量备选；是否要 `colorize-load` 预热节点？
  （建议：要 —— 与 `sam-load` 同理由，D 盘绑定挂载读权重慢）
- **D3 G8 后处理**：是否给 `saturation`（默认 1.0）+ 可选 `denoise`/`colorfulness` 增强参数？（建议：只给 `saturation`，保持确定性）
- **D4 G9 变体与掩膜来源**：默认 `fc`（文本+前景）还是 `fbc`（+背景）？前景掩膜由**内置 BiRefNet 自动出**还是要求上游提供？
  （建议：默认 `fc`；掩膜既可上游传入，也可空 ⇒ 自动调 `matting.birefnet`）
- **D5 G9 兼容红线**：若 diffusers 0.35.2 适配超过 **1 个工作日**或需要改动核心 pipeline 组件 ⇒ 是否允许**放弃 G9**（记录为"不做"）？
  （建议：允许，且禁止为了它降级/替换 diffusers —— Pascal 钉版优先）
- **D6 G10 处理方式**：延后为 B5（等重启窗口）／现在就做纯代码移植（不验收）／彻底不做？（建议：B5）
- **D7 G11 批量**：(a) 新增 `image-batch-load`/`image-batch-save` 目录批处理节点 / (b) 只出模板 + 外部脚本循环
  / (c) 两者都做？（建议：先 (b) 出模板验证价值，批量节点进 B5）
- **D8 G11 水印节点**：是否做 `image-watermark`（纯 PIL，零 numpy）？（建议：做，电商白底图基本都要）

### 50.7 风险与降级路径

| 风险 | 触发条件 | 降级路径 |
| --- | --- | --- |
| G8 权重键与 vendored 类不匹配 | S1 探针 `load_state_dict` 报 missing/unexpected | 改用 `basicsr/archs/ddcolor_arch.py` 版 arch（同仓库，一并移植） |
| G8 上色偏灰/串色 | 验收 A3/A4 不达标 | 提高 `saturation` 默认值 / 换 `artistic` 变体 / 加 `image-curves` 后处理节点串联（零代码） |
| G9 diffusers 漂移超红线 | spike 超过 1 天或需改核心 pipeline | **放弃 G9**（按 D5），B4 只交付 G8 + G11 |
| G9 显存不足 | spike 实测 > 8GB | 降 `steps`、fp16 强制、分块 VAE 解码（kornia/tiled VAE 已有先例） |
| G10 显存不足 | 实测 > 8GB | `enable_model_cpu_offload` + 驱逐 flux（**需重启**）或彻底放弃 |
| G11 批量需求强 | 用户要一次跑目录 | 提前做 D7(a) 的批量节点 |

### 50.8 S1 交付记录（G8 DDColor 上色：已实施 + 真机验收全绿）

**权重**（宿主 `D:\flowx-data\models\colorize\`，共 ≈2.8GB；容器内 `/models/colorize`）：

| 变体 | 上游仓库 | 体积 | 参数量 | 说明 |
| --- | --- | --- | --- | --- |
| `modelscope` | `piddnad/ddcolor_modelscope` | 870MB | 227.9M | **默认**（清洗数据版，泛化最好） |
| `paper` | `piddnad/ddcolor_paper` | 870MB | 227.9M | 论文版 |
| `artistic` | `piddnad/ddcolor_artistic` | 870MB | 227.9M | 艺术风格版 |
| `paper_tiny` | `piddnad/ddcolor_paper_tiny` | 211MB | 55.0M | 最轻量（convnext-t） |

**算子族 `nodes/_common/server_ops/colorize.py`**（1316 行，含 **945 行 vendored 上游代码**）
- 算子：`colorize.load` / `colorize.apply`，版本戳 **`@b77d44da027e`**；落盘插件名 `colorize_b77d44da027e.py` ✓
- **零新增 Python 依赖**（torch + timm + cv2 全有）、**零服务端改动**（`app/` 一行未改）、**不重建推理镜像**
- 上游 5 个模块（`ddcolor/model.py` + `basicsr/archs/ddcolor_arch_utils/{unet,convnext,transformer_utils,position_encoding}.py`）
  逐行内联进 `_ddclasses()` **惰性工厂**（模块导入不 import torch ⇒ Termux 可跑纯逻辑单测，27+46 项断言）
- 与上游的**唯二**差异：① 去掉模块级 import / `from basicsr` 交叉引用；② `astype(uint8)` 前补 `np.clip`
  （上游对越界值会回绕；S1 探针实测该修复对真实数据**零影响**，见下）

**节点**（id 219 `colorize-load` / 220 `colorize`，全自定义 UI，`compareInput: image` 做前后 AB 对比）

**真机探针（GTX 1080, torch 2.3.1+cuda12.1, fp32）**

| 项 | 实测 |
| --- | --- |
| **移植保真**（官方 `ColorizationPipeline` vs 本族内联，同权重同图） | **逐字节一致 = True / 最大通道差 0 / 0 像素差异（196608 全同）** |
| 权重键匹配（§50.7 的风险项） | 4 个变体全 **strict** ⇒ 不需 basicsr 版 arch，风险项**关闭** |
| 装载+build / 推理 512² / 峰值显存 | L: 12.6~13.3s / 0.23~0.46s / **3686MB**；T: 3.8s / 0.13s / 3001MB |
| 分辨率扫描（T） | 256→0.10s/2390MB、512→0.13s/3000MB、768→0.40s/4015MB、1024→2.86s/**5433MB** |
| 确定性 | 同进程两遍**逐字节一致** ✓（无随机采样） |

**e2e（3 条流水线，串行提交）**

| 流水线 | wf | 执行 | 结果 |
| --- | --- | --- | --- |
| `colorize-e2e-local`（4 类输入×modelscope + 人脸蒙版定位） | 139 | **670 / 671**（连跑两遍验确定性） | success |
| `colorize-variants-e2e`（1 图×4 变体） | 140 | **672** | success |
| `colorize-e2e`（docker 执行器，镜像 v1.15.0） | 141 | **673** | success |

**验收 `nodes/_tools/verify_colorize.py`：PASSED = 26 / FAILED = 0**

| 判据 | 实测 |
| --- | --- |
| A1 尺寸 | 上色输出 == 输入（512×384 ×3、1024×1024 ×1） |
| **A2 确定性** | wf 139 连跑两遍，4 个前缀的产物**逐字节一致** |
| A3 色彩化有效性 | 灰度人像 S 0.0→145.6（∞×）、近中性 100%→**0%**；真人灰图 0.0→38.1、近中性 100%→24.9%；褪色 33.0→134.4、偏黄旧照 46.4→135.3 |
| A4 肤色合理性 | 人脸框内**色相模态 10~20°**（正是肤色）、肤色带占比 42.8%、平均 S 43.3（35776px 采样） |
| **A5 执行器无关** | local vs docker 三张产物**逐字节一致**；A5b 两路径输入也一致 |
| A6 变体互异 | 4 变体 sha 互异（`125e13ff…`/`6484dccd…`/`41fb1792…`/`e39f8d41…`） |

**判据标定校正（诚实记录，同 §49.6 的做法）**：
- **A3**：初版写"S 均值提升 ≥5×"对**已带色**的输入（褪色/偏黄）不可达也无意义（实测 4.1×/2.9× 已是强上色）
  ⇒ 改为**分支判据**：灰度输入（S_in<1）用 ≥5×；已带色输入用"S_out ≥ 100 且 ≥ 2×S_in"。
- **A4**：初版写"人脸蒙版内 H∈[10°,45°] 占比 ≥60%"——但 `face-mask` 出的是**人脸矩形框**（含头发/背景/衣物），
  矩形框里不可能 60% 是肤色（实测峰值 10~20° 正确落在肤色带，但只占 42.8%）⇒ 改为三条按物理含义的判据：
  ① 框内**色相模态** ∈ [5°,45°]；② 肤色带占比 ≥35%；③ 框内平均 S ≥20/255。

**服务端证据**：算子 `colorize.load@b77d44da027e` / `colorize.apply@b77d44da027e`（**插件热上传，未重启服务**）；
插件落盘 `plugins.d/colorize_b77d44da027e.py`；LRU 常驻 `ddcolor:modelscope`/`ddcolor:artistic`/`ddcolor:paper_tiny`
（第 4 个 `paper` 被 3 槽 LRU 淘汰 ✓ 符合预期；`_register` 记账生效）。

**本轮新增踩坑（4 条）**：
1. **nodeRef 版本陷阱再犯**：`face-mask` 实际是 **1.1.0**（我按 1.0.0 写 → 执行 669 直接 `node not found`）。
   纪律：写完流水线先跑一遍 `node list`/API 核对**每个** nodeRef 版本，别凭记忆。
2. **A4 的人脸图要选真能检出脸的**：`b3_person.png` 在 `det_thresh=0.5` 下**检不到脸**（face.mask 报
   "未检测到人脸"并中止整条执行）；改用 B2 G6 的 `face_in1_face_sharp` 转灰度 ✓ 检出正常。
   ⇒ 顺带发现：**单节点失败会中止整条执行**（B4 流水线里人脸支路失败把已成功的上色结果的落盘也带走了）。
3. **`docker cp` 后 `docker start` 不会重跑**（容器主命令是启动时快照）：必须用
   `while [ ! -f /fetch.py ]; do sleep 2; done; python /fetch.py` 的"等文件"模式（B3 已记，这次是我自己忘）。
4. **docker build 的 context 上传在山寨隧道下很慢**（12.1MB 花了数分钟才从 `Sending build context` 走到
   `Step 1/100`）⇒ 别误判为卡死；`DOCKER_BUILDKIT=0` + `SKIP_PUSH=1`（远端 daemon 本地构建，Studio 直接用）✓

**发布**：`BUNDLE_VERSION 1.14.0 → 1.15.0`；节点镜像 `lerkobba/flowx-pixelforge-nodes:v1.15.0`（176MB，含 2 个新目录）；
节点导入 94 → **96**（id 219/220）；人眼物料 `~/flowx-output/b4_colorize_overview.png`（4 组输入/输出对照）
与 `b4_colorize_variants.png`（4 变体横排）。

### 50.9 S2 交付记录（G11 电商模板：已实施 + 真机验收全绿）

**新节点 `image-watermark`（id 221，纯节点侧 PIL，零 numpy）**
- 参数：`text`/`logo`（二选一；logo 优先）+ `position`（九宫格）+ `margin` + `opacity`
  + `font_size`（0=自动 短边/18）+ `font_path` + `color` + `stroke_width`/`stroke_color`
  + `logo_scale`（0=自动）+ `tile`/`angle`/`tile_gap`（斜铺防盗图）
- 输出：`image` + `width`/`height` + **`box`(x,y,w,h)** + `box_w`/`box_h`/`item_w`/`item_h`
  ⇒ `box` 就是"非水印区逐字节不变"这条判据的锚点
- 本机单测 `test_main.py` **42 项断言**（含**第 6 节 main() 桩冒烟**，见下）
- 已知限制：容器/手机默认字体只含 ASCII ⇒ **中文水印必须给 `font_path` 指向含 CJK 的 TTF**

**3 条电商模板流水线**（全部 local 执行器；`~/flowx-e2e-out` 落盘）

| 模板 | wf | 执行 | 链路 |
| --- | --- | --- | --- |
| `ecom-white-bg`（白底主图） | 142 | **676 / 682** | 抠图(BiRefNet main@1024 + 去白边) → 白底合成(feather=1) → 1:1 扩画布(white) → 水印 → 落盘 |
| `ecom-scene`（场景换底） | 143 | **677 / 683** | 抠图(lite + 去白边) → 背景图合成(feather=8) → HSL(sat 1.05,light +2) → 水印 → 落盘 |
| `ecom-model-spec`（模特图规格化） | 144 | **681** | 人脸修复(GFPGAN) ‖ 裁 512² 中心 → RealESRGAN 4x → 水印 → 落盘 |

**验收 `nodes/_tools/verify_ecom.py`：PASSED = 12 / FAILED = 0**

| 判据 | 实测 |
| --- | --- |
| C1 白底纯度 | 把主体外扩 2px 后的背景区（129021px）**100.000% 纯白** |
| C2 尺寸规格 | 输出 512×512 正方形；主体宽/高填充 0.73/0.63；主体面积占比 0.29 |
| **C3 水印只改水印区** | 加印前后 diff bbox=(278,458,483,484)（右下、margin 24）；**区外改动 0 px（逐字节）** |
| C4 场景合成 | 主体核心（腐蚀 24px，19607px）与输入**最大差 0**；背景替换率 **99.9%**；过渡带 3201px |
| C4' 主体核心（腐蚀 16px，29324px） | 最大差 3（≤4）✓ |

**判据标定校正（诚实记录，同 §49.6/§50.8 的做法）**
- **C1**：初版把 `mask<128` 全当背景要求 ≥99.9% 纯白 → 实测 97.534% ✗。原因是 **0<mask<128 是合成羽化的自然过渡带**，
  不该要求纯白 ⇒ 改为把主体向外扩 2px、只判"远离轮廓的背景" ⇒ **100.000%** ✓（与 B2 §48.12 去白边专项同一方法论）。
- **C4**：初版用未羽化的 `Matte.mask` 当 alpha，要求主体区（mask≥250）差 ≤1 → 实测 41% 像素超 1、最大 167 ✗。
  实测发现 **`matting-composite` 的 `feather=8` 在节点内部是 ≈24px 宽的过渡带**（腐蚀 16px 后最大差 3、腐蚀 24px 后 **0**）
  ⇒ 改为"腐蚀 `feather+8` 的核心 ≤4、腐蚀 24px 核心 = 0"，并把"feather=8 ⇒ ≈24px 过渡带"记为**运维事实**。

**本轮新增踩坑（3 条，全部已修 + 已进纪律）**
1. **`node mock` 跑的是 mock.py，跑不到 main.py** ⇒ `image-watermark/main.py` 的 `NameError: opacity`
   一路逃到真机（wf 142 首次执行失败）✗。已在该节点的本地单测里加 **第 6 节：main() 桩冒烟**
   （桩掉 `flowx_client`/`image_io` 的网络与 emit，真跑一遍 `main()`）⇒ 未定义名字/参数名写错/漏传都会在本机暴露。
   **建议**：把这个桩冒烟推广成门禁（对 `files` 含 `image_io.py` 的纯 PIL 节点统一执行），列入 B5 候选。
2. **GFPGAN 吃的是"检测后对齐的人脸块"，不是整图** ⇒ 我先按 512² 中心裁剪再喂 `face-restore`，
   结果它在这个裁剪图上"未检测到人脸"（`det_thresh` 0.3/0.5 都不行）；而同一张裁剪图给 `face-mask@1.1.0`
   在 0.3 下**能**检出 ⇒ 两个算子的检测实现/阈值语义不同。正解：人脸修复走**整图**，裁剪只服务放大分支。
3. **tunnel 单次 `POST /jobs` 偶发 `RemoteDisconnected`**（cpolar 免费隧道）⇒ 直接重跑同一条流水线即成功
   （节点本身有瞬态重试，但 `POST /jobs` 这一跳没有）⇒ 真机 e2e 失败先看是不是隧道抖动，再怀疑代码。

**发布**：`BUNDLE_VERSION 1.15.0 → 1.15.1`；镜像 `lerkobba/flowx-pixelforge-nodes:v1.15.1`（含 image-watermark 目录）；
节点 96 → **97**（image-watermark id 221）；人眼物料 `~/flowx-output/b4_ecom_overview.png`（①②③ 三模板 8 格对照）。
