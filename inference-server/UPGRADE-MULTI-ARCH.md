# 多架构（多代显卡）支持方案

> 状态：**已定稿**（2026-10-18 起草；同日用户确认五项决议，见 §十）
> 范围：含 fp8（阶段 4 为本次范围内，非可选）
> 本文件只回答一个问题：如何让**一份代码**同时支持 10/20/30/40/50 系显卡，并让 20/30/40/50 吃到各自的精度优势。
> 相关文档：`ROADMAP.md`（阶段与不做清单）、`DEPLOY.md`（部署运维）、`../docs/dev-plan.md`（待办/已关闭）。

---

## 一、目标与非目标

### 目标

1. 同一份 `app/` + `nodes/_common/server_ops/` 代码，跑在**两套依赖栈**上，不维护分支。
2. Pascal（10 系）维持现状不变，行为零回归。
3. 20/30/40/50 系共用**一个**新镜像，并吃到各自硬件支持的精度档：
   - 20 系：fp16 + 张量核（Turing **无 bf16 硬件**）
   - 30 系：fp16 / bf16 / tf32
   - 40/50 系：bf16，并（可选阶段）解锁 fp8
4. 构建期就能发现"某代卡没有对应 kernel"，而不是到真机才报 `no kernel image is available`。

### 非目标

- 不为每代卡建独立镜像（**只有 2 个镜像**：pascal / modern）。
- 不在本次方案里引入 TensorRT / xformers / flash-attn（另立项）。
- 不改 Studio、flowx 引擎、节点共享镜像（`nodes/Dockerfile` 是 `python:3.11-slim` 纯 HTTP 客户端，与显卡无关）。
- 不承诺"未真机验证过的卡可用"——见 §七 验证矩阵。

---

## 二、硬约束（方案的地基）

| 代 | 算力 | torch 要求 | bf16 | fp8 | 归属镜像 |
|---|---|---|---|---|---|
| 10 系 Pascal（GTX 10xx / P4/P40/P100） | sm_60/61 | **≤ 2.6**（2.7 官方轮子移除 sm_60/61） | ❌ | ❌ | **A: pascal** |
| 20 系 Turing | sm_75 | ≥2.1 | ❌（硬件不支持） | ❌ | **B: modern** |
| 30 系 Ampere | sm_80/86 | ≥2.1 | ✅ | ❌（无 fp8 张量核） | **B** |
| 40 系 Ada | sm_89 | ≥2.1 | ✅ | ✅（e4m3） | **B** |
| 50 系 Blackwell | sm_100/120 | **≥ 2.7 + CUDA 12.8** | ✅ | ✅ | **B** |

**为什么 2 个就够**：torch 的 cu128 轮子一份覆盖 sm_75~sm_120（CUDA 二进制在同 major 内向后兼容：8.6 的 kernel 可在 8.9 上跑）。只有 Pascal 因"被 2.7 移除"必须单独一套。

**为什么 Pascal 不能并进来**：`diffusers ≥0.36` 引用 `torch≥2.5` 的 `flex_attention`（导入期崩），而 Pascal 锁死 torch 2.3.1 ⇒ 0.35.2 是它的天花板。这是既有事实，本方案不挑战。

---

## 三、总体结构

```
同一份代码                     两套依赖                两个镜像
────────────────────────────────────────────────────────────────
app/                     ┌─ requirements-pascal.txt ── Dockerfile.pascal ─→ flowx-inference-server:pascal-partA5（现状不变）
nodes/_common/           │      torch 2.3.1 / cu121              sm_60/61
  server_ops/*.py   ─────┤      diffusers 0.35.2                 MAX_RESIDENT_MODELS=1
                         │
                         └─ requirements-modern.txt ── Dockerfile.modern ─→ flowx-inference-server:modern-partA5（新）
                                torch 2.7+ / cu128                     sm_75/80/86/89/90/100/120
                                diffusers 0.41.0                       MAX_RESIDENT_MODELS 按显存自适应
```

**分流点只有两个，且都在构建层**（pip 没有"按显卡型号条件安装"的能力，见 §五.3）：
1. 选哪个基础镜像（决定 torch/CUDA 二进制与可支持的算力）
2. 选哪个 requirements（决定 diffusers 及其依赖代次）

运行期**不分流**：同一份代码靠能力探测决定 dtype / 补丁 / 量化路径。

---

## 四、代码层改动（唯一需要动代码的部分）

原则：**只做能力探测，不写版本号 if**。理由是同一份文件要同时跑在 torch 2.3.1 与 2.7+ 上，版本号判断会随时间腐化，能力探测不会。

### 4.1 必改：FLUX 的 SDPA shim 加能力门（阻塞项）

- **位置**：`nodes/_common/server_ops/flux.py` → `_install_torch_compat()`
- **现状**：`torch.nn.RMSNorm` 有 `hasattr` 判断（新栈自动跳过），但 `F.scaled_dot_product_attention` 的 shim **无条件安装**，且 `enable_gqa=True` 时**直接抛错**。
- **后果**：modern 栈上 diffusers 新版会恒传 `enable_gqa`，FLUX 系全部算子当场失败。
- **改法**（约 3~5 行）：

```python
import inspect
_sdpa_params = inspect.signature(F.scaled_dot_product_attention).parameters
if "enable_gqa" not in _sdpa_params:      # 只有老 torch 才需要 shim
    ...安装 shim...
```

- **必须重导节点**：改的是共享件（`_common`），按纪律"改了共享件必须重导节点"，并且要 bump 受影响节点版本。
- **负例自检**：摘掉门控 ⇒ 在 modern 栈上 FLUX 单测必须失败（证明门控真的在起作用）。

### 4.2 必改：`dtype=auto` 按算力分档

- **位置**：`app/model_manager.py`（`_resolve_dtype` / VAE 相关默认值）
- **现状**：`auto` 固定 → `fp16`；VAE 默认 fp32（`VAE_FP32=1` 或外挂 VAE）
- **目标行为**：

| 算力 | `auto` 解析为 | 理由 |
|---|---|---|
| sm < 7.0 | `fp32` | Pascal 非张量 fp16 极差（既有结论） |
| sm 7.5 | `fp16` | Turing 无 bf16 |
| sm ≥ 8.0 | `bf16` | 动态范围优于 fp16，大模型/视频更稳 |

- **约束**：
  - 显式传 `dtype` 时**永远尊重用户**（不得被 auto 策略覆盖）。
  - 缓存键必须带上实际解析后的 dtype（现有 `_dtype_suffix` 机制），否则同名不同精度会串缓存。
  - `/health` 增加 `compute_capability` 与 `default_dtype`，便于排障。
- **风险**：bf16 在部分管道上与 fp16 输出不完全一致 ⇒ 需回归（见 §七）。
- **负例自检**：mock 一个 capability=7.5 的探针，断言解析为 fp16 而非 bf16。

### 4.3 必改（阶段 4）：fp8 量化（已纳入本次范围）

- 现状：`QUANTIZATION=fp8` 只做字符串识别 + 告警，**未实现**。
- 实现路径（推荐 torchao，而非自研）：
  1. 引入 `torchao`，用 weight-only / dynamic fp8 量化包装线性层；
  2. 仅在 capability ≥ 8.9 时启用，低于则降级并打印明确原因；
  3. `app/vram.py` 的体积估算要认识 fp8（1 字节/参数），否则淘汰决策会错；
  4. 退出路径：`QUANTIZATION=none` 必须仍然可用（默认）。
- **验收口径**：同 seed 同参数下 fp8 vs fp16 出图人眼可接受 + 清晰度指标不显著下降；**不接受"状态全绿但出噪声"**。
- 这一项独立开关、独立验收，不阻塞阶段 1~3。

### 4.4 明确不改

- `app/sniff.py`、`app/ops.py` 的采样分派、对象仓库、jobs/lease、监控页：与算力无关。
- 节点侧 `flowx_client.py`：纯 HTTP，无 torch。

---

## 五、镜像层改动

### 5.1 新增 `inference-server/requirements-modern.txt`

版本以"diffusers 0.41.0 的实际依赖要求"为准，构建期锁死（**不写宽范围**，与 pascal 版同样纪律）。预期对照：

| 包 | pascal（现） | modern（建议） | 备注 |
|---|---|---|---|
| torch | 2.3.1 | 2.7.x（cu128） | 需先核实 wheel 的 arch 列表（§5.3） |
| diffusers | 0.35.2 | **0.41.0** | 2026-10-06 最新 |
| transformers | 4.44.2 | 4.5x（按 diffusers 0.41 要求） | H3/Wan 新编码器 |
| huggingface-hub | 0.34.4 | 1.x | 1.x 有破坏性变更，需验证下载路径 |
| numpy | <2 | 2.x | 需连带 opencv/accelerate 一起换 |
| accelerate | 0.33.0 | 新版（跟随 torch） | offload 路径 |
| opencv-python-headless | 4.11.0.86 | 新版 | 仍须 `--no-deps` + 强制重装（老坑） |
| kornia | 0.7.4 | 0.8+ | 0.8 要求 py≥3.11 ⇒ modern 栈 Python 用 3.12 |
| timm | 0.9.16 | 新版 | BiRefNet 依赖 `timm.layers`，需回归 |
| onnxruntime | 1.19.2 | 新版 | insightface 走 CPU |
| insightface | 0.7.3 | 保持 | 无 manylinux wheel，仍需 g++ 编译 |
| controlnet_aux | 0.0.10 | 保持或新版 | 仍会拉非 headless opencv（老坑） |
| spandrel | 0.4.2 | 新版 | 超分模型识别 |
| gguf | 0.17.1 | 新版 | 配合 `kernels` 可走融合反量化（sm≥7.5） |
| peft | 0.17.1 | 新版 | LoRA 挂载 |

### 5.2 新增 `inference-server/Dockerfile.modern`

- 基础镜像：**Python 3.12**（已拍板，因 kornia 0.8+ 要求 py≥3.11，且新 timm/transformers 代次均以 3.12 为基线）
  - 优先：官方 `pytorch/pytorch:2.7.x-cuda12.8-cudnn9-runtime` **若其内置 py3.12**；
  - 否则：`nvidia/cuda:12.8-cudnn9-runtime` + 自装 python3.12 + `pip install torch --index-url .../cu128`（多一层构建，但可控且版本透明）。
  - **具体 tag 与 python 版本在阶段 0 取证确认**（这是阶段 0 的第一项产出）。
- 保留 pascal 版的全部既有构建纪律：
  - 构建期临时装 `g++` 编译 insightface，同层内卸载；
  - `pip uninstall opencv-python` → `--force-reinstall --no-deps opencv-python-headless`；
  - **import smoke 门禁**（cv2 / spandrel / controlnet_aux / insightface / onnxruntime / gguf / sentencepiece / hub / timm / kornia / einops），断言版本号为新值。
- **新增 arch 断言门禁**（关键）：

```dockerfile
RUN python -c "import torch; a=torch.cuda.get_arch_list(); \
  need=('sm_75','sm_80','sm_86','sm_89','sm_90','sm_120'); \
  miss=[x for x in need if x not in a]; \
  print('archs:', a); assert not miss, f'缺少算力支持: {miss}'"
```

  注：CUDA 二进制向后兼容，`sm_89` 可能由 `sm_86` 覆盖；断言按**实测 arch 列表**校准，不要照抄候选清单。

### 5.3 新增 `inference-server/docker-compose.modern.yml`

- 容器名 `flowx-inference-server-modern`，与 pascal 版并存（不同端口或不同宿主机）；
- `MAX_RESIDENT_MODELS` 不再写死 1，由显存预算式淘汰决定（既有机制）；
- 模型目录挂载与 pascal 一致（`D:\flowx-data\models` 约束不变）。

### 5.4 为什么不能"一份 requirements 通吃"

pip 的环境标记只认 `python_version` / `sys_platform` / `platform_machine` / `os_name`，**没有显卡维度**；`torch==2.3.1` 与 `>=2.7` 也无法共存于一次解析。分流必须在构建层完成。

---

## 六、门禁与 CI

| 门禁 | 内容 | 什么时候跑 |
|---|---|---|
| 构建期 import smoke | 两栈各自断言关键依赖可导入 + 版本符合预期 | 每次 build |
| 构建期 arch 断言 | modern 栈的 `torch.cuda.get_arch_list()` 覆盖目标算力 | 每次 build |
| 算子契约一致性 | `_tools/check-op-signatures.py`、`check-ops.py` | 两栈各跑一次（import 级，无需 GPU） |
| 节点参数一致性 | `check-node-params.py` | 共享件变更后 |
| 节点入口桩冒烟 | `test-main-smoke.py`（stub 模式） | 两栈各跑一次 |
| 服务端单测 | `inference-server/tests/`（现有 ad-hoc 测试） | 两栈各跑一次 |

**纪律**：任何新增门禁必须带负例自检（摘掉被测修复后门禁必须失败）。

---

## 七、分阶段实施与验收标准

### 阶段 0：取证（0.5 天，不改代码）

产出"可行性结论"，否则后续都是猜：

1. 在 `pytorch/pytorch:2.7-*` 基础镜像里打印 `torch.cuda.get_arch_list()` 与 `torch.version.cuda`，确认覆盖 20/30/40/50；
2. 试装 `diffusers==0.41.0`，打印其 `Requires-Dist` 列表，据此锁定 transformers/hub/numpy/accelerate；
3. 试装 `insightface==0.7.3` + `controlnet_aux==0.0.10` + `ultralytics`，确认 py3.12 + numpy2 下能否构建成功；
4. 结论写回本文件 §5.1 对照表（把"建议"改成"确定"）。

**验收**：四步全部有明确结论，不确定项显式列为"待定/不支持"。

### 阶段 1：modern 镜像（1~3 天）

- 新增三个文件（requirements-modern / Dockerfile.modern / docker-compose.modern）；
- 构建通过 + 两项构建期门禁通过。

**验收**：
- `docker build` 成功，smoke 与 arch 断言全绿；
- `curl /health` 返回 `diffusers_version: 0.41.x`、`torch_version: 2.7.x`、`status: ok`；
- 与 pascal 镜像**同时存在于同一台 daemon**，互不影响。

### 阶段 2：代码能力探测改造（1 天）

- §4.1 shim 能力门 + §4.2 dtype 分档；重导受影响节点（bump 版本）；
- 两栈跑 §六 的全部 import 级门禁。

**验收**：
- pascal 栈：FLUX 算子行为不变（原有单测全过）；
- modern 栈：FLUX 算子可用（不再因 `enable_gqa` 抛错）；
- `dtype=auto` 在 capability 7.5 / 8.6 / 8.9 的 mock 下分别解析为 fp16 / bf16 / bf16；
- 负例自检：去掉 shim 门控后 modern 栈单测失败；去掉分档后 mock 断言失败。

### 阶段 3：真机回归（1~2 天 GPU 机时）

在 modern 栈上按 §八 矩阵执行，至少覆盖：

- SD1.5 / SDXL 出图（含 ControlNet、LoRA）
- FLUX（GGUF 路径，验证新 torch 下反量化是否变快 / 是否需要 `kernels` 融合）
- 视频链（SVD i2v、Wan TI2V、vae 编解码往返）
- 人像（InstantID、GFPGAN/RestoreFormer）、抠图（BiRefNet）、预处理器（openpose/depth）
- 上色 / 重打光 / 分块超分
- 取消与显存淘汰（长采样中途 `/interrupt`，检查租约与 `/leases`）

**验收**：每条链路落盘产物 + 人眼复核（不许只看状态）；显存账本与实际占用误差在既有口径内。

### 阶段 4：fp8（2~3 天，本次范围内）

见 §4.3。独立开关（`QUANTIZATION=none` 仍为默认）、独立验收；**无 40/50 系真机时只能到“单元级 + mock 能力探测通过”，必须在文档中标注未验收**。

---

## 八、验证矩阵（"支持"的定义）

| 卡 | 栈 | 状态目标 | 验收方式 |
|---|---|---|---|
| GTX 1080（现役） | pascal | 已验收（现状） | 既有 e2e |
| 20 系 | modern | 代码就绪、**未验证**（无硬件）：fp16 档 | 单元/mock 级；上线前必跑一次真机 |
| 30 系 | modern | 代码就绪、**未验证**（无硬件）：bf16 档 | 同上 |
| 40 系 | modern | 代码就绪、**未验证**（无硬件）：bf16 档 + fp8 开关 | 单元/mock 级；上线前必跑一次真机 |
| 50 系 | modern | 代码就绪、**未验证**（无硬件） | 同上（有卡时） |

> **2026-10-18 确认：当前没有任何 20/30/40/50 系真机。** 因此本方案交付时，这四代卡在文档中一律写“代码路径就绪、未验收”，不得写“支持”。
> 可提前做的替代验证：构建期 arch 断言、capability 模拟单测（7.5/8.6/8.9/12.0 四档）、modern 栈上的 import 级 op 契约一致性；这些能挡住“代码根本没适配”，但**挡不住**“真机 kernel 缺失/性能异常/出图变异”。

---

## 九、风险与回滚

| 风险 | 影响 | 应对 |
|---|---|---|
| py3.12 + numpy2 下 insightface / controlnet_aux 装不上 | 阻塞阶段 1 | 阶段 0 先试；失败则**先报告**（Python 3.12 已拍板），备选：自建 py3.12 base 编译 insightface / 把 face.analyze 找成可选依赖 |
| torch cu128 轮子缺某算力 | 该代卡不可用 | 构建期 arch 断言兜住；必要时换 cu126 基础镜像（覆盖 40 系，50 系另论） |
| bf16 在某些管道上与 fp16 输出不一致 | 出图差异 | 回归对比；对个别管道允许钉 `dtype=fp16` |
| `diffusers 0.41` 行为变更（VAE fp32、attention、视频链） | 回归面大 | 已拍板**直接新版**，不降版本；逐条定位修复，必要时对单条链路回退到等效实现 |
| 两栈并存导致执行器混乱 | 跑错栈 | compose 容器名/端口/执行器命名显式区分，DEPLOY 写清选择规则 |

**回滚**：pascal 镜像与 `requirements-pascal.txt` 全程不动；modern 出问题直接停容器、执行器切回 pascal，不影响现有业务。

---

## 十、已确认事项（2026-10-18 拍板）

| # | 事项 | 决议 |
|---|---|---|
| 1 | modern 栈 Python 版本 | **3.12**（kornia 0.8+ / 新 timm 基线） |
| 2 | modern 栈 diffusers | **直接 0.41.0**（不做过版本过渡） |
| 3 | fp8 | **纳入本次范围**（阶段 4，+2~3 天） |
| 4 | 目标真机 | **目前没有** ⇒ 四代卡只交“代码就绪、未验收”；上线前必须在目标卡补一次真机验收 |
| 5 | 文档同步 | **需要**：`DEPLOY.md` 增加镜像选择章节并与本文件互链（清单见 §十一） |

由第 2 项衍生的已知代价（接受）：diffusers 0.35.2 → 0.41.0 是大跨度，`VAE fp32` / 视频链 / attention 路径都要按“可能行为变更”对待，回归面按最坏情况估。

---

## 十一、文档同步清单（`DEPLOY.md`）

阶段 1 完成后，`DEPLOY.md` 需增加/修改：

1. **新增“镜像选择”一节**（放在“构建镜像”之前）：
   - 按显卡代次选 `Dockerfile.pascal` / `Dockerfile.modern` 的对照表（引用本文件 §二）；
   - 两个镜像的 tag 命名与容器名（`flowx-inference-server` vs `flowx-inference-server-modern`）；
   - 明确“同一台 daemon 可并存，但一个执行器只能指向一个栈”。
2. **修正现有 Pascal 段落**：把“Pascal 是唯一镜像”的表述改成“两栈之一”。
3. **环境变量表**补 `QUANTIZATION`（fp8）与 `compute_capability`/`default_dtype` 相关说明。
4. **排障**加三条：`no kernel image is available`（选错栈）、`enable_gqa` 相关报错（shim 门控失效）、fp8 在 sm<8.9 上被降级时的日志含义。
5. 文首文档地图加一条指向本文件。

---

## 十二、工期合计

| 阶段 | 内容 | 估时 |
|---|---|---|
| 0 | 取证（arch 列表 / 依赖可行性 / py3.12 装配方式） | 0.5 天 |
| 1 | modern 镜像三件套 + 两项构建期门禁 | 1~3 天 |
| 2 | 代码能力探测（shim 门控 + dtype 分档）+ 两栈 import 门禁 | 1 天 |
| 3 | 真机回归（有卡时） | 1~2 天 GPU 机时 |
| 4 | fp8（torchao + 显存账本 + 质量口径） | 2~3 天 |
| 5 | `DEPLOY.md` 与文档互链 | 0.5 天 |
| **合计** | | **6~10 天**（无目标真机时，阶段 3 顺延到有卡为止） |
