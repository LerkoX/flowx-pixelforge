# inference-server 部署文档

FlowX 推理服务 = 远程能力运行时（FastAPI + diffusers），部署在 GPU 机器上，
为 FlowX 侧节点提供 ComfyUI 风格的文生图算子能力（`/ops`、`/op`、`/graph`）。

---

## 1. 环境要求

| 项目 | 要求 |
| --- | --- |
| 硬件 | NVIDIA GPU（SD1.5 单模型建议显存 ≥ 6 GB；多模型常驻按 `MAX_RESIDENT_MODELS` × 约 4 GB 估算） |
| 系统 | Linux x86_64（Ubuntu 20.04+ 推荐） |
| NVIDIA 驱动 | **按显卡代次分档**：10 系/V100 ≥ 530（旧栈 cu121）或 ≥450（cu118 变体）；20/30/40/50 系 ≥ **580**（modern 栈 cu13.2）。详见 §2.0 |
| Docker | 20.10+ |
| nvidia-container-toolkit | 必须，Docker 容器访问 GPU |
| 磁盘 | 镜像约 8 GB + 模型文件（SD1.5 约 4 GB/个） |

### 1.1 安装 NVIDIA Container Toolkit（Ubuntu/Debian 示例）

```bash
# 确认宿主机驱动正常
nvidia-smi

curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt update && sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# 验证容器内 GPU 可见
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

### Windows（WSL2）

可在 Windows 上通过 WSL2 部署，WSL2 通过 GPU 半虚拟化（GPU-PV）共享宿主机的 NVIDIA GPU。

前置条件：

1. Windows 10 21H2+ / Windows 11，`wsl --install` 装好 WSL2（Ubuntu 发行版）
2. **只在 Windows 宿主侧装/更新 NVIDIA 驱动**（支持 WSL 的版本，官网最新即可）；WSL 内**不要**装 Linux NVIDIA 驱动
3. WSL 内验证：`nvidia-smi` 可见 GPU、`/dev/dxg` 存在

Docker 二选一：

- **Docker Desktop**（推荐）：开启 "Use the WSL 2 based engine"，并在 Settings → Resources → WSL Integration 里启用你的发行版。容器内 GPU 开箱可用
- **WSL 内原生 Docker**：按 1.1 节安装 nvidia-container-toolkit

之后部署步骤与 Linux 完全一致（`docker compose up -d --build`）。Windows 浏览器直接访问 `http://localhost:8100`（WSL2 的 localhost 与 Windows 互通）。

注意：

- WSL2 默认只用宿主机一半内存，如需更多在 `C:\Users\<用户名>\.wslconfig` 配置 `[wsl2] memory=16GB` 后 `wsl --shutdown` 重启
- 模型文件放 WSL 原生路径（如 `~/models`），**不要**放 `/mnt/c/...`（跨文件系统 IO 极慢）
- 推理性能损失很小（约 5% 内），但显存与 Windows 图形系统共享，避免同时跑大型游戏等吃 GPU 的程序

---

## 2. 部署步骤

### 2.0 一键安装（推荐，自动识别显卡选栈）

`install.sh` 会探测显卡与驱动 → 自动选「栈」→ 拉取/构建镜像或在本机装依赖 → 起服务 →
**跑健康验收**（断言 torch/CUDA 版本与栈一致）。支持 **10/20/30/40/50 全系**，两种形态：

```bash
cd inference-server

# ① 先空跑，看清判定结果与将执行的命令（不改动系统）
bash install.sh --dry-run

# ② 容器模式（默认，Linux / WSL2）
bash install.sh

# ③ 裸机模式（micromamba 自包含 Python + systemd 常驻，不动系统 python）
sudo bash install.sh --mode native

# ④ GPU 在另一台机器（远程 docker daemon）
bash install.sh --docker-host tcp://192.168.5.60:2375
```

#### 两栈矩阵（为什么不是一个版本通吃）

CUDA 13.x 的 release notes 明确 **移除了 Maxwell/Pascal/Volta**（`Dropped support for
pre-Turing architectures`），所以 10 系与 V100 永远上不了 CUDA 13；而 **50 系（sm_120）
只存在于 CUDA 13**。只能在构建层分两套依赖栈：

| 栈 | 显卡 | 算力 | Python | torch | CUDA / cuDNN | 最低驱动 | 镜像 tag |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **P** | 10 系 GTX 10xx、P4/P40/P100、V100 | sm_60/61/70 | 3.10 | 2.3.1 | 12.1 / 8 | ≥530 | `lerkobba/flowx-inference:p-2.3.1-cu121` |
| **P-lite** | 同上，但驱动只到 450~529 | sm_60/61/70 | 3.10 | 2.3.1+cu118 | 11.8 / 8 | ≥450 | `lerkobba/flowx-inference:p-2.3.1-cu118`（需 `--build`） |
| **M** | 20/30/40/50 系、T4、A100、L40S、H100 | sm_75 ~ sm_120 | 3.12 | 2.13.0+cu132 | 13.2 / 9.20 | ≥ **580** | `lerkobba/flowx-inference:m-2.13.0-cu132` |
| **X** | Maxwell（GTX 9xx 及更老）；或 50 系 + 驱动 <580 | — | — | — | — | — | 脚本直接拒绝并给人话指引 |

分档**只认 compute capability + 驱动版本**，不认型号名：GTX 1650 是 sm_75 走 M 栈，
GTX 1080 是 sm_61 走 P 栈；多卡混插时按**算力最低**的那张定栈。

三个已取证的事实（实测于 2026-11）：

- modern 基础镜像 `pytorch/pytorch:2.13.0-cuda13.2-cudnn9-runtime`：
  `torch.cuda.get_arch_list() = ['sm_75','sm_80','sm_86','sm_90','sm_100','sm_120']`
  —— 无 `sm_60/61/70`（Pascal/Volta 不可用）、**无 `sm_89`**。
- 40 系（Ada, sm_89）靠 `sm_86` cubin 的 minor 版本兼容跑起来，代价是**本栈上拿不到 fp8**
  （fp8 需原生 sm_89 kernel）。要用 fp8 得换 cu128 变体自行编译。
- 驱动门槛：CUDA 13.x 最低 ≥580；CUDA 12.1 GA 需 **≥530.30.02**（§1 旧文档写的 525 是
  CUDA 12.0 的门槛，已修正）。

#### 脚本会做什么（8 步，幂等）

1. 环境闸门：Linux / WSL2 + x86_64（`--docker-host` 时校验 daemon 架构）。
2. 探卡：`nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version`
   （老驱动无 `compute_cap` 字段时按型号推断）。
3. 决策：输出栈 + 原因（例如「sm_86 且驱动 570 < 580，降级 P 栈，无 bf16/fp8」）。
4. 依赖：docker / nvidia-container-toolkit；缺则按 apt/dnf 安装（`--no-install-deps` 只打印）。
5. 数据：本地用绑定挂载目录；**远程 daemon 自动改用命名卷**（远端看不到本机路径）。
6. 令牌：生成随机 `INFERENCE_TOKEN`；按显存推算 `MAX_RESIDENT_MODELS` / `OFFLOAD_MODE`
   （<12GB → 1 + model；12~24GB → 2；≥24GB → 3）。
7. 启动：`docker run`（不依赖 compose 插件）或 systemd 服务 `flowx-inference`。
8. 验收：轮询 `/health` 断言 `cuda_available`、`torch_version` 与栈一致；
   失败时自动抓容器日志并翻译常见错误（`no kernel image` = 栈选错、
   `CUDA driver version is insufficient` = 驱动太老、`cuda_available=false` = toolkit 未装）。

安全默认：**已存在同名容器 / 已有 `/etc/flowx-inference/env` 时默认拒绝覆盖**（需 `--force`）；
卸载用 `--uninstall`，模型数据默认保留。

#### 常用参数

```bash
bash install.sh --port 8100 --token mytoken     # 自定义端口与令牌
bash install.sh --stack M                        # 强制某栈（跨栈迁移/排障用）
bash install.sh --max-resident 3 --offload none  # 覆盖显存参数
bash install.sh --image lerkobba/flowx-inference@sha256:... # 锁定到具体 digest
bash install.sh --build                          # 强制本地构建（不用预构建镜像）
bash install.sh --uninstall --keep-data          # 卸载，保留模型
```

> 只想手动 compose 部署？跳过本节，按 §2.1 起用 `docker-compose.yml` +
> `docker-compose.pascal.yml`（P 栈）或 `docker-compose.modern.yml`（M 栈）。

#### 自行构建与推送镜像

```bash
# 两个栈一起构建（P 栈若本地已有同栈基底镜像，走「秒级重打包」只覆盖 app/ 一层）
DOCKER_HOST=tcp://<host>:2375 bash scripts/build-images.sh

# 全量重建（自包含，慢，约 10 分钟/栈）
DOCKER_HOST=tcp://<host>:2375 bash scripts/build-images.sh --full

# 推送到 Docker Hub 并生成 IMAGES.lock（tag → digest）
DOCKERHUB_USER=lerkobba DOCKERHUB_TOKEN=<access-token> bash scripts/push-images.sh
```

### 2.1 手动部署：放模型并启动

```bash
cd inference-server

# 1. 放置模型文件
mkdir -p models loras
cp /path/to/v1-5-pruned-emaonly.safetensors models/
# 也支持 diffusers 目录格式：把含 model_index.json 的整个目录拷进 models/
# （视频模型 / 未来架构走这条路径，管道类按目录内容自动分派）
# LoRA 文件（可选）放到 loras/，在 workflow / 算子调用里用文件名引用

# 2.（可选）编辑 docker-compose.yml 调整环境变量，见第 3 节

# 3. 构建并启动
docker compose up -d --build

# 4. 验证
curl http://127.0.0.1:8100/health
# 期望返回：{"status": "ok", "cuda_available": true, ...}
```

首次 `sd.checkpoint.load` 调用时才会把模型载入显存（服务启动本身不加载模型）。

### 2.2 Pascal 显卡（GTX 10xx / Tesla P4/P40/P100 / V100，sm_60/61/70）

PyTorch 2.7+（CUDA 12.8+ 构建）的官方二进制已移除 Pascal 支持，标准镜像在 Pascal 上推理会报
`no kernel image is available`。使用仓库自带的 Pascal 兼容镜像（torch 2.3.1 + cuDNN 8）：

```bash
docker compose -f docker-compose.yml -f docker-compose.pascal.yml up -d --build
```

说明：

- `Dockerfile.pascal` 基于 `pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime`，`requirements-pascal.txt` 固定 `torch==2.3.1`，防止 pip 升级到不含 Pascal 支持的新版本
- 应用代码只用 fp16，无 bf16 / flash-attn / xformers 依赖，Pascal 上无需改动
- 宿主机驱动需 >= 530（CUDA 12.1 GA 实测门槛；>= 450 可用 cu118 变体）。旧驱动（>= 450）的机器把 `Dockerfile.pascal` 里的基础镜像换成 `pytorch/pytorch:2.3.1-cuda11.8-cudnn8-runtime` 即可（镜像内注释有说明）
- 8 GB 显存的卡：`MAX_RESIDENT_MODELS` 保留 3 即可 —— 淘汰已改为**显存预算式**
  （见下 `VRAM_BUDGET`），条目数只是兜底上限；真正决定"能不能同时常驻"的是可用显存

---

## 3. 配置项（环境变量）

在 `docker-compose.yml` 的 `environment` 段配置：

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `MODELS_DIR` | `/models` | checkpoint 目录（compose 默认挂载 `./models`） |
| `LORAS_DIR` | `/loras` | LoRA 目录（compose 默认挂载 `./loras`） |
| `MAX_RESIDENT_MODELS` | `2` | 常驻条目数**兜底上限**（超出按 LRU 淘汰）；`<=0` 表示不限条目数，只看显存预算 |
| `VRAM_BUDGET` | `1` | 显存预算式淘汰开关：加载前按"可用显存 vs 待加载模型体积估算"淘汰 LRU。设 `0` 退回纯条目数 LRU |
| `VRAM_RESERVE_MB` | `1024` | 预留余量（MB）：留给采样激活/中间张量/碎片；可用显存低于它即 `/health` 报 `vram_low` |
| `VRAM_LOAD_FACTOR` | `1.25` | 体积估算放大系数（权重之外的 CUDA 上下文/缓冲） |
| `OFFLOAD_MODE` | `none` | 显存治理档位：`none` 整模型驻留 / `model` 子模块级搬移 / `sequential` 逐层搬移（最省显存、吞吐最低，视频模型用）。旧开关 `ENABLE_CPU_OFFLOAD=1` 仍兼容（等价 `model`） |
| `QUANTIZATION` | `none` | 权重量化：`fp8` 接口已预留（当前版本识别配置但未实现，加载时告警并按原 dtype 继续） |
| `INFERENCE_TOKEN` | 无（必填） | **必填**：算子全面插件化（节点 server_op.py 经 /admin/plugins 自注册），无 token 服务启动即失败；同时作为 Bearer 鉴权令牌 |
| `OBJECT_TTL_SECONDS` | `3600` | 对象仓库（中间张量）的 TTL |
| `VRAM_STRICT` | `0` | **物理超订硬闸门**（exec 795）：`1` = 权重全量上卡物理上不可行（或已在超订态）时**拒绝加载**并给指引，不再把它交给 host memory 兜底（WSL2 下不报 OOM，采样会静默退化到分钟级/步）。可用 `POST /model/policy {"vram_strict":1}` 热改（重启即丢） |
| `OFFLOAD_VRAM_FLOOR_MB` | `1024` | `offload=model/sequential` 条目计入**显存预算**的 floor（MB）—— 权重在 CPU，不按全量占额（旧行为按全量记，SDXL 6616MB×1.25 > 整卡 8GB ⇒ 预算永远不满足 ⇒ 每次都把别的条目全淘汰） |
| `OBJECT_TTL_ORPHAN_SECONDS` | `1800` | 孤儿对象回收：条目所属**租约已结束**且龄期超此值 ⇒ 回收（其钉住的模型本体随之可淘汰）。`0` = 立即回收；**不设**（或未注入租约回调）= 关闭。长任务靠 job 进度心跳续租，不会被误回收 |
| `INPUT_DIR` | `/input` | image.load 算子读取服务端本地图片的目录（compose 默认挂载 `./input`） |
| `VIDEO_DIR` | `/videos` | VIDEO 对象 mp4 落盘目录（compose 默认挂载 `./videos`），`GET /videos/{id}` 下载 |
| `VIDEO_INPUT_DIR` | `$INPUT_DIR/videos` | **POST /videos** 上传原片的落盘目录（load-video 节点把本机视频送入服务端；与 VIDEO_DIR 分工：前者放"原片"，后者放"仓库产物"） |
| `MAX_UPLOAD_MB` | `32` | POST /images 上传体积上限 |
| `MAX_VIDEO_UPLOAD_MB` | `512` | POST /videos 上传体积上限（原片动辄数百 MB，故与图片分开） |
| `PORT` | `8100` | 容器内监听端口（在 Dockerfile ENV / compose 端口映射里改） |

启用鉴权后，所有业务端点需携带：

```bash
curl -H "Authorization: Bearer <token>" http://127.0.0.1:8100/ops
```

FlowX 侧节点把 `INFERENCE_TOKEN` 填进对应参数（`api_key` / token 字段）即可。

---

## 4. 网络暴露方式

| 场景 | 做法 |
| --- | --- |
| 同机 / 内网 | compose 已映射 `8100:8100`，防火墙放行 8100 端口即可 |
| 公网（推荐最小暴露） | **SSH 隧道**：`ssh -L 8100:127.0.0.1:8100 user@gpu-host`，FlowX 侧 `service_url` 用 `http://127.0.0.1:8100`；同时把 compose 端口改为 `127.0.0.1:8100:8100` 避免直接对外 |
| 公网直连 | 必须设置 `INFERENCE_TOKEN`，建议再套一层 HTTPS 反代（Caddy/Nginx） |

### 仅本机监听（compose 端口段改为）

```yaml
ports:
  - "127.0.0.1:8100:8100"
```

---

## 远程 Docker 执行器（Termux 侧踩坑，2026-10-13 实测）

节点镜像 `lerkobba/flowx-pixelforge-nodes` 在**远端 daemon**（Windows/Docker Desktop）上构建，
Studio 与 CLI 都要连 `cpolar` 的 dockerApi 隧道。两个坑：

1. **域名解析不了（真因）**：Termux 上 Go 的纯解析器解析不了 `5.tcp.cpolar.top`
   （`getent hosts` 也是空；系统解析器只给透明代理的 fake-IP `198.18.x.x`）。
   用域名连 docker 会报 `Cannot connect to the Docker daemon`，而 `curl http://域名:端口/_ping`
   却是 200（curl 走系统解析）。**修法：用真实 IP**：

   ```sh
   # 真实 IP（fake-IP 会变，CPOLAR 隧道换端口时用 DoH 重查）
   curl -s "https://dns.alidns.com/resolve?name=5.tcp.cpolar.top&type=A"
   # -> 8.162.7.178
   docker -H tcp://8.162.7.178:12952 version          # ✅
   ```

   Studio 侧：`/api/v1/executors` 里 `docker-cpolar` 的 `config.host` 也必须是真实 IP
   （`PUT /api/v1/executors/8`），改完 `POST /api/v1/executors/8/test` 应返回
   `serverVersion 27.3.1`。执行器 id 7（`docker-remote-211b`）的备注就是这条老经验。

2. **镜像构建**：`cd nodes && SKIP_PUSH=1 DOCKER_HOST=tcp://<真实IP>:<端口> bash build-image.sh`
   （必须 `bash`；脚本用了 `set -o pipefail`，`sh` 会报 Illegal option）。
   `SKIP_PUSH=1` = 远端 daemon 本地构建即可，无需推仓库。
   上下文只传节点代码（`.dockerignore` 挡掉 `ui/`/`mock.py`/`_tools/`），31MB ⇒ 隧道下几分钟。
   构建完核验镜像内容：

   ```sh
   docker -H tcp://<真实IP>:<端口> run --rm --entrypoint sh \
     lerkobba/flowx-pixelforge-nodes:vX.Y.Z -c \
     'ls -d /opt/flowx-nodes/image-shadow && grep -c submit_job /opt/flowx-nodes/flux-unet-load/main.py'
   ```

3. **换隧道端口时的清单**（cpolar 每次重建都会变）：执行器 `config.host`、各 pipeline YAML 的
   `service_url` / `service_url_internal` / `service_url_host`、容器内可达地址是
   `http://172.17.0.1:8100`（docker 节点用）。


## 5. 常用运维操作

### 监控页面

浏览器打开 `http://<host>:8100/monitor`（单文件页面，无需构建/额外部署），
右上角填入 `INFERENCE_TOKEN`（只存浏览器 localStorage，不经服务端）后自动每 2s 刷新：

- GPU：**四段显存分账条**（本进程已分配 / 缓存保留可回收 / 其它进程·驱动 / 空闲=轨道色）
  + 条外图例与明细（NVML 已用、torch reserved/allocated、可回收缓存、物理余量、显存预算）；
  SM 利用率/温度/功耗/风扇（经 NVML，`GET /gpu`；pynvml 不可用时自动降级只显示显存）；
  显存/利用率/功耗/温度/PCIe/时钟/系统资源各一张折线（服务端采样历史，`GET /metrics/history`）
- 常驻模型：体积估算/空闲时长/pin 与租约归属，单个卸载按钮
- 执行租约：持有模型、空闲时长、TTL 滑动倒计时（快过期红色提示）、单个释放
- 任务队列：worker 忙闲与队头等待汇总；每条含算子名（op 名 / graph·N 节点 + 算子清单）、
  状态、进度（`12/25 步 · 48%` + 当前节点 `flux.sampler · 3/5 节点`；加载阶段走流动条纹）、
  速度与 ETA（服务端 EMA `sec/步` → 前端差分 → 节点均值粗估）、租约、排队时长、耗时；
  done 任务按产出数出 `🖼 n / 🎬 n`，可弹层预览输出图/视频（带 Bearer 拉 blob），
  failed 可展开错误全文；可单个取消、可「只看未完成」
- OOM 自愈统计；运维按钮：/gc、卸载全部常驻模型、中断全部任务、释放全部租约
  （危险操作**页内二次确认**：按钮变「确认？」，3s 内再点生效；结果与时间写入按钮下方，
  卸载被租约/pin 跳过时逐条给原因与「释放租约」按钮）

> 监控页说明：`app/static/monitor.html` 随镜像 `COPY` 进容器，改页面需重建镜像才生效
> （免重启临时验证：`docker cp app/static/monitor.html flowx-inference-server:/app/app/static/monitor.html`）。

对应 API：`GET /gpu`（NVML 硬件指标，带 auth）是除 `/health` 外的新增数据源。

```bash
# 命令行等效：GPU 利用率/温度/功耗/风扇
curl -H "Authorization: Bearer $TOKEN" http://127.0.0.1:8100/gpu
# 执行租约观测
curl -H "Authorization: Bearer $TOKEN" http://127.0.0.1:8100/leases
```

### 命令行操作

```bash
# 查看日志
docker compose logs -f inference

# 重启 / 停止
docker compose restart
docker compose down

# 更新代码后重新构建
docker compose up -d --build

# 新增模型：把文件（或 diffusers 目录）放进 models/ 后无需重启，下次 sd.checkpoint.load 即用
# 下载模型（远程 daemon 也适用，走 hf-mirror；只拉 fp16 权重 + 配置）：
./scripts/download-model.sh svd    # SVD-XT 1.1 图生视频（~9.6GB）
./scripts/download-model.sh Org/some-model
# 查看当前常驻模型
curl -H "Authorization: Bearer $TOKEN" http://127.0.0.1:8100/models

# 手动清理对象仓库 + 图执行缓存（注意：/gc **不卸常驻模型**，只清缓存）
curl -X POST -H "Authorization: Bearer $TOKEN" http://127.0.0.1:8100/gc

# 显式卸载常驻模型腾显存（等价 model.unload 算子 / model-unload 节点）
curl -X POST -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d '{}' http://127.0.0.1:8100/model/unload          # {} = 全部
curl -X POST -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d '{"target":"stable-video-diffusion-img2vid-xt-1-1"}' \
  http://127.0.0.1:8100/model/unload                  # 指定名字（可省略扩展名）
```

### 5.1 显存治理（为什么不再需要手动重启容器）

淘汰不只看"条目个数"（`while len(pipes) >= MAX_RESIDENT`），而是**按显存预算**：
加载前用 `torch.cuda.mem_get_info()` 的可用量对比待加载模型的体积估算
（读 safetensors header 精确算，不是文件大小），不足就按 LRU 淘汰；正在被算子使用的
模型会被 pin 住不淘汰；OOM 时自动淘汰非在用条目并重试一次，仍失败则报错并给出
可执行指引（降档 offload / 卸载 / 调 reserve）。

历史坑与对应的行为约定：

| 现象 | 现在 |
| --- | --- |
| `MAX_RESIDENT_MODELS=3` + 视频模型 + 图像模型同池 ⇒ `vram_free_mb: 0`，采样从 0.5s/步 退化到 17s/步 | 加载时就按预算淘汰（视频模型进池不会把图像模型挤到 host memory） |
| `POST /gc` 清了缓存但模型照样占显存 | `/gc` 只管缓存；腾显存用 `/model/unload` |
| 采样后 `mem_get_info` 报 free=0（PyTorch 缓存分配器留着复用）被误判成"没空间" | `/health` 的 `vram_free_mb` = 驱动余量 + **可回收缓存**（`reserved-allocated`）；另给 `vram_driver_free_mb` / `vram_reclaimable_mb` 供诊断 |
| 两条 GPU 流水线**并发**跑同一张卡 | **不支持**：预算式淘汰会把对方正在用的模型顶掉（`on_evict` 会顺手断对象仓库视图），对方随后报 `object 'xxx' not found (expired or never created)`（实测）。服务端 `EXEC_LOCK` 只保证算子串行，不保证模型常驻。**一次只跑一条 GPU 流水线**；要多条并行请给服务配更大的卡或 `OFFLOAD_MODE=sequential` 后降低并发 |
| 出现分钟级/步只能靠人看日志发现 | `/health` 给 `vram_low`（可用量 < reserve 即 `status: degraded`）+ `resident_weights_mb` + `degraded`（OOM 自愈计数）；流水线可用 `inference-ensure` 的 `min_vram_mb` 在首节点挡住 |

## 6. 验证部署

```bash
# 1. 健康检查
curl http://127.0.0.1:8100/health

# 2. 算子注册表
curl http://127.0.0.1:8100/ops

# 3. 端到端冒烟：加载模型
curl -X POST http://127.0.0.1:8100/op \
  -H "Content-Type: application/json" \
  -d '{"name": "sd.checkpoint.load", "inputs": {"ckpt": "v1-5-pruned-emaonly.safetensors"}}'
```

无 GPU 的本地开发环境可跑引擎单元测试：

```bash
cd inference-server && python3 tests/test_engine.py && python3 tests/test_jobs.py
```

### 6.1 异步任务（分钟级任务通道，视频生成用）

同步 `/graph`、`/op` 保持不变；长任务走异步任务体系：

```bash
# 1. 提交（body 同 /graph；也支持 {"name", "inputs"} 单算子任务）
curl -X POST http://127.0.0.1:8100/jobs \
  -H "Content-Type: application/json" \
  -d '{"nodes": {...}}'
# → {"job_id": "...", "status": "pending"}

# 2. 轮询状态/进度（done 时 result 与 /graph 返回同构）
curl http://127.0.0.1:8100/jobs/<job_id>
# → {"status": "running", "progress": {"current": 12, "total": 20, "percent": 60.0}, ...}

# 3. 取消：指定 job_id 取消单个；空 body 取消当前 running + 全部 pending
curl -X POST http://127.0.0.1:8100/interrupt -H "Content-Type: application/json" \
  -d '{"job_id": "..."}'
curl -X POST http://127.0.0.1:8100/interrupt   # 全部取消
```

任务状态机：`pending → running → done / failed / cancelled`。任务串行执行
（GPU 是串行资源，与同步调用经全局执行锁互斥）；取消是协作式的，
running 任务在下一个检查点（图节点间 / 采样每步）生效。

---

## 7. 故障排查

| 现象 | 排查 |
| --- | --- |
| `cuda_available: false` | 宿主机 `nvidia-smi` 是否正常 → toolkit 是否安装并 `restart docker` → compose 里 `deploy.resources` 段是否保留 |
| 启动时报 OOM / CUDA out of memory | 先用 `POST /model/unload {}` 卸载常驻模型；仍不够再 `OFFLOAD_MODE=model`（还不够上 `sequential`）/ 降低 `VRAM_RESERVE_MB`；确认没有其他进程占显存 |
| 报错 "CUDA out of memory in op '...'：已淘汰全部可淘汰的非在用模型仍不足" | 这是自愈失败后的指引：降分辨率/步数、`offload=model|sequential`、卸载其他模型后重试 |
| `sd.checkpoint.load` 报文件不存在 | 确认文件名（含扩展名）与 `models/` 内一致；容器内路径是 `/models` |
| 401 invalid token | `INFERENCE_TOKEN` 设置后，请求头必须是 `Authorization: Bearer <token>`（注意 Bearer 后一个空格） |
| 构建拉取基础镜像慢 | `pytorch/pytorch` 镜像约 8 GB，可预先 `docker pull`，或配置镜像加速器 |
| 首次推理很慢 | 属正常：首次调用才加载模型到显存；之后命中常驻缓存 |
| `no kernel image is available`（任何卡） | **栈选错了**：Pascal 用了 modern 镜像，或 50 系用了旧栈。跑 `bash install.sh --dry-run` 看判定，或对镜像执行 `python -c "import torch;print(torch.cuda.get_arch_list())"` 对照 |
| `CUDA driver version is insufficient for CUDA runtime version` | 宿主机驱动低于该栈门槛：cu13.2 需 ≥580、cu12.1 需 ≥530、cu11.8 需 ≥450。升级驱动后重跑 `install.sh`（脚本会自动改判到能用的栈） |
| 40 系开了 `QUANTIZATION=fp8` 但没生效 | modern 镜像的 arch 列表只到 `sm_86`/`sm_120`，没有 `sm_89`，40 系走的是 sm_86 cubin —— **fp8 在 40 系上不可用**（需原生 sm_89 kernel）。这不是 bug |
| `install.sh` 报「已存在容器 X」 | 默认不覆盖生产容器。确认要重建（模型卷保留）后加 `--force` |
| 远程 daemon 上模型目录是空的 | 远程 daemon 看不到本机路径，脚本已自动改用命名卷 `flowx-inference-models`；模型要用 `scripts/download-model.sh`（`TARGET=flowx-inference-models ...`）灌进卷里 |

---

## 8. 架构说明（供二次开发参考）

```
Docker 容器 flowx-inference-server（uvicorn + FastAPI）
├─ app/main.py          端点 + 算子注册（新能力在这里加一行）
├─ app/jobs.py          异步任务：FIFO 队列 + 单 worker + 进度/取消
├─ app/execution.py     执行上下文：线程本地 job 绑定 + 取消检查点
├─ app/ops.py           算子实现（纯算法，不碰网络/对象 ID）
├─ app/engine.py        /graph 图执行引擎：拓扑排序 + 输入哈希缓存 + 执行锁
├─ app/model_manager.py checkpoint/LoRA 加载，LRU 常驻缓存，按嗅探结果分派管道类
├─ app/offload.py       显存治理配置：OFFLOAD_MODE（none/model/sequential）+ fp8 量化接口
├─ app/vram.py          显存预算计算（体积估算 + 淘汰决策，纯 stdlib 可单测）
├─ app/sniff.py         模型架构嗅探：safetensors 头部 key / diffusers 目录 model_index.json
├─ app/object_store.py  MODEL/CLIP/VAE/COND/LATENT/IMAGE/VIDEO 对象仓库（UUID，TTL；VIDEO 落盘）
├─ app/video.py         VIDEO 对象 mp4 编码（imageio-ffmpeg 内置静态 ffmpeg）
└─ app/registry.py      算子注册表（名称/端口类型/描述）
```

- 节点间只传对象 ID，张量驻留 GPU 内存；`?thumb=256` 可取 JPEG 缩略图
- 采样器（sampler_name × scheduler 自由组合）：`euler` / `euler_a` / `ddim` / `lms` / `dpmpp_2m` / `dpmpp_2m_sde` / `uni_pc` × `normal` / `karras` / `exponential` / `beta`（euler_a、ddim 仅 normal；兼容旧名 `dpmpp_2m_karras`）
- 多机器扩展：在 FlowX 节点前加一层按模型名路由的反向代理即可，节点零改动
