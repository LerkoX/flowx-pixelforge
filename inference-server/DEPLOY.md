# inference-server 部署文档

FlowX 推理服务 = 远程能力运行时（FastAPI + diffusers），部署在 GPU 机器上，
为 FlowX 侧节点提供 ComfyUI 风格的文生图算子能力（`/ops`、`/op`、`/graph`）。

---

## 1. 环境要求

| 项目 | 要求 |
| --- | --- |
| 硬件 | NVIDIA GPU（SD1.5 单模型建议显存 ≥ 6 GB；多模型常驻按 `MAX_RESIDENT_MODELS` × 约 4 GB 估算） |
| 系统 | Linux x86_64（Ubuntu 20.04+ 推荐） |
| NVIDIA 驱动 | 支持 CUDA 12.1 的版本（`nvidia-smi` 可见 GPU） |
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

首次 `checkpoint.load` 调用时才会把模型载入显存（服务启动本身不加载模型）。

### 2.1 Pascal 显卡（GTX 10xx / Tesla P4/P40/P100，sm_60/61）

PyTorch 2.7+（CUDA 12.8+ 构建）的官方二进制已移除 Pascal 支持，标准镜像在 Pascal 上推理会报
`no kernel image is available`。使用仓库自带的 Pascal 兼容镜像（torch 2.3.1 + cuDNN 8）：

```bash
docker compose -f docker-compose.yml -f docker-compose.pascal.yml up -d --build
```

说明：

- `Dockerfile.pascal` 基于 `pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime`，`requirements-pascal.txt` 固定 `torch==2.3.1`，防止 pip 升级到不含 Pascal 支持的新版本
- 应用代码只用 fp16，无 bf16 / flash-attn / xformers 依赖，Pascal 上无需改动
- 宿主机驱动需 >= 525（CUDA 12.1）。旧驱动（>= 450）的机器把 `Dockerfile.pascal` 里的基础镜像换成 `pytorch/pytorch:2.3.1-cuda11.8-cudnn8-runtime` 即可（镜像内注释有说明）
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
| `INFERENCE_TOKEN` | 空 | **非空则启用 Bearer 鉴权**，公网暴露时强烈建议设置 |
| `OBJECT_TTL_SECONDS` | `3600` | 对象仓库（中间张量）的 TTL |
| `INPUT_DIR` | `/input` | image.load 算子读取服务端本地图片的目录（compose 默认挂载 `./input`） |
| `VIDEO_DIR` | `/videos` | VIDEO 对象 mp4 落盘目录（compose 默认挂载 `./videos`），`GET /videos/{id}` 下载 |
| `MAX_UPLOAD_MB` | `32` | POST /images 上传体积上限 |
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

## 5. 常用运维操作

```bash
# 查看日志
docker compose logs -f inference

# 重启 / 停止
docker compose restart
docker compose down

# 更新代码后重新构建
docker compose up -d --build

# 新增模型：把文件（或 diffusers 目录）放进 models/ 后无需重启，下次 checkpoint.load 即用
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
  -d '{"name": "checkpoint.load", "inputs": {"ckpt": "v1-5-pruned-emaonly.safetensors"}}'
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
| `checkpoint.load` 报文件不存在 | 确认文件名（含扩展名）与 `models/` 内一致；容器内路径是 `/models` |
| 401 invalid token | `INFERENCE_TOKEN` 设置后，请求头必须是 `Authorization: Bearer <token>`（注意 Bearer 后一个空格） |
| 构建拉取基础镜像慢 | `pytorch/pytorch` 镜像约 8 GB，可预先 `docker pull`，或配置镜像加速器 |
| 首次推理很慢 | 属正常：首次调用才加载模型到显存；之后命中常驻缓存 |
| `no kernel image is available`（Pascal 显卡） | 镜像内 torch 版本不支持 sm_60/61，改用 Pascal 镜像，见 2.1 节 |

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
