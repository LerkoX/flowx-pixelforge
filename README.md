# flowx-txt2img

FlowX Studio 上的 ComfyUI 风格文生图/图生图方案：真实推理（diffusers）+ 常驻推理服务。

```
文生图：Checkpoint Loader → CLIP Text Encode(正/反) → Empty Latent → KSampler → VAE Decode → Save Image
图生图：Checkpoint Loader → CLIP Text Encode(正/反) → Load Image → VAE Encode → KSampler(denoise<1) → VAE Decode → Save Image
```

## 架构

```
Termux (FlowX Server, 本地 executor)
  └─ 11 个节点 = Python 瘦客户端（stdlib HTTP，零依赖）
          │ HTTP（节点间只传对象 ID，张量驻留 GPU 内存）
远程 GPU 机器
  └─ Docker: flowx-inference-server (FastAPI + diffusers)
      ├─ 算子注册表 + 图执行引擎（/op /graph，ComfyUI 式能力运行时）
      ├─ 模型缓存：checkpoint 常驻显存，LRU 淘汰（MAX_RESIDENT_MODELS）
      └─ 对象仓库：MODEL/CLIP/VAE/COND/LATENT/IMAGE 以 UUID 登记，TTL 1h
```

## 目录

| 路径 | 说明 |
| --- | --- |
| `inference-server/` | 推理服务 = 远程能力运行时（FastAPI + diffusers），含 Dockerfile / docker-compose.yml，部署见 [inference-server/DEPLOY.md](inference-server/DEPLOY.md) |
| `nodes/` | 11 个 FlowX 节点包：`inference-ensure` / `checkpoint-loader` / `lora-loader` / `clip-text-encode` / `empty-latent` / `load-image` / `vae-encode` / `ksampler` / `vae-decode` / `save-image` / `inference-op`（通用算子调用） |
| `pipeline/txt2img.yaml` | 文生图 workflow |
| `pipeline/i2i.yaml` | 图生图 workflow（LoadImage → VAEEncode → KSampler denoise<1） |
| `pipeline/txt2img-lora.yaml` | 文生图 + LoRA workflow（LoadCheckpoint → LoraLoader → 后续链路） |

## GPU 机器部署

```bash
cd inference-server
mkdir -p models loras && cp /path/to/v1-5-pruned-emaonly.safetensors models/
# LoRA 文件放 loras/（可选），在 workflow 里用 lora_name 引用
# 可选：编辑 docker-compose.yml 设置 INFERENCE_TOKEN
docker compose up -d --build
curl http://127.0.0.1:8100/health   # 确认 status=ok, cuda_available=true
```

Pascal 显卡（GTX 10xx / Tesla P 系列）改用兼容镜像（见 DEPLOY.md 2.1 节）：

```bash
docker compose -f docker-compose.yml -f docker-compose.pascal.yml up -d --build
```

前置要求：NVIDIA 驱动 + nvidia-container-toolkit；防火墙开放 8100 端口（或走 SSH tunnel：
`ssh -L 8100:127.0.0.1:8100 user@gpu-host`，然后 service_url 用 `http://127.0.0.1:8100`）。

## FlowX 侧操作（本仓库 Termux 环境已完成）

```bash
# 节点导入（改代码后重新 import 同名同版本需 --overwrite）
flowx-studio node import --type folder --path nodes/<name> [--overwrite]
flowx-studio node mock --id <N>          # 单节点 mock 测试（无需 GPU）

# workflow
flowx-studio workflow create --name sd-txt2img --file pipeline/txt2img.yaml
flowx-studio workflow run --id <N> --follow
# 图像输出在 SaveImage 节点日志/metadata 的 file_path（默认 ~/flowx-output/）
```

运行前编辑 `pipeline/txt2img.yaml`（图生图用 `pipeline/i2i.yaml`）的
`Param.service_url` / `ckpt_name` 指向你的 GPU 机器与模型
（改 Param 后需 `workflow update --id <N> --file pipeline/txt2img.yaml`）。

## 能力运行时（远程 ComfyUI 公共能力层）

服务端**没有固定功能端点**，所有计算都通过算子注册表 + 图执行引擎暴露（全部 FlowX 节点内部也都是调 `/op`）：

- `GET /ops`：列出全部已注册算子（名称/输入输出端口类型/描述）
- `POST /op`：单算子调用。对象端口用 `{"$id": "uuid"}` 引用，字面量直接传值
- `POST /graph`：整图执行（对齐 ComfyUI `/prompt` 语义）。对象端口用 `["节点ID", "端口"]` 引用，
  引擎做拓扑排序、类型校验、逐算子调度，并按「输入内容哈希」缓存中间结果——
  改一个 seed 重跑时上游编码/加载全部命中缓存
- 系统端点：`/health`（健康检查 + GPU/显存上报）、`POST /images`（上传图片登记为 IMAGE 对象，供图生图）、`/images/{id}`（PNG 下载，`?thumb=256` 取 JPEG 缩略图）、`/gc`（清对象仓库 + 执行缓存）、`/models`（常驻模型列表）

**新增能力 = 在 `app/main.py` 注册一个算子函数**（纯算法，不碰网络/ID），
FlowX 侧用通用节点 `inference-op` 即可立即使用，无需新增节点包；
高频能力再补专属瘦节点（参数面板 + 画布 UI 更友好）。

对象类型系统（对齐 ComfyUI）：`MODEL` / `CLIP` / `VAE` / `COND` / `LATENT` / `IMAGE`，
字面量类型：`INT` / `FLOAT` / `STRING` / `BOOL`。

已注册算子：`checkpoint.load` / `lora.apply`（ModelRef 补丁视图，可串联叠加）/
`clip.encode` / `latent.empty` / `image.load`（INPUT_DIR 服务端本地图）/ `vae.encode`（图生图入口）/
`sample`（自动识别裸 pipe 或带补丁的 ModelRef，denoise<1 即图生图）/ `vae.decode`

引擎测试（无 GPU 可跑）：`cd inference-server && python3 tests/test_engine.py`

## 采样器支持

`euler` / `euler_a` / `ddim` / `lms` / `dpmpp_2m` / `dpmpp_2m_karras` / `dpmpp_2m_sde` / `uni_pc`

## 后续演进

完整演进计划见 [inference-server/ROADMAP.md](inference-server/ROADMAP.md)（对标 ComfyUI 的 8 阶段路线）。

- 多模型：服务端已内置 LRU 常驻缓存，checkpoint.load 换 ckpt 名即可
- 多机器：在节点前加一层按 model 名路由的反向代理，节点零改动
- 图生图 / ControlNet：`app/ops.py` 加函数 + `main.py` 注册一行，FlowX 侧用 `inference-op` 节点立即可用
