"""InstantID 算子（SDXL 身份保持生成，对标 ComfyUI InstantID 节点包）。

机制（InstantX 官方管线逆向核实；diffusers 0.35.2 无内置 InstantID 支持）：
- UNet 侧：ip-adapter.bin 的 ip_adapter 部分（to_k_ip/to_v_ip）走既有
  IPAdapter 注入路径（ops.sample 的 ipa 机制），insightface 512 维人脸特征
  经 image_proj（Resampler：512→16×2048）投影为 IP token 注入交叉注意力；
  CFG 负向 = 零特征（同官方 _encode_prompt_image_emb 的 zeros_like）。
- ControlNet 侧：InstantID ControlNet（cross_attention_dim=2048）的
  encoder_hidden_states **不是文本**，而是同一份投影后的人脸 token。
  注意官方语义：先把 [零特征, 人脸特征] 拼批再过 Resampler，然后切半——
  负向 token = resampler(zeros)（LayerNorm+bias 作用下 ≠ 零 token），
  本实现严格照此执行。controlnet_cond = 5 关键点图（官方 draw_kps 布局）。
- added_cond_kwargs（text_embeds/time_ids）沿用 SDXL 常规链路
  （InstantID ControlNet config 为 addition_embed_type="text_time"）。

权重（复用既有加载算子，零新加载代码）：
- ipadapter.load：IPADAPTER_DIR/instantid-ip-adapter.bin
  （嵌套 {image_proj, ip_adapter}，InstantX/InstantID 官方权重）
- controlnet.load：MODELS_DIR/instantid-controlnet/（diffusers 组件目录）
- insightface 模型包：INSIGHTFACE_HOME（默认 /models/insightface）下的
  models/antelopev2/（antelopev2.zip 官方包解出；运行期零网络）

License 注意：InstantID 权重官方声明仅限研究/非商用（代码 Apache-2.0）。
"""
import math
import os

import torch

from app import ops as _core_ops  # 插件可见性约定：可复用核心原语

# ---------------------------------------------------------------------------
# Resampler（perceiver 投影器）——移植自 InstantX/InstantID 仓库
# ip_adapter/resampler.py（Apache-2.0），保持键名与官方权重严格一致。
# 不用 diffusers IPAdapterPlusImageProjection：那是 load_ip_adapter 内部
# 转换格式（键重映射），我们要直接吃官方 state dict。
# 类经 _resampler_cls() 惰性构造：本文件按插件契约需在无 torch 环境可导入
# （模块级仅 import torch；torch.nn 子模块本地契约测试桩不住）。
# ---------------------------------------------------------------------------

_RESAMPLER_CLS = None


def _resampler_cls():
    global _RESAMPLER_CLS
    if _RESAMPLER_CLS is not None:
        return _RESAMPLER_CLS
    from torch import nn

    def _feed_forward(dim, mult=4):
        inner = int(dim * mult)
        return nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, inner, bias=False),
            nn.GELU(),
            nn.Linear(inner, dim, bias=False),
        )

    def _reshape_heads(x, heads):
        bs, length, width = x.shape
        return x.view(bs, length, heads, -1).transpose(1, 2).reshape(
            bs, heads, length, -1)

    class _PerceiverAttention(nn.Module):
        def __init__(self, dim, dim_head=64, heads=8):
            super().__init__()
            self.dim_head = dim_head
            self.heads = heads
            inner = dim_head * heads
            self.norm1 = nn.LayerNorm(dim)
            self.norm2 = nn.LayerNorm(dim)
            self.to_q = nn.Linear(dim, inner, bias=False)
            self.to_kv = nn.Linear(dim, inner * 2, bias=False)
            self.to_out = nn.Linear(inner, dim, bias=False)

        def forward(self, x, latents):
            x = self.norm1(x)
            latents = self.norm2(latents)
            b, l, _ = latents.shape
            q = self.to_q(latents)
            k, v = self.to_kv(torch.cat((x, latents), dim=-2)).chunk(2, dim=-1)
            q = _reshape_heads(q, self.heads)
            k = _reshape_heads(k, self.heads)
            v = _reshape_heads(v, self.heads)
            scale = 1 / math.sqrt(math.sqrt(self.dim_head))
            w = (q * scale) @ (k * scale).transpose(-2, -1)
            w = torch.softmax(w.float(), dim=-1).type(w.dtype)
            out = (w @ v).permute(0, 2, 1, 3).reshape(b, l, -1)
            return self.to_out(out)

    class _Resampler(nn.Module):
        """InstantX Resampler（官方 __init__ 参数同名，forward 逐行一致）。"""

        def __init__(self, dim=1024, depth=8, dim_head=64, heads=16,
                     num_queries=8, embedding_dim=768, output_dim=1024,
                     ff_mult=4):
            super().__init__()
            self.latents = nn.Parameter(torch.randn(1, num_queries, dim)
                                        / dim ** 0.5)
            self.proj_in = nn.Linear(embedding_dim, dim)
            self.proj_out = nn.Linear(dim, output_dim)
            self.norm_out = nn.LayerNorm(output_dim)
            self.layers = nn.ModuleList([
                nn.ModuleList([
                    _PerceiverAttention(dim=dim, dim_head=dim_head,
                                        heads=heads),
                    _feed_forward(dim, mult=ff_mult),
                ]) for _ in range(depth)
            ])

        def forward(self, x):
            latents = self.latents.repeat(x.size(0), 1, 1)
            x = self.proj_in(x)
            for attn, ff in self.layers:
                latents = attn(x, latents) + latents
                latents = ff(latents) + latents
            return self.norm_out(self.proj_out(latents))

    _RESAMPLER_CLS = _Resampler
    return _Resampler


def _build_resampler(sd):
    """从 state dict 形状推断 Resampler 结构参数（不写死官方 1280/16/20，
    兼容未来变体）；strict 加载，键不符直接报错。"""
    import re
    try:
        lat = sd["latents"]               # [1, num_queries, dim]
        proj_in = sd["proj_in.weight"]    # [dim, embedding_dim]
        proj_out = sd["proj_out.weight"]  # [output_dim, dim]
    except KeyError as e:
        raise ValueError(
            f"InstantID image_proj 缺键 {e}（需含 latents/proj_in/proj_out，"
            f"即 InstantX Resampler 结构）")
    depth = 1 + max(int(m.group(1)) for k in sd
                    for m in [re.match(r"layers\.(\d+)\.", k)] if m)
    heads = sd["layers.0.0.to_q.weight"].shape[0] // 64
    r = _resampler_cls()(dim=lat.shape[2], depth=depth, dim_head=64,
                         heads=heads, num_queries=lat.shape[1],
                         embedding_dim=proj_in.shape[1],
                         output_dim=proj_out.shape[0], ff_mult=4)
    r.load_state_dict(sd)  # strict=True
    r.eval()
    return r


# ---------------------------------------------------------------------------
# face.analyze：insightface 人脸检测 + 身份特征 + 关键点图
# ---------------------------------------------------------------------------

_FACE_APP = None


def _face_app():
    """FaceAnalysis 惰性单例（onnxruntime CPU；模型包来自 INSIGHTFACE_HOME，
    运行期零网络）。"""
    global _FACE_APP
    if _FACE_APP is None:
        from insightface.app import FaceAnalysis  # 懒加载：无 GPU 契约测试可导入
        home = os.environ.get("INSIGHTFACE_HOME", "/models/insightface")
        app = FaceAnalysis(name="antelopev2", root=home,
                           providers=["CPUExecutionProvider"])
        app.prepare(ctx_id=-1, det_size=(640, 640))
        _FACE_APP = app
        print(f"[face.analyze] insightface antelopev2 ready (home={home})",
              flush=True)
    return _FACE_APP


_KPS_COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
               (255, 0, 255)]
_KPS_LIMBS = [(0, 2), (1, 2), (3, 2), (4, 2)]  # 眼/耳→鼻连接（官方 draw_kps）


def draw_kps(image, kps):
    """官方 draw_kps 等价实现：黑底、同输入尺寸；4 条眼鼻连接短棒（×0.6 调暗）
    + 5 个关键点圆（半径 10，五色）。kps: (5,2) 像素坐标。"""
    import cv2  # 懒加载
    import numpy as np
    w, h = image.size
    out = np.zeros([h, w, 3])
    kps = np.asarray(kps, dtype=np.float64)
    for a, b_ in _KPS_LIMBS:
        x = kps[[a, b_], 0]
        y = kps[[a, b_], 1]
        length = ((x[0] - x[1]) ** 2 + (y[0] - y[1]) ** 2) ** 0.5
        angle = math.degrees(math.atan2(y[0] - y[1], x[0] - x[1]))
        poly = cv2.ellipse2Poly((int(np.mean(x)), int(np.mean(y))),
                                (int(length / 2), 4), int(angle), 0, 360, 1)
        out = cv2.fillConvexPoly(out.copy(), poly, _KPS_COLORS[a])
    out = (out * 0.6).astype(np.uint8)
    for i, (x, y) in enumerate(kps):
        out = cv2.circle(out.copy(), (int(x), int(y)), 10, _KPS_COLORS[i], -1)
    from PIL import Image
    return Image.fromarray(out)


def face_analyze(image, face_index=-1, det_thresh=0.5):
    """Face Analyze：检测输入图人脸，输出 FACE 对象（512 维身份特征 +
    关键点图）。face_index：-1=最大脸（官方语义），0..N-1=按面积降序第 N 张。
    det_thresh：检测置信度阈值（默认 0.5；AI 生成的人脸置信度常仅
    0.35~0.45，回测出图时建议传 0.2）。
    人脸特征为 antelopev2 的原始 embedding（InstantID 训练尺度）；
    FACE 对象另存 normed_embedding 供余弦相似度回测（点积即余弦）。"""
    from PIL import Image
    if not isinstance(image, Image.Image):
        raise ValueError(
            f"face.analyze: image 需为单张 PIL 图像，got {type(image).__name__}")
    import cv2
    import numpy as np
    face_index = int(face_index)
    app = _face_app()
    dt = float(det_thresh)
    if not 0.01 <= dt <= 1.0:
        raise ValueError(f"face.analyze: det_thresh 需在 [0.01,1]，got {dt}")
    app.det_model.det_thresh = dt
    arr = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    faces = app.get(arr)
    if not faces:
        raise ValueError("face.analyze: 未检测到人脸（换张清晰正脸照，"
                         "或对 AI 生成图调低 det_thresh 到 0.2）")
    faces = sorted(faces, key=lambda f: (f.bbox[2] - f.bbox[0])
                   * (f.bbox[3] - f.bbox[1]), reverse=True)
    idx = 0 if face_index < 0 else face_index
    if idx >= len(faces):
        raise ValueError(
            f"face.analyze: face_index={face_index} 越界（共 {len(faces)} 张脸，"
            f"按面积降序）")
    f = faces[idx]
    # 官方用原始 embedding（非 normed_embedding！glintr100 原始模长 ~20，
    # InstantID 训练时吃的就是这个尺度；喂归一化特征会让身份信号弱 ~20x，
    # 真机实测相似度甚至低于基线）
    emb = torch.from_numpy(f.embedding.copy()).float().unsqueeze(0)  # [1,512]
    kps_img = draw_kps(image, f.kps)
    face = {"embeds": emb, "kps_image": kps_img,
            "count": len(faces), "bbox": [float(v) for v in f.bbox],
            "normed": torch.from_numpy(f.normed_embedding.copy()).float()}
    print(f"[face.analyze] faces={len(faces)} picked={idx} "
          f"bbox={[round(v) for v in face['bbox']]}", flush=True)
    return {"face": face, "kps": kps_img}


def face_similarity(image_a, image_b, face_index_a=-1, face_index_b=-1,
                    det_thresh=0.5):
    """Face Similarity：两张图各自取目标人脸（默认最大脸），返回身份余弦
    相似度（antelopev2 normed_embedding 点积）。验收/调试量化用：
    同人典型 >=0.5，陌生人典型 <0.3。任一侧未检出人脸明确报错。"""
    import cv2
    import numpy as np
    from PIL import Image
    app = _face_app()
    dt = float(det_thresh)
    if not 0.01 <= dt <= 1.0:
        raise ValueError(f"face.similarity: det_thresh 需在 [0.01,1]，got {dt}")
    app.det_model.det_thresh = dt
    embeds = []
    for img, fidx, tag in ((image_a, face_index_a, "a"), (image_b, face_index_b, "b")):
        if not isinstance(img, Image.Image):
            raise ValueError(
                f"face.similarity: image_{tag} 需为 PIL 图像，"
                f"got {type(img).__name__}")
        arr = cv2.cvtColor(np.asarray(img.convert("RGB")), cv2.COLOR_RGB2BGR)
        faces = app.get(arr)
        if not faces:
            raise ValueError(f"face.similarity: image_{tag} 未检测到人脸")
        faces = sorted(faces, key=lambda f: (f.bbox[2] - f.bbox[0])
                       * (f.bbox[3] - f.bbox[1]), reverse=True)
        idx = 0 if int(fidx) < 0 else int(fidx)
        if idx >= len(faces):
            raise ValueError(
                f"face.similarity: image_{tag} face_index={fidx} 越界"
                f"（共 {len(faces)} 张脸）")
        embeds.append(faces[idx].normed_embedding)
    sim = float(np.dot(embeds[0], embeds[1]))
    print(f"[face.similarity] cos={sim:.4f}", flush=True)
    return {"similarity": sim}


# ---------------------------------------------------------------------------
# instantid.apply：人脸特征 × IPAdapter（UNet 身份侧）× ControlNet（姿态侧）
# ---------------------------------------------------------------------------

class InstantIDModel(_core_ops.ModelRef):
    """instantid.apply 的 MODEL 产物（sample 靠 getattr(model,'ipa') 鸭子识别，
    与 ipadapter_ops.IPABundle 同构——不复用是避免插件间 import 依赖加载顺序）。"""

    def __init__(self, pipe, patches, ipa):
        super().__init__(pipe, patches)
        self.ipa = ipa


def instantid_apply(model, ipadapter, controlnet, face, weight=0.8,
                    cn_strength=0.8, start_percent=0.0, end_percent=1.0,
                    cn_start_percent=0.0, cn_end_percent=1.0):
    """InstantID Apply：FACE 身份特征 + InstantID 权重捆绑进采样链。
    输出 model（IPA 注入：人脸特征驱动身份）+ control（CN 注入：关键点图
    驱动面部位置/姿态），分别接 sample 的 model / control 端口。
    weight=IPA 身份强度（官方默认 0.8）；cn_strength=关键点控制强度
    （官方默认 0.8）；两对 start/end_percent 分别限定 IPA / CN 的生效步窗口。
    仅支持 SDXL 底模（InstantID 权重按 SDXL 训练，cross_attention_dim=2048）。"""
    pipe, patches = _core_ops.resolve_pipe(model)
    if not hasattr(pipe, "unet"):
        raise ValueError(
            f"instantid.apply: model 需为 SDXL 管道（带 unet），"
            f"got {type(pipe).__name__}")
    if getattr(pipe.unet.config, "cross_attention_dim", None) != 2048:
        raise ValueError(
            f"instantid.apply: InstantID 仅支持 SDXL 底模"
            f"（cross_attention_dim=2048），got "
            f"{getattr(pipe.unet.config, 'cross_attention_dim', None)}"
            f"（SD1.x/SD3 不可用）")
    if not isinstance(ipadapter, dict) or "state_dict" not in ipadapter:
        raise ValueError(
            "instantid.apply: ipadapter 需为 ipadapter.load 的产物"
            "（instantid-ip-adapter.bin）")
    sd = ipadapter["state_dict"]
    if "image_proj" not in sd or "latents" not in sd.get("image_proj", {}):
        raise ValueError(
            "instantid.apply: 该 IPAdapter 权重的 image_proj 不是 Resampler"
            "（缺 latents 键）——InstantID 需要 instantid-ip-adapter.bin，"
            "标准/plus 版 IPAdapter 请用 ipadapter.apply")
    if not isinstance(face, dict) or "embeds" not in face:
        raise ValueError("instantid.apply: face 需为 face.analyze 的产物")
    weight = float(weight)
    cn_strength = float(cn_strength)
    for nm, v in (("weight", weight), ("cn_strength", cn_strength)):
        if not 0.0 <= v <= 2.0:
            raise ValueError(f"instantid.apply: {nm} 需在 [0,2]，got {v}")

    # Resampler 投影：官方语义——先拼 [零特征, 人脸特征] 再整体过 Resampler
    # 后切半（负向 = resampler(zeros)，不是零 token）。CPU fp32 一次算完
    # （16 token，量小），存执行设备供采样期零成本复用。
    resampler = _build_resampler(sd["image_proj"])
    emb = face["embeds"].float()
    if emb.dim() != 2 or emb.shape[1] != 512:
        raise ValueError(
            f"instantid.apply: 人脸特征需为 [1,512]，got {tuple(emb.shape)}")
    device = _core_ops.exec_device_of(pipe)
    dtype = pipe.unet.dtype
    # Resampler 输入带 seq 维（[2,1,512]，同官方 reshape([1,-1,512]) 后拼批）
    with torch.no_grad():
        emb3 = emb.unsqueeze(1)
        toks = resampler(torch.cat([torch.zeros_like(emb3), emb3]))  # [2,16,2048]
    neg_tok = toks[0:1].to(device=device, dtype=dtype)
    pos_tok = toks[1:2].to(device=device, dtype=dtype)

    ctl = _core_ops.controlnet_apply(
        controlnet, face["kps_image"], strength=cn_strength,
        start_percent=float(cn_start_percent),
        end_percent=float(cn_end_percent))["control"]
    ctl.hidden_neg = neg_tok  # CN cross-attention 覆写（ops.sample 消费）
    ctl.hidden_pos = pos_tok

    # IPAdapter 侧人脸特征必须带 seq 维（[1,1,512]）：
    # MultiIPAdapterImageProjection 把 [b,num_images,seq,dim] 前两维合并后
    # 喂 Resampler，缺 seq 维会在 cat(x, latents) 处炸维度（真机实测）
    emb_dev = emb.to(device=device, dtype=dtype).unsqueeze(1)
    ipa = {"name": ipadapter.get("name", "?"),
           "state_dict": sd,
           "pos_embeds": emb_dev, "neg_embeds": torch.zeros_like(emb_dev),
           "weight": weight,
           "start_percent": float(start_percent),
           "end_percent": float(end_percent)}
    print(f"[instantid.apply] '{ipa['name']}' weight={weight} "
          f"window=[{start_percent},{end_percent}) cn_strength={cn_strength} "
          f"cn_window=[{cn_start_percent},{cn_end_percent}) "
          f"tokens={tuple(pos_tok.shape)}", flush=True)
    return {"model": InstantIDModel(pipe, patches, ipa), "control": ctl}


def register(registry):
    registry.register(
        "face.analyze",
        inputs={"image": "IMAGE", "face_index": "INT", "det_thresh": "FLOAT"},
        outputs={"face": "FACE", "kps": "IMAGE"},
        description="Face Analyze（InstantID 配套）：insightface antelopev2 "
                    "检测+识别（onnxruntime CPU，不占显存），输出 512 维身份"
                    "特征 + 5 关键点控制图；face_index=-1 取最大脸，"
                    "0..N-1 按面积降序选脸")(face_analyze)
    registry.register(
        "face.similarity",
        inputs={"image_a": "IMAGE", "image_b": "IMAGE",
                "face_index_a": "INT", "face_index_b": "INT",
                "det_thresh": "FLOAT"},
        outputs={"similarity": "FLOAT"},
        description="Face Similarity：两图目标人脸（默认各自最大脸）的身份余弦"
                    "相似度（antelopev2 normed_embedding 点积）——InstantID 验收/"
                    "调试量化：同人典型 >=0.5，陌生人典型 <0.3")(face_similarity)
    registry.register(
        "instantid.apply",
        inputs={"model": "MODEL", "ipadapter": "IPADAPTER",
                "controlnet": "CONTROL_NET", "face": "FACE",
                "weight": "FLOAT", "cn_strength": "FLOAT",
                "start_percent": "FLOAT", "end_percent": "FLOAT",
                "cn_start_percent": "FLOAT", "cn_end_percent": "FLOAT"},
        outputs={"model": "MODEL", "control": "CONTROL"},
        description="InstantID Apply：身份保持生成（SDXL 专用）——model 注入"
                    "人脸身份特征（weight，官方默认 0.8），control 注入关键点"
                    "姿态（cn_strength，官方默认 0.8）；ipadapter 用 "
                    "instantid-ip-adapter.bin，controlnet 用 "
                    "instantid-controlnet；输出接 sample 的 model/control 端口"
                    )(instantid_apply)
