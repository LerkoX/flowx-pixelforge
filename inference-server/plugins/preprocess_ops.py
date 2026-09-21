"""ControlNet 预处理器算子（第三档批次 2，插件化交付）。

- preprocess.canny：cv2.Canny 边缘线稿（黑底白线，ControlNet canny 出图惯例）
- preprocess.openpose：OpenPose 骨架图（controlnet_aux，body 常驻，hand/face 可选）

用途：闭环"人物替换保动作"——load-image(真人照片) → preprocess.openpose
→ controlnet.apply(openpose 权重) → sample(换服装/风格提示词)，姿势由骨架锁定。

部署：本文件放 PLUGINS_DIR（默认 /models/plugins.d，bind-mount 持久化）重启自动
扫描注册，或经 POST /admin/plugins 热上传。使用：经 inference-op 通用节点调用。

依赖：opencv-python-headless（canny）、controlnet_aux（openpose），均烘焙进镜像；
numpy/cv2/controlnet_aux 一律函数内懒加载（本地无 GPU 契约测试可导入）。
权重：openpose 预处理器从 PREPROCESSOR_MODELS_DIR（默认 /models/preprocessors）
本地加载，不访问网络（宿主机直下 body_pose_model.pth / hand_pose_model.pth）。
"""
import os

from PIL import Image

PREPROC_DIR = os.environ.get("PREPROCESSOR_MODELS_DIR", "/models/preprocessors")


def _as_image(image, op_name):
    if not isinstance(image, Image.Image):
        raise ValueError(
            f"{op_name}: image 需为 IMAGE 对象，got {type(image).__name__}")
    return image


def preprocess_canny(image, low_threshold=100, high_threshold=200):
    """Canny 边缘检测：IMAGE → 黑底白线线稿 IMAGE（ControlNet canny 提示图惯例）。
    low/high_threshold 为 cv2.Canny 双阈值（0-255，low < high）；线条太密调高 low，
    断线太多调低 low。"""
    img = _as_image(image, "preprocess.canny")
    low, high = int(low_threshold), int(high_threshold)
    if not (0 <= low < high <= 255):
        raise ValueError(
            f"preprocess.canny: 需 0 <= low < high <= 255，got ({low},{high})")
    import cv2  # 懒加载
    import numpy as np
    arr = np.asarray(img.convert("RGB"))
    edges = cv2.Canny(arr, low, high)  # 单通道 0/255
    out = Image.fromarray(edges).convert("RGB")
    ratio = float((np.asarray(edges) > 0).mean())
    print(f"[preprocess.canny] {img.size} thr=({low},{high}) "
          f"edge_ratio={ratio:.3f}", flush=True)
    return {"image": out}


_openpose_detector = None


def _get_openpose():
    """openpose 检测器进程级单例（权重常驻内存 ~360MB，避免每次调用重载）。"""
    global _openpose_detector
    if _openpose_detector is None:
        from controlnet_aux import OpenposeDetector  # 懒加载
        path = os.path.join(PREPROC_DIR, "openpose")
        if not os.path.isdir(path):
            raise FileNotFoundError(
                f"preprocess.openpose: 预处理器权重目录不存在：{path}"
                f"（需 body_pose_model.pth / hand_pose_model.pth）")
        _openpose_detector = OpenposeDetector.from_pretrained(
            path, local_files_only=True)
        print(f"[preprocess.openpose] 检测器已加载：{path}", flush=True)
    return _openpose_detector


def preprocess_openpose(image, include_hand=False, include_face=False):
    """OpenPose 骨架检测：IMAGE(真人照片) → 黑底彩色骨架图 IMAGE。
    include_hand/include_face 为可选精化（hand 权重已下载；face 需要额外
    facenet 权重，未下载时传 true 会报错）。"""
    img = _as_image(image, "preprocess.openpose")
    det = _get_openpose()
    out = det(img.convert("RGB"),
              include_hand=bool(include_hand),
              include_face=bool(include_face))
    if not isinstance(out, Image.Image):
        import numpy as np
        out = Image.fromarray(np.asarray(out))
    print(f"[preprocess.openpose] {img.size} -> {out.size} "
          f"hand={bool(include_hand)} face={bool(include_face)}", flush=True)
    return {"image": out.convert("RGB")}


def register(registry):
    registry.register(
        "preprocess.canny",
        inputs={"image": "IMAGE", "low_threshold": "INT",
                "high_threshold": "INT"},
        outputs={"image": "IMAGE"},
        description="Canny 边缘检测：IMAGE → 黑底白线线稿 IMAGE（ControlNet "
                    "canny 提示图）；low/high_threshold 为 cv2 双阈值 "
                    "（默认 100/200）")(preprocess_canny)
    registry.register(
        "preprocess.openpose",
        inputs={"image": "IMAGE", "include_hand": "BOOL",
                "include_face": "BOOL"},
        outputs={"image": "IMAGE"},
        description="OpenPose 骨架检测：真人照片 → 黑底彩色骨架图（ControlNet "
                    "openpose 提示图）；include_hand/include_face 可选精化，"
                    "权重从 /models/preprocessors/openpose 本地加载")(preprocess_openpose)
