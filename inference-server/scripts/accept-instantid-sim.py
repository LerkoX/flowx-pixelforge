#!/usr/bin/env python3
"""InstantID 身份相似度回测（容器内运行，accept-instantid.py t2b 的量化部分）。

用法（容器内）：
  python3 accept-instantid-sim.py /input/test_person.jpg /input/instantid_out.png
输出：余弦相似度（antelopev2 normed_embedding 点积）。
参考口径：同人 >= 0.5 为身份保持良好；0.35~0.5 可接受；< 0.25 基本失效。
"""
import sys

import cv2
import numpy as np
from insightface.app import FaceAnalysis


def largest_emb(app, path):
    img = cv2.imread(path)
    if img is None:
        raise SystemExit(f"读不到图像 {path}")
    faces = app.get(img)
    if not faces:
        raise SystemExit(f"{path} 未检出人脸")
    faces = sorted(faces, key=lambda f: (f.bbox[2] - f.bbox[0])
                   * (f.bbox[3] - f.bbox[1]), reverse=True)
    return faces[0].normed_embedding


def main():
    app = FaceAnalysis(name="antelopev2", root="/models/insightface",
                       providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(640, 640))
    e1 = largest_emb(app, sys.argv[1])
    e2 = largest_emb(app, sys.argv[2])
    sim = float(np.dot(e1, e2))
    verdict = "身份保持良好" if sim >= 0.5 else \
              "可接受" if sim >= 0.35 else "身份保持疑似失效"
    print(f"cosine similarity = {sim:.4f}  ({verdict})")


if __name__ == "__main__":
    main()
