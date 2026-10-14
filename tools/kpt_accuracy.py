#!/usr/bin/env python3
"""关键点精度诊断：自训模型输出的四个角点 vs rm_sim 真值角点，逐块比对。

为什么单独测这个：框的 IoU 好（0.887）不代表角点好。角点才是喂给 solvePnP 的东西，
角点偏几个像素换来的就是瞄准角的系统性偏差 —— 表现出来是"检测召回高了，命中率反而降"。
真值四点的顺序和 solver.cpp 的 SMALL_ARMOR_POINTS 一致，模型也是按这个顺序训的，
所以这里只做一次 sort_keypoints 式的重排（图像系 左上/右上/右下/左下）再比。
"""
import glob
import json
import os
import sys

import cv2
import numpy as np

WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(WS, "sp_vision_25", "sim"))
from export_sim_detector import decode, letterbox  # noqa: E402
from eval_detect import IOU_THR, gt_box, iou  # noqa: E402

MODEL = os.path.join(WS, "sp_vision_25", "assets", "sim_armor.onnx")


def sort_kp(pts):
  """和 sim_ort.cpp 的 sort_keypoints 一致：图像系 左上/右上/右下/左下。"""
  p = np.asarray(pts, float)
  idx = np.argsort(p[:, 1])
  top, bot = p[idx[:2]], p[idx[2:]]
  top = top[np.argsort(top[:, 0])]
  bot = bot[np.argsort(bot[:, 0])]
  return np.stack([top[0], top[1], bot[1], bot[0]])


def main():
  import onnxruntime as ort
  frames = sorted(glob.glob("/tmp/eval_frames/*.jpg"))
  sess = ort.InferenceSession(MODEL, providers=["CPUExecutionProvider"])
  errs, box_iou = [], []
  for f in frames:
    bgr = cv2.imread(f)
    canvas, scale, dw, dh = letterbox(bgr, 640)
    blob = canvas[:, :, ::-1].astype(np.float32).transpose(2, 0, 1)[None] / 255.0
    out = sess.run(None, {sess.get_inputs()[0].name: np.ascontiguousarray(blob)})[0][0]
    ds = decode(out, scale, dw, dh, 0.25, 0.45)
    gts = json.load(open(f[:-4] + ".json"))["detections"]
    used = set()
    for g in gts:
      gb = gt_box(g)
      if (gb[2] - gb[0]) * (gb[3] - gb[1]) < 400 or g.get("facing", 1) < 0.35:
        continue
      best, bi = 0.0, -1
      for i, d in enumerate(ds):
        if i in used:
          continue
        v = iou(gb, d["xyxy"])
        if v > best:
          best, bi = v, i
      if best < IOU_THR:
        continue
      used.add(bi)
      box_iou.append(best)
      gk = sort_kp(np.array(g["points"], float))
      dk = sort_kp(np.array(ds[bi]["kpts"], float))
      errs.append(np.linalg.norm(gk - dk, axis=1))
  if not errs:
    print("没有匹配上的板")
    return 1
  e = np.concatenate(errs)
  print("匹配 %d 块板 | 平均框 IoU %.3f" % (len(box_iou), np.mean(box_iou)))
  print("角点误差(px, 原图 1280x960): 中位 %.2f  均值 %.2f  P90 %.2f  最大 %.2f"
        % (np.median(e), e.mean(), np.percentile(e, 90), e.max()))
  # 每块板的"平均角点误差"，看有多少板是整体偏的
  per = np.array([x.mean() for x in errs])
  print("按板统计: 中位 %.2f  P90 %.2f  >5px 的板占 %.1f%%"
        % (np.median(per), np.percentile(per, 90), 100.0 * (per > 5).mean()))
  return 0


if __name__ == "__main__":
  sys.exit(main())
