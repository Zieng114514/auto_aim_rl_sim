#!/usr/bin/env python3
"""把训练好的 .pt 导出成 ONNX，并做一次 Python 侧召回评测。

评测口径和 sp_vision/sim/eval_detect.py 完全一致（可见板过滤 + IoU 0.30 匹配），
这样"新模型 vs 现役 yolov5"的数字可以直接对比。

预处理也刻意对齐 **ultralytics 的训练/验证**：居中 letterbox + 114 灰边。
上游 C++ 后端（yolov5.cpp / yolov8.cpp）用的是左上角对齐补 0，对自训模型是实打实
的精度损失，所以 sim_ort.cpp 复刻的是居中版本，这里也照它算。

用法:
    python3 tools/export_sim_detector.py runs/sim_armor_xxx/weights/best.pt
"""
import argparse
import glob
import json
import os
import sys
import time

import cv2
import numpy as np

WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(WS, "sp_vision_25", "sim"))
from eval_detect import IOU_THR, MIN_AREA, MIN_FACING, gt_box, iou  # noqa: E402


def letterbox(img, size, pad_value=114):
  """和 ultralytics LetterBox 一致：等比缩放 + 居中 + 灰边。"""
  h, w = img.shape[:2]
  scale = min(size / h, size / w)
  nw, nh = int(round(w * scale)), int(round(h * scale))
  dw, dh = (size - nw) // 2, (size - nh) // 2
  canvas = np.full((size, size, 3), pad_value, np.uint8)
  canvas[dh:dh + nh, dw:dw + nw] = cv2.resize(img, (nw, nh))
  return canvas, scale, dw, dh


def decode(pred, scale, dw, dh, conf_thr, iou_thr):
  """YOLO-pose 解码。pred 是 [C, N]，坐标先减 padding 再除 scale 回到原图。"""
  scores = pred[4]
  keep = np.nonzero(scores >= conf_thr)[0]
  if keep.size == 0:
    return []
  cx, cy, bw, bh = (pred[i][keep] for i in range(4))
  x0, y0 = (cx - bw / 2 - dw) / scale, (cy - bh / 2 - dh) / scale
  x1, y1 = (cx + bw / 2 - dw) / scale, (cy + bh / 2 - dh) / scale
  nk = (pred.shape[0] - 4 - 1) // 4
  kp = pred[5:5 + 4 * nk].reshape(4, nk, -1)[:, :2, keep]        # (4, 2, n)
  kp = np.stack([(kp[:, 0] - dw) / scale, (kp[:, 1] - dh) / scale], -1).transpose(1, 0, 2)
  boxes = np.stack([x0, y0, x1, y1], 1)
  nms = cv2.dnn.NMSBoxes(
    [[float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])] for b in boxes],
    [float(s) for s in scores[keep]], conf_thr, iou_thr)
  return [{"xyxy": [float(v) for v in boxes[i]], "conf": float(scores[keep][i]),
           "kpts": [[float(p[0]), float(p[1])] for p in kp[i]]}
          for i in np.array(nms).reshape(-1)]


def evaluate(onnx_path, frames, size, providers, conf_thr=0.25, iou_thr=0.45, limit=0):
  import onnxruntime as ort
  sess = ort.InferenceSession(onnx_path, providers=providers)
  files = sorted(glob.glob(os.path.join(frames, "*.jpg")))
  if limit:
    files = files[:limit]
  tp = fp = fn = n_vis = 0
  ious, times = [], []
  for f in files:
    bgr = cv2.imread(f)
    t0 = time.perf_counter()
    canvas, scale, dw, dh = letterbox(bgr, size)
    blob = canvas[:, :, ::-1].astype(np.float32).transpose(2, 0, 1)[None] / 255.0
    out = sess.run(None, {sess.get_inputs()[0].name: np.ascontiguousarray(blob)})[0][0]
    ds = decode(out, scale, dw, dh, conf_thr, iou_thr)
    times.append((time.perf_counter() - t0) * 1e3)

    gj = f[:-4] + ".json"
    allg = json.load(open(gj))["detections"] if os.path.exists(gj) else []
    boxes_all = [gt_box(g) for g in allg]
    vis = []
    for g, b in zip(allg, boxes_all):
      if (b[2] - b[0]) * (b[3] - b[1]) < MIN_AREA:
        continue
      if "facing" in g and g["facing"] < MIN_FACING:
        continue
      vis.append(b)
    n_vis += len(vis)
    used = set()
    for g in vis:
      best, bi = 0.0, -1
      for i, d in enumerate(ds):
        if i in used:
          continue
        v = iou(g, d["xyxy"])
        if v > best:
          best, bi = v, i
      if best >= IOU_THR:
        tp += 1
        used.add(bi)
        ious.append(best)
      else:
        fn += 1
    for i, d in enumerate(ds):
      if i in used:
        continue
      if any(iou(b, d["xyxy"]) >= IOU_THR for b in boxes_all):
        continue
      fp += 1
  p = tp / (tp + fp) if tp + fp else 0.0
  r = tp / (tp + fn) if tp + fn else 0.0
  f1 = 2 * p * r / (p + r) if p + r else 0.0
  times.sort()
  return dict(frames=len(files), visible=n_vis, tp=tp, fp=fp, fn=fn, p=p, r=r, f1=f1,
              mean_iou=sum(ious) / len(ious) if ious else 0.0,
              ms_median=times[len(times) // 2] if times else 0.0)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("weights", help="runs/.../weights/best.pt")
  ap.add_argument("--out", default=os.path.join(WS, "sp_vision_25/assets/sim_armor.onnx"))
  ap.add_argument("--imgsz", type=int, default=640)
  ap.add_argument("--frames", default="/tmp/eval_frames", help="带真值 json 的评测帧目录")
  ap.add_argument("--limit", type=int, default=0)
  ap.add_argument("--opset", type=int, default=12)
  args = ap.parse_args()

  from ultralytics import YOLO
  m = YOLO(args.weights)
  # simplify 需要 onnxslim（要联网装），而且 ORT 自己会做图优化，这里直接关掉；
  # 少一个网络依赖就少一个半夜挂掉的理由。
  try:
    onnx_path = m.export(format="onnx", imgsz=args.imgsz, simplify=False, opset=args.opset,
                         dynamic=False, half=False)
  except Exception as e:  # noqa: BLE001
    print("opset=%d 导出失败（%s），退回 opset=18" % (args.opset, str(e)[:120]))
    onnx_path = m.export(format="onnx", imgsz=args.imgsz, simplify=False, opset=18,
                         dynamic=False, half=False)
  os.makedirs(os.path.dirname(args.out), exist_ok=True)
  if os.path.abspath(onnx_path) != os.path.abspath(args.out):
    os.replace(onnx_path, args.out)
  print("✓ 导出:", args.out, "%.1f MB" % (os.path.getsize(args.out) / 1e6))

  if os.path.isdir(args.frames):
    print("\n== 召回评测（口径同 eval_detect.py）==")
    for prov, tag in ((["CPUExecutionProvider"], "ORT CPU"),
                      (["CUDAExecutionProvider", "CPUExecutionProvider"], "ORT CUDA(30HX)")):
      try:
        res = evaluate(args.out, args.frames, args.imgsz, prov, limit=args.limit)
      except Exception as e:  # noqa: BLE001
        print("  %-16s 跑不了：%s" % (tag, str(e)[:140]))
        continue
      print("  %-16s 帧 %d 可见板 %d | P=%.3f R=%.3f F1=%.3f | 平均IoU=%.3f | %.1f ms/帧"
            % (tag, res["frames"], res["visible"], res["p"], res["r"], res["f1"],
               res["mean_iou"], res["ms_median"]))
  return 0


if __name__ == "__main__":
  sys.exit(main())
