#!/usr/bin/env python3
"""从 rm_sim 生成装甲板检测数据集（YOLO-pose 格式），用来在 30HX 上微调检测器。

为什么要有这个东西
------------------
现役的装甲板检测器（assets/yolov5.xml）是在**真实**比赛数据上训的，喂给它仿真画面
只有约 69% 召回 —— 这就是"域差"。rm_sim 本来就带完整真值，能自动标注任意多张图，
所以正解是拿它训一个**仿真域**检测器，再导出 ONNX 给 sp_vision 用。

标签约定（关键，和 sp_vision 的 C++ 侧必须一致）
-----------------------------------------------
* 4 个关键点 = 装甲板的四个角，顺序**就是** GT 里的顺序，也就是 solver.cpp 里
  SMALL_ARMOR_POINTS 的顺序：
      [(+W/2,+L/2), (-W/2,+L/2), (-W/2,-L/2), (+W/2,-L/2)]
  这样 C++ 后端拿到关键点可以直接塞进 Armor.points，不需要任何重排；
  弄错顺序的后果是 solvePnP 点对应对不上，解出来的 yaw 会整体翻 90/180 度。
* 类别只有 1 类（armor）。颜色/号码不在这里判 —— 号码由 tiny_resnet 分类器给，
  颜色由 configs/sim.yaml 的 enemy_color 给。

用法
----
    ~/mjlab-env/venv/bin/python gen_dataset.py --out datasets/sim --frames 3000
"""
import argparse
import json
import math
import os
import sys
import time

import mujoco
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "sp_vision_25", "sim"))

from PIL import Image  # noqa: E402
import rm_link as R  # noqa: E402


def build_args(argv=None):
  """借用 rm_link 的解析器，保证数据集和联调跑的是**同一套**渲染/物理配置。"""
  ap = R.build_parser()
  args = ap.parse_args([])                      # 先拿一份默认值
  gen = argparse.ArgumentParser(description="rm_sim 装甲板数据集生成")
  gen.add_argument("--out", required=True, help="数据集输出目录")
  gen.add_argument("--frames", type=int, default=3000, help="总帧数（保存张数）")
  gen.add_argument("--save-every", type=int, default=1, help="每 N 个仿真 tick 存一帧")
  gen.add_argument("--cars", type=int, default=3)
  gen.add_argument("--seed", type=int, default=7)
  gen.add_argument("--val-ratio", type=float, default=0.08)
  gen.add_argument("--empty-ratio", type=float, default=0.10,
                   help="有多大比例的帧刻意不瞄准靶车（纯背景负样本）")
  gen.add_argument("--retarget", type=int, default=10, help="每 N 个 tick 重选一次瞄准点")
  gen.add_argument("--jit-yaw", type=float, default=0.13, help="瞄准偏置范围 ±rad")
  gen.add_argument("--jit-pitch", type=float, default=0.09)
  # 遮挡剔除默认**关**。本来想用 mj_ray 从相机往板中心打射线，把被前车挡住的板丢掉，
  # 但实测这条射线会先打到 z≈0 那个 floor **平面**：MuJoCo 的平面渲染是单面的
  # （从背面看不见，所以画面上没有），射线却是双面的 —— 于是每一块板都被判成"被挡住"，
  # 400 帧全空。要用它得先把平面 geom 排除在射线之外，收益不值这个复杂度：
  # 靶车有 keep-out 不会叠在一起，真被挡住的情况很少。
  gen.add_argument("--occlusion", action="store_true", default=False,
                   help="用 mj_ray 剔除被别的车挡住的板（本场景下平面会误判，默认关）")
  g = gen.parse_args(argv)

  # 数据集模式下的固定覆盖
  args.cars = g.cars
  args.seed = g.seed
  args.overview = False            # 不渲全局视角，省一半渲染开销
  args.save_dir = ""               # 不用 rm_link 自带的抓图（我们自己写标签）
  return args, g


def label_line(pts: np.ndarray, w: int, h: int, cls: int = 0) -> str | None:
  """四点 -> YOLO-pose 的一行标签。坐标归一化到 [0,1]。"""
  x0, y0 = float(pts[:, 0].min()), float(pts[:, 1].min())
  x1, y1 = float(pts[:, 0].max()), float(pts[:, 1].max())
  bw, bh = x1 - x0, y1 - y0
  if bw < 1.5 or bh < 1.5:                       # 退化成一两条像素的板，训了反而是噪声
    return None
  cx, cy = (x0 + x1) / 2 / w, (y0 + y1) / 2 / h
  parts = [f"{cls}", f"{cx:.6f}", f"{cy:.6f}", f"{bw / w:.6f}", f"{bh / h:.6f}"]
  for px, py in pts:
    parts += [f"{px / w:.6f}", f"{py / h:.6f}", "2"]     # v=2 表示"可见且有标签"
  return " ".join(parts)


def occluded(model, data, cam_pos, target) -> bool:
  """从相机往板中心打一条射线，看看第一个撞到的是不是它自己那台车。"""
  d = np.asarray(target, dtype=float) - np.asarray(cam_pos, dtype=float)
  dist = float(np.linalg.norm(d))
  if dist < 1e-6:
    return True
  d /= dist
  geomid = np.zeros(1, dtype=np.int32)
  hit = mujoco.mj_ray(model, data, np.asarray(cam_pos, dtype=float), d, None, 1, -1, geomid)
  # 命中的距离比到板中心还近 -> 中间有东西挡着
  return 0.0 < hit < dist - 0.05


def main() -> int:
  args, g = build_args()
  out = g.out if os.path.isabs(g.out) else os.path.join(os.path.dirname(_HERE), g.out)
  for sub in ("images/train", "images/val", "labels/train", "labels/val"):
    os.makedirs(os.path.join(out, sub), exist_ok=True)

  rng = np.random.default_rng(g.seed)
  sim = R.LinkSim(args)
  sim.renderer = mujoco.Renderer(sim.model, height=args.height, width=args.width)

  n_val = max(1, int(g.frames * g.val_ratio))
  saved = 0
  plates_total = 0
  empty = 0
  t0 = time.time()

  aim = {"car": 0, "jy": 0.0, "jp": 0.0, "bg": False}

  while saved < g.frames:
    # ---- 重选瞄准点 ------------------------------------------------------
    if sim.frame_idx % g.retarget == 0:
      alive = [i for i, c in enumerate(sim.cars.cars) if c.alive]
      aim["bg"] = bool(rng.random() < g.empty_ratio)
      aim["car"] = int(rng.choice(alive)) if alive else 0
      aim["jy"] = float(rng.uniform(-1, 1) * g.jit_yaw)
      aim["jp"] = float(rng.uniform(-1, 1) * g.jit_pitch)

    # ---- 算瞄准角 --------------------------------------------------------
    # 世界系下：yaw 关节角 == 目标方位角；抬头为负（同 sim_link 里的约定）。
    mp, _ = sim.demo.muzzle()
    if aim["bg"]:
      yaw = float(rng.uniform(-math.pi, math.pi))
      pitch = float(rng.uniform(-0.35, 0.35))
    else:
      cars = sim.cars.cars
      c = cars[aim["car"]] if aim["car"] < len(cars) else cars[0]
      t = np.array(sim.data.xpos[c.bid], dtype=float) + np.array([0.0, 0.0, 0.15])
      d = t - mp
      yaw = math.atan2(d[1], d[0]) + aim["jy"]
      pitch = -math.atan2(d[2], math.hypot(d[0], d[1])) + aim["jp"]
    with sim.cmd_lock:
      sim.cmd_yaw, sim.cmd_pitch = yaw, pitch

    sim.tick()

    if sim.frame_idx % max(1, g.save_every) != 0:
      continue

    # ---- 写图 + 写标签 ---------------------------------------------------
    stem = f"{saved:06d}"
    split = "val" if saved < n_val else "train"
    with open(os.path.join(out, "images", split, stem + ".jpg"), "wb") as f:
      f.write(sim.jpeg)

    cam_pos = np.array(sim.data.cam_xpos[sim.cam_id], dtype=float)
    lines = []
    for det in sim.gt:
      pts = np.array(det["points"], dtype=float)
      if g.occlusion:
        face = pts.mean(axis=0)
        # 用板在 gimbal 系下的真值中心回投成世界点不划算，
        # 这里退而求其次：用板到相机的视线，被"任何"更近的 geom 挡住就丢。
        xyz = det.get("xyz_in_world")
        if xyz is not None and occluded(sim.model, sim.data, cam_pos, xyz):
          continue
      ln = label_line(pts, args.width, args.height)
      if ln:
        lines.append(ln)
    with open(os.path.join(out, "labels", split, stem + ".txt"), "w") as f:
      f.write("\n".join(lines) + ("\n" if lines else ""))
    plates_total += len(lines)
    empty += 1 if not lines else 0
    saved += 1
    if saved % 200 == 0:
      print(f"  {saved}/{g.frames} 帧  板 {plates_total}  空帧 {empty}  "
            f"{(saved)/(time.time()-t0):.1f} 帧/s  仿真 {sim.data.time:.0f}s", flush=True)

  yaml_path = os.path.join(out, "data.yaml")
  with open(yaml_path, "w") as f:
    # flip_idx 必须写对：我们的四点顺序是 [右上, 左上, 左下, 右下]（GT 的
    # corners = [(+1,+1), (-1,+1), (-1,-1), (+1,-1)]），左右镜像后
    # 右上<->左上、右下<->左下，所以置换是 [1,0,3,2]。
    # 留空的话 fliplr 增广会把同一个物理角点有时标成 0、有时标成 1，
    # 关键点回归的监督信号自相矛盾，训出来的角点会飘。
    f.write(f"path: {out}\ntrain: images/train\nval: images/val\n"
            "names:\n  0: armor\nkpt_shape: [4, 3]\nflip_idx: [1, 0, 3, 2]\n")
  print(f"完成：{saved} 帧（val {n_val}），装甲板 {plates_total} 块，"
        f"空帧 {empty}（{100.0*empty/max(saved,1):.1f}%），用时 {time.time()-t0:.0f}s")
  print(f"  {yaml_path}")
  return 0


if __name__ == "__main__":
  sys.exit(main())
