#!/usr/bin/env python3
"""靶车小陀螺机动的最小验证（rm_cars.GyroPlan）。

跑法：  python3 rm_sim/test_car_motion.py
（纯状态机单测，不需要 MuJoCo，秒级）

设计要点（也解释了为什么检查项长这样）：GyroPlan 返回的是**最终**角速度，
它把"巡逻自转"和"甩头正弦"按模式混合，再过一道**角加速度限幅** ——
所以模式切换附近一定有一段过渡（ramp），量范围/周期时要把 ramp 排除掉。

检查项：
  1. 模式在 patrol / gyro 之间切换，时间占比符合参数；
  2. 巡逻态收敛到调用方给的巡逻自转角速度；
  3. 小陀螺**内部**（排除 ramp）ω 落在 [base-amp, base+amp]，且峰值快于巡逻自转；
  4. 单段小陀螺里能测出正弦周期，且比巡逻自转的 1.2~2.0 s 更小；
  5. 角加速度限幅生效（不突跳），且不削正弦本身；
  6. 参数可覆盖（改 base/amp/period 立刻反映到输出）。
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rm_cars import GyroParams, GyroPlan  # noqa: E402

DT = 0.001
PATROL_OMEGA = 4.0        # 巡逻自转的"替身"值（真机上来自原来的自转模型）
RAMP_SKIP = 0.40          # 过渡段长度：量范围/周期时跳过
FAILED: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
  print("  %s %s%s" % ("✓" if ok else "✗", name, ("  " + detail) if detail else ""))
  if not ok:
    FAILED.append(name)


def run(p: GyroParams, seconds: float, seed: int = 0):
  g = GyroPlan(p, np.random.default_rng(seed))
  n = int(seconds / DT)
  modes, omegas = [], np.zeros(n)
  for i in range(n):
    omegas[i] = g.step(DT, PATROL_OMEGA)
    modes.append(g.mode)
  return modes, omegas


def segments(modes, want: str, skip: float = RAMP_SKIP):
  """切出连续片段，并去掉每段开头 skip 秒（过渡段）。"""
  out, cur = [], []
  for m, i in zip(modes, range(len(modes))):
    if m == want:
      cur.append(i)
    elif cur:
      out.append(cur); cur = []
  if cur:
    out.append(cur)
  k = int(skip / DT)
  return [np.array(s[k:]) for s in out if len(s) > k + 50]


def main() -> int:
  p = GyroParams()
  modes, omegas = run(p, 120.0)

  print("== 1. 模式切换与时间占比 ==")
  n = len(modes)
  gyro_frac = modes.count("gyro") / n
  exp = ((p.dwell_min + p.dwell_max) / 2) / (
    (p.dwell_min + p.dwell_max) / 2 + (p.patrol_min + p.patrol_max) / 2)
  check("小陀螺时间占比符合参数", abs(gyro_frac - exp) < 0.05,
        "实测 %.1f%% 期望 %.1f%%" % (100 * gyro_frac, 100 * exp))
  switches = sum(1 for a, b in zip(modes, modes[1:]) if a != b)
  check("120 s 内多次切换", switches >= 20, "%d 次" % switches)

  print("== 2. 巡逻态收敛到调用方给的自转角速度 ==")
  patrol_segs = segments(modes, "patrol")
  tail = np.concatenate([omegas[s[-int(0.5 / DT):]] for s in patrol_segs])
  check("巡逻态尾部 ω ≈ 巡逻自转值", abs(float(np.median(tail)) - PATROL_OMEGA) < 0.05,
        "实测中位 %.3f 期望 %.1f" % (float(np.median(tail)), PATROL_OMEGA))

  print("== 3. 小陀螺内部的角速度范围 ==")
  gyro_segs = segments(modes, "gyro")
  check("切出足够多的小陀螺片段", len(gyro_segs) >= 5, "%d 段" % len(gyro_segs))
  gw = np.concatenate([omegas[s] for s in gyro_segs])
  lo, hi = p.spin_base - p.spin_amp, p.spin_base + p.spin_amp
  check("ω 落在 [base-amp, base+amp]", gw.min() >= lo - 1e-6 and gw.max() <= hi + 1e-6,
        "实测 [%.2f, %.2f] 期望 [%.1f, %.1f]" % (gw.min(), gw.max(), lo, hi))
  check("峰值明显快于巡逻自转（更快）", gw.max() > 7.02,
        "%.2f rad/s vs 巡逻上限 7.02" % gw.max())

  print("== 4. 正弦周期 ==")
  pers = []
  for s in gyro_segs:
    w = omegas[s]
    thr = float(np.median(w))
    up = int(np.sum((w[:-1] < thr) & (w[1:] >= thr)))
    if up >= 2:
      pers.append(len(w) * DT / up)
  per = float(np.median(pers)) if pers else None
  check("能测出正弦周期且比巡逻自转周期小",
        per is not None and p.period_min - 0.1 < per < p.period_max + 0.1,
        "估计周期 %.2f s（期望 %.1f~%.1f s；巡逻自转是 1.2~2.0 s）"
        % (per if per else -1, p.period_min, p.period_max))

  print("== 5. 角加速度限幅 ==")
  max_jump = float(np.abs(np.diff(omegas)).max())
  limit = p.accel * DT * 1.02
  check("单步变化不超限幅（不突跳）", max_jump <= limit,
        "最大单步跳变 %.4f rad/s（限幅 %.4f = accel %.0f × dt %.3f）"
        % (max_jump, limit, p.accel, DT))
  slope = p.spin_amp * 2 * math.pi / ((p.period_min + p.period_max) / 2)
  check("限幅不会削掉正弦本身", p.accel > slope * 1.2,
        "正弦斜率峰值 %.0f < accel %.0f" % (slope, p.accel))

  print("== 6. 参数可覆盖 ==")
  p2 = GyroParams(spin_base=10.0, spin_amp=2.0, period_min=0.25, period_max=0.25)
  modes2, omegas2 = run(p2, 60.0, seed=3)
  g2 = np.concatenate([omegas2[s] for s in segments(modes2, "gyro")])
  check("改 base/amp 立刻生效",
        abs(g2.min() - 8.0) < 0.2 and abs(g2.max() - 12.0) < 0.2,
        "实测 [%.2f, %.2f] 期望 [8.0, 12.0]" % (g2.min(), g2.max()))

  print()
  if FAILED:
    print("✗ %d 项未通过：%s" % (len(FAILED), ", ".join(FAILED)))
    return 1
  print("✓ 全部通过")
  return 0


if __name__ == "__main__":
  sys.exit(main())
