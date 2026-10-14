#!/usr/bin/env python3
"""rm_gun 的最小验证脚本：弹速 / 热量 / 散布三个模型的行为是否合乎预期。

跑法：  python3 rm_sim/test_gun_model.py
（不需要 MuJoCo，纯模型单测，秒级完成）

检查项：
  1. 弹速从冷枪 22 单调爬升、饱和在 24 附近，停机后能降温回落；
  2. 弹速随机波动 ≈ ±0.5 m/s（按 2σ 解释，即 σ=0.25）；
  3. 散布：发射方向相对炮管的夹角标准差 ≈ spread_deg；
  4. 热量：连射 N 发后进入"1 发/秒"的冷却预算（burst then sustain）；
     硬阈值模式不扣血、被拒次数被记账；软超限模式按超出量扣血；
  5. 开关：三个模型各自关掉后行为退回旧基线（恒速/无热量/无散布）。
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rm_gun import GunModel, GunParams  # noqa: E402

FRAME = 1.0 / 30.0
STEPS_PER_FRAME = 33          # 1 kHz 物理步
FAILED: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
  print("  %s %s%s" % ("✓" if ok else "✗", name, ("  " + detail) if detail else ""))
  if not ok:
    FAILED.append(name)


def run(g: GunModel, frames: int, tries_per_frame: int = 1):
  """跑 frames 帧；每帧尝试开火 tries_per_frame 次。返回 (开火数, 被拒数)。"""
  fired = blocked = 0
  for _ in range(frames):
    for _ in range(STEPS_PER_FRAME):
      g.step(1.0 / (30.0 * STEPS_PER_FRAME))
    for _ in range(tries_per_frame):
      if g.can_fire():
        g.take_shot()
        fired += 1
      else:
        g.shots_blocked += 1
        blocked += 1
  return fired, blocked


def main() -> int:
  print("== 1. 弹速：升温爬升 + 饱和 + 降温回落 ==")
  g = GunModel(GunParams(), seed=1)
  speeds = [g.current_speed()]
  for _ in range(120):                      # 120 发
    g.take_shot()
    speeds.append(g.current_speed())
  p = g.p
  check("冷枪速度 = 标称", abs(speeds[0] - p.speed_cold) < 1e-6, "%.3f" % speeds[0])
  check("单调不降", all(b >= a - 1e-9 for a, b in zip(speeds, speeds[1:])))
  check("饱和趋近热枪速度", abs(speeds[-1] - p.speed_warm) < 0.2,
        "%.3f -> %.3f（目标 %.1f）" % (speeds[0], speeds[-1], p.speed_warm))
  warm_before = g.current_speed()
  for _ in range(60 * 300):                 # 300 秒不射击
    g.step(1.0 / 60.0)
  check("停机后能降温", g.current_speed() < warm_before - 0.3,
        "%.3f -> %.3f" % (warm_before, g.current_speed()))

  print("== 2. 弹速随机波动 ≈ ±0.5 m/s（2σ）==")
  g = GunModel(GunParams(), seed=2)
  vs = [g.take_shot()[0] for _ in range(4000)]
  sd = float(np.std(vs))
  check("σ ≈ jitter/2", abs(sd - p.jitter / 2) < 0.03, "实测 σ=%.3f 期望 %.3f" % (sd, p.jitter / 2))

  print("== 3. 枪口散布 ==")
  # 口径：spread_deg 是"每个正交轴各自的 1σ"，所以这里分别投影到两个正交轴上量
  g = GunModel(GunParams(), seed=3)
  base = np.array([1.0, 0.0, 0.0])
  tmp = np.array([0.0, 0.0, 1.0])
  u = np.cross(base, tmp); u /= np.linalg.norm(u)
  w = np.cross(base, u)
  a1s, a2s, angs = [], [], []
  for _ in range(4000):
    d = g.perturb_dir(base)
    a1s.append(math.degrees(math.atan2(float(d @ u), float(d @ base))))
    a2s.append(math.degrees(math.atan2(float(d @ w), float(d @ base))))
    angs.append(math.degrees(math.acos(float(np.clip(np.dot(d, base), -1, 1)))))
  sd1, sd2 = float(np.std(a1s)), float(np.std(a2s))
  mean_ang = float(np.mean(angs))
  check("散布：两轴 σ ≈ spread_deg", abs(sd1 - p.spread_deg) < 0.02 and abs(sd2 - p.spread_deg) < 0.02,
        "σ_u=%.3f° σ_w=%.3f° 期望 %.2f°" % (sd1, sd2, p.spread_deg))
  check("合成总偏差均值 ≈ 1.25×spread_deg", abs(mean_ang - 1.25 * p.spread_deg) < 0.02,
        "均值 %.3f°" % mean_ang)

  print("== 4. 热量：连射 -> 冷却预算 ==")
  g = GunModel(GunParams(), seed=4)
  burst = 0
  while g.can_fire():
    g.take_shot()
    burst += 1
  check("首发连射发数 = heat_max/heat_per_shot",
        burst == int(p.heat_max // p.heat_per_shot), "%d 发" % burst)
  # 稳态射速要从"首发连射打完"之后开始量：前 10 发是 burst，会把平均值拉高
  g2 = GunModel(GunParams(), seed=4)
  run(g2, 300)                              # 前 10 s：度过 burst
  fired, blocked = run(g2, 300)             # 后 10 s：稳态
  rate = fired / 10.0
  check("稳态射速 ≈ 冷却/每发热量", abs(rate - p.heat_cool / p.heat_per_shot) < 0.1,
        "%.2f 发/s（期望着 %.1f）" % (rate, p.heat_cool / p.heat_per_shot))
  check("硬阈值模式不扣血", g2.overheat_damage == 0.0, "扣血 %.1f" % g2.overheat_damage)

  g3 = GunModel(GunParams(), seed=4)
  run(g3, 300, tries_per_frame=3)
  fired3, blocked3 = run(g3, 300, tries_per_frame=3)   # 同样只量后 10 s 的稳态
  check("狂按扳机不会提高射速（资源约束真实存在）",
        abs(fired3 - fired) <= 1 and blocked3 > blocked,
        "开火 %d vs %d，被拒 %d vs %d" % (fired3, fired, blocked3, blocked))

  g4 = GunModel(GunParams(hard_block=False), seed=4)
  run(g4, 600)
  check("软超限模式会按超出量扣血", g4.overheat_damage > 1.0, "扣血 %.1f" % g4.overheat_damage)

  print("== 5. 开关：三个模型各自关掉 ==")
  g5 = GunModel(GunParams(model_speed=False, model_heat=False, model_spread=False), seed=5)
  for _ in range(50):
    g5.take_shot()
  check("弹速模型关掉 -> 恒为标称", abs(g5.current_speed() - p.speed_cold) < 1e-9)
  check("热量模型关掉 -> 永不锁枪", g5.can_fire() and g5.heat == 0.0)
  d = g5.perturb_dir(np.array([1.0, 0.0, 0.0]))
  check("散布模型关掉 -> 方向不变", np.allclose(d, [1.0, 0.0, 0.0]))

  print()
  if FAILED:
    print("✗ %d 项未通过：%s" % (len(FAILED), ", ".join(FAILED)))
    return 1
  print("✓ 全部通过")
  return 0


if __name__ == "__main__":
  sys.exit(main())
