#!/usr/bin/env python3
"""靶车运动基准脚本的最小验证（rm_car_script）。

跑法：  python3 rm_sim/test_car_script.py
（纯状态机单测，不需要 MuJoCo，秒级）

检查项：
  1. 每段时长严格等于 dwell（不能飘）；
  2. 同一颗种子 -> 同一套顺序（**可复现**，这是这个模块存在的全部意义）；
     不同种子 -> 顺序不同（避免"总先打最简单的靶型"的顺序偏差）；
  3. 每一轮**默认那四种**靶型各出现一次（默认集合 = 既有基线口径，不能变）；
     另测：含轮腿的 6 段集合（benchmark_2026_wl）也能正常播放、升降字段正确；
  4. 靶型 -> (平移方式, 自转方式) 的映射正确；
  5. 分段计数：段边界的 (发射, 命中, 击杀) 增量算得对（含"进行中"那段）。
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rm_car_script import (  # noqa: E402
  MODES, MODES_LEGACY, MODES_WHEEL_LEGGED, CarScript, CarScriptParams)

DT = 1.0 / 30.0
FAILED: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
  print("  %s %s%s" % ("✓" if ok else "✗", name, ("  " + detail) if detail else ""))
  if not ok:
    FAILED.append(name)


def play(p: CarScriptParams, seconds: float, seed: int, counts=None):
  """跑 seconds 秒；counts(t) 返回当时的 (发射, 命中, 击杀)。"""
  sc = CarScript(p, np.random.default_rng(seed))
  n = int(seconds / DT)
  specs = []
  for i in range(n):
    c = counts(i) if counts else (0, 0, 0)
    specs.append(sc.step(DT, c))
  return sc, specs


def main() -> int:
  p = CarScriptParams(dwell=10.0)

  print("== 1. 段时长严格等于 dwell ==")
  sc, _ = play(p, 95.0, seed=1)
  # 每段 10 s -> 95 s 内应有 9 个完整段 + 1 个进行中的段
  segs = sc.segments
  full = [s for s in segs if s['t1'] is not None]
  check("完成段数符合预期", len(full) == 9, "%d 段" % len(full))
  check("每段时长都等于 dwell",
        all(abs(s['dur'] - p.dwell) < 1e-6 for s in full),
        "最大偏差 %.6f s" % max(abs(s['dur'] - p.dwell) for s in full))
  check("段边界连续（首尾相接）",
        all(abs(full[i]['t1'] - full[i + 1]['t0']) < 1e-6 for i in range(len(full) - 1)))

  print("== 2. 乱序但可复现 ==")
  sc_a, _ = play(p, 95.0, seed=7)      # 跑够 2 轮（8 段），才能查"每轮四种各一次"
  sc_b, _ = play(p, 95.0, seed=7)
  sc_c, _ = play(p, 95.0, seed=8)
  check("同种子 -> 同顺序", sc_a.order[:8] == sc_b.order[:8],
        " / ".join(sc_a.order[:4]))
  check("不同种子 -> 顺序不同", sc_a.order[:8] != sc_c.order[:8],
        "seed7=%s  seed8=%s" % (sc_a.order[:4], sc_c.order[:4]))
  check("关掉乱序时按固定顺序",
        play(CarScriptParams(shuffle=False), 20.0, 1)[0].order[:4] == list(MODES_LEGACY))
  # 默认集合必须还是那 4 种（换了会让全部既有基线不可比）
  check("默认靶型集合 = 老的 4 种", tuple(CarScriptParams().modes) == MODES_LEGACY)
  # 轮腿 6 段集合：能播、且升降字段正确
  wl = CarScriptParams(modes=MODES_LEGACY + MODES_WHEEL_LEGGED)
  sc, _ = play(wl, 120.0, seed=3)
  modes_seen = set(sc.order)
  check("轮腿集合能播放", modes_seen.issuperset(set(MODES_WHEEL_LEGGED)), str(sorted(modes_seen)))
  from rm_car_script import _SPEC  # noqa: E402
  check("轮腿段的升降模式正确",
        _SPEC["wheel_legged_periodic"].lift == "periodic"
        and _SPEC["wheel_legged_random"].lift == "random"
        and _SPEC["inplace_spin"].lift == "off")

  print("== 3. 每轮四种靶型各一次 ==")
  for r in range(3):
    chunk = sc_a.order[r * 4:(r + 1) * 4]
    if sorted(chunk) != sorted(MODES_LEGACY):
      check("第 %d 轮包含四种靶型" % (r + 1), False, str(chunk))
      break
  else:
    check("每轮都是四种靶型各一次", True, "前 3 轮正确")

  print("== 4. 靶型 -> 动作映射 ==")
  want = {
    "move_spin_rand": ("patrol", "mod"),
    "move_spin_const": ("patrol", "const"),
    "inplace_spin": ("stop", "gyro"),
    "dash_halt": ("dashhalt", "mod"),
  }
  _, specs = play(p, 40.0, seed=3)
  seen = {}
  for s in specs:
    seen[s.mode] = (s.translation, s.spin)
  ok = all(seen.get(m) == w for m, w in want.items())
  check("四种靶型的平移/自转映射正确", ok, str(seen))

  print("== 5. 分段计数 ==")
  # 每秒 2 发、1 中；每 27 s 击杀一台。
  # **注意别把击杀时刻放在段边界上**：段边界那一刻的事件会算进"上一段"
  # （先关旧段、再开新段），放整数倍 dwell 上会得到一个纯属测试自找的假失败。
  def counts(i):
    t = i * DT
    return (int(t * 2), int(t * 1), int(t // 27))
  sc2, _ = play(p, 45.0, seed=5, counts=counts)
  full2 = [s for s in sc2.segments if s['t1'] is not None]
  # 第 1 段（0~10 s）：发射 20、命中 10、击杀 0
  s0 = full2[0]
  check("第 1 段发射数正确", s0['shots'] == 20, "%d（期望 20）" % s0['shots'])
  check("第 1 段命中数正确", s0['hits'] == 10, "%d（期望 10）" % s0['hits'])
  # 第 3 段（20~30 s）里有 t=27 的击杀
  s2 = full2[2]
  check("含击杀的段计数正确（20~30s 应为 20/10/1）",
        (s2['shots'], s2['hits'], s2['kills']) == (20, 10, 1),
        "%s" % ((s2['shots'], s2['hits'], s2['kills']),))
  check("不含击杀的段 kills 为 0", full2[0]['kills'] == 0 and full2[1]['kills'] == 0)
  st = sc2.status(counts(int(45.0 / DT)))
  check("status() 里含进行中的段", st['segments'] and st['segments'][-1]['t1'] is None,
        "当前 %s（已 %.1fs）" % (st['current_desc'], st['elapsed']))

  print()
  if FAILED:
    print("✗ %d 项未通过：%s" % (len(FAILED), ", ".join(FAILED)))
    return 1
  print("✓ 全部通过")
  return 0


if __name__ == "__main__":
  sys.exit(main())
