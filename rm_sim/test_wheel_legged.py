#!/usr/bin/env python3
"""轮腿靶车升降的自检：行程/占空比/限幅/配置层，以及**防嵌地**这条硬约束。

背景（都是实测踩出来的，别删）：
  * 负责人 2026-10-14 发现"装甲板嵌地上"：我们的靶车是单刚体，"贴地姿态"就是几何最低点
    落在地面（ride_z ≈ -4mm），所以任何下行程都等于往地下钻。行程下沿因此固定为 0。
  * 负责人还要求"不期望全部时间装甲板都在上下移动"，于是有了活动/静止窗口（占空比 ~0.4）。
"""
import math
import sys
import os

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rm_lift import BodyLift, LiftParams   # noqa: E402
import rm_config                            # noqa: E402
import rm_car_script as CS                  # noqa: E402

DT = 1.0 / 500.0
FAILED = []


def check(name: str, ok: bool, detail: str = "") -> None:
  print("  %s %s%s" % ("✓" if ok else "✗", name, ("  " + detail) if detail else ""))
  if not ok:
    FAILED.append(name)


def run(p: LiftParams, seconds: float, seed: int = 0):
  b = BodyLift(p, np.random.default_rng(seed))
  rows = []
  for _ in range(int(seconds / DT)):
    pv, pa = b.vel, b.active
    z = b.step(DT)
    rows.append((b.t, z * 1000, b.vel * 1000, (b.vel - pv) / DT * 1000, pa, b.active))
  return b, rows


def col(rows, i):
  return np.array([r[i] for r in rows])


DEF = dict(mode="periodic", amplitude_mm=150.0, period_s=2.0,
           accel_limit_mps2=2.0, vel_limit_mps=0.5)
A = DEF["amplitude_mm"] / 1000.0

print("== 1. 默认档：行程 0 ~ +300mm，干净正弦 ==")
b, rows = run(LiftParams(**DEF), 60.0)
z, v, a = col(rows, 1), col(rows, 2), col(rows, 3)
check("行程下沿 = 0（= 贴地姿态，绝不嵌地）", z.min() >= -0.05, "实测最低 %.1f mm" % z.min())
check("行程上沿 = +300mm（bottom + 2A）", abs(z.max() - 2 * A * 1000) < 0.5,
      "实测最高 %.1f mm" % z.max())
check("峰值速度 ≈ A·ω（0.471 m/s）", abs(b.peak_vel - A * math.pi) < 0.03,
      "实测 %.3f m/s" % b.peak_vel)
check("峰值加速度 ≈ A·ω²（1.48）且未顶到上限 2.0",
      abs(b.peak_accel - A * math.pi ** 2) < 0.08, "实测 %.3f m/s²" % b.peak_accel)
# 注意单位：col(rows,3) 是 mm/s²，限幅值在 DEF 里是 m/s²（踩过：直接比 => 恒真）
sat = float(np.mean(np.abs(a) >= DEF["accel_limit_mps2"] * 1000.0 - 1e-9)) * 100
check("限幅在默认档不介入（饱和步 ≈ 0%）", sat < 0.5, "实测 %.1f%%" % sat)
# 周期：活动窗口内找高位的峰，量峰间距
idx = [i for i in range(1, len(z) - 1) if z[i] > 290 and z[i] > z[i - 1] and z[i] >= z[i + 1]]
# 峰可能落在**不同的活动窗口**里（中间隔着 2.5~5 s 的静止），所以只取间距 < 1.5 个周期的
# 相邻峰对来量周期 —— 直接用全部峰的均值会量出"周期 + 静止"（实测 4.10 s，踩过）。
gaps = np.diff(idx) * DT
gaps = gaps[gaps < 1.5 * DEF["period_s"]]
check("活动窗口内的周期 = 2.000 s", len(gaps) >= 5 and abs(gaps.mean() - 2.0) < 0.02,
      "实测 %.3f s（%d 对相邻峰）" % (gaps.mean() if len(gaps) else float("nan"), len(gaps)))

print("\n== 2. 占空比：不是全程都在动 ==")
check("在动的时间占比 ~40%（0.25~0.65）", 0.25 < b.duty < 0.65, "实测 %.2f" % b.duty)
check("静止窗口确实停住（有连续 ≥1s 的 0 偏移）",
      max(len(x) for x in "".join("1" if abs(v_) < 0.5 else "0" for v_ in z).split("0")) * DT > 1.0,
      "最长静止 %.1f s" % (max(len(x) for x in "".join("1" if abs(v_) < 0.5 else "0" for v_ in z).split("0")) * DT))
# 窗口切换点的速度决定"进出是否平滑"（正弦起止都落在贴地姿态且速度为 0）
sw = [i for i in range(1, len(rows)) if rows[i][5] != rows[i - 1][5]]
vsw = [abs(rows[i][2]) for i in sw]
check("窗口切换处速度 ≈ 0（平滑进出，不突跳）", len(sw) >= 6 and max(vsw) < 20.0,
      "%d 次切换，最大 |v| %.0f mm/s" % (len(sw), max(vsw) if vsw else 0.0))

print("\n== 3. 无窗口档 = 全程都在动（老行为，向后兼容）==")
b0, rows0 = run(LiftParams(**{**DEF, "pause_s": ()}), 30.0)
z0 = col(rows0, 1)
check("占空比 = 1.00", abs(b0.duty - 1.0) < 1e-6, "实测 %.2f" % b0.duty)
check("行程仍是 0 ~ +300mm", z0.min() >= -0.05 and abs(z0.max() - 2 * A * 1000) < 0.5,
      "实测 %.1f ~ %.1f mm" % (z0.min(), z0.max()))

print("\n== 4. 限幅是真旋钮（调紧才生效）==")
bt, rowst = run(LiftParams(**{**DEF, "accel_limit_mps2": 1.0}), 40.0)
zt = col(rowst, 1)
# 注意：上限收紧**不会**把行程压小 —— 下沿被地面钉住、上沿被行程钉住，
# 所以高度范围还是 0~300，变化的是**轨迹形状**（顶在上沿的时间变长）。
pinned_t = float(np.mean(zt > 299.0)) * 100
check("加速度上限 1.0 生效：峰加速 = 上限、轨迹被削平（顶在上沿）",
      abs(bt.peak_accel - 1.0) < 0.02 and pinned_t > 2.0 and zt.min() >= -0.05,
      "峰加速 %.2f，顶在上沿 %.1f%% 的步（默认档 1.5%%），最低 %.1f mm"
      % (bt.peak_accel, pinned_t, zt.min()))
bv, rowsv = run(LiftParams(**{**DEF, "vel_limit_mps": 0.3}), 40.0)
check("速度上限 0.3 生效", abs(bv.peak_vel - 0.3) < 0.02, "峰速 %.3f m/s" % bv.peak_vel)

print("\n== 5. random 模式：行程同样不下地 ==")
br, rowsr = run(LiftParams(mode="random", amplitude_mm=150.0, accel_limit_mps2=2.0,
                           vel_limit_mps=0.5), 60.0)
zr = col(rowsr, 1)
check("行程 0 ~ +300mm", zr.min() >= -0.05 and zr.max() <= 2 * A * 1000 + 0.5,
      "实测 %.1f ~ %.1f mm" % (zr.min(), zr.max()))
check("限幅生效（峰速/峰加速贴上限）",
      abs(br.peak_vel - 0.5) < 0.02 and abs(br.peak_accel - 2.0) < 0.02,
      "峰速 %.2f / 峰加速 %.2f" % (br.peak_vel, br.peak_accel))
check("也有静止窗口（占空比 < 0.7）", br.duty < 0.7, "实测 %.2f" % br.duty)

print("\n== 6. 防嵌地：负的下沿会被夹回 0 ==")
bn, rowsn = run(LiftParams(**{**DEF, "bottom_mm": -150.0}), 30.0)
zn = col(rowsn, 1)
check("bottom_mm=-150 被夹到 0，行程不会低于贴地面", zn.min() >= -0.05,
      "实测最低 %.1f mm（若真按 -150 走会埋进地里）" % zn.min())

print("\n== 7. off 模式恒为 0（默认行为不能变）==")
bo, rowso = run(LiftParams(mode="off"), 5.0)
check("off 恒为 0.0", float(np.abs(col(rowso, 1)).max()) == 0.0)

print("\n== 8. 配置层 ==")
c = rm_config.load()
bl = c.target("infantry_wheel_legged").body_lift
check("targets.yaml 读出升幅/周期/下沿/占空比",
      bl.amplitude_mm == 150.0 and bl.period_s == 2.0 and bl.bottom_mm == 0.0
      and tuple(bl.active_cycles) == (1, 2) and len(tuple(bl.pause_s)) == 2,
      "%s" % (bl,))
check("默认靶车不带升降", c.target("infantry_omni").body_lift.mode == "off")
check("默认预设仍是 4 段（保基线可比）", len(c.preset().modes) == 4, str(c.preset().modes))
lift_preset = c.preset("benchmark_2026_lift")
check("新预设 benchmark_2026_lift = 负责人点名的四种靶型",
      tuple(lift_preset.modes) == CS.MODES_LIFT, str(lift_preset.modes))
check("四种靶型的运动学配置正确（固定/移动/旋转/随机）",
      CS._SPEC["lift_static"].translation == "stop" and CS._SPEC["lift_static"].spin == "none"
      and CS._SPEC["lift_linear"].translation == "patrol" and CS._SPEC["lift_linear"].spin == "none"
      and CS._SPEC["lift_spin"].translation == "stop" and CS._SPEC["lift_spin"].spin == "const"
      and CS._SPEC["lift_random"].translation == "patrol" and CS._SPEC["lift_random"].spin == "mod"
      and all(CS._SPEC[m].lift == "periodic" for m in CS.MODES_LIFT), "")
check("全部靶型的 lift 都写了（漏一个就会静默不动）",
      all(CS._SPEC[m].lift in ("off", "periodic", "random") for m in CS.MODES), "")
check("靶车血量 600（负责人指定口径）", c.target("infantry_wheel_legged").hp == 600.0)
check("规则集可切换（rmuc_2026）", rm_config.load(ruleset="rmuc_2026").ruleset_name == "rmuc_2026")

if FAILED:
  print("\n✗ %d 项未通过：%s" % (len(FAILED), "、".join(FAILED)))
  sys.exit(1)
print("\n✓ 全部通过")
