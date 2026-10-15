#!/usr/bin/env python3
"""弹道模型自检：验证"仿真里真的落了这条曲线"，以及默认行为没被动过。

跑法：  python3 rm_sim/test_ballistics.py          （需要 MuJoCo，几秒~十几秒）

检查项：
  1. 参数自洽：YAML 默认关；YAML 的直径/质量与仿真弹丸 geom 一致（不一致就是隐患）；
  2. **默认（关闭气动）**：仿真弹道 == 纯重力参考解（< 1 mm）；
     且 `apply_drag()` 在没打开时**不动物理**（返回 False）；
  3. **打开阻力**：仿真弹道 == 参考积分器（同款半隐式 Euler、dt=1 ms）（< 2 mm）；
  4. 数值误差量级：参考的 RK4（细步，视作解析真值）vs Euler（1 ms，MuJoCo 用的那个）的差；
  5. 单调性：开阻力后落点确实比关阻力时低，并打印 3/5/8/10 m × 22/24 m/s 对照表。

为什么能比得这么紧：MuJoCo 的 `integrator="Euler"` 是**半隐式**（先更新速度再更新位置），
阻力是在 `mj_step` **之前**写进 qvel 的 ⇒ 与参考解里"加速度 = 阻力 + 重力，v += a·dt，x += v·dt"
逐步同序，两者应当只差浮点舍入（自检里用 1~2 mm 的阈值兜住插值误差）。
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rm_ballistics as B  # noqa: E402
import rm_gimbal_demo as R  # noqa: E402
from rm_config import load as load_config  # noqa: E402

FAILED: list[str] = []
DISTANCES = (3.0, 5.0, 8.0, 10.0)
SPEEDS = (22.0, 24.0)          # 冷枪 / 热枪标称弹速
DT = 0.001                     # 与仿真同一个物理步长


def check(name: str, ok: bool, detail: str = "") -> None:
  print("  %s %s%s" % ("✓" if ok else "✗", name, ("  " + detail) if detail else ""))
  if not ok:
    FAILED.append(name)


def settle_and_aim(demo, steps: int = 250, pitch_deg: float = 8.0) -> None:
  """把云台推到 +y（**避开靶车**：简易模型里车在 +x 方向 1.6 m 处）、并**抬头 8°**，
  再推进若干步让伺服稳住。

  ⚠️ 符号：本项目的 `cmd.pitch` 沿用 **sp_vision 的约定「抬头为负」**（见 sim/README §3.2），
  所以"抬头 8°"要写 `-radians(8)`；写成正的会**朝下打**，弹丸一出膛就扎地
  （第一版就是栽在这儿，幸好加了"触地"守卫把它抓出来）。

  为什么要抬头：出膛点只比地面高 0.17 m，平射时弹丸 4~5 m 就落地，
  后面测到的会是"反弹后的球"而不是自由飞行；抬头 8° 后 10 m 内都在空中。
  具体角度不关心 —— 出膛方向是打之前从 muzzle() 现读的，参考解用的是同一个 θ。
  """
  demo.cmd.yaw = math.pi / 2.0
  demo.cmd.pitch = -math.radians(pitch_deg)      # 抬头为负！
  for _ in range(steps):
    demo.step(demo.model, demo.data)


def shoot_and_measure(demo, v0: float, distance: float, max_steps: int = 6000):
  """打一发（理想枪：显式给弹速 ⇒ 绕过枪模型与散布），返回

      (相对出膛点的竖直位移 m, 飞行时间 s, 俯仰角 rad, 弹丸槽号)

  水平位移首次 >= distance 时按线性插值取那一刻的值（与参考积分器同一口径）。
  """
  p0, dirv = demo.muzzle()
  dirv = dirv / float(np.linalg.norm(dirv))
  theta = math.asin(float(np.clip(dirv[2], -1.0, 1.0)))
  horiz = np.array([dirv[0], dirv[1], 0.0])
  hn = float(np.linalg.norm(horiz))
  if hn < 1e-9:
    return None
  horiz /= hn
  slot = int(demo.next_ball)
  if not demo.fire(speed=v0):
    return None                                  # 热量模型没开，不该被拒
  qa, da = demo.ball_qadr[slot], demo.ball_dadr[slot]

  t = 0.0
  prev = None                                    # (dx, dz, t)
  for _ in range(max_steps):
    demo.step(demo.model, demo.data)
    t += DT
    bp = np.array(demo.data.qpos[qa : qa + 3], dtype=float)
    dx = float(np.dot(bp - p0, horiz))
    dz = float(bp[2] - p0[2])
    # 守卫：弹丸一旦蹭到地面/车体就不再是自由飞行，测到的没有意义 —— 直接标出来，
    # 免得出一个"看起来像实现错了"的数字（第一版就是栽在这儿：水平射击 4~5 m 落地反弹）。
    if float(bp[2]) < 0.05:
      return None, "弹丸触地（z=%.3f m）" % float(bp[2])
    if dx >= distance:
      if prev is not None:
        d0, z0, t0 = prev
        frac = 0.0 if dx <= d0 else (distance - d0) / (dx - d0)
        frac = min(max(frac, 0.0), 1.0)
        return (z0 + (dz - z0) * frac, t0 + frac * DT, theta, slot), None
      return (dz, t, theta, slot), None
    prev = (dx, dz, t)
  return None, "超过 %d 步还没到 %.1f m" % (max_steps, distance)



def main() -> int:
  cfg = load_config(None)                        # None ⇒ 仓库里的 configs/
  raw_drag = cfg.ballistics.get("drag", {})
  print("== 1. 配置与参数自洽 ==")
  check("ballistics.yaml 默认关闭阻力", raw_drag.get("enabled") is False,
        "enabled=%r" % raw_drag.get("enabled"))
  p_off = B.from_config(cfg)
  check("from_config 读到了参数", p_off is not None)
  p_on = B.BallisticsParams(enabled=True, rho=p_off.rho, cd=p_off.cd,
                            diameter_mm=p_off.diameter_mm, mass_g=p_off.mass_g)
  check("YAML 直径与仿真弹丸 geom 一致（2*BALL_RADIUS）",
        abs(p_on.diameter_mm - 2.0 * R.BALL_RADIUS * 1000.0) < 1e-6,
        "yaml=%.2f mm，geom=%.2f mm" % (p_on.diameter_mm, 2.0 * R.BALL_RADIUS * 1000.0))
  check("YAML 质量与仿真弹丸质量一致（BALL_MASS）",
        abs(p_on.mass_g - R.BALL_MASS * 1000.0) < 1e-6,
        "yaml=%.2f g，geom=%.2f g" % (p_on.mass_g, R.BALL_MASS * 1000.0))
  print("    k = %.5f 1/m（a_drag = k·|v|·v；v0=24 时出膛阻力加速度 %.1f m/s² = g 的 %.0f%%）"
        % (p_on.k, p_on.k * 24.0 ** 2, 100 * p_on.k * 24.0 ** 2 / B.G))

  print()
  print("== 2. 关闭气动（默认）：仿真 == 纯重力参考解 ==")
  demo_off = R.Demo(dt=DT, ballistics_params=B.BallisticsParams(enabled=False))
  check("弹丸质量来自模型且等于 YAML",
        abs(float(demo_off.model.body_mass[demo_off.ball_bids_sorted[0]])
            - p_on.mass_g / 1000.0) < 1e-9,
        "model=%.4f g" % (demo_off.model.body_mass[demo_off.ball_bids_sorted[0]] * 1000))
  probe = np.array([24.0, 0.0, 0.0])
  check("apply_drag() 在关闭时不动速度（返回 False）",
        (demo_off.ballistics.apply_drag(probe, DT) is False) and float(probe[0]) == 24.0)
  settle_and_aim(demo_off)
  p_muz = demo_off.muzzle()[0]
  print("    出膛点高度 z=%.3f m（云台已抬 +8°，保证 10 m 内弹丸都在空中）" % float(p_muz[2]))
  worst_off = 0.0
  for dist in DISTANCES:
    for v0 in SPEEDS:
      got, err = shoot_and_measure(demo_off, v0, dist)
      if got is None:
        check("关闭阻力：打 %.0f m @ %.0f m/s" % (dist, v0), False, err or "测量失败")
        continue
      z_sim, _t_sim, theta, _ = got
      z_ref, _t_ref = B.fly(v0, theta, 0.0, dist, dt=DT, drag=False, scheme="euler")
      err = abs(z_sim - z_ref) * 1000.0
      worst_off = max(worst_off, err)
      print("    d=%4.0f m v0=%2.0f m/s | 仿真 %+7.2f cm | 参考 %+7.2f cm | 差 %5.2f mm"
            % (dist, v0, z_sim * 100, z_ref * 100, err))
  check("关闭阻力时仿真与纯重力参考解一致（< 1 mm）", worst_off < 1.0,
        "最大差 %.2f mm" % worst_off)

  print()
  print("== 3. 打开阻力：仿真 == 参考积分器（同款半隐式 Euler）==")
  demo_on = R.Demo(dt=DT, ballistics_params=p_on)
  settle_and_aim(demo_on)
  worst_on = 0.0
  rows = []
  for dist in DISTANCES:
    for v0 in SPEEDS:
      got, err = shoot_and_measure(demo_on, v0, dist)
      if got is None:
        check("开阻力：打 %.0f m @ %.0f m/s" % (dist, v0), False, err or "测量失败")
        continue
      z_sim, t_sim, theta, _ = got
      z_ref, _t_ref = B.fly(v0, theta, p_on.k, dist, dt=DT, drag=True, scheme="euler")
      z_rk, t_rk = B.fly(v0, theta, p_on.k, dist, dt=1e-5, drag=True, scheme="rk4")
      err = abs(z_sim - z_ref) * 1000.0
      worst_on = max(worst_on, err)
      rows.append((dist, v0, z_sim, z_rk, t_sim, t_rk))
      print("    d=%4.0f m v0=%2.0f m/s | 仿真 %+7.2f cm | 参考(Euler) %+7.2f cm | 差 %5.2f mm"
            % (dist, v0, z_sim * 100, z_ref * 100, err))
  check("打开阻力时仿真与参考积分器一致（< 2 mm）", worst_on < 2.0,
        "最大差 %.2f mm" % worst_on)

  print()
  print("== 4. 数值误差：参考 Euler(1 ms) vs RK4(细步)（MuJoCo 用的是 Euler）==")
  worst_num = 0.0
  for (_dist, _v0, z_sim, z_rk, _t_sim, _t_rk) in rows:
    worst_num = max(worst_num, abs(z_rk - z_sim) * 1000.0)
  check("Euler(1 ms) 的截断误差 < 5 mm（够用；再小就得换积分器）", worst_num < 5.0,
        "最大 %.2f mm" % worst_num)

  print()
  print("== 5. 对照表：同一条无阻力俯仰角下，开/关阻力的落点（解析 RK4）==")
  print("    %-6s %-6s %11s %11s %11s %10s %10s"
        % ("距离", "弹速", "无阻力", "有阻力", "差", "t 无阻力", "t 有阻力"))
  ok_mono = True
  for dist in DISTANCES:
    for v0 in SPEEDS:
      th = B.pitch_no_drag(dist, v0)             # 用"只有重力"的俯仰角去打
      z0, t0 = B.fly(v0, th, 0.0, dist, dt=1e-5, drag=False, scheme="rk4")
      z1, t1 = B.fly(v0, th, p_on.k, dist, dt=1e-5, drag=True, scheme="rk4")
      ok_mono = ok_mono and (z1 < z0)
      print("    %-6.0f %-6.0f %+10.2f %+10.2f %+10.2f %9.1f %9.1f"
            % (dist, v0, z0 * 100, z1 * 100, (z1 - z0) * 100, t0 * 1000, t1 * 1000))
  print("    （单位：落点 cm（相对出膛点，负=低）；时间 ms）")
  check("开阻力后落点一致性地更低（单调）", ok_mono)

  print()
  if FAILED:
    print("❌ 失败 %d 项：%s" % (len(FAILED), "；".join(FAILED)))
    return 1
  print("✅ 全部通过（默认关闭 ⇒ 物理逐位不变；打开后仿真 = 参考解）")
  return 0


if __name__ == "__main__":
  sys.exit(main())
