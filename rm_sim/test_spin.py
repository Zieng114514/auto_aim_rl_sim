#!/usr/bin/env python3
"""自旋（马格努斯力）自检：默认关时逐位不变；打开后横向漂移与独立 3D 参考解一致。

跑法：  python3 rm_sim/test_spin.py            （需要 MuJoCo，几秒~十几秒）

检查项：
  1. 配置与模型：`spin` 段能读到；默认关；`C_L(S)` 单调、会饱和；
  2. **默认关**：出膛后弹丸的角速度仍是 0（历史行为），横向漂移为 0（< 0.1 mm）；
  3. **打开自旋**（配空气阻力）：横向漂移与**独立 3D RK4 参考解**一致（< 2 mm）；
  4. 物理方向性：`axis=barrel_up` ⇒ 漂移在**横向**、竖直落点几乎不变；
     `axis=barrel_side` ⇒ 反过来作用在**竖直**方向；
  5. 打印一张表（距离 → 横漂），供文档与真机标定参考（横漂 ∝ d²）。

为什么能比得这么紧：与阻力一样，马格努斯力是"在 `mj_step` 之前改弹丸 qvel"，
MuJoCo 的 Euler 是半隐式 ⇒ 与参考解的发力顺序一致（差异只剩离散化 ~mm 级）。
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rm_ballistics as B  # noqa: E402
import rm_gimbal_demo as R  # noqa: E402
import test_ballistics as TB  # noqa: E402  复用"抬头 8°、避开靶车"的瞄准助手
from rm_config import load as load_config  # noqa: E402

FAILED: list[str] = []
DT = 0.001
DISTANCES = (3.0, 5.0, 8.0, 10.0)
V0 = 24.0


def check(name: str, ok: bool, detail: str = "") -> None:
  print("  %s %s%s" % ("✓" if ok else "✗", name, ("  " + detail) if detail else ""))
  if not ok:
    FAILED.append(name)


# ---------------------------------------------------------------- 独立 3D 参考解
def ref_fly3d(v0_vec, omega_vec, bp: B.BallisticsParams, distance: float, dt: float = 1e-4):
  """独立参考（RK4）：a = -g ẑ - k|v|v + ½ρA·C_L(S)/m·|v|²·(ω̂ × v̂)，推进到水平距离 distance。

  返回（竖直位移 dz、侧向位移 dlat、飞行时间 t）；侧向轴 = ẑ × 水平方向（右手）。
  """
  v = np.array(v0_vec, dtype=float)
  w = np.array(omega_vec, dtype=float)
  k = bp.k if bp.enabled else 0.0
  m = bp.mass_g * 1e-3
  r = bp.diameter_mm * 1e-3 / 2.0
  horiz = np.array([v[0], v[1], 0.0])
  horiz /= np.linalg.norm(horiz)
  lat = np.cross([0.0, 0.0, 1.0], horiz)

  def acc(v_, w_):
    a = np.array([0.0, 0.0, -B.G])
    s = float(np.linalg.norm(v_))
    if k > 0.0 and s > 0.0:
      a -= k * s * v_
    ws = float(np.linalg.norm(w_))
    if bp.spin.enabled and s > 0.0 and ws > 0.0:
      cl = bp.spin.cl(ws * r / s)
      a += (0.5 * bp.rho * bp.area_m2 * cl * s * s / m) * np.cross(w_ / ws, v_ / s)
    return a

  p = np.zeros(3)
  t = 0.0
  prev = None
  for _ in range(4_000_000):
    a1 = acc(v, w)
    a2 = acc(v + 0.5 * dt * a1, w)
    a3 = acc(v + 0.5 * dt * a2, w)
    a4 = acc(v + dt * a3, w)
    v = v + dt * (a1 + 2 * a2 + 2 * a3 + a4) / 6.0
    p = p + v * dt
    t += dt
    dx = float(np.dot(p, horiz))
    if prev is not None and dx >= distance:
      d0, z0, l0, t_0 = prev
      frac = 0.0 if dx <= d0 else (distance - d0) / (dx - d0)
      frac = min(max(frac, 0.0), 1.0)
      return (z0 + (float(p[2]) - z0) * frac,
              l0 + (float(np.dot(p, lat)) - l0) * frac, t_0 + (t - t_0) * frac)
    prev = (dx, float(p[2]), float(np.dot(p, lat)), t)
  raise RuntimeError("参考解没推进到 %.1f m" % distance)


# ---------------------------------------------------------------- 仿真侧测量
def shoot3d(demo, v0: float, distance: float, max_steps: int = 8000):
  """打一发（理想枪），沿出膛水平方向推进到 distance，返回 (dz, dlat, t, v0_vec, omega)。

  侧向轴 lat = ẑ × 水平方向（与参考解同一约定）；正负号一致才有可比性。
  """
  p0, dirv = demo.muzzle()
  dirv = dirv / float(np.linalg.norm(dirv))
  horiz = np.array([dirv[0], dirv[1], 0.0])
  horiz /= float(np.linalg.norm(horiz))
  lat = np.cross(np.array([0.0, 0.0, 1.0]), horiz)
  slot = int(demo.next_ball)
  if not demo.fire(speed=v0):
    return None
  qa, da = demo.ball_qadr[slot], demo.ball_dadr[slot]
  w0 = np.array(demo.data.qvel[da + 3 : da + 6], dtype=float)
  v0_vec = dirv * v0

  t = 0.0
  prev = None
  for _ in range(max_steps):
    demo.step(demo.model, demo.data)
    t += DT
    bp = np.array(demo.data.qpos[qa : qa + 3], dtype=float)
    rel = bp - p0
    dx = float(np.dot(rel, horiz))
    dz = float(rel[2])
    dl = float(np.dot(rel, lat))
    if float(bp[2]) < 0.05:
      return None                      # 触地：不再是自由飞行
    if dx >= distance:
      if prev is not None:
        d0, z0, l0, t0 = prev
        frac = 0.0 if dx <= d0 else (distance - d0) / (dx - d0)
        frac = min(max(frac, 0.0), 1.0)
        return (z0 + (dz - z0) * frac, l0 + (dl - l0) * frac, t0 + frac * DT, v0_vec, w0)
      return dz, dl, t, v0_vec, w0
    prev = (dx, dz, dl, t)
  return None


def main() -> int:
  cfg = load_config(None)
  p0 = B.from_config(cfg)
  print("== 1. 配置与模型 ==")
  check("ballistics.yaml 默认关闭自旋", p0.spin.enabled is False,
        "enabled=%r" % p0.spin.enabled)
  check("C_L(S) 单调且在 cl_max 处饱和",
        p0.spin.cl(0.01) < p0.spin.cl(0.1) < p0.spin.cl(0.5) <= p0.spin.cl_max + 1e-12,
        "C_L(0.1)=%.3f, C_L(0.5)=%.3f, cl_max=%.2f"
        % (p0.spin.cl(0.1), p0.spin.cl(0.5), p0.spin.cl_max))
  m = p0.mass_g * 1e-3
  r = p0.diameter_mm * 1e-3 / 2.0
  s_ref = p0.spin.omega_rad_s * r / V0
  a_ref = 0.5 * p0.rho * p0.area_m2 * p0.spin.cl(s_ref) * V0 * V0 / m
  print("    ω=%.0f rad/s ⇒ S=%.3f、C_L=%.3f ⇒ 横向加速度 %.2f m/s²"
        % (p0.spin.omega_rad_s, s_ref, p0.spin.cl(s_ref), a_ref))

  # 两个工况：① 只开阻力 ② 阻力+自旋(up)（自旋轴在 §4 里用 attribute 切换，避免多 demo 姿态漂移）
  common = dict(rho=p0.rho, cd=p0.cd, diameter_mm=p0.diameter_mm, mass_g=p0.mass_g)
  p_drag = B.BallisticsParams(enabled=True, **common)
  p_spin = B.BallisticsParams(enabled=True, spin=B.SpinParams(
    enabled=True, omega_rad_s=p0.spin.omega_rad_s, axis="barrel_up",
    cl_slope=p0.spin.cl_slope, cl_max=p0.spin.cl_max), **common)

  print()
  print("== 2. 默认关：出膛角速度必须是 0、横向漂移必须是 0 ==")
  demo_off = R.Demo(dt=DT, ballistics_params=B.BallisticsParams(enabled=False))
  TB.settle_and_aim(demo_off)
  got = shoot3d(demo_off, V0, 5.0)
  check("出膛 ω = 0", got is not None and float(np.linalg.norm(got[4])) == 0.0,
        "|ω|=%.6f" % (float(np.linalg.norm(got[4])) if got else -1.0))
  check("横向漂移 = 0（<0.1 mm）", got is not None and abs(got[1]) < 1e-4,
        "dlat=%.6f m" % (got[1] if got else 0.0))

  print()
  print("== 3. 阻力 + 自旋（barrel_up）：与独立 3D 参考解一致 ==")
  demo = R.Demo(dt=DT, ballistics_params=p_spin)
  TB.settle_and_aim(demo)
  print("    %-6s %-13s %-13s %-9s %-13s %-13s %s"
        % ("距离", "横漂(仿真)", "横漂(参考)", "差(mm)", "竖直(仿真)", "竖直(参考)", "结论"))
  worst = 0.0
  for dist in DISTANCES:
    got = shoot3d(demo, V0, dist)
    if got is None:
      check("阻力+自旋 打 %.0f m" % dist, False, "测量失败")
      continue
    dz, dl, _t, v0_vec, w0 = got
    rz, rl, _rt = ref_fly3d(v0_vec, w0, p_spin, dist)
    err_l = abs(dl - rl) * 1000.0
    worst = max(worst, err_l)
    print("    %-6.0f %-13.2f %-13.2f %-9.2f %-13.2f %-13.2f %s"
          % (dist, dl * 100, rl * 100, err_l, dz * 100, rz * 100,
             "一致" if err_l < 2.0 else "**偏了**"))
  check("横向漂移与 3D 参考解一致（< 2 mm）", worst < 2.0, "最大差 %.2f mm" % worst)

  print()
  print("== 4. 方向性：barrel_up ⇒ 漂在横向；barrel_side ⇒ 作用在竖直 ==")
  print("    （**同一次瞄准下切换自旋开关/轴**，排除\"三个 demo 各自 stable 后姿态略有差异\"的干扰 ——")
  print("      上一版就是这样被污染的：跨 demo 比较时炮口仰角差 0.6° ⇒ 8 m 处就差了 8 cm）")
  demo_t = R.Demo(dt=DT, ballistics_params=p_drag)   # 先只开阻力，自旋靠 attribute 切换
  TB.settle_and_aim(demo_t)

  def one(spin_on: bool, axis: str):
    demo_t.ballistics.spin.enabled = bool(spin_on)
    demo_t.ballistics.spin.axis = axis
    return shoot3d(demo_t, V0, 8.0)

  gd = one(False, "barrel_up")     # 无自旋
  gu = one(True, "barrel_up")      # 自旋轴 = 炮管的"上"（力 = ω̂×v̂ ⟂ 两者 ⇒ 横向）
  gs = one(True, "barrel_side")    # 自旋轴 = 炮管的"右" ⇒ 力竖直
  print("    8 m：只有阻力       dz=%+6.2f cm  dlat=%+6.2f cm" % (gd[0] * 100, gd[1] * 100))
  print("    8 m：阻力+自旋(up)   dz=%+6.2f cm  dlat=%+6.2f cm" % (gu[0] * 100, gu[1] * 100))
  print("    8 m：阻力+自旋(side) dz=%+6.2f cm  dlat=%+6.2f cm" % (gs[0] * 100, gs[1] * 100))
  check("barrel_up：横漂显著（>10 cm@8m）而竖直几乎不变（<0.5 cm）",
        abs(gu[1]) > 0.10 and abs(gu[0] - gd[0]) < 0.005,
        "dlat=%.2f cm，Δdz=%.2f cm" % (gu[1] * 100, (gu[0] - gd[0]) * 100))
  check("barrel_side：竖直被显著改变（>5 cm@8m）而横漂几乎不变（<0.5 cm）",
        abs(gs[0] - gd[0]) > 0.05 and abs(gs[1] - gd[1]) < 0.005,
        "Δdz=%.2f cm，Δdlat=%.2f cm" % ((gs[0] - gd[0]) * 100, (gs[1] - gd[1]) * 100))

  print()
  print("== 5. 横漂随距离（真机标定参考：横漂 ∝ d²）==")
  print("    %-6s %s" % ("距离", "横漂(cm)"))
  for dist in DISTANCES:
    got = shoot3d(demo, V0, dist)
    if got is not None:
      print("    %-6.0f %+8.2f" % (dist, got[1] * 100))

  print()
  if FAILED:
    print("❌ 失败 %d 项：%s" % (len(FAILED), "；".join(FAILED)))
    return 1
  print("✅ 全部通过（默认关 ⇒ 逐位不变；打开后与 3D 参考解一致）")
  return 0


if __name__ == "__main__":
  sys.exit(main())
