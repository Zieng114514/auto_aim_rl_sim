#!/usr/bin/env python3
"""弹丸弹道模型：参数、逐步施力、以及一个**参考积分器**（给自检和文档当解析真值）。

三条设计约束（与项目其它模块一致）：

1. **默认什么都不做**：`BallisticsParams()`（`enabled=False`）时 `apply_*` 立即返回，
   既有基线与黄金轨迹 hash 逐位不变。
2. **数值只在 YAML 里**（`configs/ballistics.yaml`），代码不写死 ρ/Cd/直径。
3. 这里额外实现一份**参考积分器**（RK4 与"和 MuJoCo 同款的半隐式 Euler"），
   用途只有一个：让 `test_ballistics.py` 能验证"仿真里真的落了这条曲线"。

单位与坐标系（本项目踩过坐标系的坑，写清楚）：
* 全部 SI：m / s / kg / rad；角度用弧度。
* 速度 = MuJoCo 的 `qvel`（世界系线速度，前 3 个分量）。
* 参考积分器只在**竖直平面**里算：(x=水平前进方向, z=竖直向上)，返回相对出膛点的位移。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

G = 9.81          # 与 rm_gimbal_demo 的 <option gravity="0 0 -9.81"> 一致。
                  # （sp_vision 解算器用 9.7833，差 0.27%：5 m 处落点差 0.6 mm，可忽略）


@dataclass
class BallisticsParams:
  """气动阻力参数。`enabled=False` ⇒ 完全不参与物理（默认）。"""

  enabled: bool = False
  rho: float = 1.225        # 空气密度 kg/m^3
  cd: float = 0.47          # 阻力系数（无量纲）
  diameter_mm: float = 17.0  # 弹丸直径 mm（与 BALL_RADIUS 取齐）
  mass_g: float = 3.2        # 弹丸质量 g

  @property
  def area_m2(self) -> float:
    r = self.diameter_mm * 1e-3 / 2.0
    return math.pi * r * r

  @property
  def k(self) -> float:
    """阻力系数 k：a_drag = -k·|v|·v（单位 1/m）。"""
    m = self.mass_g * 1e-3
    return 0.5 * self.rho * self.cd * self.area_m2 / m

  def describe(self) -> str:
    if not self.enabled:
      return "弹道：只有重力（气动阻力关闭）"
    return ("弹道：重力 + 空气阻力  k=%.5f 1/m（ρ=%.3f kg/m³, Cd=%.2f, d=%.1f mm, m=%.1f g）"
            % (self.k, self.rho, self.cd, self.diameter_mm, self.mass_g))

  # ---------------------------------------------------------------- 施力
  def apply_drag(self, qvel_lin: np.ndarray, dt: float) -> bool:
    """对**一段**线速度就地施加一个 Euler 步的阻力：v ← v - k·|v|·v·dt。

    返回是否真的动了（关闭时返回 False，调用方可用它做自检）。
    符号与量纲：k 单位 1/m，|v|·v 是 m²/s²，故 k|v|v 是 m/s²（加速度）✓
    """
    if not self.enabled:
      return False
    s = float(np.linalg.norm(qvel_lin))
    if s <= 0.0:
      return False
    qvel_lin[:] = qvel_lin - (self.k * s * dt) * qvel_lin
    return True

  def apply_to_balls(self, data, qadr, dadr, ball_t, dt: float) -> int:
    """对弹丸池里**飞行中**（TTL>0）的每一发施加阻力。返回改动了几发。

    注意：MuJoCo 的 Euler 是半隐式（先更新速度再更新位置），
    所以"在 mj_step 之前改 qvel"= 这一步的加速度里含阻力，与参考积分器的
    `scheme="euler"` 完全同序 ✓（自检里就是这么对齐的）。
    """
    if not self.enabled:
      return 0
    n = 0
    for i in range(len(ball_t)):
      if ball_t[i] <= 0.0:
        continue
      da = dadr[i]
      if self.apply_drag(data.qvel[da : da + 3], dt):
        n += 1
    return n


def from_config(cfg) -> BallisticsParams | None:
  """从配置层（rm_config.Config）读 `configs/ballistics.yaml`。

  cfg 为 None 或没这段配置时返回 None（= 不做任何气动物理，与历史行为一致）。
  """
  if cfg is None:
    return None
  raw = getattr(cfg, "ballistics", None)
  if not isinstance(raw, dict):
    return None
  d = raw.get("drag") or {}
  if "enabled" not in d:
    return None
  return BallisticsParams(
    enabled=bool(d.get("enabled", False)),
    rho=float(d.get("rho_kg_m3", 1.225)),
    cd=float(d.get("cd", 0.47)),
    diameter_mm=float(d.get("diameter_mm", 17.0)),
    mass_g=float(d.get("mass_g", 3.2)),
  )


# ================================================================ 参考积分器
def fly(v0: float, pitch: float, k: float, distance: float,
        dt: float = 1e-5, drag: bool = True, scheme: str = "rk4") -> tuple[float, float]:
  """水平推进到 `distance`，返回（相对出膛点的竖直位移 m，飞行时间 s）。

  * `scheme="rk4"`  ：4 阶（默认 dt=1e-5，视作解析真值）
  * `scheme="euler"`：半隐式 Euler —— **与 MuJoCo `integrator="Euler"` 同款**，
                     用 dt=0.001 就能和仿真逐步对齐（自检用的就是这一支）

  `k=0` 或 `drag=False` 时退化成纯重力抛物线。
  """
  vx = v0 * math.cos(pitch)
  vz = v0 * math.sin(pitch)
  x = z = t = 0.0

  def acc(vx_: float, vz_: float) -> tuple[float, float]:
    if not drag or k <= 0.0:
      return 0.0, -G
    s = math.hypot(vx_, vz_)
    return -k * s * vx_, -G - k * s * vz_

  def cross(x0, z0, t0, vx_, vz_):
    """跨过目标距离时按线性插值给出那一刻的 (z, t)。"""
    if vx_ <= 1e-12:
      return z0, t0
    frac = (distance - x0) / (vx_ * dt) if scheme == "euler" else (distance - x0) / (vx_ * dt)
    frac = min(max(frac, 0.0), 1.0)
    return z0 + vz_ * dt * frac, t0 + dt * frac

  guard = 0
  while x < distance and guard < 20_000_000:
    guard += 1
    if scheme == "euler":
      ax, az = acc(vx, vz)
      vx += ax * dt
      vz += az * dt
      x0, z0, t0 = x, z, t
      x += vx * dt
      z += vz * dt
      t += dt
      if x >= distance:
        return cross(x0, z0, t0, vx, vz)
    else:
      ax1, az1 = acc(vx, vz)
      ax2, az2 = acc(vx + 0.5 * dt * ax1, vz + 0.5 * dt * az1)
      ax3, az3 = acc(vx + 0.5 * dt * ax2, vz + 0.5 * dt * az2)
      ax4, az4 = acc(vx + dt * ax3, vz + dt * az3)
      x0, z0, t0 = x, z, t
      vx += dt * (ax1 + 2 * ax2 + 2 * ax3 + ax4) / 6
      vz += dt * (az1 + 2 * az2 + 2 * az3 + az4) / 6
      x += vx * dt
      z += vz * dt
      t += dt
      if x >= distance:
        return cross(x0, z0, t0, vx, vz)
  return z, t


def pitch_no_drag(distance: float, v0: float, g: float = G) -> float:
  """纯重力下的低伸弹道俯仰角（与 sp_vision `trajectory.cpp` 同式），单位 rad。

  仅用于自检/对照：给"打这个距离需要抬多少"一个解析参照。
  """
  a = g * distance * distance / (2.0 * v0 * v0)
  disc = distance * distance - 4.0 * a * a
  if disc <= 0.0:
    return float("nan")
  return math.atan((distance - math.sqrt(disc)) / (2.0 * a))
