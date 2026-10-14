#!/usr/bin/env python3
"""枪的物理：摩擦轮弹速模型 + 枪口热量模型 + 弹丸随机散布。

为什么要有这个模块
------------------
以前的仿真里弹丸永远是 25 m/s 的理想直线，枪也不会热 —— 这三点在真机上都成立不了：

* **摩擦轮弹速**：3508 摩擦轮靠摩擦把弹丸甩出去。冷枪标称 22 m/s（为了不超官方
  25 m/s 上限），连打之后摩擦轮温度上来、摩擦系数变大，弹速慢慢升到 ~24 m/s 饱和；
  停机又会冷却回落。更关键的是：**视觉侧只知道操作手设定的标称值**，实际出膛速度
  和它对不上 —— 弹道解算因此周期性失配，这正是 RL 要学着扛的扰动。
* **枪口热量**：每发都会加热枪管，持续冷却。热量到上限就锁枪；继续超限按裁判系统
  语义扣我方血量。所以"能不能一直乱射"在真机上是生存问题，不是纪律问题
  —— 这也是我们把"击杀质量"定义成**热量约束下的 TTK** 的原因。
* **弹丸散布**：真枪每发都有枪口精度散布（真机部署时同样存在），它决定命中率的
  天花板；仿真理想要是没有这一项，RL 会学出"精确瞄准就必中"的错误世界观。

三个模型**各自可开关**（见 GunParams 里的 model_*），关掉即可复现旧基线。

参数全部是拍脑袋默认值（CLI 可覆盖），标注 **待真机校准**。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class GunParams:
  """枪的参数。全部可被 CLI 覆盖；带「待真机校准」的参数先给的是工程默认值。"""

  # ---- 弹速（摩擦轮）----
  speed_cold: float = 22.0     # 冷枪标称速度 m/s（也是发给视觉的标称值）
  speed_warm: float = 24.0     # 热枪饱和速度 m/s              【待真机校准】
  tau_warm: float = 20.0       # 升温时间常数，单位=发（20 发到 63%，60 发到 95%）
  tau_cool: float = 90.0       # 冷却时间常数，单位=秒          【待真机校准】
  jitter: float = 0.5          # 弹速随机波动 ±（按 2σ 解释，即 σ=jitter/2）

  # ---- 热量 ----
  heat_max: float = 100.0      # 热量上限                    【待真机校准】
  heat_per_shot: float = 10.0  # 每发加热量                   【待真机校准】
  heat_cool: float = 10.0      # 冷却速率 /s（满热到 0 约 10s）【待真机校准】
  # 超限伤害**与超出量成正比**（裁判语义：按超出热量扣血），单位 1/s：
  #   超出 0.5 点 -> 1 HP/s（贴着上限打几乎不疼）
  #   超出 10 点  -> 20 HP/s（乱射很快就没血）
  # 用比例而不是固定速率，是因为固定速率会让"贴着上限打"和"超一大截"代价一样，
  # 那样策略学不到"别乱射"这件事。                                【待真机校准】
  overheat_k: float = 2.0
  self_hp_max: float = 600.0   # 我方血量上限（仅用于记录超热量代价）
  # 硬阈值 vs 软超限（真机规则待负责人确认，两种都实现了，默认硬阈值）：
  #   hard_block=True  ：这一发会让热量**超过上限**时直接打不出去（真机"锁枪"语义）。
  #                      好处是"贴着上限稳打"不会掉血，射速预算就是纯粹的资源约束；
  #   hard_block=False ：允许最后一发顶到上限之上，超出的部分按比例扣血。
  #                      这时"狂按扳机"才有代价，但也让正常作战一直在掉血（不真实）。
  # 实测教训：一开始用的是软超限 + 锁枪的混合，结果**狂按扳机零代价**
  # （被锁的尝试不加热量、也不扣血），RL 会学出"永远请求开火"这种没意义的行为。
  hard_block: bool = True

  # ---- 散布 ----
  # 枪口角散布：每个正交轴各自的 1σ（度）。合成总偏差均值 ≈1.25×、σ≈0.655×。
  spread_deg: float = 0.15                                                       # 【待真机校准】

  # ---- 开关（关掉即回到旧行为，用于 A/B 与复现旧基线）----
  model_speed: bool = True
  model_heat: bool = True
  model_spread: bool = True


class GunModel:
  """一把枪的运行时状态。Demo 每步调 step()，每次开火调 take_shot()。"""

  def __init__(self, params: GunParams | None = None, rng=None, seed: int | None = None):
    self.p = params or GunParams()
    self.rng = rng if rng is not None else np.random.default_rng(seed)
    # warm 用"发当量"表示：每开一发 +1，停机按 tau_cool 指数衰减
    self.warm = 0.0
    self.heat = 0.0
    self.self_hp = float(self.p.self_hp_max)
    self.locked = False
    self.shots_fired = 0
    self.shots_blocked = 0        # 因为锁枪没打出去的
    self.overheat_time = 0.0      # 累计超限秒数
    self.overheat_damage = 0.0    # 累计扣掉的我方血量
    self.speed_prev = float(self.p.speed_cold)

  # ------------------------------------------------------------------ 查询
  @property
  def nominal_speed(self) -> float:
    """**发给视觉的标称速度**。真机上视觉只知道操作手设的这个值，不知道实际弹速。"""
    return float(self.p.speed_cold)

  def current_speed(self) -> float:
    """当前的实际出膛速度（含升温），不含每发随机波动。"""
    if not self.p.model_speed:
      return float(self.p.speed_cold)
    frac = 1.0 - math.exp(-max(self.warm, 0.0) / max(self.p.tau_warm, 1e-6))
    return float(self.p.speed_cold + (self.p.speed_warm - self.p.speed_cold) * frac)

  def heat_fraction(self) -> float:
    return float(self.heat / max(self.p.heat_max, 1e-6))

  def can_fire(self) -> bool:
    """现在允许开火吗？热量模型关掉时永远允许。"""
    p = self.p
    if not p.model_heat:
      return True
    if p.hard_block:
      # 硬阈值：判据是"这一发会不会顶破上限"，不是"当前是否已超限" ——
      # 否则会永远停在"超限 -> 被拒 -> 冷却到刚好不超 -> 又打出去"的锯齿上
      return self.heat + p.heat_per_shot <= p.heat_max
    return self.heat < p.heat_max

  # ------------------------------------------------------------------ 每步
  def step(self, dt: float) -> None:
    p = self.p
    if p.model_speed and p.tau_cool > 0:
      self.warm = max(0.0, self.warm - self.warm * dt / p.tau_cool)
    if not p.model_heat:
      return
    self.heat = max(0.0, self.heat - p.heat_cool * dt)
    if self.heat > p.heat_max:
      # 超限：按裁判语义持续扣我方血量（这也是"乱射"的真实代价）。
      # 伤害与超出量成正比 —— 犯错越狠罚得越重，而不是一刀切。
      self.overheat_time += dt
      dmg = p.overheat_k * (self.heat - p.heat_max) * dt
      self.self_hp = max(0.0, self.self_hp - dmg)
      self.overheat_damage += dmg
    self.locked = self.heat >= p.heat_max

  # ------------------------------------------------------------------ 开火
  def take_shot(self) -> tuple[float, float]:
    """消耗一次开火，返回 (实际出膛速度, 本发的速度随机偏移)。

    注意：允许"最后一发把热量顶到上限以上"（真机也是这一发打出去之后才锁枪），
    所以超限伤害是有可能真的发生的 —— 这正是要建模的取舍。
    """
    p = self.p
    self.shots_fired += 1
    if p.model_speed:
      self.warm += 1.0
    v = self.current_speed()
    dv = float(self.rng.normal(0.0, p.jitter / 2.0)) if p.model_speed else 0.0
    v = max(v + dv, 1e-3)
    self.speed_prev = v
    if p.model_heat:
      self.heat += p.heat_per_shot
      self.locked = self.heat >= p.heat_max
    return v, dv

  def perturb_dir(self, dirv: np.ndarray) -> np.ndarray:
    """给发射方向叠加枪口散布（在垂直于炮管的平面内随机偏一个角度）。"""
    p = self.p
    d = np.asarray(dirv, dtype=float)
    if not p.model_spread or p.spread_deg <= 0.0:
      return d
    n = np.linalg.norm(d)
    if n < 1e-9:
      return d
    d = d / n
    # 构造与 d 垂直的一组基
    tmp = np.array([0.0, 0.0, 1.0]) if abs(d[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(d, tmp)
    u = u / (np.linalg.norm(u) + 1e-12)
    w = np.cross(d, u)
    # spread_deg 的口径 = **每个正交轴各自的 1σ**（最不容易误读的定义）。
    # 合成后的总角偏差 θ=√(a1²+a2²) 服从 Rayleigh 分布：
    #   尺度参数 = spread_deg，均值 ≈ 1.25×spread_deg，标准差 ≈ 0.655×spread_deg。
    # 要按"总角偏差 1σ"给参数的话，除以 0.655 即可。
    sigma = math.radians(p.spread_deg)
    a1 = float(self.rng.normal(0.0, sigma))
    a2 = float(self.rng.normal(0.0, sigma))
    out = d + math.tan(a1) * u + math.tan(a2) * w
    return out / (np.linalg.norm(out) + 1e-12)

  # ------------------------------------------------------------------ 导出
  def status(self) -> dict:
    return {
      "nominal_speed": round(self.nominal_speed, 3),
      "actual_speed": round(self.current_speed(), 3),
      "speed_last": round(self.speed_prev, 3),
      "warm": round(self.warm, 2),
      "heat": round(self.heat, 2),
      "heat_frac": round(self.heat_fraction(), 3),
      "locked": bool(self.locked),
      "self_hp": round(self.self_hp, 1),
      "overheat_time": round(self.overheat_time, 2),
      "overheat_damage": round(self.overheat_damage, 1),
      "shots_fired": int(self.shots_fired),
      "shots_blocked": int(self.shots_blocked),
      "model_speed": bool(self.p.model_speed),
      "model_heat": bool(self.p.model_heat),
      "model_spread": bool(self.p.model_spread),
    }
