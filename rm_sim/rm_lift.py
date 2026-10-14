#!/usr/bin/env python3
"""轮腿靶车的**车体升降**：四块装甲板在 z 轴上上下移动。

为什么要有它（负责人 2026-10-14）
--------------------------------
轮腿步兵能升高/降低车体。交战时它的四块装甲板会**在 z 轴上上下移动**，
而且通常**同时还在变速小陀螺**。这对自瞄的意义是：以前靶车的竖直方向几乎是"静止"的
（只有地形起伏 + 安装角偏移），现在竖直方向有了真实的预测误差来源 —— 正好打在
自瞄最弱的一环上。

行程：为什么不是 -150 ~ +150（负责人实测发现"装甲板嵌地上"后改的）
------------------------------------------------------------------
量了几何才知道：我们的靶车是**单个自由体**，"贴地姿态"（= 其它所有靶型用的那个姿态）
就是**车体几何最低点正好落在地面上**（`ride_z ≈ -4mm`，网格最低点 = 轮子底面），
所以任何**下行程都等于往地下钻**：装甲板下沿本来只有 +65mm，-150mm 就埋进去 85mm。

现在改成（默认档，单位 mm）：

    下沿 = bottom_mm        默认 0   （= 贴地姿态，与其它靶型完全一致，绝不嵌地）
    上沿 = bottom_mm + 2·A  默认 +300
    指令 = bottom + A·(1 + sinφ)      <- 干净正弦，半幅仍是 A = 150

也就是把"中位"从贴地面抬到 +150，行程 0 ~ +300。真机上"降低车体"是腿收拢、
**轮子仍在地面**；单刚体模型做不到，所以宁可少一段下行程，也不嵌地。
（想让车体也能往下走，只能先给它一个抬高的基准姿态 —— 未实现，见 configs/README。）

占空比：不期望全程都在动（负责人 2026-10-14）
--------------------------------------------
加了一层"活动窗口 + 静止窗口"：

    活动窗口 = 1~2 个完整升降周期（默认 2~4 s），结束后回到贴地姿态
    静止窗口 = pause_s（默认 2.5~5 s 随机）
    启动时先在贴地姿态停一个静止窗口（让自瞄有机会先捕获目标，也便于分段统计）

两个窗口的起止都落在"贴地姿态 + 速度为 0"（正弦相位从 -π/2 起、按**整数周期**结束），
所以进出窗口是**平滑**的，没有速度/加速度突跳。宏观上大约 40% 的时间在动，
其余时间是"普通静止靶"—— 这正是负责人的要求。

机构怎么跟指令（这才是这个模块的核心）
------------------------------------
    periodic : a_cmd = a_target(指令的解析加速度，前馈) + kp·(target−pos) + kd·(v_target−vel)
    random   : 指令是阶跃，直接给"刹得住的接近速度" v = √(2a|err|)，再折算成加速度需求
然后统一过限幅：**加速度**先截到 ±accel_limit，**速度**再截到 ±vel_limit。
能力足够时误差恒为 0、PD 项不参与 → 轨迹与解析式完全一致；能力不足才削峰。

默认取值（负责人让我先定一个、再用 run_debug.sh 确认）
----------------------------------------------------
    A = 150 mm、周期 2 s  ->  正弦本身需要
       峰值速度 = A·ω = 0.15 · π       ≈ 0.47 m/s
       峰值加速度 = A·ω² = 0.15 · π²   ≈ 1.48 m/s²
    所以默认取 vel_limit = 0.5 m/s、accel_limit = 2.0 m/s²：
    **刚好不扭曲这个正弦**（实测默认档 0% 的步饱和），但把上限留在配置里 —— 调紧它，
    轨迹就被压扁（高度变小、贴着上限跑），一眼能从曲线图上看出来。

适用边界：限幅值必须 ≥ 上面两个物理需求，否则机构跟不上指令，轨迹不再等于解析式。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class LiftParams:
  mode: str = "periodic"           # periodic | random | off
  amplitude_mm: float = 150.0      # 升降半幅（负责人指定）
  period_s: float = 2.0            # 一个完整升降周期（负责人指定）
  accel_limit_mps2: float = 2.0    # 【待校准】机械加速度上限
  vel_limit_mps: float = 0.5       # 【待校准】机械速度上限
  rand_hold_s: tuple = (0.35, 0.8) # random 模式换目标高度的间隔
  bottom_mm: float = 0.0           # 行程下沿（相对贴地姿态）；< 0 会被夹回 0（会嵌地）
  active_cycles: tuple = (1, 2)    # 每个活动窗口含几个完整升降
  pause_s: tuple = (2.5, 5.0)      # 活动窗口之间的静止时长；空/None = 全程都在动
  pos_gain: float = 25.0           # periodic 模式位置反馈增益 kp (1/s²)，只用来防止限幅时的位置漂移
  vel_gain: float = 10.0           # periodic 模式速度反馈增益 kd (1/s)，给上面的位置环提供阻尼
  phase: float = 0.0

  @property
  def amplitude(self) -> float:
    return self.amplitude_mm / 1000.0

  @property
  def bottom(self) -> float:
    """行程下沿（米）。负值 = 往地下走 = 嵌地，夹回 0（贴地姿态）。"""
    return max(self.bottom_mm, 0.0) / 1000.0

  @classmethod
  def from_config(cls, bl, **kw) -> "LiftParams":
    """从 rm_config.BodyLift 构造（配置层 -> 物理层）。"""
    return cls(mode=str(bl.mode), amplitude_mm=float(bl.amplitude_mm),
               period_s=float(bl.period_s), accel_limit_mps2=float(bl.accel_limit_mps2),
               vel_limit_mps=float(bl.vel_limit_mps),
               rand_hold_s=tuple(bl.rand_hold_s),
               bottom_mm=float(getattr(bl, "bottom_mm", 0.0)),
               active_cycles=tuple(getattr(bl, "active_cycles", (1, 2))),
               pause_s=tuple(getattr(bl, "pause_s", (2.5, 5.0))), **kw)


class BodyLift:
  """车体升降状态机。每物理步调一次 step(dt)，返回"相对贴地姿态的偏移"（米）。"""

  def __init__(self, p: LiftParams | None = None, rng=None):
    self.p = p or LiftParams()
    self.rng = rng if rng is not None else np.random.default_rng(0)
    self.t = 0.0
    self.pos = 0.0                 # 当前偏移
    self.vel = 0.0                 # 当前速度
    self.target = 0.0              # 当前目标高度
    self.v_target = 0.0            # 目标高度的**解析速度**（periodic 模式前馈用）
    self.a_target = 0.0            # 目标高度的**解析加速度**（periodic 模式前馈用）
    self._hold_left = 0.0
    self._phase = -0.5 * math.pi + self.p.phase   # 从正弦下沿起步（= 贴地姿态）
    self._phase_end = 0.0          # 本窗口的结束相位（整数周期 -> 正好回到下沿）
    self.active = False            # 当前是"活动窗口"还是"静止窗口"
    self._window_left = 0.0        # 当前窗口剩余时长（估计值，仅用于状态输出）
    self.active_time = 0.0         # 统计：真正在动的时间
    self.total_time = 0.0
    self.peak_accel = 0.0          # 统计用（自检/日志）
    self.peak_vel = 0.0
    self._started = False

  # ------------------------------------------------------------------ 占空比
  def _scheduled(self) -> bool:
    """要不要走"活动/静止窗口"。pause_s 为空 -> 全程都在动（老行为，测试用）。"""
    return bool(self.p.pause_s)

  def _draw_pause(self) -> float:
    lo, hi = self.p.pause_s if len(self.p.pause_s) > 1 else (self.p.pause_s[0],) * 2
    return float(self.rng.uniform(lo, hi))

  def _enter_pause(self) -> None:
    self.active = False
    self._window_left = self._draw_pause()
    if self._scheduled():
      self.target, self.v_target, self.a_target = self.p.bottom, 0.0, 0.0

  def _enter_active(self) -> None:
    self.active = True
    n = self.rng.integers(self.p.active_cycles[0], self.p.active_cycles[1] + 1)
    # 相位从 -π/2 起、走整数个周期：sin(-π/2) = -1 -> target = 下沿，
    # 结束时同理，且两端 cos = 0 -> 速度也是 0，所以进出窗口都平滑。
    self._phase = -0.5 * math.pi + self.p.phase
    self._phase_end = self._phase + 2.0 * math.pi * float(n)
    self._window_left = (self._phase_end - self._phase) / max(2.0 * math.pi / self.p.period_s, 1e-9)

  # ------------------------------------------------------------------ 目标
  def _update_target(self, dt: float) -> None:
    A = self.p.amplitude
    if self.p.mode == "off":
      self.target, self.v_target, self.a_target = 0.0, 0.0, 0.0
      return

    # 窗口推进：**统一用剩余时长**判断，不能只看相位 —— 相位只有 periodic 模式才走，
    # 用相位判断会让 random 模式的窗口永不结束（实测占空比 0.94、切换只有 1 次）。
    # 窗口时长本来就是按"整数个周期"算出来的，所以 periodic 结束时相位也正好落在下沿。
    if self._scheduled():
      self._window_left -= dt
      if self.active:
        if self._window_left <= 0.0:
          self._enter_pause()
      elif self._window_left <= 0.0:
        self._enter_active()
    else:
      self.active = True          # pause_s 为空 = 全程都在动（老行为，自检用）

    if self.p.mode == "periodic":
      if not self.active:
        # 静止窗口：停在**下沿**（默认贴地姿态 = 其它靶型的姿态），速度/加速度都给 0，
        # 由 PD 把它平滑地拉过去（不会一步跳）。
        self.target, self.v_target, self.a_target = self.p.bottom, 0.0, 0.0
        return
      if self.p.period_s <= 1e-6:
        self.target, self.v_target, self.a_target = self.p.bottom, 0.0, 0.0
        return
      w = 2.0 * math.pi / self.p.period_s
      self._phase += w * dt          # 只有活动窗口才推进（静止窗口停在贴地姿态）
      ph = self._phase
      self.target = self.p.bottom + A * (1.0 + math.sin(ph))
      # **指令速度用正弦的解析导数**：这样"机构能力足够时"走出来的就是干净的正弦；
      # 若用 (target-pos)/dt 这种死拍跟踪，限幅会一直饱和、轨迹变成三角波 ——
      # 那样"加速度上限"这个旋钮就失去意义了（永远在饱和，调它看不出差别）。
      self.v_target = A * w * math.cos(ph)
      self.a_target = -A * w * w * math.sin(ph)
    elif self.p.mode == "random":
      if not self.active:
        self.target, self.v_target, self.a_target = self.p.bottom, 0.0, 0.0
        return
      self._hold_left -= dt
      if self._hold_left <= 0.0:
        self._hold_left = float(self.rng.uniform(*self.p.rand_hold_s))
        self.target = self.p.bottom + float(self.rng.uniform(0.0, 2.0 * A))
      # 阶跃指令：用"能刹住的接近速度"而不是 deadbeat（err/dt）。
      # deadbeat 会以 v_max 一路撞向目标、靠限幅硬停 —— 实测冲过行程顶到安全边界；
      # v = sqrt(2·a·|err|) 是标准的制动曲线，既不过冲也不磨蹭。
      err = self.target - self.pos
      a_l = max(self.p.accel_limit_mps2, 1e-6)
      v_brake = math.sqrt(2.0 * a_l * abs(err))
      self.v_target = math.copysign(min(self.p.vel_limit_mps, v_brake), err) if abs(err) > 1e-9 else 0.0
    else:
      self.target, self.v_target, self.a_target = 0.0, 0.0, 0.0

  # ------------------------------------------------------------------ 每步
  def step(self, dt: float) -> float:
    self.t += dt
    if self.p.mode == "off" or self.p.amplitude <= 0.0:
      self.pos = self.vel = self.target = 0.0
      return 0.0

    # 先决定"现在该处于哪个窗口"，再算目标
    if self._scheduled() and not self._started:
      self.active = False                 # 启动先静止一个窗口（便于捕获/统计）
      self._window_left = self._draw_pause()
    self._update_target(dt)

    # 占空比统计必须放在窗口决策**之后**：摆在前面的话，第一帧（active 还是初值 False）
    # 会被算成"静止"，无窗口档的占空比就不是精确的 1.00 了（实测 0.99993）。
    self.total_time += dt
    if self.active:
      self.active_time += dt

    A, v_max, a_max = self.p.amplitude, self.p.vel_limit_mps, self.p.accel_limit_mps2
    top = self.p.bottom + 2.0 * A

    # 无扰启动：第一帧就站到指令位置/速度上。
    # 不做这一步，起步时"指令已经在动、机构才刚开始加速"，跟踪会一直落后，
    # 最后撞到硬边界被夹住 —— 夹逼分支又会硬造出速度尖峰（实测峰值加速度假象 296 m/s²）。
    if not self._started:
      self._started = True
      self.pos = self.target
      self.vel = max(-v_max, min(v_max, self.v_target))
      return self.pos

    # **二自由度跟踪：前馈 + 误差 PD**（别用死拍跟踪律）
    #   periodic : a_cmd = a_target(前馈) + kp·(target-pos) + kd·(v_target-vel)
    #   random   : v_target 已经是"刹得住的接近速度"，折算成等效加速度需求
    # 为什么 periodic 要这么写：
    #   * 纯前馈（只有 v_target）在机构能力不足时，被削掉的那部分速度会一路积分成
    #     位置漂移 —— 实测轨迹会顶到软边界上（看着像"幅度反而变大"）。
    #   * 只补位置项 kp·e 也不行：前馈已经解析地抵消了指令加速度，误差方程退化成
    #     ë + kp·e = 0，那是**无阻尼**振荡器，数值误差会被放大成持续振荡。
    #   * 补成 ë + kd·ė + kp·e = 0（kp=25、kd=10 -> ωn=5 rad/s、ζ=1）才既收敛又不超调。
    #   * PD 修正必须加在**加速度**上，不能走 (v_cmd-vel)/dt —— 那等于给 ė 乘 1/dt
    #     （1/0.002 = 500 倍刚度），实测 88% 的步都在饱和、峰加速度假象 235 m/s²。
    if self.p.mode == "periodic":
      a_cmd = (self.a_target
               + self.p.pos_gain * (self.target - self.pos)
               + self.p.vel_gain * (self.v_target - self.vel))
    else:
      v_req = max(-v_max, min(v_max, self.v_target))
      a_cmd = (v_req - self.vel) / max(dt, 1e-9)

    a_applied = math.copysign(a_max, a_cmd) if (a_max > 0.0 and abs(a_cmd) > a_max) else a_cmd
    v_new = max(-v_max, min(v_max, self.vel + a_applied * dt))
    step = v_new * dt

    # 行程夹逼：**只夹位置、不动速度**（改速度会造出假尖峰）。
    # 正常跟踪不会碰到它，只是防数值漂移；下沿 0 = 贴地姿态，绝不往地下走。
    z = self.pos + step
    if z > top:
      self.pos = top
    elif z < self.p.bottom:
      self.pos = self.p.bottom
    else:
      self.pos = z

    a_eff = (v_new - self.vel) / max(dt, 1e-9)   # 速度限幅也会改变实际加速度，统计用实测值
    self.vel = v_new
    self.peak_accel = max(self.peak_accel, abs(a_eff))
    self.peak_vel = max(self.peak_vel, abs(self.vel))
    return self.pos

  # ------------------------------------------------------------------ 查询
  @property
  def duty(self) -> float:
    """真正在动的时间占比（活动窗口 / 总时长）。"""
    return float(self.active_time / self.total_time) if self.total_time > 1e-9 else 0.0

  def status(self) -> dict:
    return {
      "mode": self.p.mode,
      "offset_mm": round(self.pos * 1000.0, 2),
      "target_mm": round(self.target * 1000.0, 2),
      "vel_mps": round(self.vel, 4),
      "amplitude_mm": self.p.amplitude_mm,
      "bottom_mm": round(self.p.bottom * 1000.0, 2),
      "period_s": self.p.period_s,
      "accel_limit_mps2": self.p.accel_limit_mps2,
      "vel_limit_mps": self.p.vel_limit_mps,
      "active": bool(self.active),          # 现在在"活动窗口"还是"静止窗口"
      "duty": round(self.duty, 3),          # 至今为止在动的时间占比
      "peak_accel_mps2": round(self.peak_accel, 3),
      "peak_vel_mps": round(self.peak_vel, 3),
    }
