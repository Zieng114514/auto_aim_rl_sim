#!/usr/bin/env python3
"""靶车运动**基准脚本**：固定时长、乱序靶型 —— 让 TTK / 命中率的对比可复现。

为什么需要它
------------
原来靶车是"每轮随机抽参数"的连续随机过程：两次跑出来面对的机动序列完全不同，
于是 TTK / 命中率的差异里**既有算法差异也有运气差异**，没法比较 ——
尤其到了评估 RL 策略的时候，这种噪声会把真实增益淹没。

做法：把机动切成**固定时长的段**，每段是一种明确的"靶型"，每轮**乱序**播放：

| 靶型 | 含义 | 平移 | 自转 |
|---|---|---|---|
| `move_spin_rand` | 随机旋转移动靶（**默认靶型**） | 连续巡逻 | 随机周期的正弦调制 |
| `move_spin_const` | 固定（匀速）旋转移动靶 | 连续巡逻 | **恒定**转速（可预测） |
| `inplace_spin` | 原地随机甩头（小陀螺） | **停住** | 快速正弦甩头 |
| `dash_halt` | 急停冲刺靶 | 冲刺/急停交替 | 随机周期的正弦调制 |

乱序用**固定种子**生成：同种子 -> 同一套顺序 -> 可复现；不同种子 -> 顺序不同，
避免"总是先打最简单的靶型"这种顺序偏差。

每段边界会记录当时的 (发射, 命中, 击杀) 计数，于是能给出**分靶型的统计** ——
这才是评估 RL 时真正该看的表：不是"总体 TTK 多少"，而是"哪种机动上掉链子"。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

# 全部可选靶型。**默认只用前四种**（= 既有全部基线的口径，必须保持不变），
# 轮腿两种需要显式选（--car-script-preset benchmark_2026_wl 或 --car-script-modes）。
MODES_LEGACY = ("move_spin_rand", "move_spin_const", "inplace_spin", "dash_halt")
MODES_WHEEL_LEGGED = ("wheel_legged_periodic", "wheel_legged_random")
# 负责人 2026-10-14 点名的四种带升降靶型：经典四型（固定 / 移动 / 旋转 / 随机旋转移动）
# **各自叠加车体升降**。与上面轮腿两段的区别：这两段的升降是"顺带"，这四种是把
# "竖直方向的运动"当成主要矛盾来考（固定升降靶除了升降没有任何其它运动）。
MODES_LIFT = ("lift_static", "lift_linear", "lift_spin", "lift_random")
MODES = MODES_LEGACY + MODES_WHEEL_LEGGED + MODES_LIFT

MODE_DESC = {
  "move_spin_rand": "随机旋转移动靶",
  "move_spin_const": "固定(匀速)旋转移动靶",
  "inplace_spin": "原地随机甩头",
  "dash_halt": "急停冲刺靶",
  # ↓ 轮腿（负责人 2026-10-14）：交战时车体升降 + 变速小陀螺
  "wheel_legged_periodic": "轮腿·定频升降+甩头",
  "wheel_legged_random": "轮腿·随机升降+甩头",
  # ↓ 四种带升降靶型（负责人 2026-10-14）
  "lift_static": "固定升降靶",
  "lift_linear": "左右移动升降靶",
  "lift_spin": "原地旋转升降靶",
  "lift_random": "随机旋转移动升降靶",
}


@dataclass
class MotionSpec:
  """当前这一帧"该怎么动"。"""

  mode: str = "move_spin_rand"
  translation: str = "patrol"     # patrol | dashhalt | stop
  spin: str = "mod"               # mod（随机周期调制） | const（恒定） | gyro（快速甩头） | none（不转）
  lift: str = "off"               # off | periodic | random（轮腿车体升降，见 rm_lift.py）


_SPEC = {
  "move_spin_rand": MotionSpec("move_spin_rand", "patrol", "mod"),
  "move_spin_const": MotionSpec("move_spin_const", "patrol", "const"),
  "inplace_spin": MotionSpec("inplace_spin", "stop", "gyro"),
  "dash_halt": MotionSpec("dash_halt", "dashhalt", "mod"),
  # 轮腿：平移用巡逻（真机是边走边打）、自转用变速调制、再叠加车体升降
  "wheel_legged_periodic": MotionSpec("wheel_legged_periodic", "patrol", "mod", lift="periodic"),
  "wheel_legged_random": MotionSpec("wheel_legged_random", "patrol", "mod", lift="random"),
  # 四种带升降靶型（负责人 2026-10-14）：
  #   spin="none" = 不转（固定靶/直线靶），translation="stop" = 不平移（固定靶/旋转靶）
  "lift_static": MotionSpec("lift_static", "stop", "none", lift="periodic"),
  "lift_linear": MotionSpec("lift_linear", "patrol", "none", lift="periodic"),
  "lift_spin": MotionSpec("lift_spin", "stop", "const", lift="periodic"),
  "lift_random": MotionSpec("lift_random", "patrol", "mod", lift="periodic"),
}


@dataclass
class CarScriptParams:
  dwell: float = 10.0            # 每段时长（秒）——负责人定的 10 s
  # **默认仍是 4 种**：换成 6 种会让数字与既有基线不可比（见 configs/targets.yaml 的 presets）
  modes: tuple = MODES_LEGACY    # 参与本轮基准的靶型
  shuffle: bool = True           # 每轮乱序（固定种子 -> 可复现）


class CarScript:
  """按固定时长切段、乱序播放靶型；同时记录每段的计数快照。"""

  def __init__(self, p: CarScriptParams, rng):
    self.p = p
    self.rng = rng
    self.t = 0.0
    self.seg_index = -1
    self.order: list[str] = []
    self.segments: list[dict] = []     # 已完成/进行中的段（带计数快照）
    self._cur_start = 0.0
    self._cur_start_counts = (0, 0, 0)
    self._next_round()

  # ------------------------------------------------------------------ 内部
  def _next_round(self) -> None:
    modes = list(self.p.modes)
    if self.p.shuffle:
      self.rng.shuffle(modes)
    self.order.extend(modes)

  def _close_segment(self, counts: tuple) -> None:
    if self.seg_index < 0:
      return
    s, h, d = self._cur_start_counts
    cs, ch, cd = counts
    self.segments.append({
      "index": self.seg_index,
      "mode": self.order[self.seg_index],
      "desc": MODE_DESC[self.order[self.seg_index]],
      "t0": round(self._cur_start, 3),
      "t1": round(self.t, 3),
      "dur": round(self.t - self._cur_start, 3),
      "shots": int(cs - s),
      "hits": int(ch - h),
      "kills": int(cd - d),
    })
    if len(self.segments) > 2000:
      del self.segments[:500]

  # ------------------------------------------------------------------ 每帧
  def step(self, dt: float, counts: tuple = (0, 0, 0)) -> MotionSpec:
    """推进时间轴，返回当前靶型对应的 MotionSpec。

    counts = (发射, 命中, 击杀) 的当前值；段边界用它算本段增量。
    """
    self.t += dt
    if self.seg_index < 0 or self.t - self._cur_start >= self.p.dwell - 1e-9:
      self._close_segment(counts)
      self.seg_index += 1
      if self.seg_index >= len(self.order):
        self._next_round()
      self._cur_start = self.t
      self._cur_start_counts = tuple(counts)
    return _SPEC[self.order[self.seg_index]]

  @property
  def mode(self) -> str:
    return self.order[self.seg_index] if self.seg_index >= 0 else self.p.modes[0]

  def status(self, counts: tuple = (0, 0, 0)) -> dict:
    """导出给 /state.json：当前段 + 已完成段（含"进行中"那段的实时增量）。"""
    segs = list(self.segments)
    if self.seg_index >= 0:
      s, h, d = self._cur_start_counts
      cs, ch, cd = counts
      segs.append({
        "index": self.seg_index,
        "mode": self.order[self.seg_index],
        "desc": MODE_DESC[self.order[self.seg_index]],
        "t0": round(self._cur_start, 3), "t1": None,
        "dur": round(self.t - self._cur_start, 3),
        "shots": int(cs - s), "hits": int(ch - h), "kills": int(cd - d),
      })
    return {
      "enabled": True,
      "dwell": self.p.dwell,
      "shuffle": self.p.shuffle,
      "order": self.order[:16],
      "current": self.mode,
      "current_desc": MODE_DESC[self.mode],
      "elapsed": round(self.t - self._cur_start, 2),
      "segments": segs[-(len(self.p.modes) * 4):],
    }
