#!/usr/bin/env python3
"""我方底盘扰动（简化版）：随机颠簸 → 云台指向抖动 + IMU 噪声 + 相机视轴抖动。

为什么需要
----------
现在仿真里云台挂在一个**静止的基座**上，而真机自瞄装在移动底盘上：
过地形、加减速、转弯都会让底盘颠簸，颠簸经悬挂和结构传到云台上 ——
云台伺服追不上，世界系里的**指向就会抖**；同时 IMU 读数带噪声、
相机相对炮管还有微小振动。整条"底盘扰动下的自瞄"链目前完全没建模，
而它比靶车怎么机动更能决定落地成败。

简化版的三个通道（各自可开关，默认全关以便复现旧基线）：

1. **云台指向抖动**（主项）：给云台的关节目标叠加一个**窄带随机抖动**
   （几个正弦叠加，频带对应底盘/悬挂的一阶模态）。它直接改变世界系指向，
   所以相机看到的画面、弹丸出膛方向、以及开火门限判据全都一致地受影响
   —— 这是物理上自洽的注入点。
2. **IMU/姿态噪声**：喂给 sp_vision 的四元数叠一个小的随机旋转。
   它只影响解算，不影响物理（对应 IMU 噪声与结构柔性带来的估计误差）。
3. **相机视轴抖动**：相机相对云台再额外抖一点（只影响画面，
   对应相机支架/镜头在振动下的相对位移）。

参数全部可 CLI 覆盖，默认值是拍脑袋的工程值（【待真机校准】）。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# 抖动由几个正弦叠加而成：数量越多越接近窄带随机过程
N_TONES = 3


@dataclass
class ChassisParams:
  """底盘扰动参数（默认全关）。"""

  enabled: bool = False          # 总开关
  model_gimbal: bool = True      # 云台指向抖动（高频，0.8~4 Hz）
  model_drift: bool = True       # 云台指向**漂移**（低频随机游走，见下）
  model_imu: bool = True         # IMU/姿态噪声
  model_cam: bool = True         # 相机视轴抖动

  yaw_amp_deg: float = 0.35      # 云台 yaw 世界指向抖动 1σ（度）  【待真机校准】
  pitch_amp_deg: float = 0.25    # 俯仰 1σ（度）                  【待真机校准】
  band_lo_hz: float = 0.8        # 抖动频带下限（底盘/悬挂一阶模态附近）
  band_hi_hz: float = 4.0        # 上限
  imu_deg: float = 0.15          # IMU/姿态噪声 1σ（度）          【待真机校准】
  cam_deg: float = 0.10          # 相机视轴额外抖动 1σ（度）       【待真机校准】
  # 低频漂移：底盘在走/转向时，云台基座本身在缓慢摆动，伺服保持的是"相对底盘"的角，
  # 于是世界系指向会**慢慢漂**（画面上就是镜头在缓慢平移/俯仰）。
  # 用 OU 随机游走生成，相关时间 drift_tau。
  drift_yaw_deg: float = 1.20    # yaw 漂移 1σ（度）              【待真机校准】
  drift_pitch_deg: float = 0.80  # pitch 漂移 1σ（度）            【待真机校准】
  drift_tau: float = 3.0         # 漂移相关时间（秒）

  @classmethod
  def from_args(cls, a) -> "ChassisParams":
    return cls(
      enabled=str(getattr(a, "chassis_disturb", "off")) == "on",
      model_gimbal=bool(getattr(a, "disturb_gimbal", True)),
      model_drift=bool(getattr(a, "disturb_drift", True)),
      model_imu=bool(getattr(a, "disturb_imu", True)),
      model_cam=bool(getattr(a, "disturb_cam", True)),
      yaw_amp_deg=float(getattr(a, "disturb_yaw_deg", 0.35)),
      pitch_amp_deg=float(getattr(a, "disturb_pitch_deg", 0.25)),
      band_lo_hz=float(getattr(a, "disturb_band_lo", 0.8)),
      band_hi_hz=float(getattr(a, "disturb_band_hi", 4.0)),
      imu_deg=float(getattr(a, "disturb_imu_deg", 0.15)),
      cam_deg=float(getattr(a, "disturb_cam_deg", 0.10)),
      drift_yaw_deg=float(getattr(a, "disturb_drift_yaw_deg", 1.20)),
      drift_pitch_deg=float(getattr(a, "disturb_drift_pitch_deg", 0.80)),
      drift_tau=float(getattr(a, "disturb_drift_tau", 3.0)),
    )


class ChassisDisturb:
  """每帧推进一次：给出云台指向抖动、IMU 四元数噪声、相机视轴抖动。"""

  def __init__(self, p: ChassisParams, rng=None, seed: int | None = None):
    self.p = p
    self.rng = rng if rng is not None else np.random.default_rng(seed)
    self.t = 0.0
    self.yaw_jitter = 0.0       # 高频抖动（叠加在关节目标上）
    self.pitch_jitter = 0.0
    self.yaw_drift = 0.0        # 低频漂移（同样叠加在关节目标上）
    self.pitch_drift = 0.0
    # 每个分量用 3 个正弦叠加（频率在频带内随机、相位随机、幅度归一化）
    self._tones = {}
    for name in ("yaw", "pitch"):
      f = self.rng.uniform(p.band_lo_hz, p.band_hi_hz, N_TONES)
      ph = self.rng.uniform(0.0, 2.0 * math.pi, N_TONES)
      # 归一化：让叠加后的 RMS 等于设定的 1σ
      self._tones[name] = (f, ph, 1.0 / math.sqrt(N_TONES / 2.0))

  # ------------------------------------------------------------------ 每帧
  def step(self, dt: float) -> None:
    self.t += dt
    p = self.p

    # --- 高频抖动：窄带正弦叠加 ---
    if p.enabled and p.model_gimbal:
      out = {}
      for name, amp_deg in (("yaw", p.yaw_amp_deg), ("pitch", p.pitch_amp_deg)):
        f, ph, norm = self._tones[name]
        s = float(np.sum(np.sin(2.0 * math.pi * f * self.t + ph)))
        out[name] = math.radians(amp_deg) * norm * s
      self.yaw_jitter, self.pitch_jitter = out["yaw"], out["pitch"]
    else:
      self.yaw_jitter = self.pitch_jitter = 0.0

    # --- 低频漂移：OU 随机游走 x' = -x/tau + sigma*sqrt(2 dt/tau) N(0,1) ---
    if p.enabled and p.model_drift and p.drift_tau > 0.0:
      k = math.sqrt(2.0 * dt / p.drift_tau)
      for name, sig_deg in (("yaw", p.drift_yaw_deg), ("pitch", p.drift_pitch_deg)):
        cur = self.yaw_drift if name == "yaw" else self.pitch_drift
        cur += (-cur / p.drift_tau) * dt + math.radians(sig_deg) * k * float(self.rng.normal())
        if name == "yaw":
          self.yaw_drift = cur
        else:
          self.pitch_drift = cur
    else:
      self.yaw_drift = self.pitch_drift = 0.0

  # ------------------------------------------------------- IMU / 相机噪声
  def imu_quat_noise(self) -> np.ndarray:
    """返回一个小的随机旋转向量（轴角，弧度），调用方用它扰动报给视觉的四元数。"""
    if not (self.p.enabled and self.p.model_imu):
      return np.zeros(3)
    sigma = math.radians(self.p.imu_deg)
    axis = self.rng.normal(size=3)
    n = float(np.linalg.norm(axis))
    if n < 1e-9:
      return np.zeros(3)
    return axis / n * float(self.rng.normal(0.0, sigma))

  def cam_quat_noise(self) -> np.ndarray:
    """相机相对云台的额外抖动（轴角，弧度）。"""
    if not (self.p.enabled and self.p.model_cam):
      return np.zeros(3)
    sigma = math.radians(self.p.cam_deg)
    axis = self.rng.normal(size=3)
    n = float(np.linalg.norm(axis))
    if n < 1e-9:
      return np.zeros(3)
    return axis / n * float(self.rng.normal(0.0, sigma))

  # ------------------------------------------------------------------ 导出
  def status(self) -> dict:
    return {
      "enabled": bool(self.p.enabled),
      "yaw_jitter_deg": round(math.degrees(self.yaw_jitter), 4),
      "pitch_jitter_deg": round(math.degrees(self.pitch_jitter), 4),
      "yaw_drift_deg": round(math.degrees(self.yaw_drift), 4),
      "pitch_drift_deg": round(math.degrees(self.pitch_drift), 4),
      "model_gimbal": bool(self.p.model_gimbal),
      "model_drift": bool(self.p.model_drift),
      "model_imu": bool(self.p.model_imu),
      "model_cam": bool(self.p.model_cam),
    }


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
  """四元数乘法（w, x, y, z）。"""
  w1, x1, y1, z1 = a
  w2, x2, y2, z2 = b
  return np.array([
    w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
    w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
    w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
  ])


def axis_angle_to_quat(rotvec: np.ndarray) -> np.ndarray:
  """轴角（弧度，模长=角度）-> 四元数。零向量返回单位四元数。"""
  ang = float(np.linalg.norm(rotvec))
  if ang < 1e-12:
    return np.array([1.0, 0.0, 0.0, 0.0])
  axis = rotvec / ang
  s = math.sin(ang / 2.0)
  return np.array([math.cos(ang / 2.0), axis[0] * s, axis[1] * s, axis[2] * s])
