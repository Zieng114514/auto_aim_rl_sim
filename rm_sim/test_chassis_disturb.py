#!/usr/bin/env python3
"""我方底盘扰动的最小验证（rm_chassis）。

跑法：  python3 rm_sim/test_chassis_disturb.py
（纯模型单测，不需要 MuJoCo，秒级）

检查项：
  1. 总开关关掉时三个通道**全为零**（保证默认配置能复现旧基线）；
  2. 云台指向抖动：1σ 与设定一致、均值≈0、**频带落在设定范围内**（窄带，不是白噪声）；
  3. 抖动是时间相关的（相邻样本高度相关）—— 白噪声会让伺服"抖个不停"，
     和底盘颠簸的物理完全不同；
  4. IMU / 相机噪声：逐轴 σ 与设定一致，且各分量独立；
  5. 三个通道的独立开关各自生效；
  6. 四元数工具正确（单位元、模长守恒、轴角→四元数）。
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rm_chassis import (  # noqa: E402
  ChassisDisturb,
  ChassisParams,
  axis_angle_to_quat,
  quat_mul,
)

DT = 1.0 / 30.0
FAILED: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
  print("  %s %s%s" % ("✓" if ok else "✗", name, ("  " + detail) if detail else ""))
  if not ok:
    FAILED.append(name)


def jitter_series(p: ChassisParams, seconds: float, seed: int = 0):
  d = ChassisDisturb(p, seed=seed)
  n = int(seconds / DT)
  ys, ps = np.zeros(n), np.zeros(n)
  for i in range(n):
    d.step(DT)
    ys[i], ps[i] = d.yaw_jitter, d.pitch_jitter
  return np.degrees(ys), np.degrees(ps)


def main() -> int:
  print("== 1. 默认（关）时全为零 ==")
  off = ChassisDisturb(ChassisParams(enabled=False), seed=0)
  off.step(DT)
  check("关掉时云台抖动为 0", off.yaw_jitter == 0.0 and off.pitch_jitter == 0.0)
  check("关掉时 IMU/相机噪声为 0",
        not off.imu_quat_noise().any() and not off.cam_quat_noise().any())

  print("== 2. 云台指向抖动：幅度与频带 ==")
  p = ChassisParams(enabled=True)
  ys, ps = jitter_series(p, 200.0)
  check("yaw 1σ 与设定一致", abs(ys.std() - p.yaw_amp_deg) < 0.03 * p.yaw_amp_deg,
        "实测 %.4f° 设定 %.2f°" % (ys.std(), p.yaw_amp_deg))
  check("pitch 1σ 与设定一致", abs(ps.std() - p.pitch_amp_deg) < 0.03 * p.pitch_amp_deg,
        "实测 %.4f° 设定 %.2f°" % (ps.std(), p.pitch_amp_deg))
  check("均值为 0（无系统性偏置）", abs(ys.mean()) < 0.02 and abs(ps.mean()) < 0.02,
        "yaw 均值 %+.4f°" % ys.mean())
  # 频带：用 FFT 找能量峰所在频率
  spec = np.abs(np.fft.rfft(ys))
  freqs = np.fft.rfftfreq(len(ys), DT)
  peak = float(freqs[int(np.argmax(spec))])
  check("能量峰落在设定频带内", p.band_lo_hz <= peak <= p.band_hi_hz,
        "峰在 %.2f Hz（频带 %.1f~%.1f）" % (peak, p.band_lo_hz, p.band_hi_hz))

  print("== 3. 时间相关性（不能是白噪声）==")
  corr = float(np.corrcoef(ys[:-1], ys[1:])[0, 1])
  check("相邻样本高度相关", corr > 0.7, "相关系数 %.3f（白噪声应≈0）" % corr)

  print("== 3b. 低频漂移 ==")
  p_d = ChassisParams(enabled=True, model_gimbal=False)
  dd = ChassisDisturb(p_d, seed=11)
  xs = []
  for _ in range(int(600 / DT)):          # 600 s
    dd.step(DT)
    xs.append(math.degrees(dd.yaw_drift))
  xs = np.array(xs)
  # OU 过程的稳态标准差就是设定值
  check("漂移 1σ 符合设定", abs(xs.std() - p_d.drift_yaw_deg) < 0.25 * p_d.drift_yaw_deg,
        "实测 %.3f° 设定 %.2f°" % (xs.std(), p_d.drift_yaw_deg))
  # OU 过程的滞后 1 步自相关理论值 = exp(-dt/tau)，直接对着这个量（比拍阈值靠谱）
  corr_d = float(np.corrcoef(xs[:-1], xs[1:])[0, 1])
  exp_corr = math.exp(-DT / p_d.drift_tau)
  check("漂移时间尺度符合 OU 理论", abs(corr_d - exp_corr) < 0.005,
        "相邻相关 %.4f 理论 %.4f（tau=%.1f s, dt=%.4f s；高频抖动的相关是 %.3f）"
        % (corr_d, exp_corr, p_d.drift_tau, DT, corr))
  check("只开漂移时高频抖动为 0", dd.yaw_jitter == 0.0 and dd.pitch_jitter == 0.0)
  p_nd = ChassisParams(enabled=True, model_gimbal=False, model_drift=False)
  nd = ChassisDisturb(p_nd, seed=11)
  nd.step(DT)
  check("关掉漂移后为零", nd.yaw_drift == 0.0 and nd.pitch_drift == 0.0)

  print("== 4. IMU / 相机噪声 ==")
  d = ChassisDisturb(p, seed=7)
  imu = np.array([np.degrees(np.linalg.norm(d.imu_quat_noise())) for _ in range(4000)])
  cam = np.array([np.degrees(np.linalg.norm(d.cam_quat_noise())) for _ in range(4000)])
  # 噪声是"轴随机 × 模长 N(0,σ)"，所以分布不是各轴同方差的高斯；
  # 这里检查**模长均值**：半正态均值 = σ·sqrt(2/π)
  exp_imu = p.imu_deg * math.sqrt(2.0 / math.pi)
  exp_cam = p.cam_deg * math.sqrt(2.0 / math.pi)
  check("IMU 噪声模长均值符合设定", abs(imu.mean() - exp_imu) < 0.15 * exp_imu,
        "实测 %.4f° 理论 %.4f°（1σ=%.2f°）" % (imu.mean(), exp_imu, p.imu_deg))
  check("相机噪声模长均值符合设定", abs(cam.mean() - exp_cam) < 0.15 * exp_cam,
        "实测 %.4f° 理论 %.4f°（1σ=%.2f°）" % (cam.mean(), exp_cam, p.cam_deg))

  print("== 5. 通道独立开关 ==")
  only_g = ChassisDisturb(ChassisParams(enabled=True, model_imu=False, model_cam=False), seed=1)
  only_g.step(DT)
  check("只开云台通道：IMU/相机为零",
        only_g.yaw_jitter != 0.0 and not only_g.imu_quat_noise().any()
        and not only_g.cam_quat_noise().any())
  only_i = ChassisDisturb(
    ChassisParams(enabled=True, model_gimbal=False, model_cam=False), seed=1)
  only_i.step(DT)
  check("只开 IMU 通道：云台抖为 0、IMU 非零",
        only_i.yaw_jitter == 0.0 and only_i.imu_quat_noise().any())

  print("== 6. 四元数工具 ==")
  check("零旋转 -> 单位四元数", np.allclose(axis_angle_to_quat(np.zeros(3)), [1, 0, 0, 0]))
  rv = np.array([0.0, 0.0, math.radians(90.0)])
  q = axis_angle_to_quat(rv)
  check("轴角->四元数模长守恒", abs(np.linalg.norm(q) - 1.0) < 1e-12)
  check("单位元相乘不变", np.allclose(quat_mul([1, 0, 0, 0], q), q))
  check("乘法的模长仍为 1", abs(np.linalg.norm(quat_mul(q, q)) - 1.0) < 1e-12)

  print()
  if FAILED:
    print("✗ %d 项未通过：%s" % (len(FAILED), ", ".join(FAILED)))
    return 1
  print("✓ 全部通过")
  return 0


if __name__ == "__main__":
  sys.exit(main())
