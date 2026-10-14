#!/usr/bin/env python3
"""云台伺服：按真实电机（DJI GM6020）+ 模型自带惯量重做 PD 参数。

为什么
------
原来的执行器是 `<position kp="50" kv="6"/>`，两个问题：

1. **kv 太大，严重过阻尼。** PD 位置伺服的闭环二阶系统是
   `I·q̈ = kp·(q_des−q) − kv·q̇`，即 ωn=√(kp/I)、ζ=(kv+c_joint)/(2√(kp·I))。
   代进模型实测惯量：
       yaw   I=0.0280  ωn=42.3 rad/s  ζ=2.60
       pitch I=0.00267 ωn=136.9 rad/s ζ=8.42
   ζ 应该接近 1.0；2.6 / 8.4 意味着响应又慢又钝（实测跟随时滞后约 2°，
   PlotJuggler 上能看到阶梯状爬升）。这不是电机不行，是参数配错了。
2. **forcerange=0 等于没有力矩上限** —— 无限大力矩的电机不存在。

GM6020 参数（DJI 官方）
-----------------------
    额定电压        DC 24 V
    空载转速        320 rpm = 33.5 rad/s
    额定(持续)转矩   1.2 N·m
    额定转矩下转速   132 rpm = 13.8 rad/s
    额定电流        1.62 A
官方没给堵转转矩，用上面两点线性外推：
    堵转转矩 = 1.2 × 320/(320−132) ≈ **2.04 N·m**
（和 Kt=1.2/1.62=0.74 N·m/A、峰值电流约 3 A 一致）

能力校验：yaw 轴 I=0.028 -> 堵转角加速度 2.04/0.028 = 72.9 rad/s²，
0→10 rad/s 只需 0.137 s，够用；所以瓶颈在电流上限而不是参数保守。

参数怎么来的
------------
取 ζ=1.0（临界阻尼，无超调且最快），ωn 取真实云台的典型带宽 6~8 Hz：
    kp = I·ωn²
    kv = 2·ζ·I·ωn − c_joint      （c_joint 是关节自身的 damping=0.15）
然后 forcerange = ±2.04 N·m 让大角度机动自然进入饱和（和真车一样）。
"""

from __future__ import annotations

import re

# 模型实测的等效惯量（kg·m²），用"施加已知力矩测角加速度"测得，见 README
INERTIA = {"yaw": 0.0280, "pitch": 0.00267}
JOINT_DAMPING = 0.15          # MJCF 里 <joint damping="0.15">

# GM6020
GM6020_STALL_TORQUE = 2.04    # N·m，由额定点线性外推
GM6020_MAX_SPEED = 33.5       # rad/s（空载 320 rpm），本模型里用不到，留作参考
GM6020_RATED_TORQUE = 1.2     # N·m，持续输出上限


def servo_gains(axis: str, wn: float, zeta: float = 1.0):
  """由惯量 + 目标带宽反推 kp/kv。"""
  I = INERTIA[axis]
  kp = I * wn * wn
  kv = max(0.0, 2.0 * zeta * I * wn - JOINT_DAMPING)
  return kp, kv


# 默认带宽。实测（同种子，只统计在跟踪目标时）：
#   wn=40/50    err_yaw 中位 3.12°   <1° 占比 14.6%
#   wn=70/90    err_yaw 中位 2.85°   <1° 占比 16.1%
#   wn=100/120  err_yaw 中位 2.29°   <1° 占比 21.0%   <- 取这组
# 为什么大 kp 不算"脱离现实"：力矩上限 2.04 N·m 是硬约束，kp 大只是让控制器
# 更快地要求满力矩；实际运动仍被电机能力卡着（yaw 最高 72.9 rad/s²）。
# 真车调参也会把增益推到接近饱和，所以这是合理的。
DEFAULTS = {
  "yaw": dict(wn=100.0, zeta=1.0),
  "pitch": dict(wn=120.0, zeta=1.0),
}


def apply_gm6020(xml: str, torque_limit: float = GM6020_STALL_TORQUE,
                 wn_yaw: float = None, wn_pitch: float = None,
                 zeta: float = 1.0) -> str:
  """把 <position name="yaw_servo"/pitch_servo> 重写成 GM6020 模型。

  对已生成的 MJCF 做字符串重写，和 apply_appearance() 同一套路 ——
  不动转换器产出的资产文件，随时可以切回原参数。
  """
  cfg = {
    "yaw": dict(wn=wn_yaw or DEFAULTS["yaw"]["wn"], zeta=zeta),
    "pitch": dict(wn=wn_pitch or DEFAULTS["pitch"]["wn"], zeta=zeta),
  }

  def repl(m):
    tag, axis = m.group(0), m.group(1)
    kp, kv = servo_gains(axis, cfg[axis]["wn"], cfg[axis]["zeta"])
    # 保留 joint / ctrlrange 等原有属性，只换 kp/kv 并补 forcerange
    body = re.sub(r'\skp="[^"]*"', "", tag)
    body = re.sub(r'\skv="[^"]*"', "", body)
    body = re.sub(r'\sforcerange="[^"]*"', "", body)
    body = body[:-2].rstrip()          # 去掉结尾的 "/>"
    return '%s kp="%.3f" kv="%.4f" forcerange="%.4f %.4f"/>' % (
      body, kp, kv, -torque_limit, torque_limit)

  # 用捕获组拿到轴名，repl 里靠 m.group(1) 判断是 yaw 还是 pitch
  pat = re.compile(r'<position\s+name="(yaw|pitch)_servo"[^>]*/>')
  return pat.sub(repl, xml)


def report() -> str:
  lines = ["=== GM6020 云台伺服参数 ===",
           "  惯量: yaw %.5f  pitch %.5f kg·m²" % (INERTIA["yaw"], INERTIA["pitch"]),
           "  堵转转矩 %.2f N·m -> 堵转角加速度 yaw %.1f / pitch %.1f rad/s²"
           % (GM6020_STALL_TORQUE, GM6020_STALL_TORQUE / INERTIA["yaw"],
              GM6020_STALL_TORQUE / INERTIA["pitch"])]
  for axis in ("yaw", "pitch"):
    kp, kv = servo_gains(axis, DEFAULTS[axis]["wn"], DEFAULTS[axis]["zeta"])
    wn = DEFAULTS[axis]["wn"]
    lines.append("  %-5s kp=%7.3f kv=%6.3f  (ωn=%.0f rad/s = %.1f Hz, ζ=%.1f)"
                 % (axis, kp, kv, wn, wn / 6.2832, DEFAULTS[axis]["zeta"]))
  return "\n".join(lines)


if __name__ == "__main__":
  print(report())
  import sys
  if len(sys.argv) > 1:
    with open(sys.argv[1], encoding="utf-8") as f:
      x = f.read()
    out = apply_gm6020(x)
    for line in out.splitlines():
        if "_servo" in line:
            print("  " + line.strip()[:120])
