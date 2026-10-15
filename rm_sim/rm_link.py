#!/usr/bin/env python3
"""rm_sim <-> sp_vision 双向联调接口（把仿真当成 sp_vision 的"相机 + 下位机"）。

和 rm_stream.py 的区别：那个是**单向**的（只出图 + 真值），而且云台自己追着靶车转。
这个是对照实车链路做的**闭环**接口：

  GET  /grab       一次拿到「图像 + 同一帧的云台姿态四元数」——原子配对，不会有帧/姿态错位
  GET  /state.json 当前状态 + 装甲板真值（含 sp_vision 口径的四点）+ 命中统计
  POST /command    接收 sp_vision 的 io::Command {control, shoot, yaw, pitch}
  GET  /stream.mjpg  人看的 MJPEG

坐标约定（关键，和 configs/sim.yaml 一一对应）：
  * 云台 pitch 体 <body name="gimbal_pitch" pos="0 0 0.1765"> 挂在 yaw 轴上，
    所以它的原点是固定点，云台怎么转都不动 —— 适合当 sp_vision 的"世界系原点"。
  * 喂给 solver.set_R_gimbal2world(q) 的 q = gimbal_pitch 体在世界系下的姿态四元数，
    配合 R_gimbal2imubody = I，sp_vision 的"世界系"就等于 MuJoCo 世界系。
  * sp_vision 输出的是**世界系绝对角**：yaw = atan2(y,x)（和 MuJoCo 的 yaw 关节角同义），
    pitch = -(仰角)（抬头为负）。仿真里关节角 = 仰角反号 + 炮管安装角：
        关节 = mount_tilt + command.pitch
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import math
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import mujoco  # noqa: E402

import rm_gimbal_demo as R  # noqa: E402
from rm_appearance import PRESETS as APPEARANCE_PRESETS  # noqa: E402
from rm_cars import CarField, duplicate_cars  # noqa: E402
from rm_config import load as load_config_impl  # noqa: E402  配置层（configs/*.yaml）
from rm_ballistics import BallisticsParams  # noqa: E402
from rm_ballistics import from_config as ballistics_from_config  # noqa: E402
from rm_lift import LiftParams  # noqa: E402
from rm_gimbal_servo import apply_gm6020, report as servo_report  # noqa: E402
from rm_gun import GunParams  # noqa: E402
from rm_chassis import (  # noqa: E402
  ChassisDisturb,
  ChassisParams,
  axis_angle_to_quat,
  quat_mul,
)


def gun_params_from_args(args) -> GunParams | None:
  """把 CLI 上的枪参数收成一个 GunParams；--gun off 时返回 None（= 理想枪，旧行为）。

  三个模型各自可开关（--gun-speed/--gun-heat/--gun-spread），方便 A/B 与复现旧基线。
  """
  if getattr(args, "gun", "on") == "off":
    return None
  return GunParams(
    speed_cold=args.gun_speed_cold,
    speed_warm=args.gun_speed_warm,
    tau_warm=args.gun_tau_warm,
    tau_cool=args.gun_tau_cool,
    jitter=args.gun_jitter,
    heat_max=args.gun_heat_max,
    heat_per_shot=args.gun_heat_per_shot,
    heat_cool=args.gun_heat_cool,
    overheat_k=args.gun_overheat_k,
    spread_deg=args.gun_spread,
    model_speed=bool(args.gun_speed),
    model_heat=bool(args.gun_heat),
    model_spread=bool(args.gun_spread_model),
  )
from rm_appearance import apply_appearance  # noqa: E402
from rm_stream import inject_camera, project, read_barrel_elev  # noqa: E402

# sp_vision 假设的装甲板尺寸（tasks/auto_aim/solver.cpp 里的常量）
SV_ARMOR_W = 0.135   # 两块灯条间距
SV_ARMOR_H = 0.056   # 灯条长度
# sp_vision 装甲板局部系 -> 仿真装甲板 geom 局部系（列向量 = 各轴在 geom 系下的表示）
#   sv+X(板法线朝外) = geom+Y ,  sv+Y(宽度) = geom-X ,  sv+Z(高度) = geom+Z
SV2GEOM = np.array([[0.0, -1.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0]])


def look_at_xyaxes(cam_pos, target, fovy_deg: float = 45.0) -> tuple[str, str]:
  """算出"把相机放在 cam_pos、看向 target"所需的 xyaxes。

  MuJoCo 相机沿自身 -z 看、+x 为右、+y 为上，所以：
      zc = -视线方向,  yc = 世界上方向去掉视线分量,  xc = yc × zc
  """
  c = np.asarray(cam_pos, dtype=float)
  t = np.asarray(target, dtype=float)
  d = t - c
  d = d / (np.linalg.norm(d) + 1e-12)
  zc = -d
  up = np.array([0.0, 0.0, 1.0])
  yc = up - float(up @ d) * d
  if np.linalg.norm(yc) < 1e-6:            # 视线与上方向平行时退化，换个参考轴
    up = np.array([0.0, 1.0, 0.0])
    yc = up - float(up @ d) * d
  yc = yc / (np.linalg.norm(yc) + 1e-12)
  xc = np.cross(yc, zc)
  axes = " ".join("%.6f" % v for v in np.concatenate([xc, yc]))
  return axes, axes


def inject_overview_camera(xml: str, cam_pos, target, fovy_deg: float = 45.0) -> str:
  """插一个**固定在世界系**的全局视角相机（浏览器用），位置不随云台动。"""
  axes, _ = look_at_xyaxes(cam_pos, target, fovy_deg)
  cam = ('\n    <camera name="overview" mode="fixed" pos="%.4f %.4f %.4f" '
         'xyaxes="%s" fovy="%.1f"/>') % (cam_pos[0], cam_pos[1], cam_pos[2], axes, fovy_deg)
  return xml.replace("<worldbody>", "<worldbody>" + cam, 1)


def mat2quat(Rm: np.ndarray) -> list[float]:
  """旋转矩阵 -> [w, x, y, z]"""
  tr = Rm[0, 0] + Rm[1, 1] + Rm[2, 2]
  if tr > 0:
    s = math.sqrt(tr + 1.0) * 2
    w = 0.25 * s
    x = (Rm[2, 1] - Rm[1, 2]) / s
    y = (Rm[0, 2] - Rm[2, 0]) / s
    z = (Rm[1, 0] - Rm[0, 1]) / s
  elif Rm[0, 0] > Rm[1, 1] and Rm[0, 0] > Rm[2, 2]:
    s = math.sqrt(1.0 + Rm[0, 0] - Rm[1, 1] - Rm[2, 2]) * 2
    w = (Rm[2, 1] - Rm[1, 2]) / s
    x = 0.25 * s
    y = (Rm[0, 1] + Rm[1, 0]) / s
    z = (Rm[0, 2] + Rm[2, 0]) / s
  elif Rm[1, 1] > Rm[2, 2]:
    s = math.sqrt(1.0 + Rm[1, 1] - Rm[0, 0] - Rm[2, 2]) * 2
    w = (Rm[0, 2] - Rm[2, 0]) / s
    x = (Rm[0, 1] + Rm[1, 0]) / s
    y = 0.25 * s
    z = (Rm[1, 2] + Rm[2, 1]) / s
  else:
    s = math.sqrt(1.0 + Rm[2, 2] - Rm[0, 0] - Rm[1, 1]) * 2
    w = (Rm[1, 0] - Rm[0, 1]) / s
    x = (Rm[0, 2] + Rm[2, 0]) / s
    y = (Rm[1, 2] + Rm[2, 1]) / s
    z = 0.25 * s
  n = math.sqrt(w * w + x * x + y * y + z * z)
  return [w / n, x / n, y / n, z / n]


def _build_xml_hook(args, elev_deg: float):
  """把对 MJCF 的几处改写串成一条链。

  顺序有讲究：
    1. apply_appearance  改材质/灯光（外观）
    2. apply_gm6020      改 yaw_servo / pitch_servo 的 kp/kv/forcerange
    3. duplicate_cars    复制 chassis 子树成 3 台（必须在材质之后，否则副本漏改）
    4. inject_camera     给云台挂取流相机
    5. inject_overview   插一个世界系固定视角相机
  """
  led_color = None
  if getattr(args, "led_color", None):
    led_color = tuple(float(v) for v in str(args.led_color).split(","))

  def _hook(xml: str) -> str:
    xml = apply_appearance(xml, preset=args.appearance, led_color=led_color)
    if getattr(args, "servo", "gm6020") == "gm6020":
      xml = apply_gm6020(
        xml, wn_yaw=args.servo_wn_yaw, wn_pitch=args.servo_wn_pitch, zeta=args.servo_zeta)
    xml = duplicate_cars(xml, 3)
    xml = inject_camera(xml, fovy=args.fovy, elev_deg=elev_deg)
    return inject_overview_camera(
      xml,
      [float(v) for v in args.overview_pos.split(",")],
      [float(v) for v in args.overview_lookat.split(",")],
      fovy_deg=args.overview_fovy)
  return _hook


def load_config(args):
  """按 CLI 参数装载配置层（configs/*.yaml）。

  单独包一层是为了让**所有**调用点（联调、RL env、基准脚本）走同一份配置，
  而不是各自去拼路径 —— 训练与评测的世界配置只要有一项对不上，练出来的策略
  在评测里就是另一回事（这个坑我们在 rm_car_script 那轮已经踩过一次）。
  """
  return load_config_impl(getattr(args, "config_dir", None), getattr(args, "ruleset", None))


def ballistics_params_from_config(cfg, args) -> BallisticsParams | None:
  """弹道气动参数：**以 `configs/ballistics.yaml` 为准**，可用 `--ballistics-drag` 覆盖开关。

  默认（YAML 里 `drag.enabled: false`）返回的对象 `enabled=False` ⇒ 物理逐位不变。
  打开阻力前先看 YAML 顶部那三条前置条件（同步解算器 / 重标 pitch_offset / 重录基线）。
  """
  p = ballistics_from_config(cfg)
  if p is None:
    return None
  mode = getattr(args, "ballistics_drag", "config")
  if mode == "on":
    p.enabled = True
  elif mode == "off":
    p.enabled = False
  return p


def apply_cmd_args(demo, args) -> None:
  """把 CLI 参数灌进 demo.cmd（枪的标称弹速 / 靶车机动 / 运动基准脚本）。

  单独抽出来是为了让 **RL 训练环境**（auto_aim_rl/env）和联调链路共用同一套世界配置
  —— 训练和评测的世界只要有一项对不上，练出来的策略在评测里就是另一回事。
  """
  if demo.gun is not None:
    # 枪模型开着时**发给视觉的标称速度 = 冷枪标称值（22 m/s）**：真机上视觉只知道
    # 操作手设的标称值，实际出膛速度（升温+波动）它是不知道的。--gun off 时保持 25。
    demo.cmd.ball_speed = demo.gun.nominal_speed
    # 枪的随机源（弹丸散布 / 摩擦轮弹速波动）**必须跟 --seed 绑定**：
    # GunModel 默认 default_rng(None)，每次运行都不一样 —— 于是"同种子"其实复现不了，
    # 策略 A/B 也平白多出一份噪声（同一策略重跑数字会变）。这里统一从 seed 派生，
    # 分布不变，只是变得可复现。（踩坑记录：golden trace 两次跑出不同 hash 才发现。）
    # 注意 --seed 是**可选**的（run_debug.sh 就不传），此时它是 None ——
    # 直接 int(None) 会抛 TypeError 把整个仿真拉不起来（踩过）。
    # 没给 seed 时用 0 兜底：命令行一样就跑得出一样的结果。
    _seed = getattr(args, "seed", None)
    demo.gun.rng = np.random.default_rng((0 if _seed is None else int(_seed)) + 4242)
  demo.cmd.car_motion = args.car_motion
  demo.cmd.gyro_spin_base = args.gyro_spin_base
  demo.cmd.gyro_spin_amp = args.gyro_spin_amp
  demo.cmd.gyro_period_min = args.gyro_period_min
  demo.cmd.gyro_period_max = args.gyro_period_max
  demo.cmd.gyro_dwell_min = args.gyro_dwell_min
  demo.cmd.gyro_dwell_max = args.gyro_dwell_max
  demo.cmd.gyro_patrol_min = args.gyro_patrol_min
  demo.cmd.gyro_patrol_max = args.gyro_patrol_max
  demo.cmd.gyro_shape = args.gyro_shape
  demo.cmd.gyro_enter_prob = args.gyro_enter_prob
  demo.cmd.gyro_move_frac = args.gyro_move_frac
  demo.cmd.gyro_accel = args.gyro_accel
  demo.cmd.car_script_dwell = args.car_script_dwell
  demo.cmd.car_script_shuffle = args.car_script_shuffle

  # ---- 配置层（configs/*.yaml）：靶型预设 + 轮腿升降（负责人 2026-10-14）----
  # 默认什么都不改：不给 --car-script-preset / --car-lift 时，行为与配置层不存在时一致。
  preset = getattr(args, "car_script_preset", None)
  if preset:
    cfg = load_config(args)
    p = cfg.preset(preset)
    demo.cmd.car_script_modes = p.modes
    demo.cmd.car_script_dwell = p.dwell_s
    demo.cmd.car_script_shuffle = p.shuffle

  # 轮腿靶车：默认取 configs/targets.yaml 里 infantry_wheel_legged 的升降参数
  lift_mode = getattr(args, "car_lift", None)
  if lift_mode:
    bl = load_config(args).target("infantry_wheel_legged").body_lift
    demo.cmd.car_lift = LiftParams(
      mode=lift_mode,
      amplitude_mm=(args.car_lift_amp_mm if args.car_lift_amp_mm is not None else bl.amplitude_mm),
      period_s=(args.car_lift_period_s if args.car_lift_period_s is not None else bl.period_s),
      accel_limit_mps2=(args.car_lift_accel if args.car_lift_accel is not None
                        else bl.accel_limit_mps2),
      vel_limit_mps=(args.car_lift_vel if args.car_lift_vel is not None else bl.vel_limit_mps),
      rand_hold_s=bl.rand_hold_s,
      # 行程下沿/占空比也跟着配置走：漏了它们的话，命令行开的升降会"全程都在动"、
      # 而且下沿取默认 0（这次是对的，但语义上必须与配置文件一致，否则又是一个静默分歧）
      bottom_mm=getattr(bl, "bottom_mm", 0.0),
      active_cycles=tuple(getattr(bl, "active_cycles", (1, 2))),
      pause_s=tuple(getattr(bl, "pause_s", (2.5, 5.0))))


class LinkSim:
  def __init__(self, args) -> None:
    self.args = args
    # 配置层（configs/*.yaml）只读一次，枪 / 弹道 / 靶车都从这里取值 —— 训练与评测共用同一份
    self.cfg = load_config(args)
    self.ballistics = ballistics_params_from_config(self.cfg, args)
    self.ruleset_name = self.cfg.ruleset_name         # 导出到 /state.json，便于溯源
    self.lock = threading.Lock()
    self.jpeg = None
    self.jpeg_ov = None
    self.jpeg_id = 0
    self.frame_idx = 0
    self.stop = False
    self._rng = np.random.default_rng(20251012)

    # 下位机语义：sp_vision 发来的绝对角，默认保持在初始位
    self.cmd_lock = threading.Lock()
    self.in_control = False
    self.cmd_yaw = 0.0
    self.cmd_pitch = 0.10
    self.pending_fire = 0
    self.cmd_count = 0
    self.aim_errors: list[float] = []   # 每次开火时的瞄准误差(度)
    self._jpeg_ov = None
    self._ov_counter = 0
    self._ov_watch_until = 0.0

    elev_deg = read_barrel_elev(args.assets_dir) if args.model == "real" else 0.0
    # read_barrel_elev 返回的是**度**；这里统一存弧度，避免和关节角相加时量纲搞错
    self.barrel_elev = math.radians(elev_deg)
    self.demo = R.Demo(
      dt=args.dt,
      real_assets=(args.assets_dir if args.model == "real" else None),
      xml_hook=_build_xml_hook(args, elev_deg),
      gun_params=gun_params_from_args(args),
      ballistics_params=self.ballistics,
    )
    # 枪 / 靶车机动 / 运动基准脚本的配置统一走 apply_cmd_args（RL env 用同一份）
    apply_cmd_args(self.demo, args)
    # 靶车机动参数里带 accel 的项（apply_cmd_args 已覆盖）——
    # 这里不再重复设置，避免两处漂移。
    # 固定靶车运动随机源：标定时要能复现同一段靶车轨迹，否则抖动会淹没参数差异
    if getattr(args, "seed", None) is not None:
      self.demo._rng = np.random.default_rng(int(args.seed))
    # 运动基准脚本要按段统计击杀数，所以把 CarField 的计数接给 demo
    self.demo._deaths_fn = lambda: self.cars.deaths
    # 云台伺服增益可覆盖：用于验证"跟踪滞后"是不是命中率的主要限制。
    # PD 位置伺服跟斜坡的稳态误差 ≈ (kv/kp)·ω，kp=60/kv=6 时时间常数 0.1 s，
    # 目标以 ~18°/s 移动时就是 ~1.8° 的滞后 —— 正好和实测瞄准误差中位数吻合。
    if getattr(args, "servo_kp", None):
      kp = float(args.servo_kp)
      kv = float(args.servo_kv if args.servo_kv else 6.0)
      for i in range(self.demo.model.nu):
        self.demo.model.actuator_gainprm[i][0] = kp
        self.demo.model.actuator_biasprm[i][1] = -kp
        self.demo.model.actuator_biasprm[i][2] = -kv
    self.model, self.data = self.demo.model, self.demo.data
    self.cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "eye")
    if self.cam_id < 0:
      raise RuntimeError("取流相机没插进去：MJCF 里没有 eye 相机")
    # 我方底盘扰动（阶段 0-D）。相机基准姿态要留一份：抖动是"相对基准"的，
    # 不还原的话噪声会一帧帧累加漂走。
    self.disturb = ChassisDisturb(
      ChassisParams.from_args(args),
      rng=np.random.default_rng(int(getattr(args, "seed", 0) or 0) + 991))
    self._cam_quat0 = np.array(self.model.cam_quat[self.cam_id], dtype=float).copy()
    self._cam_quat_dirty = False
    # 一帧推进的仿真时长（tick 里物理步数 × dt），扰动按它推进
    self._frame_dt = max(1, int(round(1.0 / (args.fps * args.dt)))) * args.dt
    self.pitch_bid = int(self.model.site_bodyid[self.demo.muzzle_sid])
    self.plate_geoms = [
      g for g in range(self.model.ngeom)
      if (mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, g) or "").startswith("armor")
      and not (mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, g) or "").endswith("_vis")
    ]
    # 多靶车 + 血量（必须在 Demo 建好之后）：MJCF 恒建 3 台，每轮随机启用 1~3 台
    self.cars = CarField(self.demo, self.model, self.data, mujoco,
                         np.random.default_rng(getattr(args, "seed", None) or 0),
                         self.demo.cmd,
                         spawn_count=getattr(args, "cars", 1),
                         random_spawn=getattr(args, "cars_random", False))
    # snapshot / gt 必须在这里就建好：HTTP 服务是**先于** run() 启动的，
    # 启动瞬间的外部轮询（run_debug.sh 探测端口）会打在 run() 初始化之前，
    # 否则就是 "'LinkSim' object has no attribute 'snapshot'" 刷满日志。
    # 注意要放在 self.demo 之后 —— 下面这行用到了 demo.cmd。ball_speed。
    self.snapshot = {"frame": 0, "t": 0.0, "q": [1, 0, 0, 0], "yaw": 0.0, "pitch": 0.0,
                     "ball_speed": float(self.demo.cmd.ball_speed)}
    self.gt = []
    print(f"  取流相机 eye (fovy={args.fovy}°)，装甲板 geom {len(self.plate_geoms)} 块，"
          f"炮管安装仰角 {elev_deg:+.1f}°，瞄准解算已补偿")
    if getattr(args, "servo", "gm6020") == "gm6020":
      for _l in servo_report().splitlines():
        print("  " + _l)
    # 启动自检：把"看起来没生效"的开关直接打出来（踩过：扰动默认关，
    # 联调时以为坏了，其实是没打开）
    p = self.disturb.p
    if p.enabled:
      ch = [n for n, on in (("指向抖动", p.model_gimbal), ("指向漂移", p.model_drift),
                            ("IMU噪声", p.model_imu), ("相机视轴", p.model_cam)) if on]
      print(f"  底盘扰动：开启 [{'/'.join(ch)}]  yaw 抖 {p.yaw_amp_deg}° + 漂移 "
            f"{p.drift_yaw_deg}°/τ{p.drift_tau}s  IMU {p.imu_deg}°  相机 {p.cam_deg}°")
    else:
      print("  底盘扰动：关闭（--chassis-disturb on 可开）")
    # 弹道：默认只有重力；打开气动阻力会在启动日志里显式打出来（便于溯源）
    if self.ballistics is not None:
      print("  " + self.ballistics.describe())
    print(f"  靶车机动：{args.car_motion}"
          + (f"（甩头 ω {args.gyro_spin_base}±{args.gyro_spin_amp} rad/s、"
             f"周期 {args.gyro_period_min}~{args.gyro_period_max}s、"
             f"进入概率 {args.gyro_enter_prob}、移动比例 {args.gyro_move_frac}）"
             if args.car_motion == "gyro" else ""))
    print(f"  多靶车：模型里 {len(self.cars.cars)} 台，本轮场上 {self.cars.spawn_count} 台"
          f"（HP {__import__('rm_cars').CAR_HP_MAX}，每发 {__import__('rm_cars').BULLET_DAMAGE} 伤害）")

  # ------------------------------------------------------------------ 命令
  def set_command(self, d: dict) -> None:
    with self.cmd_lock:
      if "yaw" in d:
        self.cmd_yaw = float(d["yaw"])
      if "pitch" in d:
        self.cmd_pitch = float(d["pitch"])
      self.in_control = bool(d.get("control", True))
      if d.get("shoot"):
        self.pending_fire += 1
      self.cmd_count += 1

  def _overview_due(self) -> bool:
    """全局视角这一路要不要在本帧渲染。

    没人看的时候完全不渲（这是把仿真从 0.66x 拉回实时的关键）；
    有人看的时候也只每 overview_every 帧渲一次。
    """
    if not self.args.overview:
      return False
    if self._ov_watch_until < time.time():
      return False
    self._ov_counter += 1
    return self._ov_counter % max(1, self.args.overview_every) == 0

  def touch_overview(self) -> None:
    """HTTP 侧每次请求 /overview* 时调用，表示"有人在看"。"""
    self._ov_watch_until = time.time() + 3.0

  def _record_aim_error(self) -> None:
    p, d = self.demo.muzzle()
    best = None
    for gid in self.plate_geoms:
      v = np.array(self.data.geom_xpos[gid]) - p
      nv = np.linalg.norm(v)
      if nv < 1e-6:
        continue
      cosang = float(np.clip(np.dot(d, v / nv), -1.0, 1.0))
      ang = math.degrees(math.acos(cosang))
      if best is None or ang < best:
        best = ang
    if best is not None:
      self.aim_errors.append(best)

  # ------------------------------------------------------------------ 每帧
  def tick(self) -> None:
    d, m = self.demo, self.model
    # 我方底盘扰动（阶段 0-D，默认关）：先推进一步，再把它注入到
    # ① 云台关节目标（世界系指向抖动，物理上自洽）
    # ② 相机模型姿态（视轴额外抖动，只影响画面）
    self.disturb.step(self._frame_dt)
    if self.disturb.p.enabled and self.disturb.p.model_cam:
      dq = axis_angle_to_quat(self.disturb.cam_quat_noise())
      m.cam_quat[self.cam_id] = quat_mul(self._cam_quat0, dq)
    elif self._cam_quat_dirty:
      m.cam_quat[self.cam_id] = self._cam_quat0
      self._cam_quat_dirty = False
    with self.cmd_lock:
      cmd_yaw, cmd_pitch = self.cmd_yaw, self.cmd_pitch
      n_fire = self.pending_fire
      self.pending_fire = 0
    # sp_vision 的 pitch 抬头为负；仿真关节角 = 安装角 + command.pitch
    # 底盘颠簸直接叠在关节目标上：伺服会追它，于是世界系指向真的在抖 ——
    # 相机画面、弹丸方向、开火门限判据三者一致地受影响。
    # 高频抖动 + 低频漂移一起叠上（两者都由 rm_chassis 生成）
    joint_yaw = cmd_yaw + self.disturb.yaw_jitter + self.disturb.yaw_drift
    joint_pitch = float(np.clip(
      self.barrel_elev + cmd_pitch + self.disturb.pitch_jitter + self.disturb.pitch_drift,
      self.demo.pitch_lo, self.demo.pitch_hi))
    self.demo.cmd.aim_point = None
    self.demo.cmd.yaw, self.demo.cmd.pitch = joint_yaw, joint_pitch
    for _ in range(n_fire):
      # 记录开火瞬间的瞄准误差：炮口朝向 vs 到最近装甲板中心的夹角。
      # 命中率只能告诉我们"打没打中"，这个量能直接说明"偏了多少、往哪偏"。
      self._record_aim_error()
      self.demo.fire()
    for _ in range(max(1, int(round(1.0 / (self.args.fps * m.opt.timestep))))):
      self.cars.pre_step(m.opt.timestep)
      self.demo.step(m, self.data)
      self.cars.post_step(m.opt.timestep)

    from PIL import Image

    # 1) 云台相机：喂给 sp_vision 的那一路
    self.renderer.update_scene(self.data, camera="eye")
    buf = io.BytesIO()
    Image.fromarray(self.renderer.render()).save(buf, format="JPEG", quality=self.args.quality)
    jpeg = buf.getvalue()

    # 2) 全局固定视角：给人看的，不随云台转。**惰性渲染** —— 没人在看就完全不渲，
    #    否则 1280x960 + 1280x720 双路会把仿真拖到 0.6 倍实时，而闭环依赖实时性。
    #
    #    实测（有人看时）：
    #      800x450  every=2 -> 20.6 FPS   1024x576 every=2 -> 19.8
    #      1024x576 every=1 -> 16.1       1280x720 every=2 -> 19.8
    #    结论：**分辨率几乎不影响帧率**（每次渲染的 GL 上下文切换才是固定开销），
    #    所以画质基本白拿；真正影响仿真帧率的是 every（抽帧）。
    #    默认给 every=1 换全局视角的流畅度，代价是仿真降到约 0.53 倍实时。
    #    想换回实时性：--overview-every 2
    #
    #    注：本想用 update_scene(camera=[...]) 一次渲两路来省掉上下文切换，
    #    但本机 MuJoCo 的 classic renderer 不支持相机列表（只有 warp 版支持）。
    jpeg_ov = self._jpeg_ov
    if self._overview_due():
      self.renderer_ov.update_scene(self.data, camera="overview")
      buf_ov = io.BytesIO()
      Image.fromarray(self.renderer_ov.render()).save(
        buf_ov, format="JPEG", quality=self.args.overview_quality)
      jpeg_ov = buf_ov.getvalue()
      self._jpeg_ov = jpeg_ov

    Rm = np.array(self.data.xmat[self.pitch_bid]).reshape(3, 3)
    q = mat2quat(Rm)
    # IMU/姿态通道噪声：扰动**报给视觉的**四元数（物理状态不变）
    if self.disturb.p.enabled and self.disturb.p.model_imu:
      q = quat_mul(np.asarray(q, dtype=float),
                   axis_angle_to_quat(self.disturb.imu_quat_noise())).tolist()
    mp, md = self.demo.muzzle()
    with self.lock:
      self.jpeg = jpeg
      self.jpeg_ov = jpeg_ov
      self.jpeg_id += 1
      self.snapshot = {
        "frame": self.frame_idx,
        "t": float(self.data.time),
        "q": q,
        "yaw": float(self.data.qpos[self.demo.yaw_qadr]),
        "pitch": float(self.data.qpos[self.demo.pitch_qadr]),
        "muzzle_pos": [float(v) for v in mp],
        "muzzle_dir": [float(v) for v in md],
        "car_pos": [float(v) for v in self.data.xpos[self.demo.car_bid]],
        "car_positions": [
          [float(v) for v in self.data.xpos[c.bid]] for c in self.cars.cars if c.alive
        ],
        "shots": self.demo.shots,
        "hits": self.demo.hits,
        # 发给视觉的**标称**弹速（真机语义：操作手设定值）
        "ball_speed": float(self.demo.cmd.ball_speed),
        # 枪的真实状态：实际弹速 / 热量 / 锁枪 / 超限扣血 ——
        # 既是诊断信息，也是将来 RL 的观测来源
        "gun": None if self.demo.gun is None else self.demo.gun.status(),
        # 靶车机动状态（阶段 0-C）：当前自转角速度与模式（patrol / gyro），
        # 方便在 PlotJuggler / 浏览器里直接看"什么时候在甩头"
        "car_omega": float(getattr(self.demo, "_car_omega", 0.0)),
        "car_motion_mode": ("classic" if self.demo._gyro is None else self.demo._gyro.mode),
        # 我方底盘扰动状态（阶段 0-D）：当前指向抖动量 + 三个通道的开关
        "chassis": self.disturb.status(),
        # 轮腿靶车的车体升降状态（阶段 0-E）：当前偏移/目标/速度/限幅/峰值
        # 既是诊断信息，也是将来"竖直预测"类策略的观测量来源
        "car_lift": (None if getattr(self.demo, "_lift", None) is None
                     else self.demo._lift.status()),
        # 配置层身份：换规则集/靶型预设后，跑出来的数字属于哪一版一目了然
        "config": {"ruleset": self.ruleset_name},
        # 运动基准脚本（固定 10s 一段、乱序）：当前靶型 + 分段统计
        "car_script": (None if self.demo._script is None else
                       self.demo._script.status((self.demo.shots, self.demo.hits,
                                                 self.cars.deaths))),
      }
      self.gt = self.gt_points()
    if self.args.save_dir and self.frame_idx % max(1, self.args.save_every) == 0:
      self._save(jpeg)
    self.frame_idx += 1

  def _save(self, jpeg: bytes) -> None:
    """抓图 + 同步真值，给 eval_detect.py 做识别召回评测。

    收的是 tick() 里已经编码好的 JPEG 字节 —— 那一份就是真正喂给 sp_vision 的图，
    重编码反而会引入伪影，让评测结果偏离实机。
    """
    os.makedirs(self.args.save_dir, exist_ok=True)
    stem = f"frame_{self.frame_idx:06d}"
    with open(os.path.join(self.args.save_dir, stem + ".jpg"), "wb") as f:
      f.write(jpeg)
    with open(os.path.join(self.args.save_dir, stem + ".json"), "w") as f:
      json.dump({"frame": stem, "detections": list(self.gt)}, f, ensure_ascii=False)

  # ------------------------------------------------- sp_vision 口径的装甲板真值
  def gt_points(self) -> list[dict]:
    """把每块装甲板按 sp_vision 的尺寸约定(135x56)投影成 4 个角点。

    直接投影仿真装甲板的真实尺寸是不行的：solver.cpp 里写死了
    SMALL_ARMOR_WIDTH=135mm / LIGHTBAR_LENGTH=56mm，用别的尺寸反解出来的距离会错。
    所以这里按 sp_vision 的模型**合成**四个点，等价于"一个完美的四点检测器"。
    """
    cam_pos = np.array(self.data.cam_xpos[self.cam_id])
    cam_mat = np.array(self.data.cam_xmat[self.cam_id]).reshape(3, 3)
    fovy = float(self.model.cam_fovy[self.cam_id])
    w, h = self.args.width, self.args.height
    out = []
    # 多车：遍历所有"活着的车"的装甲板，而不是只遍历固定几何列表
    plate_list = [(car, gid, nm) for car, gid, nm in self.cars.armor_geoms()] \
      if self.cars is not None else []
    for car_state, gid, _nm in plate_list:
      name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, gid) or f"g{gid}"
      car = np.array(self.data.xpos[car_state.bid])
      Rg = np.array(self.data.geom_xmat[gid]).reshape(3, 3)      # geom -> world
      c = np.array(self.data.geom_xpos[gid])
      # 直接按几何构造 sp_vision 的装甲板系，保证：+X 朝车外、右手系、+Z 大致朝上。
      # 不写死法线方向，是因为四块板的 geom 局部朝向不一定一致。
      # 四块装甲板的 geom 局部轴约定**并不统一**：转换器给"前后板"的是
      # [宽,厚,高]，给"左右板"的是 [厚,宽,高]。所以不能写死哪个轴是法线，
      # 按**尺寸**反推：最薄的是厚度(法线)轴，最宽的是宽度轴，剩下的是高度轴。
      # （写死成 y 会让左右板的"宽度"沿法线方向，投影出来退化成一条竖线。）
      half = np.array(self.model.geom_size[gid], dtype=float)
      i_thick = int(np.argmin(half))
      i_width = int(np.argmax(half))
      i_height = 3 - i_thick - i_width
      e_thick = np.zeros(3); e_thick[i_thick] = 1.0
      e_height = np.zeros(3); e_height[i_height] = 1.0
      thick = float(half[i_thick])
      n = Rg @ e_thick
      if float(n @ (c - car)) < 0:
        n = -n
      z_sv = Rg @ e_height                                     # 板高方向
      z_sv = z_sv / (np.linalg.norm(z_sv) + 1e-12)
      y_sv = np.cross(z_sv, n)
      y_sv = y_sv / (np.linalg.norm(y_sv) + 1e-12)
      z_sv = np.cross(n, y_sv)
      # +Z 必须朝上：翻转时 y/z 一起翻，否则右手系就破了
      if z_sv[2] < 0:
        y_sv, z_sv = -y_sv, -z_sv
      R_sv2world = np.column_stack([n, y_sv, z_sv])              # 列 = sv 三轴在世界系下
      face = c + n * thick                                      # 板外表面中心
      # 顺序必须和 solver.cpp 的 SMALL_ARMOR_POINTS 完全一致，否则 solvePnP 的
      # 点对应对不上，解出来的姿态会整体转 90°/180°（实测表现为 yaw 满屏乱跳）：
      #   obj = [(+W/2,+L/2), (-W/2,+L/2), (-W/2,-L/2), (+W/2,-L/2)]
      corners = [(1, 1), (-1, 1), (-1, -1), (1, -1)]
      pts = []
      for sy, sz in corners:
        local = np.array([0.0, sy * SV_ARMOR_W * 0.5, sz * SV_ARMOR_H * 0.5])
        pts.append(face + R_sv2world @ local)
      uv, front = project(np.array(pts), cam_pos, cam_mat, fovy, w, h)
      if not front.all():
        continue
      if uv[:, 0].max() < 0 or uv[:, 1].max() < 0 or uv[:, 0].min() > w or uv[:, 1].min() > h:
        continue

      # 侧对相机的板要丢掉。真实的四点检测器看不到这种板；更重要的是，
      # 退化成细线的四点做 PnP 是病态的，解出来的姿态基本是乱的，
      # 会把 tracker 带飞（实测表现为瞄准角偶尔跳到 ±160°）。
      facing = float(n @ ((cam_pos - face) / (np.linalg.norm(cam_pos - face) + 1e-12)))
      area = 0.0
      for k in range(4):
        x1, y1 = uv[k]
        x2, y2 = uv[(k + 1) % 4]
        area += x1 * y2 - x2 * y1
      area = abs(area) * 0.5
      if facing < self.args.gt_min_cos or area < self.args.gt_min_area:
        continue
      # 亚像素噪声：**必须加**。理想矩形的 PnP 是退化的 —— IPPE 的两个解
      # 重投影误差都是 0，OpenCV 会随机挑一个（通常是镜像解，目标跑到相机侧下方）。
      # 真实检测器本来就有 ~0.3px 抖动，加上它既贴近真实，又能让 PnP 收敛到正确分支。
      if self.args.gt_noise > 0:
        uv = uv + self._rng.normal(0.0, self.args.gt_noise, uv.shape)
      # 顺手导出四个角在 gimbal_pitch 系下的真值坐标，便于外部独立复算投影、
      # 定位 sp_vision 侧约定错误（sim_geom_check 只能看出"错"，看不出"错在哪一步"）
      Rb = np.array(self.data.xmat[self.pitch_bid]).reshape(3, 3)
      ppos = np.array(self.data.xpos[self.pitch_bid])
      corners_gimbal = [Rb.T @ (p - ppos) for p in pts]
      out.append({
        "plate": name,
        "cls": self.args.label,
        "points": [[round(float(u), 2), round(float(v), 2)] for u, v in uv],
        "corners_gimbal": [[round(float(x), 5) for x in c] for c in corners_gimbal],
        "xyz_in_world": [round(float(v), 4) for v in (face - ppos)],
        "facing": round(facing, 3),
        "area": round(area, 1),
      })
    return out

  def run(self) -> None:
    self.renderer = mujoco.Renderer(self.model, height=self.args.height, width=self.args.width)
    self.renderer_ov = mujoco.Renderer(
      self.model, height=self.args.overview_height, width=self.args.overview_width)
    period = 1.0 / max(self.args.fps, 1.0)
    n = 0
    t0 = time.time()
    while not self.stop:
      s = time.time()
      self.tick()
      n += 1
      if n % 120 == 0:
        with self.lock:
          snap = dict(self.snapshot)
        el = time.time() - t0
        print(f"  {n} 帧 / {el:.1f}s = {n/el:.1f} FPS  仿真 {snap['t']:.1f}s  "
              f"关节 yaw={math.degrees(snap['yaw']):+.1f}° pitch={math.degrees(snap['pitch']):+.1f}°  "
              f"真值 {len(self.gt)} 块  发射 {snap['shots']} 命中 {snap['hits']}")
      sleep = period - (time.time() - s)
      if sleep > 0:
        time.sleep(sleep)


BOUNDARY = b"--framer"


def make_handler(sim: LinkSim):
  class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
      pass

    def _send(self, body: bytes, ctype: str, code: int = 200) -> None:
      self.send_response(code)
      self.send_header("Content-Type", ctype)
      self.send_header("Content-Length", str(len(body)))
      self.send_header("Cache-Control", "no-store")
      self.end_headers()
      self.wfile.write(body)

    def do_POST(self):  # noqa: N802
      path = self.path.split("?")[0]
      if path != "/command":
        self._send(b"not found", "text/plain", 404)
        return
      n = int(self.headers.get("Content-Length", 0))
      try:
        d = json.loads(self.rfile.read(n) or b"{}")
      except Exception as e:
        self._send(json.dumps({"ok": False, "err": str(e)}).encode(), "application/json", 400)
        return
      sim.set_command(d)
      self._send(b'{"ok":true}', "application/json")

    def do_GET(self):  # noqa: N802
      parts = self.path.split("?", 1)
      path = parts[0]
      query = parts[1] if len(parts) > 1 else ""
      if path == "/grab":
        with sim.lock:
          snap = dict(sim.snapshot)
          jpeg = sim.jpeg
          # 真值必须和图像来自同一个 tick，否则和图像一样会错帧
          snap["detections"] = list(sim.gt)
        if jpeg is None:
          self._send(b"{}", "application/json", 503)
          return
        snap["jpeg_b64"] = base64.b64encode(jpeg).decode()
        self._send(json.dumps(snap).encode(), "application/json")
      elif path == "/state.json":
        with sim.lock:
          snap = dict(sim.snapshot)
          gt = list(sim.gt)
        snap["detections"] = gt
        snap["hits_by_plate"] = sim.demo.hits_by_plate
        snap["cmd_count"] = sim.cmd_count
        # 逐发弹道记录（调瞄具用）：命中率是稀疏的 0/1 信号、噪声极大，
        # 这个列表里的"离板心最近距离"是连续量，几十发就能看出系统性偏差。
        snap["shot_log"] = list(sim.demo.shot_log)
        # 多靶车 + 血量
        snap["carfield"] = sim.cars.status()
        if sim.aim_errors:
          e = np.array(sim.aim_errors)
          snap["aim_error_deg"] = {
            "n": int(e.size), "mean": round(float(e.mean()), 3),
            "p50": round(float(np.percentile(e, 50)), 3),
            "p10": round(float(np.percentile(e, 10)), 3),
            "p90": round(float(np.percentile(e, 90)), 3),
            "lt_1deg": round(float((e < 1.0).mean()), 3),
          }
        self._send(json.dumps(snap, ensure_ascii=False).encode(), "application/json")
      elif path in ("/frame.jpg", "/overview.jpg"):
        is_ov = path.startswith("/overview")
        if is_ov:
          sim.touch_overview()
          # 惰性渲染下首帧还没出来，等一下再取，否则返回空图
          for _ in range(20):
            with sim.lock:
              body = sim.jpeg_ov
            if body:
              break
            time.sleep(0.05)
        else:
          with sim.lock:
            body = sim.jpeg
        self._send(body or b"", "image/jpeg")
      elif path in ("/stream.mjpg", "/overview.mjpg"):
        overview = path.startswith("/overview")
        if overview:
          sim.touch_overview()
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=framer")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        last = -1
        last_touch = 0.0
        try:
          while not sim.stop:
            # 关键：这里必须**每轮都续期**。之前只在请求开始时 touch 一次，
            # 3 秒后惰性渲染的看门狗就过期 → 全局画面不再更新，
            # 而循环仍按帧号重复发同一张图 → 页面上就是"画面定住"。
            now = time.time()
            if overview and now - last_touch > 0.5:
              sim.touch_overview()
              last_touch = now
            with sim.lock:
              jpeg = sim.jpeg_ov if overview else sim.jpeg
              idx = sim.jpeg_id
            if jpeg is None or idx == last:
              time.sleep(0.003)
              continue
            last = idx
            self.wfile.write(BOUNDARY + b"\r\nContent-Type: image/jpeg\r\n"
                             + f"Content-Length: {len(jpeg)}\r\n\r\n".encode()
                             + jpeg + b"\r\n")
        except (BrokenPipeError, ConnectionResetError):
          pass
      elif path in ("/", "/index.html"):
        # 页面上**只放全局固定视角**。云台那一路（/stream.mjpg）接口保留，
        # 因为 sp_vision 要靠它取图，只是不再往页面上摆。
        cam_block = ""
        ov_w = sim.args.overview_width
        self._send(
          ("<html><head><title>rm_sim</title></head>"
           "<body style='background:#111;color:#ddd;font-family:monospace;margin:16px'>"
           "<h3 style='margin:4px 0'>rm_sim &lt;-&gt; sp_vision 联调</h3>"
           "<div id='st' style='font-size:15px;color:#8fd;margin:8px 0;"
           "min-height:1.3em;letter-spacing:.3px'></div>"
           "<div style='display:flex;flex-direction:column;gap:12px'>"
           "<div><div style='color:#6cf;font-size:13px;margin-bottom:4px'>"
           "全局固定视角（靶车运动 + 命中情况）</div>"
           # image-rendering: pixelated 关掉浏览器的平滑放大 ——
           # 之前用 width:100% 把 800px 的图拉到 1500px 显示，糊得很明显
           "<img src='/overview.mjpg' style='width:100%;max-width:" + str(ov_w) +
           "px;image-rendering:auto'></div>"
           + cam_block +
           "</div>"
           "<p style='font-size:12px;margin-top:14px'>"
           "<a href='/state.json'>/state.json</a> · /grab · /overview.jpg · /overview.mjpg · "
           "/frame.jpg · /stream.mjpg · POST /command</p>"
           # 状态条的口径：发射用 demo.shots（Demo.fire() 的计数，就是弹丸数）；
           # 命中用 carfield.hits（驱动血量的那一套，覆盖所有活车）。
           # 不用 demo.hits —— 那个只统计第 0 台车，多车时偏少。
           #
           # 车体升降（轮腿靶车）：**只有真开了升降才显示**，否则状态条上会一直挂着
           # 一行恒为 0 的数字。口径 = /state.json 的 car_lift，它来自 demo._lift，
           # 也就是第 0 台车（多车时其它车没有单独导出，所以这里写死"车1"）。
           # 为什么要显示：升降是最容易被"参数加了但没生效"坑到的一环（优先级还改过一次），
           # 联调时肉眼必须能确认车体真的在上下走，而不是靠猜。
           "<script>"
           "function tick(){"
           "  fetch('/state.json').then(function(r){return r.json()}).then(function(d){"
           "    var c=d.carfield||{};var cars=c.cars||[];var s='';"
           "    var shots=d.shots||0, hits=c.hits||0;"
           "    var rate=shots>0?(100.0*hits/shots).toFixed(1):'--';"
           "    s+='场上 '+c.spawn_count+' 台';"
           "    s+='    发射 '+shots+'    命中 '+hits+'    命中率 '+rate+'%';"
           "    s+='    靶车阵亡 '+c.deaths;"
           "    for(var i=0;i<cars.length;i++){var x=cars[i];"
           "      if(x.alive){s+='    | 车'+x.idx+' HP '+x.hp+'/600';}}"
           "    var lf=d.car_lift;"
           "    if(lf&&lf.mode&&lf.mode!=='off'){"
           "      var lv=Number(lf.offset_mm)||0;"
           "      s+='    | 车1 升降 '+(lv>=0?'+':'')+lv.toFixed(0)+'mm ('+lf.mode"
           "+((lf.active===false)?'·停':'·动')+')';}"
           "    s+='    | 仿真 '+d.t.toFixed(1)+'s';"
           "    document.getElementById('st').textContent=s;"
           "  }).catch(function(){});"
           "}"
           "setInterval(tick,500);tick();"
           "</script>"
           "</body></html>").encode(), "text/html; charset=utf-8")
      else:
        self._send(b"not found", "text/plain", 404)

  return Handler


def build_parser() -> argparse.ArgumentParser:
  """参数解析单独抽出来，好让 gen_dataset.py 复用同一套渲染/物理默认值。

  数据集和联调必须跑在**完全一样**的配置上，否则训出来的检测器和部署时看到的
  画面分布对不上，白训。复制一份参数表迟早会漂移，所以这里共用。
  """
  ap = argparse.ArgumentParser(description="rm_sim <-> sp_vision 闭环联调接口")
  ap.add_argument("--model", choices=("simple", "real"), default="real")
  ap.add_argument("--assets-dir", default=os.path.join(_HERE, "assets"))
  ap.add_argument("--port", type=int, default=8099)
  # HTTP 绑定地址：默认只监听本机（与历史行为一致）。
  # 跨机联调（例如 sp_vision 跑在另一台机器、或从浏览器看流）写 --host 0.0.0.0，
  # 这样 LAN 内可用 http://<本机IP>:<port>/ 访问；sp_vision 侧对应改 configs/sim.yaml 的 sim.base_url。
  ap.add_argument("--host", default="127.0.0.1",
                  help="HTTP 监听地址；127.0.0.1=仅本机（默认），0.0.0.0=开放给局域网")
  # 默认跟 rm_gimbal_demo 的离屏分辨率对齐（1280x960）：
  # 板子在 YOLO 的 640 输入里会大一倍，识别召回明显更好；
  # 同时 draw_detections 缩 0.5 后的 debug 窗口正好是 640x480，不再是 320x240。
  ap.add_argument("--width", type=int, default=1280)
  ap.add_argument("--height", type=int, default=960)
  ap.add_argument("--overview", dest="overview", action="store_true", default=True,
                  help="启用全局固定视角（默认开，但只在有人看时才渲染）")
  ap.add_argument("--no-overview", dest="overview", action="store_false")
  # 实测：全局视角的分辨率对帧率几乎没影响（800x450 -> 1280x720 只掉 0.8 FPS），
  # 瓶颈是每次渲染的固定开销（GL 上下文切换），不是像素数。所以画质基本是白拿的。
  ap.add_argument("--overview-width", type=int, default=1280, help="全局视角分辨率")
  ap.add_argument("--overview-height", type=int, default=720)
  ap.add_argument("--overview-every", type=int, default=1,
                  help="全局视角每 N 帧渲一次；1=最流畅但仿真会降到约 0.53 倍实时")
  ap.add_argument("--overview-pos", default="-2.6,-2.2,2.3", help="全局相机位置 x,y,z")
  ap.add_argument("--overview-lookat", default="2.0,0.0,0.15", help="全局相机看向的点")
  ap.add_argument("--overview-fovy", type=float, default=45.0)
  ap.add_argument("--fps", type=float, default=30.0)
  ap.add_argument("--quality", type=int, default=85, help="云台相机 JPEG 质量")
  ap.add_argument("--overview-quality", type=int, default=92, help="全局视角 JPEG 质量")
  ap.add_argument("--fovy", type=float, default=60.0)
  ap.add_argument("--dt", type=float, default=0.001)
  ap.add_argument("--label", default="3")
  # ---- 枪（摩擦轮弹速 / 枪口热量 / 弹丸散布）----
  # 参数都是拍脑袋的工程默认值（【待真机校准】），改这里即可整体切换真机数据。
  # ---- 我方底盘扰动（阶段 0-D，默认关）----
  ap.add_argument("--chassis-disturb", choices=("on", "off"), default="off",
                  help="我方底盘扰动（颠簸->云台指向抖动 + IMU 噪声 + 相机视轴抖动）")
  ap.add_argument("--disturb-gimbal", dest="disturb_gimbal", action="store_true", default=True)
  ap.add_argument("--no-disturb-gimbal", dest="disturb_gimbal", action="store_false")
  ap.add_argument("--disturb-imu", dest="disturb_imu", action="store_true", default=True)
  ap.add_argument("--no-disturb-imu", dest="disturb_imu", action="store_false")
  ap.add_argument("--disturb-cam", dest="disturb_cam", action="store_true", default=True)
  ap.add_argument("--no-disturb-cam", dest="disturb_cam", action="store_false")
  ap.add_argument("--disturb-drift", dest="disturb_drift", action="store_true", default=True,
                  help="低频指向漂移（底盘在走/转向时云台基座缓慢摆动，画面上看得见）")
  ap.add_argument("--no-disturb-drift", dest="disturb_drift", action="store_false")
  ap.add_argument("--disturb-drift-yaw-deg", type=float, default=1.20, help="yaw 漂移 1σ（度）")
  ap.add_argument("--disturb-drift-pitch-deg", type=float, default=0.80, help="pitch 漂移 1σ（度）")
  ap.add_argument("--disturb-drift-tau", type=float, default=3.0, help="漂移相关时间（秒）")
  ap.add_argument("--disturb-yaw-deg", type=float, default=0.35, help="云台 yaw 指向抖动 1σ（度）")
  ap.add_argument("--disturb-pitch-deg", type=float, default=0.25, help="俯仰抖动 1σ（度）")
  ap.add_argument("--disturb-band-lo", type=float, default=0.8, help="抖动频带下限 Hz")
  ap.add_argument("--disturb-band-hi", type=float, default=4.0, help="抖动频带上限 Hz")
  ap.add_argument("--disturb-imu-deg", type=float, default=0.15, help="IMU 姿态噪声 1σ（度）")
  ap.add_argument("--disturb-cam-deg", type=float, default=0.10, help="相机视轴抖动 1σ（度）")
  # ---- 靶车机动（小陀螺，阶段 0-C）----
  ap.add_argument("--car-motion", choices=("classic", "gyro", "script"), default="gyro",
                  help="classic=巡逻+连续自转；gyro=加「静止快速甩头」小陀螺（默认）；"
                       "script=固定 10s 一段、四种靶型乱序的**基准脚本**（评测 RL 用）")
  ap.add_argument("--config-dir", default=None,
                  help="配置目录（默认 rm_sim/configs，见 configs/README.md）")
  ap.add_argument("--ruleset", default=None,
                  help="规则集名（默认 legacy_sim；configs/rules/<name>.yaml）")
  ap.add_argument("--car-script-preset", default=None,
                  help="基准靶型预设（configs/targets.yaml 的 presets，"
                       "如 benchmark_2026_wl = 含轮腿 6 段；默认用 4 段的老口径）")
  # ---- 轮腿靶车：车体升降（负责人 2026-10-14：±150 mm、一次约 2 s）----
  ap.add_argument("--car-lift", choices=("off", "periodic", "random"), default=None,
                  help="车体升降模式；不给就用 configs/targets.yaml 里该靶车的配置（默认 off）")
  ap.add_argument("--car-lift-amp-mm", type=float, default=None, help="升降半幅 mm（默认 150）")
  ap.add_argument("--car-lift-period-s", type=float, default=None, help="升降周期 s（默认 2.0）")
  ap.add_argument("--car-lift-accel", type=float, default=None,
                  help="升降加速度上限 m/s²（默认 2.0：【待校准】，用 run_debug 看轨迹调）")
  ap.add_argument("--car-lift-vel", type=float, default=None, help="升降速度上限 m/s（默认 0.5）")
  ap.add_argument("--car-script-dwell", type=float, default=10.0, help="基准脚本每段时长 s")
  ap.add_argument("--car-script-shuffle", dest="car_script_shuffle",
                  action="store_true", default=True, help="每轮乱序（默认；固定种子可复现）")
  ap.add_argument("--no-car-script-shuffle", dest="car_script_shuffle", action="store_false",
                  help="按固定顺序播放（调试用）")
  ap.add_argument("--gyro-spin-base", type=float, default=7.0, help="甩头角速度基准 rad/s")
  ap.add_argument("--gyro-spin-amp", type=float, default=5.0, help="正弦调制幅值 rad/s")
  ap.add_argument("--gyro-period-min", type=float, default=0.4, help="正弦周期下限 s")
  ap.add_argument("--gyro-period-max", type=float, default=0.8, help="正弦周期上限 s")
  ap.add_argument("--gyro-dwell-min", type=float, default=1.0, help="小陀螺持续下限 s")
  ap.add_argument("--gyro-dwell-max", type=float, default=3.0, help="小陀螺持续上限 s")
  ap.add_argument("--gyro-patrol-min", type=float, default=2.0, help="巡逻持续下限 s")
  ap.add_argument("--gyro-patrol-max", type=float, default=5.0, help="巡逻持续上限 s")
  ap.add_argument("--gyro-shape", type=float, default=0.45, help="甩头波形指数")
  ap.add_argument("--gyro-enter-prob", type=float, default=0.35,
                  help="巡逻后进入小陀螺的概率（默认 0.35：保留原地旋转靶但降低出现概率）")
  ap.add_argument("--gyro-accel", type=float, default=120.0,
                  help="小陀螺角加速度限幅 rad/s²（防进出机动时角速度突跳）")
  ap.add_argument("--gyro-move-frac", type=float, default=0.0,
                  help="小陀螺期间保留的线速度比例：0=停住（默认，真机静止甩头），1=边走边甩头")
  ap.add_argument("--gun", choices=("on", "off"), default="on",
                  help="off = 理想枪（恒速 25、无热量、无散布），用于复现旧基线")
  ap.add_argument("--gun-speed-cold", type=float, default=22.0, help="冷枪标称弹速 m/s")
  ap.add_argument("--gun-speed-warm", type=float, default=24.0, help="热枪饱和弹速 m/s")
  ap.add_argument("--gun-tau-warm", type=float, default=20.0, help="升温时间常数（发）")
  ap.add_argument("--gun-tau-cool", type=float, default=90.0, help="摩擦轮降温时间常数（秒）")
  ap.add_argument("--gun-jitter", type=float, default=0.5, help="弹速波动 ±m/s（按 2σ）")
  ap.add_argument("--gun-heat-max", type=float, default=100.0, help="枪口热量上限")
  ap.add_argument("--gun-heat-per-shot", type=float, default=10.0, help="每发加热量")
  ap.add_argument("--gun-heat-cool", type=float, default=10.0, help="冷却速率 /s")
  ap.add_argument("--gun-overheat-k", type=float, default=2.0,
                  help="超限扣血系数 /s：每秒扣 k×(超出热量) 点血")
  ap.add_argument("--gun-spread", type=float, default=0.15, help="枪口角散布 1σ（度）")
  # 弹道气动（空气阻力）。**默认跟 YAML**（configs/ballistics.yaml 的 drag.enabled=false
  # ⇒ 只有重力，与历史基线逐位一致）；这里给个 CLI 覆盖便于 A/B：
  #   --ballistics-drag on  = 开阻力（注意：解算器还没同步，会系统性打低）
  #   --ballistics-drag off = 强制关（即使 YAML 里开了）
  ap.add_argument("--ballistics-drag", choices=("config", "on", "off"), default="config",
                  help="覆盖 ballistics.yaml 的 drag.enabled：config=按 YAML（默认）/ on / off")
  ap.add_argument("--gun-speed-model", dest="gun_speed", action="store_true", default=True)
  ap.add_argument("--no-gun-speed", dest="gun_speed", action="store_false")
  ap.add_argument("--gun-heat-model", dest="gun_heat", action="store_true", default=True)
  ap.add_argument("--no-gun-heat", dest="gun_heat", action="store_false")
  ap.add_argument("--gun-spread-model", dest="gun_spread_model", action="store_true", default=True)
  ap.add_argument("--no-gun-spread", dest="gun_spread_model", action="store_false")
  # 多车默认关闭：所有装甲板贴纸号码相同（都是 --label），场上有几台车视觉侧就看到
  # 几组一样的号码，tracker 分不清是哪台车，瞄准会乱跳。真实比赛是 3 号 / 4 号各不同。
  ap.add_argument("--cars", type=int, default=1, help="场上靶车台数（默认 1，上限 3）")
  ap.add_argument("--cars-random", action="store_true",
                  help="恢复「每次死亡后随机 1~3 台」（号码相同，瞄准会混）")
  ap.add_argument("--led-color", type=lambda s: tuple(float(v) for v in s.split(",")),
                  default=None, help="灯条 RGB 覆盖，如 0.55,0.78,1.0（越白 luma 越高）")
  ap.add_argument("--appearance", choices=list(APPEARANCE_PRESETS), default="realistic",
                  help="渲染外观预设；original=原样，realistic*=深色车体+高亮灯条（给检测器用）")
  ap.add_argument("--save-dir", default="", help="抓图+真值到目录，供识别召回评测")
  ap.add_argument("--save-every", type=int, default=5)
  ap.add_argument("--gt-noise", type=float, default=0.35,
                  help="真值四点的亚像素噪声(px)；0 会让 PnP 退化到镜像解")
  ap.add_argument("--gt-min-area", type=float, default=250.0,
                  help="真值四点最小像素面积；更小的板视为看不到")
  ap.add_argument("--gt-min-cos", type=float, default=0.30,
                  help="板法线与视线的余弦下限；更侧对的板直接丢弃（PnP 病态）")
  ap.add_argument("--seconds", type=float, default=0.0)
  ap.add_argument("--seed", type=int, default=None, help="靶车运动随机种子（标定用，保证可复现）")
  ap.add_argument("--servo-kp", type=float, default=None, help="覆盖云台伺服 kp（验证跟踪滞后用）")
  ap.add_argument("--servo-kv", type=float, default=None, help="覆盖云台伺服 kv")
  # 云台模型：gm6020 = 按 GM6020 堵转转矩(2.04 N·m) + 模型惯量反推 kp/kv，
  # 并加 forcerange 让大机动自然饱和；legacy = 原来的 kp=50 kv=6 无限制版本
  ap.add_argument("--servo", choices=("gm6020", "legacy"), default="gm6020")
  ap.add_argument("--servo-wn-yaw", type=float, default=None,
                  help="yaw 目标带宽 rad/s（默认 100≈15.9Hz，见 rm_gimbal_servo.py）")
  ap.add_argument("--servo-wn-pitch", type=float, default=None,
                  help="pitch 目标带宽 rad/s（默认 120≈19.1Hz）")
  ap.add_argument("--servo-zeta", type=float, default=1.0, help="阻尼比（默认 1.0 临界阻尼）")
  return ap


def main() -> int:
  a = build_parser().parse_args()

  sim = LinkSim(a)
  srv = ThreadingHTTPServer((a.host, a.port), make_handler(sim))
  threading.Thread(target=srv.serve_forever, daemon=True).start()
  print(f"  /grab        http://{a.host}:{a.port}/grab        (图像 + 同帧四元数)")
  print(f"  /state.json  http://{a.host}:{a.port}/state.json  (状态 + 真值 + 命中)")
  print(f"  POST /command 绝对角 {{control,shoot,yaw,pitch}}")
  print(f"  /stream.mjpg http://{a.host}:{a.port}/stream.mjpg")
  if a.host == "0.0.0.0":
    print("  监听所有网卡：局域网内用 http://<本机IP>:%d/ 访问（sp_vision 侧改 sim.base_url）" % a.port)

  th = threading.Thread(target=sim.run, daemon=True)
  th.start()
  try:
    if a.seconds > 0:
      time.sleep(a.seconds)
      sim.stop = True
      th.join(timeout=3)
    else:
      while th.is_alive():
        th.join(timeout=1.0)
  except KeyboardInterrupt:
    sim.stop = True
  finally:
    srv.shutdown()
  print(f"\n  退出：发射 {sim.demo.shots} 发，命中 {sim.demo.hits} 发")
  return 0


if __name__ == "__main__":
  sys.exit(main())
