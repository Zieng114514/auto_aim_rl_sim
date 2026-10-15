#!/usr/bin/env python3
"""
RoboMaster 云台互动仿真（mjlab 同款技术栈）
================================================================
技术栈与 mjlab 一致：viser（Web 3D 查看器）+ mjviser（mjlab 自带的查看器模块）+ MuJoCo 物理。
场景用 mjlab 同款 MuJoCo XML 描述，可直接搬进 mjlab 做批量 RL 训练（见文件末尾说明）。

交互
----
  鼠标左键【拖拽绿色准星】  → 控制云台 yaw / pitch（连续跟手瞄准）
  鼠标左键【单击场景任意处】→ 发射一颗 17mm 弹丸（沿当前炮管朝向）
  鼠标右键拖拽 / 滚轮       → 视角旋转 / 缩放（viser 默认相机操作）
  左侧面板                  → yaw/pitch 微调、弹丸初速、单发/连发、靶车开关、重置

物理要点（这类仿真真正的坑）
----
  * 17mm 弹丸初速 25 m/s，dt=1ms 时每步前进 2.5cm ≈ 直径的 1.5 倍，
    必须开 CCD（<option ccd_iterations>），否则会瞬间穿过装甲板（隧穿）。
  * 每颗弹丸是带自由关节的刚体，从弹丸池循环取用，不动态增删 body。

运行
----
  # 先装依赖（很轻，不需要 torch）
  pip install mujoco mjviser viser

  python rm_gimbal_demo.py             # 然后浏览器打开 http://localhost:8080
  python rm_gimbal_demo.py --port 0    # 自动挑空闲端口
  python rm_gimbal_demo.py --selftest  # 无界面自检：跑物理+自动开火+打印命中统计
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import math
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ASSETS = os.path.join(_HERE, "assets")   # 转换产物；整目录可随意搬移

import sys
import time
from dataclasses import dataclass, field

import numpy as np
import mujoco

import rm_cars as R_Cars                  # 小陀螺状态机（GyroPlan/GyroParams）
from rm_car_script import CarScript, CarScriptParams, MODES_LEGACY  # 运动基准脚本（固定段长/乱序）
from rm_lift import BodyLift, LiftParams  # 轮腿靶车：车体升降（±150mm/2s，见 configs/targets.yaml）
from rm_gun import GunModel, GunParams   # 枪：弹速/热量/散布（三者各自可开关）

# ----------------------------------------------------------------------------
# 可调参数
# ----------------------------------------------------------------------------
BALL_RADIUS = 0.0085  # 17mm 直径（RM 标准弹丸）
BALL_MASS = 0.0032  # RM 弹丸约 3.2 g
BALL_SPEED = 25.0  # 初速 m/s
BALL_POOL = 40  # 弹丸池大小
BALL_TTL = 4.0  # 弹丸存活秒数，超时回收
PARK = (60.0, 60.0)  # 未使用弹丸的停放点（远处，不影响场景）
# 靶车活动范围 (x_min, x_max, y_min, y_max)：留在云台前方视野里。
# 原来取 (1.1, 3.3, -1.4, 1.4)，但靶车有动量：限加速度 6 m/s²、冲刺 2.5 m/s 时
# 刹车距离约 v²/(2a) ≈ 0.5 m，实际位置会冲出目标框，最近能贴到云台跟前。
# 所以把下限推到 1.7 m，另外再用 CAR_KEEPOUT 做一道硬保护（见 step()）。
CAR_BOX = (1.7, 3.6, -1.6, 1.6)
# 靶车中心到云台 yaw 轴的最小距离（m）。低于这个值就直接把速度改成径向外推，
# 不再靠"目标点选得远"这种软约束 —— 有动量时软约束保证不了不撞。
CAR_KEEPOUT = 1.45
PITCH_MIN, PITCH_MAX = -0.35, 0.45  # 俯仰限位 ≈ -20° ~ +26°
DEFAULT_DT = 0.001  # 1ms（配合 CCD 足以拦住 25 m/s 弹丸）


# ----------------------------------------------------------------------------
# 场景（mjlab 同款 MuJoCo XML）
# ----------------------------------------------------------------------------
def _sticker_png_dir() -> str:
  """把 4 张数字贴纸生成为 PNG（供 MuJoCo <texture type="2d"> 使用）。"""
  d = os.path.join(os.path.expanduser("~"), ".cache", "rm_gimbal_demo")
  os.makedirs(d, exist_ok=True)
  for label in "1234":
    p = os.path.join(d, f"sticker_{label}.png")
    if not os.path.exists(p):
      make_sticker(label).save(p)
  return d


def build_xml(dt: float = DEFAULT_DT, nball: int = BALL_POOL, realistic_armor: bool = True) -> str:
  """realistic_armor=True 时给装甲板加数字贴纸纹理 + 上下发光 LED 灯条（纯视觉，不参与碰撞）。"""
  sticker_dir = _sticker_png_dir() if realistic_armor else ""
  assets = ""
  if realistic_armor:
    assets = "<asset>\n" + "".join(
      f'    <texture name="sticker{i}" type="2d" file="{sticker_dir}/sticker_{i}.png"/>\n'
      f'    <material name="mat_sticker{i}" texture="sticker{i}" specular="0.1" shininess="0.1"/>\n'
      for i in "1234"
    ) + (
      # 注意：MuJoCo 的 emission 是【标量强度】，颜色来自 rgba；自发光在这里表现为"灯条自亮"
      '    <material name="led_red" rgba="1 0.13 0.13 1" emission="1"/>\n'
      '    <material name="led_blue" rgba="0.2 0.45 1 1" emission="1"/>\n'
      "  </asset>\n"
    )
  plates = ""
  for nm, sx, sy, px, py, rgba, label, team in (
    ("front", 0.068, 0.012, 0.135, 0.0, "0.90 0.16 0.16 1", "1", "red"),
    ("back", 0.068, 0.012, -0.135, 0.0, "0.16 0.38 0.90 1", "3", "blue"),
    ("left", 0.012, 0.068, 0.0, 0.135, "0.90 0.16 0.16 1", "2", "red"),
    ("right", 0.012, 0.068, 0.0, -0.135, "0.16 0.38 0.90 1", "4", "blue"),
  ):
    mat = f' material="mat_sticker{label}"' if realistic_armor else ""
    plates += (
      f'<geom name="armor_{nm}" type="box" size="{sx} {sy} 0.042" pos="{px} {py} 0.03" '
      f'rgba="{rgba}"{mat} mass="0.3" condim="3" friction="0.6 0.005 0.0001"/>'
    )
    if realistic_armor:  # 上下两条 LED 灯条（contype=0 → 纯视觉）
      ax = "0.001" if sx > sy else "0.070"   # 沿板长方向
      ay = "0.070" if sx > sy else "0.001"
      for dz in (0.052, -0.052):
        plates += (
          f'<geom name="led_{nm}_{"t" if dz > 0 else "b"}" type="box" '
          f'size="{ax} {ay} 0.0065" pos="{px} {py} {0.03 + dz}" '
          f'material="led_{team}" contype="0" conaffinity="0"/>'
        )
  wheels = "".join(
    f'<geom name="wheel_{nm}" type="cylinder" size="0.058 0.022" euler="90 0 0" '
    f'pos="{x} {y} -0.03" rgba="0.12 0.12 0.12 1" contype="0" conaffinity="0"/>'
    for nm, x, y in (("fl", 0.10, 0.145), ("fr", 0.10, -0.145), ("rl", -0.10, 0.145), ("rr", -0.10, -0.145))
  )
  balls = "".join(
    f'<body name="ball{i}" pos="{PARK[0] + 1.2 * i} {PARK[1]} 0.05">'
    f'<freejoint name="bj{i}"/>'
    f'<geom name="bg{i}" type="sphere" size="{BALL_RADIUS}" mass="{BALL_MASS}" '
    f'rgba="1.0 0.85 0.20 1" condim="3" friction="0.5 0.005 0.0001" '
    f'solref="0.002 1" solimp="0.95 0.99 0.001"/></body>'
    for i in range(nball)
  )
  return f"""<mujoco model="rm_gimbal_demo">
  <option timestep="{dt}" integrator="Euler" ccd_iterations="50"
          iterations="20" ls_iterations="10" gravity="0 0 -9.81"/>
  <size njmax="3000" nconmax="500"/>
{assets}  <visual>
    <headlight diffuse="0.7 0.7 0.7" ambient="0.35 0.35 0.35" specular="0.2 0.2 0.2"/>
    <global azimuth="120" elevation="-18" offwidth="1280" offheight="960"/>
  </visual>
  <worldbody>
    <light pos="0 0 4" dir="0 0 -1" directional="true"/>
    <geom name="floor" type="plane" size="0 0 0.05" rgba="0.34 0.37 0.42 1"
          condim="3" friction="0.8 0.005 0.0001"/>

    <!-- ===== 靶车：浮基底盘 + 4 块装甲板 + 4 个轮子（轮子仅视觉，不参与碰撞）===== -->
    <body name="car" pos="1.6 0 0.075">
      <freejoint name="car_free"/>
      <geom name="chassis" type="box" size="0.14 0.14 0.075" mass="7"
            rgba="0.26 0.26 0.30 1" condim="3" friction="0.8 0.005 0.0001"/>
      {plates}
      {wheels}
    </body>

    <!-- ===== 云台：底座 → yaw(绕 z) → pitch(绕 y) → 炮管 + muzzle site ===== -->
    <body name="gimbal_yoke" pos="0 0 0.15">
      <geom name="yoke" type="cylinder" size="0.05 0.03" rgba="0.20 0.20 0.22 1"
            contype="0" conaffinity="0"/>
      <body name="yaw_link">
        <joint name="g_yaw" type="hinge" axis="0 0 1" damping="0.15" armature="0.02"/>
        <geom name="yaw_body" type="box" size="0.045 0.045 0.035" mass="1.0"
              rgba="0.30 0.30 0.34 1" contype="0" conaffinity="0"/>
        <geom name="yaw_fin" type="box" size="0.02 0.01 0.03" pos="-0.055 0 0"
              rgba="0.30 0.30 0.34 1" contype="0" conaffinity="0"/>
        <body name="pitch_link" pos="0 0 0.05">
          <joint name="g_pitch" type="hinge" axis="0 1 0" damping="0.15" armature="0.01"
                 range="{math.degrees(PITCH_MIN):.1f} {math.degrees(PITCH_MAX):.1f}" limited="true"/>
          <geom name="barrel" type="cylinder" size="0.016 0.10" euler="0 90 0" pos="0.10 0 0"
                mass="0.7" rgba="0.22 0.22 0.25 1" contype="0" conaffinity="0"/>
          <geom name="muzzle_ring" type="cylinder" size="0.021 0.012" euler="0 90 0" pos="0.20 0 0"
                mass="0.05" rgba="0.62 0.62 0.66 1" contype="0" conaffinity="0"/>
          <!-- 弹丸出膛点；site 的局部 +x 即炮管朝向 -->
          <site name="muzzle" pos="0.215 0 0" size="0.01" rgba="1 0.4 0 1"/>
        </body>
      </body>
    </body>

    {balls}
  </worldbody>

  <actuator>
    <position name="yaw_servo" joint="g_yaw" kp="60" kv="6" ctrlrange="-3.1416 3.1416"/>
    <!-- 注意：这个 XML 用 MuJoCo 默认的角度制（没写 compiler angle），
         所以角度类属性都要按"度"给；PITCH_MIN/MAX 在 Python 里是弧度，
         这里必须换算，否则限位会变成 ±0.5° 这种荒唐值。 -->
    <position name="pitch_servo" joint="g_pitch" kp="60" kv="6" ctrlrange="{math.degrees(PITCH_MIN):.1f} {math.degrees(PITCH_MAX):.1f}"/>
  </actuator>
</mujoco>"""


# ----------------------------------------------------------------------------
# 写实装甲板：数字贴纸（2D 纹理）+ 上下发光 LED 灯条（自发光材质）
#
#   注意：mjviser 只把 MuJoCo 的 geom_rgba 转给 viser —— 纹理和 material/emission
#   不会自动透传。所以这里自己造 GLB（trimesh 导出；glTF 原生支持 textures 与
#   emissiveFactor），再挂到车体上，每帧按车体位姿跟随。
#   MuJoCo 自己的渲染器（离屏相机/视觉观测）则原生支持 <texture type="2d"> 与
#   <material emission="标量强度">（注意 emission 是标量，不是 RGB）。
# ----------------------------------------------------------------------------
PLATE_W, PLATE_H = 0.136, 0.084  # 贴纸尺寸
LED_LEN, LED_OFF, LED_H = 0.142, 0.052, 0.013  # 灯条长度 / 距中心高度 / 粗细
# 每块装甲板：贴纸中心的车体局部坐标、朝向（绕 z 弧度）、队伍色
ARMOR_LOCAL = {
  "front": ((0.149, 0.000, 0.030), 0.0, (1.0, 0.15, 0.15), "1"),
  "left": ((0.000, 0.149, 0.030), math.pi / 2, (1.0, 0.15, 0.15), "2"),
  "back": ((-0.149, 0.000, 0.030), math.pi, (0.18, 0.45, 1.0), "3"),
  "right": ((0.000, -0.149, 0.030), -math.pi / 2, (0.18, 0.45, 1.0), "4"),
}


def make_sticker(label: str, size: int = 256):
  """程序化生成"数字贴纸"纹理（白底 + 黑边 + 大号数字），返回 PIL.Image。"""
  from PIL import Image, ImageDraw, ImageFont

  img = Image.new("RGB", (size, size), (232, 232, 232))
  d = ImageDraw.Draw(img)
  d.rectangle([5, 5, size - 6, size - 6], outline=(18, 18, 18), width=9)
  try:
    font = ImageFont.load_default(size=int(size * 0.62))
    bbox = d.textbbox((0, 0), label, font=font)
    d.text(((size - bbox[2] + bbox[0]) / 2, (size - bbox[3] + bbox[1]) / 2), label, fill=(22, 22, 22), font=font)
  except Exception:  # 老版本 Pillow 兜底
    d.rectangle([size * 0.3, size * 0.2, size * 0.7, size * 0.8], fill=(22, 22, 22))
  return img


def _quad_glb(w: float, h: float, texture=None, emissive=None, uv_rect=(0, 0, 1, 1)) -> bytes:
  """造一个朝 +x 的矩形面：带贴图（数字）或自发光（LED 灯条），返回 GLB 字节。"""
  import trimesh
  from trimesh.visual import TextureVisuals
  from trimesh.visual.material import PBRMaterial, SimpleMaterial

  y, z = w / 2, h / 2
  verts = np.array([[0, -y, -z], [0, y, -z], [0, y, z], [0, -y, z]], dtype=np.float64)
  faces = np.array([[0, 1, 2], [0, 2, 3]])
  if texture is not None:
    u0, v0, u1, v1 = uv_rect
    uv = np.array([[u0, v0], [u1, v0], [u1, v1], [u0, v1]], dtype=np.float64)
    vis = TextureVisuals(uv=uv, material=SimpleMaterial(image=texture))
  else:
    color = [int(255 * c) for c in (emissive or (1, 1, 1))] + [255]
    pbr = PBRMaterial(baseColorFactor=color, metallicFactor=0.0, roughnessFactor=0.35)
    pbr.emissiveFactor = [min(1.0, c * 1.15) for c in (emissive or (1, 1, 1))]  # glTF: 0~1
    vis = TextureVisuals(material=pbr)
  mesh = trimesh.Trimesh(vertices=verts, faces=faces, visual=vis, process=False)
  return trimesh.exchange.gltf.export_glb(mesh)


class RealisticArmor:
  """把 4 块装甲板的贴纸与 LED 灯条挂上车体，每帧跟随车体位姿。"""

  def __init__(self, server, demo: "Demo", enabled: bool = True) -> None:
    self.items = []  # [(handle, local_pos, local_quat)]
    for name, (pos, ang, color, label) in ARMOR_LOCAL.items():
      q = np.zeros(4)
      mujoco.mju_axisAngle2Quat(q, np.array([0.0, 0.0, 1.0]), ang)
      # 贴纸（略微向外偏移，避免与物理盒面重叠闪烁）
      off = np.array([0.003, 0.0, 0.0])
      self.items.append(
        (server.scene.add_glb(f"/armor_deco/{name}_sticker", _quad_glb(PLATE_W, PLATE_H, texture=make_sticker(label)),
                              position=(0, 0, 0), visible=enabled),
         np.array(pos) + off, q)
      )
      # 上下两条 LED
      for tag, dz in (("top", LED_OFF), ("bot", -LED_OFF)):
        self.items.append(
          (server.scene.add_glb(f"/armor_deco/{name}_led_{tag}", _quad_glb(LED_LEN, LED_H, emissive=color),
                                position=(0, 0, 0), visible=enabled),
           np.array(pos) + off + np.array([0.0, 0.0, dz]), q)
        )

  def update(self, data: mujoco.MjData, car_bid: int) -> None:
    """按车体位姿计算每个装饰件的世界位姿。"""
    cp = data.xpos[car_bid]
    qc = np.zeros(4)
    mujoco.mju_mat2Quat(qc, np.asarray(data.xmat[car_bid], dtype=np.float64))
    rot = np.asarray(data.xmat[car_bid], dtype=np.float64).reshape(3, 3)
    for handle, lp, lq in self.items:
      wq = np.zeros(4)
      mujoco.mju_mulQuat(wq, qc, lq)
      handle.position = tuple(cp + rot @ lp)
      handle.wxyz = tuple(wq)


# ----------------------------------------------------------------------------
# 真实模型（rm_glb_to_mjcf.py 的产物）+ 弹丸池/地面 → 一个可直接仿真的 MJCF
# ----------------------------------------------------------------------------
def write_merged_atomic(assets_dir: str, xml: str) -> str:
  """把合并后的 MJCF 加锁 + 原子落盘，返回真正用于加载的文件名。

  **所有**对合并 XML 的写入都必须走这里（含 xml_hook 的改写）：RL 训练会同时起
  8 个 worker，每个都要写/读这个文件。之前 xml_hook 分支用 open(...,"w") 原地重写，
  别的进程正好在 from_xml_path 读 → 读到写了一半的文件，报
  "XML parse error 7 ... XMLElement name=joint"。

  文件名带内容指纹：不同配置（不同 --cars / 贴纸 / dt）互不覆盖；相同配置的多进程
  共享同一份文件，内容一致所以谁写都一样。"""
  import fcntl
  digest = hashlib.sha256(xml.encode("utf-8")).hexdigest()[:12]
  lock_path = os.path.join(assets_dir, ".merge.lock")
  path = os.path.join(assets_dir, f"rm_demo_merged_{digest}.xml")
  with open(lock_path, "w") as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    if not os.path.exists(path):                 # 内容相同就不必重写
      tmp = f"{path}.{os.getpid()}.tmp"
      with open(tmp, "w", encoding="utf-8") as f:
        f.write(xml)
      os.replace(tmp, path)
    # 人类查看用的稳定名快照（程序加载一律用带指纹的文件名）
    snap = os.path.join(assets_dir, "rm_demo_merged.xml")
    tmp = f"{snap}.{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
      f.write(xml)
    os.replace(tmp, snap)
    stale = sorted(glob.glob(os.path.join(assets_dir, "rm_demo_merged_*.xml")),
                   key=os.path.getmtime)
    for old in stale[:-6]:                       # 只留最近 6 份，避免越攒越多
      try:
        os.remove(old)
      except OSError:
        pass
  return path


def merge_real_assets(assets_dir: str, dt: float, xml_hook=None) -> str:
  """把转换出的 rm_vehicle_real.xml 与弹丸池、地面合并成完整场景，返回合并后的 XML 路径。

  必须落盘再用 from_xml_path 加载：MJCF 里引用的是 obj/*.obj 相对路径，
  用 from_xml_string 没有基准目录会报 Error opening file。

  xml_hook: 可选的 str->str 钩子；**在锁内**应用后再原子落盘，见 write_merged_atomic。"""
  xml_path = os.path.join(assets_dir, "rm_vehicle_real.xml")
  if not os.path.exists(xml_path):
    raise FileNotFoundError(
      f"找不到 {xml_path}\n请先运行:  python rm_glb_to_mjcf.py")
  xml = open(xml_path, encoding="utf-8").read()
  xml = xml.replace('timestep="0.001"', f'timestep="{dt}"')

  # 地面 + 弹丸池（与简单模型一致，便于命中判定复用）
  balls = "".join(
    f'<body name="ball{i}" pos="{PARK[0] + 1.2 * i} {PARK[1]} 0.05"><freejoint name="bj{i}"/>'
    f'<geom name="bg{i}" type="sphere" size="{BALL_RADIUS}" mass="{BALL_MASS}" '
    f'rgba="1.0 0.85 0.20 1" condim="3" friction="0.5 0.005 0.0001" '
    f'solref="0.002 1" solimp="0.95 0.99 0.001"/></body>'
    for i in range(BALL_POOL)
  )
  floor = ('<geom name="floor" type="plane" size="0 0 0.05" rgba="0.34 0.37 0.42 1" '
           'condim="3" friction="0.8 0.005 0.0001"/>')
  xml = xml.replace("</worldbody>", f"    {floor}\n    {balls}\n  </worldbody>")
  xml = apply_terrain(xml)
  if xml_hook:
    xml = xml_hook(xml)              # 在锁内改 + 原子落盘：多进程同时构建也不能互相踩
  return write_merged_atomic(assets_dir, xml)


# ----------------------------------------------------------------------------
# 地形起伏：高度场（hfield）代替理想平面
# ----------------------------------------------------------------------------
TERRAIN_AMP = 0.08          # 起伏半幅（米）；main() 里可被 --terrain-amp 覆盖
TERRAIN_HALF = 8.0          # 地形半边长（米）
TERRAIN_PNG = os.path.join(os.path.expanduser("~/.cache/rm_gimbal_demo"), "terrain.png")


def make_terrain_heightmap(path: str = TERRAIN_PNG, n: int = 128, seed: int = 7) -> str:
  """生成平滑起伏的高度图：4 组低频正弦叠加。

  特意用正弦而不是随机噪声：128 格跨 16 m、0.6~3 个周期，坡度只有 2~3°，
  车能顺畅开过去；换成白噪声就成了"搓板路"，车会被卡住、弹丸落点也乱跳。
  """
  from PIL import Image
  # **只生成一次 + 原子落盘**：RL 训练会同时起多个进程构建世界，而 MuJoCo 解析 hfield
  # 时要读这个 PNG；原地重写会让别的进程读到写了一半的图（地形乱掉/加载报错）。
  if os.path.exists(path) and os.path.getsize(path) > 0:
    return path
  rng = np.random.default_rng(seed)
  os.makedirs(os.path.dirname(path), exist_ok=True)
  yy, xx = np.mgrid[0:1:complex(n), 0:1:complex(n)]
  h = np.zeros_like(xx)
  for k in range(4):
    fx, fy = rng.uniform(0.6, 3.0, 2)     # 空间频率：最短波长约 5 m，远大于 12.6cm 网格
    px, py = rng.uniform(0.0, 2.0 * math.pi, 2)
    h += np.sin(2 * np.pi * (fx * xx + px)) * np.sin(2 * np.pi * (fy * yy + py)) / (k + 1)
  h = (h - h.min()) / max(h.max() - h.min(), 1e-9)
  # **必须显式给 format**：PIL 是从扩展名推断格式的，写到 "xxx.png.1234.tmp"
  # 会直接报 "unknown file extension: .tmp"（只在"缓存文件还不存在"的全新环境上才暴露，
  # 本地因为早就生成过 terrain.png 一直没踩到 —— 部署到新机器才发现）。
  tmp = f"{path}.{os.getpid()}.tmp"
  Image.fromarray((h * 255).astype(np.uint8), mode="L").save(tmp, format="PNG")
  os.replace(tmp, path)
  return path


def _terrain_parts(amp: float) -> tuple[str, str]:
  """返回 (hfield 资源声明, 地形 geom 串)。amp<=0 时退回平面。"""
  plane = ('<geom name="floor" type="plane" size="0 0 0.05" rgba="0.34 0.37 0.42 1" '
           'condim="3" friction="0.8 0.005 0.0001"/>')
  if amp <= 1e-6:
    return "", plane
  tex = make_terrain_heightmap()
  # size 需要 4 个值：半径 x/y + 高度 + base。base 必须为正（给 0 会报
  # "size parameter is not positive in hfield"），这里给 1 mm。
  asset = (f'    <hfield name="terrain" file="{tex}" '
           f'size="{TERRAIN_HALF} {TERRAIN_HALF} {2 * amp:.4f} 0.001"/>')
  geom = (f'<geom name="floor" type="hfield" hfield="terrain" pos="0 0 {-amp:.4f}" '
          f'rgba="0.34 0.37 0.42 1" condim="3" friction="0.8 0.005 0.0001"/>\n'
          f'    <!-- 远处兜底平面：停在 (60,60) 的弹丸不至于掉出世界 -->\n'
          f'    <geom name="floor_far" type="plane" size="0 0 0.05" '
          f'pos="0 0 {-amp - 0.02:.4f}" rgba="0.34 0.37 0.42 1" '
          f'condim="3" friction="0.8 0.005 0.0001"/>')
  return asset, geom


class TerrainField:
  """高度场的查询接口：给定 (x, y) 返回地形高度与法向。

  姿态伺服需要"车底下那块地的倾角"，直接从 hfield 数据双线性插值求梯度即可，
  不用打射线（射线会被车体自己挡住）。
  """

  def __init__(self, model) -> None:
    self.ok = model.nhfield > 0
    self.z0 = 0.0
    if not self.ok:
      return
    self.nrow = int(model.hfield_nrow[0])
    self.ncol = int(model.hfield_ncol[0])
    self.size = np.array(model.hfield_size[0], dtype=float)      # sx sy sz base
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    self.z0 = float(model.geom_pos[gid][2]) + float(self.size[3])
    self.h = np.array(model.hfield_data, dtype=float).reshape(self.nrow, self.ncol) * self.size[2]
    # ---- 热路径加速（纯 Python float，**表达式一字不改**）------------------------
    # height()/normal() 每个物理步被调用 5 次（0-D numpy 标量运算每次 ~1µs 起），
    # 训练时占单步开销可观比例。这里把参数与高度场转成 float/list：
    # 每一步仍是 IEEE-754 double 的同序运算 ⇒ 结果与用 np.float64 **逐位相同**。
    # 验证：auto_aim_rl/scratch/golden_trace.py（改前/改后 qpos hash 必须一致）。
    self._sx = float(self.size[0])
    self._sy = float(self.size[1])
    self._col1 = float(self.ncol - 1)
    self._row1 = float(self.nrow - 1)
    self._z0f = float(self.z0)
    self._hl = [[float(v) for v in row] for row in self.h]

  def _rc(self, x: float, y: float):
    """世界 (x,y) → 高度场浮点行列号；越界返回 None。"""
    sx, sy = self._sx, self._sy
    if abs(x) > sx or abs(y) > sy:
      return None
    u = (x + sx) / (2 * sx) * self._col1            # 列：x 方向
    v = (y + sy) / (2 * sy) * self._row1            # 行：y 方向
    return v, u

  def height(self, x: float, y: float) -> float:
    if not self.ok:
      return self.z0 if hasattr(self, "z0") else 0.0
    rc = self._rc(float(x), float(y))
    if rc is None:
      return self._z0f
    v, u = rc
    i0, j0 = math.floor(v), math.floor(u)
    i1, j1 = min(i0 + 1, self.nrow - 1), min(j0 + 1, self.ncol - 1)
    tv, tu = v - i0, u - j0
    hl = self._hl
    h = ((1 - tv) * ((1 - tu) * hl[i0][j0] + tu * hl[i0][j1])
         + tv * ((1 - tu) * hl[i1][j0] + tu * hl[i1][j1]))
    return self._z0f + h

  def normal(self, x: float, y: float) -> np.ndarray:
    """当地地形法向（世界系单位向量，z 分量为正）。"""
    if not self.ok:
      return np.array([0.0, 0.0, 1.0])
    d = 0.35                                        # 差分步长（米）
    hx = (self.height(x + d, y) - self.height(x - d, y)) / (2 * d)
    hy = (self.height(x, y + d) - self.height(x, y - d)) / (2 * d)
    n = np.array([-hx, -hy, 1.0])
    return n / max(float(np.linalg.norm(n)), 1e-9)


def apply_terrain(xml: str, amp: float | None = None) -> str:
  """把 XML 里的平面地板换成高度场地形（两条模型路径共用）。"""
  if amp is None:
    amp = TERRAIN_AMP
  asset, geom = _terrain_parts(amp)
  if amp <= 1e-6 or 'name="floor"' not in xml:
    return xml
  if "<asset>" in xml:                      # MJCF 只能有一个 asset 块，合并进去
    xml = xml.replace("<asset>", "<asset>\n" + asset, 1)
  else:
    xml = xml.replace("<worldbody>",
                      "<asset>\n" + asset + "\n  </asset>\n  <worldbody>", 1)
  i = xml.index('name="floor"')
  k = xml.rindex("<geom", 0, i)
  j = xml.index("/>", i) + 2
  return xml[:k] + geom + xml[j:]


# ----------------------------------------------------------------------------
# 共享控制指令：viser 事件回调（其他线程）只写这里，物理线程读
# ----------------------------------------------------------------------------
@dataclass
class Cmd:
  yaw: float = 0.0
  pitch: float = 0.10
  aim_point: np.ndarray | None = None  # 拖拽准星后指向的世界坐标点
  fire: bool = False  # 单发请求，消费后清零
  auto: bool = False
  auto_hz: float = 6.0
  car_move: bool = True
  # 移动靶：绕自身 yaw 轴持续自转，转速按正弦变化。
  # base > amp 保证"一直朝同一方向转"（不会来回摆）。
  car_tilt: bool = True           # 车体姿态跟随地形（False = 永远水平，便于对照）
  car_spin: bool = True
  car_spin_base: float = 4.32     # rad/s，自转基准角速度
  car_spin_amp: float = 2.70      # rad/s，正弦调制幅值（ω ∈ [1.62, 7.02]）
  car_spin_hz: float = 0.8333     # Hz，固定周期模式用（1.2 s）
  car_spin_random: bool = True    # 每个正弦周期随机取周期，让转速不可预测
  car_spin_period_min: float = 1.2
  car_spin_period_max: float = 2.0
  # 平移：随机"急停 + 冲刺"，让瞄准点不断丢失
  car_dash: bool = True
  # 冲刺速度（2026-10-13 按负责人要求整体 +40%：真机靶车更快、更难预判）
  car_dash_speed_min: float = 2.55  # m/s 冲刺速度下限（原 1.82）
  car_dash_speed_max: float = 3.53  # m/s 冲刺速度上限（原 2.52）
  car_accel: float = 6.0            # m/s² 加速度/减速度限幅（起步与急刹都有过程）
  car_turn_rate: float = 3.8        # rad/s 转向速率限幅（≈218°/s，不能瞬间掉头）
  car_spin_shape: float = 0.45      # 自转波形指数 q：1=纯正弦，越小低谷越短、回升越陡
  car_dash_time: tuple = (0.45, 1.2)   # s 冲刺时长范围
  car_halt_time: tuple = (0.30, 0.9)   # s 急停时长范围
  # ---- 靶车运动基准脚本（固定 10 s 一段、四种靶型乱序）----
  car_script_dwell: float = 10.0   # 每段时长 s
  car_script_shuffle: bool = True  # 每轮乱序（固定种子 -> 可复现）
  # 参与基准的靶型集合。默认仍是**原来那 4 种**（既有基线可比）；
  # 含轮腿的 6 段集合见 configs/targets.yaml 的 presets.benchmark_2026_wl
  car_script_modes: tuple = MODES_LEGACY
  # ---- 小陀螺机动（阶段 0-C）----
  # classic = 原来的巡逻+连续自转（零行为变化）；gyro = 巡逻与「静止快速甩头」交替。
  car_motion: str = "gyro"
  gyro_patrol_min: float = 2.0
  gyro_patrol_max: float = 5.0
  gyro_dwell_min: float = 1.0
  gyro_dwell_max: float = 3.0
  gyro_spin_base: float = 7.0    # 甩头角速度基准 rad/s
  gyro_spin_amp: float = 5.0     # 正弦调制幅值 -> ω ∈ [2, 12]
  gyro_period_min: float = 0.4   # 正弦周期（比巡逻自转 1.2~2.0 更小）
  gyro_period_max: float = 0.8
  gyro_shape: float = 0.45
  # 巡逻结束后进入小陀螺的概率。**不能太高**：静止甩头在本仿真里反而更好打
  # （平移才是主要脱靶源），但训练开火策略又需要一定比例的原地旋转靶，
  # 所以保留但降低出现概率（负责人 2026-10-13：可以减少固定旋转靶出现的概率）。
  gyro_enter_prob: float = 0.35
  # ---- 轮腿靶车：车体升降（负责人 2026-10-14）----
  # 真机交战时轮腿会一边升降一边变速小陀螺，四块装甲板在 z 轴上上下动。
  # 参数在 configs/targets.yaml 的 infantry_wheel_legged.body_lift 里，这里只是运行时载体。
  # 默认必须是 off —— 保证"不配置就是老行为"（黄金轨迹回归要求逐位一致）
  car_lift: LiftParams = field(default_factory=lambda: LiftParams(mode="off"))
  ball_speed: float = BALL_SPEED


class Demo:
  """单世界：云台瞄准并发射 17mm 弹丸击打随机移动的靶车。"""

  def __init__(self, dt: float = DEFAULT_DT, realistic_armor: bool = True,
               real_assets: str | None = None, xml_hook=None, gun_params=None,
               ballistics_params=None) -> None:
    """xml_hook: 可选的 str->str 钩子，在加载前改 MJCF（例如给云台挂取流相机）。

    gun_params: rm_gun.GunParams；给 None 表示保持旧行为（理想枪：恒速、无热量、无散布）。
    ballistics_params: rm_ballistics.BallisticsParams；给 None 或 enabled=False
      表示弹道只有重力（历史行为，逐位不变）。
    """
    self.ballistics = ballistics_params
    # 枪：摩擦轮弹速 + 枪口热量 + 弹丸散布（三个模型各自可开关）
    self.gun = GunModel(gun_params) if gun_params is not None else None
    self.last_shot_speed = float(self.gun.nominal_speed) if self.gun else float(BALL_SPEED)
    # 小陀螺状态机 / 运动基准脚本：都懒构建（rm_link 会在构造之后再覆盖 Cmd 参数）
    self._gyro = None
    self._script = None
    self._lift = None          # 轮腿升降状态机（懒构建，模式变了就重建）
    # 击杀数由外部（CarField）提供；没接上就按 0 算（分段统计里就只看发射/命中）
    self._deaths_fn = None
    self.real = real_assets is not None
    if self.real:
      # 返回路径：obj 是相对路径，必须落盘再加载。钩子在锁内生效（见 merge_real_assets）。
      merged = merge_real_assets(real_assets, dt, xml_hook)
      self.model = mujoco.MjModel.from_xml_path(merged)
    else:
      xml = apply_terrain(build_xml(dt, realistic_armor=realistic_armor))
      self.model = mujoco.MjModel.from_xml_string(xml_hook(xml) if xml_hook else xml)
    self.data = mujoco.MjData(self.model)
    m = self.model

    def jid(n: str) -> int:
      return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)

    self.yaw_jid, self.pitch_jid, self.car_jid = jid("g_yaw"), jid("g_pitch"), jid("chassis_free" if self.real else "car_free")
    self.muzzle_sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "muzzle")
    # 简单模型 / 真实模型（vehicle.glb 转换产物）用不同的 body 名
    self.car_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "chassis" if self.real else "car")
    self.yoke_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "gimbal_yaw" if self.real else "gimbal_yoke")
    self.car_dadr = m.jnt_dofadr[self.car_jid]
    self.car_qadr = int(m.jnt_qposadr[self.car_jid])
    # pitch 关节的真实限位（真实模型因炮管自带安装角，限位也整体平移过）
    if bool(m.jnt_limited[self.pitch_jid]):
      self.pitch_lo, self.pitch_hi = (float(m.jnt_range[self.pitch_jid][0]),
                                      float(m.jnt_range[self.pitch_jid][1]))
    else:
      self.pitch_lo, self.pitch_hi = PITCH_MIN, PITCH_MAX
    # 炮口相对 yaw 体的高度（零俯仰时）：瞄准解算要用它，不能硬编码
    _pb = int(m.site_bodyid[self.muzzle_sid])          # muzzle 所在的 pitch 体
    self.muzzle_up = float(m.site_pos[self.muzzle_sid][2] + m.body_pos[_pb][2])
    self.pitch_dz = float(m.body_pos[_pb][2])                 # 俯仰轴相对 yaw 体的高度
    self.muzzle_local = np.array(m.site_pos[self.muzzle_sid], dtype=float).copy()
    self.yaw_qadr = int(m.jnt_qposadr[self.yaw_jid])
    self.pitch_qadr = int(m.jnt_qposadr[self.pitch_jid])

    if self.real:   # 转换器生成的 4 块装甲板几何
      self.armor_gid2name = {
        mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"armor{i}"): f"板{i}"
        for i in range(4)
      }
    else:
      self.armor_gid2name = {
        mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"armor_{n}"): n
        for n in ("front", "back", "left", "right")
      }
    # 瞄准点要落在装甲板中心：不同模型车体高度不同，硬编码 +0.05 会让弹丸
    # 从板子上方擦过去（实测简单模型就差 1.4 cm）。
    mujoco.mj_forward(m, self.data)
    if self.armor_gid2name:
      self.armor_dz = float(np.mean([self.data.geom_xpos[g][2]
                                     for g in self.armor_gid2name])) - float(self.data.xpos[self.car_bid][2])
    else:
      self.armor_dz = 0.0

    self.ball_bids = set()
    self.ball_bids_sorted = []
    self.ball_qadr, self.ball_dadr = [], []
    for i in range(BALL_POOL):
      self.ball_bids_sorted.append(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"ball{i}"))
      self.ball_qadr.append(m.jnt_qposadr[jid(f"bj{i}")])
      self.ball_dadr.append(m.jnt_dofadr[jid(f"bj{i}")])
    self.geom_body = np.asarray(m.geom_bodyid)

    self.cmd = Cmd()
    self.gravity_comp = True  # 弹道重力补偿
    self.ball_t = [0.0] * BALL_POOL
    self.next_ball = 0
    self.body2ball = {b: i for i, b in enumerate(self.ball_bids_sorted)}
    # 弹丸 geom 集合：命中判定要遍历**所有**接触点（场上 40 颗停放的弹丸一直压在地面上，
    # 每步几十个接触），先在集合里过一遍比"每个接触都查 geom→body→ball 两张表"快得多。
    self.ball_geoms = {int(g) for g, b in enumerate(self.geom_body) if int(b) in self.body2ball}
    self.contacting: set = set()   # 上一步仍在接触的 (弹丸, 装甲板) 对
    self.ball_scored: set = set()  # 已经计过命中的弹丸序号
    self.shots = 0
    self.hits = 0
    self.hits_by_plate: dict[str, int] = {}
    # 逐发记录：调瞄具参数时，命中率是 0/1 的稀疏信号，噪声极大（同参数重复跑
    # 能差一倍）。这里记录每一发**弹道离板心的最近距离**和那一刻在板局部系下的
    # 竖直/水平偏移 —— 连续量，几十发就能看出系统性偏差，扫描参数快得多。
    self.shot_log: list[dict] = []
    self._ball_track: dict[int, dict] = {}
    # 逐发弹道追踪每步要为在飞的弹丸算到各装甲板的距离。实测开销约 1.5%
    # （22.9/23.1 开 vs 23.5/23.2 关，落在噪声里），所以默认开着。
    # 不需要标定数据时可以用 RM_SIM_SHOT_LOG=0 关掉。
    self.track_shots_env = os.environ.get("RM_SIM_SHOT_LOG", "1")
    self.track_shots = self.track_shots_env != "0"
    # 板局部哪个轴是"宽"哪个是"高"：转换器给前后板的是 [宽,厚,高]、给左右板的是
    # [厚,宽,高]，不统一，所以按尺寸反推（和 gt_points 里同一套规则）。
    self.armor_axes: dict[int, tuple[int, int]] = {}
    for gid in self.armor_gid2name:
      half = np.asarray(m.geom_size[gid], dtype=float)
      i_thick = int(np.argmin(half))
      i_width = int(np.argmax(half))
      i_height = 3 - i_thick - i_width
      self.armor_axes[gid] = (i_width, i_height)
    self.markers: list[dict] = []  # {"pos": ndarray, "ttl": float, "handle": handle|None}
    self.reset_request = False  # viser 线程置位，物理线程消费
    self._car_vel = np.zeros(3)
    self._car_phase = 0.0
    self._last_fire = -1e9
    self._sim_time = 0.0
    # 靶车运动状态：相位累加器 + 急停/冲刺状态机 + 独立随机源
    self._rng = np.random.default_rng()
    self._spin_phase = 0.0
    self._spin_period = 0.8333
    self._car_env = 1.0          # 速度包络 0=停 1=全速
    self._car_dash_state = "dash"
    self._car_timer = 0.0
    self._car_speed = 2.6          # 本次冲刺的目标速度
    self._car_dir = np.array([1.0, 0.0])   # 当前行进方向（受转向速率限幅）
    self._car_v = 0.0              # 当前速度大小（受加速度限幅）
    self._car_omega = 0.0          # 当前自转角速度（物理步后重新断言用）
    self.terrain = TerrainField(self.model)
    self._car_tilt_w = (0.0, 0.0)  # 姿态伺服输出的 roll/pitch 角速度（步后重新断言用）
    self._ride_z = self._measure_ride_z()   # 车体原点离地高度（按几何最低点算，见下）
    self._tilt_kp = 10.0           # 姿态伺服增益（≈0.1 s 时间常数）
    self._car_target = np.array([2.2, 0.0])
    self.reset()
    # 炮管安装仰角：静止位 site 的 +x 轴的实际仰角（真实模型自带 +25°，简单模型为 0）。
    # 瞄准解算算的是"世界仰角"，而 qpos 是相对安装位的，必须补这个零点。
    _d0 = self.data.site_xmat[self.muzzle_sid].reshape(3, 3)[:, 0]
    self.mount_tilt = float(math.asin(max(-1.0, min(1.0, float(_d0[2])))))
    if abs(self.mount_tilt) > 1e-3:
      print(f"  炮管安装仰角 {math.degrees(self.mount_tilt):+.1f}°，已在瞄准解算中补偿")

  # ------------------------------------------------------------------ 重置
  def reset(self) -> None:
    m, d = self.model, self.data
    mujoco.mj_resetData(m, d)
    for i in range(BALL_POOL):
      qa = self.ball_qadr[i]
      d.qpos[qa : qa + 3] = (PARK[0] + 1.2 * i, PARK[1], 0.05)
      d.qpos[qa + 3 : qa + 7] = (1.0, 0.0, 0.0, 0.0)
    d.qpos[m.jnt_qposadr[self.car_jid] : m.jnt_qposadr[self.car_jid] + 3] = (1.6, 0.0, 0.02 if self.real else 0.075)
    d.qpos[m.jnt_qposadr[self.car_jid] + 3 : m.jnt_qposadr[self.car_jid] + 7] = (1.0, 0.0, 0.0, 0.0)
    d.ctrl[:] = (self.cmd.yaw, self.cmd.pitch)
    mujoco.mj_forward(m, d)
    self.ball_t = [0.0] * BALL_POOL
    self.next_ball = 0
    self.contacting = set()
    self.ball_scored = set()
    self.shots = self.hits = 0
    self.hits_by_plate = {}
    self._car_vel = np.zeros(3)
    self._car_phase = 0.0
    self._last_fire = -1e9
    self.reset_request = False

  # ------------------------------------------------------------------ 瞄准
  def muzzle(self) -> tuple[np.ndarray, np.ndarray]:
    """弹丸出膛点 + 炮管朝向（世界坐标）。"""
    p = self.data.site_xpos[self.muzzle_sid].copy()
    rot = self.data.site_xmat[self.muzzle_sid].reshape(3, 3)
    return p, rot[:, 0].copy()  # 局部 +x

  def _measure_ride_z(self) -> float:
    """车体原点贴地时离地面的高度 = 车体几何在机体系里的最低点。

    不能拿"第一步时车的位置减地形高度"来当偏移——车初始 z 是 0.02（还没落地），
    那个悬空量会被永久固化，车就一直飘在地面上（截图里就是这样）。
    """
    m = self.model
    lows = []
    for g in range(m.ngeom):
      if int(m.geom_bodyid[g]) != self.car_bid:
        continue
      gt = m.geom_type[g]
      if gt == mujoco.mjtGeom.mjGEOM_BOX:
        lows.append(float(m.geom_pos[g][2] - m.geom_size[g][2]))
      elif gt == mujoco.mjtGeom.mjGEOM_MESH:
        mid = int(m.geom_dataid[g])
        a, n = int(m.mesh_vertadr[mid]), int(m.mesh_vertnum[mid])
        V = np.array(m.mesh_vert[a:a + n], dtype=float)
        Rg = np.zeros(9)
        mujoco.mju_quat2Mat(Rg, m.geom_quat[g])
        lows.append(float(m.geom_pos[g][2] + (V @ Rg.reshape(3, 3).T)[:, 2].min()))
    return -min(lows) - 0.004 if lows else 0.0      # 再压 4 mm，消除斜坡上的视觉缝隙

  def aim_to_point(self, pt, gravity_comp: bool | None = None) -> tuple[float, float]:
    """世界坐标点 → (yaw, pitch) 目标角。

    两件必须补偿的事：
      ① 重力下坠（否则瞄中心会打低）；
      ② 炮口随俯仰摆动——炮口离俯仰轴 0.226 m，转 30° 高度就差 0.11 m。
         把炮口高度当常数会有约 3° 的系统偏差（2.4 m 处偏 12 cm，直接打飞），
         所以这里对俯仰角做定点迭代。
    """
    if gravity_comp is None:
      gravity_comp = self.gravity_comp
    base = self.data.xpos[self.yoke_bid]
    dx, dy = float(pt[0] - base[0]), float(pt[1] - base[1])
    yaw = math.atan2(dy, dx)
    horiz = max(math.hypot(dx, dy), 1e-6)
    tz = float(pt[2])
    # 俯仰轴在 yaw 体上的高度 + 炮口在俯仰系里的偏置
    pz = float(base[2]) + self.pitch_dz
    mx, mz = float(self.muzzle_local[0]), float(self.muzzle_local[2])
    pitch = -math.atan2(tz - pz - mz, horiz)          # 初值：把炮口当固定点
    speed = max(self.cmd.ball_speed, 1e-6)
    for _ in range(6):
      q = pitch + self.mount_tilt                     # 关节角 = 世界仰角 + 安装角
      cq, sq = math.cos(q), math.sin(q)
      muzzle_z = pz + (-sq * mx + cq * mz)            # R_y(q) 作用于炮口偏置的 z 分量
      drop = 0.0
      if gravity_comp and self.cmd.ball_speed > 2.0:
        t = horiz / speed
        drop = 0.5 * 9.81 * t * t
        t_prev = t
        for _ in range(2):
          t = horiz / (speed * max(math.cos(pitch), 1e-3))
          if abs(t - t_prev) < 1e-4:
            break
          t_prev = t
          drop = 0.5 * 9.81 * t * t
      pitch = -math.atan2(tz - muzzle_z + drop, horiz)
    # +mount_tilt：把"世界仰角"换算成相对安装位的关节角
    return yaw, float(np.clip(pitch + self.mount_tilt, self.pitch_lo, self.pitch_hi))

  # ------------------------------------------------------------------ 发射
  def fire(self, speed: float | None = None) -> bool:
    """打一发。返回是否真的打出去了（枪口热量超限会锁枪）。

    speed 显式给值时走"理想枪"路径（自检/标定脚本用），忽略弹速模型与散布。
    """
    if self.gun is not None and speed is None and not self.gun.can_fire():
      self.gun.shots_blocked += 1
      return False
    i = self.next_ball
    self.next_ball = (i + 1) % BALL_POOL
    p, dirv = self.muzzle()
    if speed is None and self.gun is not None:
      # 实际出膛速度来自摩擦轮模型；方向叠加枪口散布
      speed = self.gun.take_shot()[0]
      dirv = self.gun.perturb_dir(dirv)
      self.last_shot_speed = float(speed)
    qa, da = self.ball_qadr[i], self.ball_dadr[i]
    self.data.qpos[qa : qa + 3] = p
    self.data.qpos[qa + 3 : qa + 7] = (1.0, 0.0, 0.0, 0.0)
    self.data.qvel[da : da + 3] = dirv * (self.cmd.ball_speed if speed is None else speed)
    # 自旋（摩擦轮夹着甩出去 ⇒ 出膛必然带 ω）：写进弹丸的**真实角速度**，
    # 之后马格努斯力由 rm_ballistics 从 ω 读出来算（方向 = ω̂ × v̂）。
    # 关闭自旋时写 0 = 历史行为，逐位不变。
    self.data.qvel[da + 3 : da + 6] = 0.0
    if (self.ballistics is not None and self.ballistics.spin.enabled
        and self.ballistics.spin.omega_rad_s != 0.0):
      sp = self.ballistics.spin
      # 炮口 site 的局部轴（列向量）：x = 炮管朝向、y = 侧向、z = 炮管的"上"
      R_ms = self.data.site_xmat[self.muzzle_sid].reshape(3, 3)
      axis = R_ms[:, 1] if sp.axis == "barrel_side" else R_ms[:, 2]
      self.data.qvel[da + 3 : da + 6] = axis * sp.omega_rad_s
    mujoco.mj_forward(self.model, self.data)
    self.ball_t[i] = BALL_TTL
    self.shots += 1
    self._last_fire = self._sim_time
    # 记下这一发：出膛时的指令角 + 整条弹道离板心的最近距离
    self._ball_track[i] = {
      "id": self.shots, "t": float(self._sim_time), "hit": False,
      "cmd_yaw": float(self.cmd.yaw), "cmd_pitch": float(self.cmd.pitch),
      "min_dist": float("inf"), "plate": None,
      "off_v": 0.0, "off_h": 0.0, "off_n": 0.0, "range": 0.0, "stopped": False,
      "speed": float(speed) if speed is not None else float(self.cmd.ball_speed),
      # 当时是哪种靶型（基准脚本模式下）—— 用来解释"某类靶型为什么打不中"
      "motion": (self._script.mode if self._script is not None else "-"),
    }
    return True

  # ------------------------------------------------------------------ 每步
  def step(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
    if self.reset_request:
      self.reset()
      return
    self._sim_time += model.opt.timestep
    if self.gun is not None:
      self.gun.step(model.opt.timestep)   # 枪口冷却 / 超限扣血 / 摩擦轮降温
    c = self.cmd

    # 0) 靶型：car_motion=script 时按基准脚本走（固定时长、乱序），否则用原来的随机过程
    frame_dt = model.opt.timestep
    motion = getattr(c, "car_motion", "classic")
    spec = None
    if c.car_move and motion == "script":
      if self._script is None:
        self._script = CarScript(
          CarScriptParams(dwell=float(getattr(c, "car_script_dwell", 10.0)),
                          modes=tuple(getattr(c, "car_script_modes", MODES_LEGACY)),
                          shuffle=bool(getattr(c, "car_script_shuffle", True))),
          self._rng)
      deaths = int(self._deaths_fn()) if self._deaths_fn else 0
      spec = self._script.step(frame_dt, (self.shots, self.hits, deaths))

    # 小陀螺：car_motion=gyro 用自由状态机；script 模式里 inplace_spin 段用"全程保持"
    gyro_active = False
    need_hold = bool(spec is not None and spec.spin == "gyro")
    if c.car_move and (motion == "gyro" or need_hold):
      if self._gyro is None or self._gyro.hold != need_hold:
        self._gyro = R_Cars.GyroPlan(
          R_Cars.GyroParams.from_cmd(c), self._rng, hold=need_hold)
      gyro_active = self._gyro.spinning
    elif self._gyro is not None and self._gyro.hold:
      # 离开「全程甩头」段：**必须把 hold 放掉**。不然它会一直生效 —— 实测基准脚本里
      # 第 2 段 inplace_spin 之后，后面每一段的 car_motion_mode 都还是 gyro，
      # 分靶型统计被整体污染（固定/随机旋转移动靶其实都在按小陀螺甩）。
      self._gyro.release()

    # 轮腿升降：靶型（基准脚本）决定模式；与自转**同时**发生，物理上就是车体高度在变。
    # 优先级：**命令行显式给的 --car-lift > 靶型自带的 lift > 默认 off**。
    # 反过来的话会很难查：跑了 --car-motion script（基准脚本）时，老 4 段靶型的 spec.lift 都是 off，
    # 于是 --car-lift periodic 被静默吃掉、靶车根本不动 —— 表现为"参数加了没效果"。
    lift_mode = c.car_lift.mode if c.car_lift.mode != "off" else (
      spec.lift if spec is not None else "off")
    if self._lift is None or self._lift.p.mode != lift_mode:
      self._lift = BodyLift(
        LiftParams(mode=lift_mode, amplitude_mm=c.car_lift.amplitude_mm,
                   period_s=c.car_lift.period_s,
                   accel_limit_mps2=c.car_lift.accel_limit_mps2,
                   vel_limit_mps=c.car_lift.vel_limit_mps,
                   rand_hold_s=c.car_lift.rand_hold_s,
                   bottom_mm=c.car_lift.bottom_mm,          # 默认 0 = 不往地下走（防嵌地）
                   active_cycles=c.car_lift.active_cycles,  # 占空比：活动窗口几个周期
                   pause_s=c.car_lift.pause_s),             # 占空比：窗口间静止时长
        self._rng)

    # 1) 靶车"随机移动"：平滑随机游走的速度伺服（仍然受物理碰撞约束）
    if c.car_move:
      self._car_phase += model.opt.timestep
      t = self._car_phase
      # 目标点：绕 (2.2, 0) 附近做有界随机游走；速度用位置反馈追踪，避免跑出场地
      tx = 2.2 + 1.05 * math.sin(0.45 * t) + 0.40 * math.sin(1.10 * t + 0.7)
      ty = 0.0 + 1.25 * math.sin(0.31 * t + 0.4) + 0.45 * math.sin(0.83 * t)
      pos = data.xpos[self.car_bid]
      dt = model.opt.timestep
      # 基准脚本模式：平移方式由当前靶型决定（patrol=连续巡逻 / dashhalt=急停冲刺 / stop=停住）
      trans = spec.translation if spec is not None else ("dashhalt" if c.car_dash else "patrol")
      if trans == "patrol":
        goal = 1.0
        self._car_speed = 3.08        # 巡航速度（与冲刺上限同步 +40%）
        self._car_dash_state = "dash" # 复位，切回急停冲刺模式时从冲刺开始
      elif trans == "stop":
        goal = 0.0
        self._car_speed = 0.0
      elif c.car_dash:
        # 状态机：冲刺一段时间 → 急停一段时间 → 再来，时长与冲刺速度都随机
        self._car_timer -= dt
        if self._car_timer <= 0.0:
          if self._car_dash_state == "dash":
            self._car_dash_state = "halt"
            self._car_timer = float(self._rng.uniform(*c.car_halt_time))
          else:
            self._car_dash_state = "dash"
            self._car_timer = float(self._rng.uniform(*c.car_dash_time))
            self._car_speed = float(self._rng.uniform(c.car_dash_speed_min,
                                                     c.car_dash_speed_max))
            # 每次冲刺重新抽一个目标点：方向瞬间改变，云台很难预判。
            # 抽点时就把"离云台太近"的位置剔掉，省得一直往云台身上撞。
            yoke_xy = np.asarray(data.xpos[self.yoke_bid][:2], dtype=float)
            for _ in range(24):
              cand = np.array([
                self._rng.uniform(CAR_BOX[0], CAR_BOX[1]),
                self._rng.uniform(CAR_BOX[2], CAR_BOX[3]),
              ])
              if float(np.linalg.norm(cand - yoke_xy)) >= CAR_KEEPOUT:
                break
            self._car_target = cand
        goal = 1.0 if self._car_dash_state == "dash" else 0.0
      else:
        # trans == "dashhalt" 但 car_dash 被关掉：退化成连续巡逻
        goal = 1.0
        self._car_speed = 3.08   # 非冲刺模式的巡航速度（与冲刺速度同步 +40%）
      if gyro_active:
        # 小陀螺期间按 move_frac 缩放线速度：默认 0 = 车体**停住**（真机"静止甩头"），
        # 加速度限幅让它自然刹停；设成 1 就是"边走边甩头"（更难的一档）。
        # 基准脚本里的 inplace_spin 段用 stop 已经停住了，这里再乘一次也无妨。
        frac = self._gyro.p.move_frac
        goal *= frac
        self._car_speed *= frac
      aim_xy = self._car_target if c.car_dash else np.array([tx, ty])
      d_xy = np.asarray(aim_xy, dtype=float) - pos[:2]
      nd = float(np.linalg.norm(d_xy))
      # ① 方向按有限转向速率转过去（掉头需要时间，不会瞬间反向）
      if nd > 1e-6:
        want_ang = math.atan2(float(d_xy[1]), float(d_xy[0]))
        cur_ang = math.atan2(float(self._car_dir[1]), float(self._car_dir[0]))
        d_ang = (want_ang - cur_ang + math.pi) % (2.0 * math.pi) - math.pi
        step = max(-c.car_turn_rate * dt, min(c.car_turn_rate * dt, d_ang))
        cur_ang += step
        self._car_dir = np.array([math.cos(cur_ang), math.sin(cur_ang)])
      # ② 速度大小按有限加速度逼近目标（起步/刹车都有过程）
      v_des = goal * self._car_speed
      if nd < 0.35:                     # 快到目标点自然减速
        v_des *= max(nd / 0.35, 0.0)
      dv = max(-c.car_accel * dt, min(c.car_accel * dt, v_des - self._car_v))
      self._car_v += dv
      self._car_vel[:2] = self._car_dir * self._car_v
      # 硬防撞：一旦进到 keepout 圈内，直接覆盖成"沿径向向外"的速度。
      # 位置反馈那个软约束在有动量时会失效（刹车要距离），这里必须硬来。
      to_g = np.asarray(pos[:2], dtype=float) - np.asarray(data.xpos[self.yoke_bid][:2], dtype=float)
      dg = float(np.linalg.norm(to_g))
      if dg < CAR_KEEPOUT:
        out = to_g / dg if dg > 1e-6 else np.array([1.0, 0.0])
        self._car_vel[:2] = out * max(c.car_dash_speed_min, 1.2)
        self._car_dir = out
        self._car_v = float(np.linalg.norm(self._car_vel[:2]))
        # 同时把下一个目标点也挪到圈外，避免刚推出去又被吸回来
        self._car_target = np.asarray(data.xpos[self.yoke_bid][:2]) + out * (CAR_KEEPOUT + 0.8)
      data.qvel[self.car_dadr : self.car_dadr + 3] = self._car_vel
    else:
      data.qvel[self.car_dadr : self.car_dadr + 3] = 0.0

    # 靶车自转：真实移动靶是绕自身 yaw 轴不停转，转速按正弦变化。
    # 自由关节的角速度在**机体局部系**里，车子保持直立时局部 z 就是竖直轴，
    # 所以直接写 qvel[5]；同时清零另外两个分量，保证它不会翻倒。
    if c.car_spin:
      # 自转方式：mod = 随机周期调制；const = 恒定转速（可预测，基准里的"固定旋转靶"）
      spin_mode = spec.spin if spec is not None else ("mod" if c.car_spin_random else "const")
      if spin_mode == "none":
        # 「固定靶 / 直线移动靶」：完全不转。仍然走同一个出口（下面还要过小陀螺的
        # 角加速度限幅），所以从"正在转"的上一段切过来时是**平滑刹停**，不是一步跳成 0。
        omega = 0.0
      else:
        # 相位累加器：周期只在正弦过零时更换，相位始终连续（不会速度突跳）
        if spin_mode != "const":
          self._spin_phase += 2.0 * math.pi * model.opt.timestep / max(self._spin_period, 1e-3)
          if self._spin_phase >= 2.0 * math.pi:
            self._spin_phase -= 2.0 * math.pi
            self._spin_period = float(self._rng.uniform(c.car_spin_period_min,
                                                       c.car_spin_period_max))
        else:
          self._spin_phase = 2.0 * math.pi * c.car_spin_hz * self._sim_time   # 固定周期
        # 波形整形：g = 0.5(1-cos φ) ∈ [0,1] 取 q 次幂（q<1 → 低谷很短、回升很陡），
        # 再映射回完整区间 [base-amp, base+amp]，保证速度上下限与原来一致。
        g = 0.5 * (1.0 - math.cos(self._spin_phase))
        omega = ((c.car_spin_base - c.car_spin_amp)
                 + 2.0 * c.car_spin_amp * (g ** max(c.car_spin_shape, 1e-3)))
      if self._gyro is not None:
        # 把"巡逻自转"作为目标喂进去，由 GyroPlan 做模式混合 + 角加速度限幅：
        # 小陀螺期间目标是甩头正弦（script 的 inplace_spin 段是全程保持），
        # 其余时候就是本来的自转，切换处自动平滑。
        omega = self._gyro.step(model.opt.timestep, omega)
      if c.car_tilt:
        # 姿态伺服：目标姿态 = 机体 z 轴对齐当地地形法向，yaw 保持当前自转方向。
        # 直接把姿态交给物理的话，7 rad/s 自旋 + 起伏会直接翻滚（实测 roll 跑到 180°）。
        _n = self.terrain.normal(float(pos[0]), float(pos[1]))
        _mat = np.zeros(9)
        mujoco.mju_quat2Mat(_mat, data.qpos[self.car_qadr + 3 : self.car_qadr + 7])
        _R = _mat.reshape(3, 3)
        _xd = _R[:, 0] - float(_R[:, 0] @ _n) * _n            # 保持当前朝向的切向分量
        _nx = float(np.linalg.norm(_xd))
        _xd = _xd / _nx if _nx > 1e-6 else np.array([1.0, 0.0, 0.0])
        _yd = np.cross(_n, _xd)
        _Rd = np.column_stack([_xd, _yd, _n])
        # 用 MuJoCo 自带的四元数相对旋转向量算姿态误差。
        # 手写矩阵版公式极易搞错（我第一版把 Rd^T·R 当成误差，方向反了 → 平地都能
        # 把车推成 145° 倾角），subQuat 已经处理好符号与帧。
        _qc = np.zeros(4); _qd = np.zeros(4)
        mujoco.mju_mat2Quat(_qc, np.ascontiguousarray(_R).ravel())
        mujoco.mju_mat2Quat(_qd, np.ascontiguousarray(_Rd).ravel())
        _err = np.zeros(3)
        mujoco.mju_subQuat(_err, _qd, _qc)      # 当前 → 期望，机体局部系
        _w = np.clip(self._tilt_kp * _err, -4.0, 4.0)
        self._car_tilt_w = (float(_w[0]), float(_w[1]))
      else:
        self._car_tilt_w = (0.0, 0.0)
      data.qvel[self.car_dadr + 3] = self._car_tilt_w[0]
      data.qvel[self.car_dadr + 4] = self._car_tilt_w[1]
      data.qvel[self.car_dadr + 5] = omega
      self._car_omega = omega
    else:
      data.qvel[self.car_dadr + 3 : self.car_dadr + 6] *= 0.9
      self._car_omega = float(data.qvel[self.car_dadr + 5])

    # 2) 瞄准（拖拽准星优先，否则用面板滑条）
    if c.aim_point is not None:
      c.yaw, c.pitch = self.aim_to_point(c.aim_point)
    data.ctrl[0], data.ctrl[1] = c.yaw, c.pitch

    # 3) 开火
    if c.fire:
      c.fire = False
      self.fire()
    if c.auto and (self._sim_time - self._last_fire) > 1.0 / max(c.auto_hz, 0.1):
      self.fire()

    # 3b) 气动阻力（默认关；`configs/ballistics.yaml` 的 drag.enabled）
    #     只改**飞行中**弹丸的 qvel，仍交给 MuJoCo 的 Euler 积分推进。
    #     放在 mj_step 之前 = "这一步的加速度里含阻力"，与 rm_ballistics.fly(scheme="euler")
    #     同序（自检 test_ballistics.py 就是按这个前提对齐的）。
    if self.ballistics is not None:
      self.ballistics.apply_to_balls(
        data, self.ball_qadr, self.ball_dadr, self.ball_t, model.opt.timestep)

    # 4) 物理推进
    mujoco.mj_step(model, data)
    # 4b) 重新断言靶车速度：贴地接触摩擦会在 mj_step 里改写 qvel，
    #     不补这一步的话实测加速度/转向速率会远超限幅（看着像"瞬停瞬起"）。
    data.qvel[self.car_dadr : self.car_dadr + 2] = self._car_vel[:2]
    data.qvel[self.car_dadr + 5] = self._car_omega
    # 姿态角速度也要重新断言：接触冲量会在 mj_step 里把它改写掉（只断言线速度和
    # yaw 的话，姿态伺服根本压不住，实测会翻滚到 180°）。
    data.qvel[self.car_dadr + 3] = self._car_tilt_w[0]
    data.qvel[self.car_dadr + 4] = self._car_tilt_w[1]
    # 贴地滑行：竖直方向直接按地形高度解算。让盒体靠接触去爬 ±8cm 的三角网格
    # 地形时，棱角撞台阶的冲量根本压不住（实测被弹到 +44 cm 并翻滚），
    # 所以竖直分量改为运动学跟随，姿态仍由上面的伺服跟随地形。
    _px, _py = float(data.xpos[self.car_bid][0]), float(data.xpos[self.car_bid][1])
    _lift_z = self._lift.step(model.opt.timestep) if self._lift is not None else 0.0
    data.qpos[self.car_qadr + 2] = self.terrain.height(_px, _py) + self._ride_z + _lift_z
    data.qvel[self.car_dadr + 2] = 0.0

    # 5) 命中判定：边沿触发（同一个弹丸贴在装甲板上持续接触只算一次撞击）
    now_contacting: set = set()
    cg = data.contact.geom if data.ncon else None    # (ncon,2) 数组视图，见下面的说明
    armor_gid2name = self.armor_gid2name
    body2ball = self.body2ball
    geom_body = self.geom_body
    for k in range(data.ncon):
      # 用 d.contact.geom 数组视图（见 rm_cars._score 的说明）：每个 contact[k]
      # 对象要 2.3 µs，一步 ~94 个接触，光构造对象就是 200 µs。
      g1, g2 = int(cg[k, 0]), int(cg[k, 1])
      # 先判"这个接触里有没有装甲板"，**再**去查 geom→body→ball
      # （原来的顺序对每个接触点都要多查两张表 + 一次 int()，纯属浪费）
      plate = armor_gid2name.get(g1)
      if plate is None:
        plate = armor_gid2name.get(g2)
        if plate is None:
          continue
        other = g1
      else:
        other = g2
      bidx = body2ball.get(int(geom_body[other]))
      if bidx is None:
        continue
      key = (bidx, plate)
      now_contacting.add(key)
      if key in self.contacting:
        continue  # 仍是同一次接触
      self.hits_by_plate[plate] = self.hits_by_plate.get(plate, 0) + 1
      self.markers.append({"pos": np.array(data.contact[k].pos, dtype=float).copy(), "ttl": 0.7, "handle": None})
      if bidx not in self.ball_scored:  # 这一"发"首次命中
        self.ball_scored.add(bidx)
        self.hits += 1
        if bidx in self._ball_track:
          self._ball_track[bidx]["hit"] = True
    # 命中判定只扫"装甲板 vs 弹丸"的接触，但**任何**接触都意味着自由飞行段结束。
    # 不掐掉这段的话，"离板心最近距离"会记录到弹丸撞车体/落地后滚到车边的位置 ——
    # 实测能把中位值从 3 cm 抬到 12 cm，参数扫描的梯度全被淹掉。
    if data.ncon:
      ball_geoms = self.ball_geoms
      for k in range(data.ncon):
        g1, g2 = int(cg[k, 0]), int(cg[k, 1])
        if g1 not in ball_geoms and g2 not in ball_geoms:
          continue                    # 绝大多数接触是地面/车身，一步排掉
        for g in (g1, g2):
          b = body2ball.get(int(geom_body[g]))
          if b is not None and b in self._ball_track:
            self._ball_track[b]["stopped"] = True
    self.contacting = now_contacting

    # 6) 弹丸回收 + 命中标记计时
    for i in range(BALL_POOL):
      if self.ball_t[i] > 0.0:
        # 飞行中：更新这一发离各装甲板中心的最近距离（连续量，用来标定瞄具）
        rec = self._ball_track.get(i) if self.track_shots else None
        # 快速排除：弹丸离车还远的时候没必要逐板算距离。
        # 不加这一条的话，脱靶的弹丸会在整整 4s 的 TTL 里持续被追踪，
        # 每步 4 次距离计算 × 33 步/帧 —— 实测能把仿真帧率拉低约 10%。
        # 注意这里**不能用 continue**：下面的回收逻辑会被跳过，弹丸就再也回不了池子。
        near = False
        if rec is not None and not rec.get("stopped"):
          bp = data.qpos[self.ball_qadr[i] : self.ball_qadr[i] + 3]
          near = float(np.linalg.norm(bp - data.xpos[self.car_bid])) <= 1.2
        if near:
          for gid in self.armor_gid2name:
            c = data.geom_xpos[gid]
            rel_w = bp - c
            d = float(np.linalg.norm(rel_w))
            if d < rec["min_dist"]:
              iw, ih = self.armor_axes[gid]
              i_thick = 3 - iw - ih
              # 板的局部轴 = geom 姿态矩阵的列，把偏移换算到板自己的坐标系
              Rg = data.geom_xmat[gid].reshape(3, 3)
              rel_l = Rg.T @ rel_w
              rec["min_dist"] = d
              rec["plate"] = self.armor_gid2name[gid]
              rec["off_h"] = float(rel_l[iw])   # 沿板宽方向
              rec["off_v"] = float(rel_l[ih])   # 沿板高方向（正 = 偏高）
              rec["off_n"] = float(rel_l[i_thick])  # 沿板法线（诊断用）
              # 出膛点到板心的水平距离，用来判断偏差是不是随距离变化
              mp, _ = self.muzzle()
              rec["range"] = float(np.hypot(c[0] - mp[0], c[1] - mp[1]))

        self.ball_t[i] -= model.opt.timestep
        if self.ball_t[i] <= 0.0:
          qa = self.ball_qadr[i]
          data.qpos[qa : qa + 3] = (PARK[0] + 1.2 * i, PARK[1], 0.05)
          data.qpos[qa + 3 : qa + 7] = (1.0, 0.0, 0.0, 0.0)
          data.qvel[self.ball_dadr[i] : self.ball_dadr[i] + 6] = 0.0
          self.ball_scored.discard(i)
          self.contacting = {k for k in self.contacting if k[0] != i}
          done = self._ball_track.pop(i, None)
          if done is not None:
            if done["min_dist"] == float("inf"):
              done["min_dist"] = -1.0     # 整条弹道飞出界/停住都没接近过
            self.shot_log.append(done)
            if len(self.shot_log) > 4000:
              del self.shot_log[:1000]
    for mk in self.markers:
      mk["ttl"] -= model.opt.timestep
    self.markers = [mk for mk in self.markers if mk["ttl"] > 0.0]


# ----------------------------------------------------------------------------
# 无界面自检
# ----------------------------------------------------------------------------
def selftest(seconds: float = 6.0, assets: str | None = None) -> int:
  demo = Demo(real_assets=assets)
  demo.cmd.car_move = True
  demo.cmd.auto, demo.cmd.auto_hz = True, 5.0
  track = True  # 持续跟踪瞄准（模拟玩家用鼠标跟枪）
  n = int(seconds / demo.model.opt.timestep)
  t0 = time.perf_counter()
  for _ in range(n):
    if track:
      car = demo.data.xpos[demo.car_bid].copy()
      demo.cmd.aim_point = car + np.array([0.0, 0.0, demo.armor_dz])  # 瞄装甲板中心
    demo.step(demo.model, demo.data)
  wall = time.perf_counter() - t0
  sps = n / wall
  print(f"  物理：{n:,} 步 / {wall:.2f}s = {sps:,.0f} 步/秒（实时需 {1/demo.model.opt.timestep:,.0f} 步/秒）"
        f"  → {sps * demo.model.opt.timestep:.2f}x 实时")
  print(f"  发射 {demo.shots} 发，命中 {demo.hits} 发（{100 * demo.hits / max(demo.shots, 1):.0f}%）"
        f"  装甲板分布 {demo.hits_by_plate}")
  print(f"  靶车位置 {demo.data.xpos[demo.car_bid].round(2)}"
        f"  云台 yaw={demo.data.qpos[demo.yaw_qadr]:+.2f} pitch={demo.data.qpos[demo.pitch_qadr]:+.2f}")
  ok = demo.shots > 0 and demo.hits > 0
  print("  自检：" + ("✓ 通过（能发射并命中）" if ok else "✗ 未命中，检查瞄准/CCD"))
  return 0 if ok else 1


# ----------------------------------------------------------------------------
# 交互界面（viser + mjviser）
# ----------------------------------------------------------------------------
def render_preview(path: str, assets: str | None = None, width: int = 1500, height: int = 900) -> int:
  """用 MuJoCo 自带渲染器出一张离屏图，用来核对贴纸与发光灯条是否正确。"""
  from PIL import Image

  demo = Demo(real_assets=assets)
  d = demo.data
  d.qpos[demo.car_qadr : demo.car_qadr + 3] = (2.0, 0.0, 0.02 if assets else 0.075)
  demo.cmd.yaw, demo.cmd.pitch = demo.aim_to_point(np.array([2.0, 0.0, 0.25]))
  d.ctrl[0], d.ctrl[1] = demo.cmd.yaw, demo.cmd.pitch
  for _ in range(400):  # 让云台伺服稳定下来
    demo.step(demo.model, d)
  renderer = mujoco.Renderer(demo.model, height, width)
  cam = mujoco.MjvCamera()
  mujoco.mjv_defaultCamera(cam)
  car = d.xpos[demo.car_bid].copy()
  cam.lookat[:] = (car[0] + 0.02, car[1] - 0.02, car[2] + 0.06)
  cam.distance, cam.azimuth, cam.elevation = 1.55, 150.0, -22.0
  renderer.update_scene(d, camera=cam)
  Image.fromarray(renderer.render()).save(path)
  print(f"  已渲染预览: {path}")
  return 0


def disable_viser_lod(viser) -> None:
  """关掉 mjviser 的网格 LOD 抽稀。

  mjviser 传 lod=((2.0, 1000/顶点数),)：顶点数 > 2000 时 lod_ratio < 0.5，
  相机距离超过 2.0 就会把网格抽稀到 ~1000 顶点。本模型整车 60 万顶点，
  远看时装甲板/数字/灯条会被削掉，整台车糊成灰白团。
  单台车场景全量网格无压力，直接强制 lod="off"。
  """
  _orig = viser.SceneApi.add_batched_meshes_trimesh

  def _patched(self, *a, **kw):  # type: ignore[no-untyped-def]
    kw["lod"] = "off"
    # 注：add_batched_meshes_trimesh **没有**颜色参数（batched_colors 属于
    # add_batched_meshes_simple），传了会直接 TypeError。颜色走网格自带的
    # 逐顶点 COLOR_0，已核对 GLB 里存在且数值正确。
    return _orig(self, *a, **kw)

  # 注：viser 的两个网格 API 都**没有** side 参数（试过，直接 TypeError），
  # 无法在 viser 端强制双面，双面只能靠网格自身的绕序。
  viser.SceneApi.add_batched_meshes_trimesh = _patched


def run_viewer(port: int, assets: str | None = None, realistic: bool = True,
               keep_lod: bool = False) -> None:
  import viser
  import mjviser

  if not keep_lod:
    disable_viser_lod(viser)

  demo = Demo(real_assets=assets)
  server = viser.ViserServer(port=port, label="RM gimbal demo")
  c = demo.cmd

  # ---- 写实装甲板：数字贴纸 + 发光 LED 灯条（跟随车体）----
  # 真实模型（--model real）自带装甲板贴纸与 LED，这里必须关掉：装饰件是按
  # 简单模型的装甲板位置算的，叠上去会错位 + 图案重叠。
  armor = RealisticArmor(server, demo, enabled=(not demo.real) and realistic)

  # ---- 绿色准星：左键拖拽它即控制云台 yaw/pitch ----
  aim = server.scene.add_icosphere(
    "/aim_crosshair", radius=0.05, color=(40, 235, 90), opacity=0.9, position=(1.6, 0.0, 0.55)
  )

  @aim.on_drag("left")
  def _(_ev) -> None:  # type: ignore[no-untyped-def]
    if _ev.phase == "end":
      return
    p = np.array(_ev.end_position, dtype=float)
    c.aim_point = p
    c.yaw, c.pitch = demo.aim_to_point(p)

  # ---- 左键单击场景任意处 = 发射 ----
  state = {"point_and_shoot": False}

  @server.scene.on_click()
  def _(_ev) -> None:  # type: ignore[no-untyped-def]
    if state["point_and_shoot"]:
      o = np.array(_ev.ray_origin, dtype=float)
      v = np.array(_ev.ray_direction, dtype=float)
      car = demo.data.xpos[demo.car_bid]
      z_plane = float(car[2])
      if abs(v[2]) > 1e-6:  # 与过靶车的水平面求交
        t = (z_plane - o[2]) / v[2]
        if t > 0:
          c.aim_point = o + v * t
          c.yaw, c.pitch = demo.aim_to_point(c.aim_point)
    c.fire = True

  # ---- 面板 ----
  with server.gui.add_folder("云台 / 瞄准"):
    sl_yaw = server.gui.add_slider("yaw (rad)", -math.pi, math.pi, 0.01, c.yaw)
    sl_pitch = server.gui.add_slider("pitch (rad)", demo.pitch_lo, demo.pitch_hi, 0.005, c.pitch)
    sl_speed = server.gui.add_slider("弹丸初速 (m/s)", 8.0, 30.0, 0.5, BALL_SPEED)
    sl_hz = server.gui.add_slider("连发频率 (Hz)", 1.0, 15.0, 0.5, 6.0)

    @sl_yaw.on_update
    def _(_, __=None) -> None:  # type: ignore[no-untyped-def]
      c.aim_point = None
      c.yaw = float(sl_yaw.value)

    @sl_pitch.on_update
    def _(_, __=None) -> None:  # type: ignore[no-untyped-def]
      c.aim_point = None
      c.pitch = float(sl_pitch.value)

    @sl_speed.on_update
    def _(_, __=None) -> None:  # type: ignore[no-untyped-def]
      c.ball_speed = float(sl_speed.value)

    @sl_hz.on_update
    def _(_, __=None) -> None:  # type: ignore[no-untyped-def]
      c.auto_hz = float(sl_hz.value)

  with server.gui.add_folder("开火"):
    btn_fire = server.gui.add_button("单发（单击场景也可以）", color=(230, 90, 60))
    cb_auto = server.gui.add_checkbox("连发", False)
    cb_pts = server.gui.add_checkbox("点哪打哪（单击时先瞄准再发射）", False)
    cb_gc = server.gui.add_checkbox("弹道重力补偿（自动抬高炮口）", True)

    @btn_fire.on_click
    def _(_, __=None) -> None:  # type: ignore[no-untyped-def]
      c.fire = True

    @cb_auto.on_update
    def _(_, __=None) -> None:  # type: ignore[no-untyped-def]
      c.auto = bool(cb_auto.value)

    @cb_pts.on_update
    def _(_, __=None) -> None:  # type: ignore[no-untyped-def]
      state["point_and_shoot"] = bool(cb_pts.value)

    @cb_gc.on_update
    def _(_, __=None) -> None:  # type: ignore[no-untyped-def]
      demo.gravity_comp = bool(cb_gc.value)

  with server.gui.add_folder("靶车 / 场景"):
    cb_move = server.gui.add_checkbox("靶车随机移动", True)
    cb_spin = server.gui.add_checkbox("靶车自转（转速正弦变化）", c.car_spin)
    cb_tilt = server.gui.add_checkbox("车体姿态跟随地形", c.car_tilt)
    sl_spin_base = server.gui.add_slider("自转基准速度 (rad/s)", 0.0, 13.0, 0.05, c.car_spin_base)
    sl_spin_amp = server.gui.add_slider("自转正弦幅值 (rad/s)", 0.0, 9.0, 0.05, c.car_spin_amp)
    sl_spin_hz = server.gui.add_slider("自转调制频率 (Hz)", 0.0, 2.0, 0.01, c.car_spin_hz)
    cb_spin_rand = server.gui.add_checkbox("自转周期随机 1.2~2.0 s", c.car_spin_random)
    sl_period_max = server.gui.add_slider("自转周期上限 (s)", 1.2, 3.0, 0.05, c.car_spin_period_max)
    cb_dash = server.gui.add_checkbox("靶车急停 / 冲刺", c.car_dash)
    sl_dash_speed = server.gui.add_slider("冲刺速度上限 (m/s)", 2.0, 8.0, 0.1, c.car_dash_speed_max)
    sl_accel = server.gui.add_slider("加速度限幅 (m/s²)", 0.5, 8.0, 0.1, c.car_accel)
    sl_turn = server.gui.add_slider("转向速率限幅 (rad/s)", 1.0, 8.0, 0.1, c.car_turn_rate)
    sl_shape = server.gui.add_slider("自转波形指数 q", 0.2, 1.0, 0.05, c.car_spin_shape)
    cb_armor = server.gui.add_checkbox("写实装甲板（数字贴纸 + 发光 LED）", True)
    btn_reset = server.gui.add_button("重置")

    @cb_armor.on_update
    def _(_, __=None) -> None:  # type: ignore[no-untyped-def]
      for handle, _, _ in armor.items:
        handle.visible = bool(cb_armor.value)

    @cb_move.on_update
    def _(_, __=None) -> None:  # type: ignore[no-untyped-def]
      c.car_move = bool(cb_move.value)

    @cb_tilt.on_update
    def _(_, __=None) -> None:  # type: ignore[no-untyped-def]
      c.car_tilt = bool(cb_tilt.value)

    @cb_spin.on_update
    def _(_, __=None) -> None:  # type: ignore[no-untyped-def]
      c.car_spin = bool(cb_spin.value)

    @sl_spin_base.on_update
    def _(_, __=None) -> None:  # type: ignore[no-untyped-def]
      c.car_spin_base = float(sl_spin_base.value)

    @sl_spin_amp.on_update
    def _(_, __=None) -> None:  # type: ignore[no-untyped-def]
      c.car_spin_amp = float(sl_spin_amp.value)

    @sl_spin_hz.on_update
    def _(_, __=None) -> None:  # type: ignore[no-untyped-def]
      c.car_spin_hz = float(sl_spin_hz.value)

    @cb_spin_rand.on_update
    def _(_, __=None) -> None:  # type: ignore[no-untyped-def]
      c.car_spin_random = bool(cb_spin_rand.value)

    @sl_period_max.on_update
    def _(_, __=None) -> None:  # type: ignore[no-untyped-def]
      c.car_spin_period_max = max(float(sl_period_max.value), c.car_spin_period_min)

    @sl_accel.on_update
    def _(_, __=None) -> None:  # type: ignore[no-untyped-def]
      c.car_accel = float(sl_accel.value)

    @sl_turn.on_update
    def _(_, __=None) -> None:  # type: ignore[no-untyped-def]
      c.car_turn_rate = float(sl_turn.value)

    @sl_shape.on_update
    def _(_, __=None) -> None:  # type: ignore[no-untyped-def]
      c.car_spin_shape = float(sl_shape.value)

    @cb_dash.on_update
    def _(_, __=None) -> None:  # type: ignore[no-untyped-def]
      c.car_dash = bool(cb_dash.value)

    @sl_dash_speed.on_update
    def _(_, __=None) -> None:  # type: ignore[no-untyped-def]
      c.car_dash_speed_max = float(sl_dash_speed.value)
      c.car_dash_speed_min = 0.7 * float(sl_dash_speed.value)

    @btn_reset.on_click
    def _(_, __=None) -> None:  # type: ignore[no-untyped-def]
      demo.reset_request = True

  stats = server.gui.add_markdown("初始化中…")

  # ---- 渲染回调 ----
  line = server.scene.add_line_segments(
    "/aim_ray", np.array([[[0, 0, 0], [0, 0, 0]]], dtype=float), (255, 145, 0),
    thickness=2.0, thickness_units="screen",
  )

  def render(scene) -> None:  # type: ignore[no-untyped-def]
    scene.update_from_mjdata(demo.data)
    armor.update(demo.data, demo.car_bid)  # 贴纸/LED 跟随车体
    mp, dirv = demo.muzzle()
    line.points = np.array([[mp, mp + dirv * 3.0]])
    if c.aim_point is not None:
      aim.position = tuple(c.aim_point)

    # 命中标记：新出现的建节点，已过期的删节点（键为 marker 的 id，随 marker 存活）
    for mk in demo.markers:
      if mk["handle"] is None:
        mk["handle"] = server.scene.add_icosphere(
          f"/hit/{id(mk)}", radius=0.03, color=(255, 70, 70), position=tuple(mk["pos"])
        )
      hit_handles[id(mk)] = mk["handle"]
    live = {id(mk) for mk in demo.markers}
    for key in list(hit_handles):
      if key not in live:
        hit_handles.pop(key).remove()

    stats.content = (
      f"**发射** {demo.shots}　**命中** {demo.hits}　**命中率** "
      f"{100 * demo.hits / max(demo.shots, 1):.0f}%\n\n"
      f"装甲板：{demo.hits_by_plate or '—'}\n\n"
      f"云台 yaw={demo.data.qpos[demo.yaw_qadr]:+.2f}　pitch={demo.data.qpos[demo.pitch_qadr]:+.2f}　"
      f"物理实时倍率 {viewer.actual_realtime:.2f}x"
    )

  hit_handles: dict[int, object] = {}
  port_used = port if port else server.get_port()
  print("=" * 68, flush=True)
  print(" RoboMaster 云台互动仿真（viser + MuJoCo，mjlab 同款技术栈）")
  print("=" * 68)
  print(f"  在浏览器打开：  http://localhost:{port_used}", flush=True)
  print("  左键拖拽绿色准星 → 控制云台 yaw / pitch")
  print("  左键单击场景处   → 发射 17mm 弹丸")
  print("  右键拖拽 / 滚轮  → 旋转 / 缩放视角")
  print("  Ctrl+C 退出")
  print("=" * 68, flush=True)

  viewer = mjviser.Viewer(
    demo.model, demo.data, step_fn=demo.step, render_fn=render,
    reset_fn=lambda m, d: demo.reset(), server=server,
  )
  viewer.run()


def main() -> int:
  ap = argparse.ArgumentParser(description="RoboMaster 云台互动仿真")
  ap.add_argument("--port", type=int, default=8080, help="viser 端口，0 = 自动选择")
  ap.add_argument("--selftest", action="store_true", help="无界面自检")
  ap.add_argument("--seconds", type=float, default=6.0, help="自检时长")
  ap.add_argument("--model", choices=("simple", "real"), default="simple",
                  help="simple=内置方块模型；real=由 vehicle.glb 转换出的真实模型")
  ap.add_argument("--assets-dir", default=DEFAULT_ASSETS,
                  help="rm_glb_to_mjcf.py 的输出目录")
  ap.add_argument("--terrain-amp", type=float, default=0.08,
                  help="地形起伏半幅（米）；0 = 理想平面")
  ap.add_argument("--keep-lod", action="store_true",
                  help="保留 viser 的网格 LOD 抽稀（默认关闭：远距离会把装甲板削成灰块）")
  ap.add_argument("--armor", choices=("realistic", "simple"), default="realistic",
                  help="装甲板外观：realistic=数字贴纸+发光LED，simple=纯色块")
  ap.add_argument("--render-preview", metavar="PNG", help="离屏渲染一张预览图后退出")
  a = ap.parse_args()
  global TERRAIN_AMP
  TERRAIN_AMP = max(float(a.terrain_amp), 0.0)
  assets = a.assets_dir if a.model == "real" else None
  if a.render_preview:
    return render_preview(a.render_preview, assets)
  return (selftest(a.seconds, assets) if a.selftest
          else (run_viewer(a.port, assets, a.armor == "realistic", a.keep_lod) or 0))


if __name__ == "__main__":
  sys.exit(main())
