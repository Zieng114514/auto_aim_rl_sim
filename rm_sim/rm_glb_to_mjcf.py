#!/usr/bin/env python3
"""
vehicle.glb  →  MuJoCo 模型转换器
================================================================
把 bevy_robomaster_simulator 的 assets/vehicle.glb（标准 RM 步兵机器人）拆成
MuJoCo 可直接使用的部件，并生成 MJCF：

  chassis.obj     底盘（含 4 块装甲板块体，不含灯条）
  armor_1..4.obj  4 块装甲板（各自独立，便于按板判定命中）
  led_red.obj     所有红色灯条（配自发光材质）
  led_blue.obj    所有蓝色灯条（配自发光材质）
  gimbal.obj      云台总装

要点 / 踩过的坑
----------------
  * glTF 是 Y-up、MuJoCo 是 Z-up → 需要绕 X 轴 -90°（用 --up-axis 控制）
  * MuJoCo 不支持 glTF，只认 OBJ/STL → 按部件导出 OBJ，材质在 MJCF 里指定
  * 装甲板左右灯条是成对的 `_L_L` / `_L_R`，且红蓝两套（按队伍切换）
  * SHOT_DIRECTION 节点是真实出膛参考帧，用作 MJCF 的 muzzle site
  * 部件顶点会被烘焙成"以各自 body 原点为基准"，MJCF 里 geom pos 都用 0

用法
----
  python rm_glb_to_mjcf.py                       # 转换 + 生成 MJCF + 预览图
  python rm_glb_to_mjcf.py --up-axis z           # 若模型本来就是 Z-up
  python rm_glb_to_mjcf.py --src <vehicle.glb> --out-dir <dir>
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ASSETS = os.path.join(_HERE, "assets")   # 转换产物；整目录可随意搬移

import struct
import sys

import numpy as np
import trimesh

# RM 真实参数（取自 bevy_robomaster_simulator/config.toml）
REAL_PARAMS = {
  "projectile_speed": 25.0,      # m/s
  "projectile_diameter": 0.017,  # m (17mm)
  "projectile_mass": 0.0032,     # kg (3.2g)
  "projectile_friction": 1.1,
  "gimbal_pitch_limit": 0.785,   # rad ≈45°
  "gimbal_rotation_speed": 3.0,  # rad/s
  "chassis_max_speed": 6.0,      # m/s
  "chassis_rotation_speed": 7.42,
  "mecanum_wheel_radius": 0.076,
  "mecanum_half_wheelbase": 0.18,
  "mecanum_half_trackwidth": 0.15,
}


def rot_x(deg: float) -> np.ndarray:
  c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
  T = np.eye(4)
  T[1, 1], T[1, 2], T[2, 1], T[2, 2] = c, -s, s, c
  return T


HELPERS = ("激光检测", "cam_direction", "shot_direction")
PITCH_UP, PITCH_DOWN = -0.785, 0.35   # 负=抬炮口(45°)、正=低头(20°)
STICKER_LABELS = ("1", "2", "3", "4", "5", "B", "G", "O")
STICKER_FRAME = ("2", "3", "4")        # 装甲板贴纸的边框件后缀
STICKER_VARIANTS = ("B", "G", "O")     # 三套互斥号码图案的后缀


def classify(node_name: str) -> str:
  """节点名 → 部件类别。"""
  n = node_name.lower()
  base = node_name.split("_")[0]                      # 贴纸网格名可能带 hash 后缀
  if any(h in node_name for h in HELPERS):
    return "skip"                                     # 视觉验证工具的辅助物体，不属于车体
  if len(node_name) >= 2 and node_name[0].isdigit() and node_name[1] in STICKER_LABELS:
    return "sticker"                                  # 形如 "3O"：3 号装甲板上的"前哨 O"贴纸
  if node_name in STICKER_LABELS:
    # 第 4 块装甲板（_ARMOR_ROOT，-x 向）的贴纸用的是**单字符**命名（2/3/4/5/B/G/O），
    # 以前不认这种名字，7 片全被当成"底盘"塞进底盘桶，于是那块板显示的是别的槽位图案，
    # 看上去像"板子自带图案、缺号码"。
    return "sticker"
  if "_l_l" in n or "_l_r" in n:            # 灯条（左右成对）
    return "led_red" if "red" in n else "led_blue"
  if n.endswith("_marker") or "vertex" in n:
    return "skip"                            # 视觉标注用的角点/标记，不参与渲染
  if n.startswith("gimbal"):
    return "gimbal"
  if n.endswith("_armor") or n.endswith("_armor_base") or n.endswith("_armor_center"):
    return "armor"
  if n == "chassis":
    return "chassis"
  if n.endswith("_armor_root") or n == "base":
    return "skip"                            # 纯父节点
  return "chassis"                           # 其余归入底盘


def node_matrix(nd: dict) -> np.ndarray:
  """glTF 节点的 TRS → 4x4（numpy 行主序）。"""
  M = np.eye(4)
  if "matrix" in nd:
    return np.array(nd["matrix"], dtype=float).reshape(4, 4).T
  if "rotation" in nd:
    x, y, z, w = nd["rotation"]
    M[:3, :3] = np.array([
      [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
      [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
      [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
  M[:3, 3] = nd.get("translation", [0, 0, 0])
  if "scale" in nd:
    M[:3, :3] = M[:3, :3] @ np.diag(nd["scale"])
  return M


def mat_color(g):
  """源材质 baseColorFactor → (r,g,b)；材质存在但无该字段时按 glTF 默认白。

  这个模型完全没有纹理（MAT_COLOR 是 128x128 纯白），外观全靠 23 个材质的
  baseColorFactor：MAT_1≈0.008 近黑机体、材质=白装甲板、BLUE/RED 带 emissive。
  """
  mat = getattr(getattr(g, "visual", None), "material", None)
  if mat is None:
    return None
  for attr in ("baseColorFactor", "base_color_factor"):
    v = getattr(mat, attr, None)
    if v is None:
      continue
    v = np.asarray(v, dtype=float).reshape(-1)
    if v.size < 3:
      continue
    v = v[:3]
    if v.max() > 1.5:
      v = v / 255.0
    return tuple(np.round(v, 4))
  return (1.0, 1.0, 1.0)          # 有材质但无 baseColorFactor → glTF 默认白


def print_buckets(bucket_colors, bucket_mats):
  """诊断：每个部件桶里各材质的颜色与面数占比。"""
  for k in sorted(bucket_mats):
    tot = sum(bucket_mats[k].values()) or 1
    print(f"    [{k:9s}] 材质构成:")
    for (mname, col), fc in bucket_mats[k].most_common(5):
      print(f"        {mname[:22]:22s} 色 {np.round(col,3)}  面 {fc:>7,} ({100*fc/tot:5.1f}%)")


def dominant_color(counter, default):
  if not counter:
    return default
  return max(counter.items(), key=lambda kv: kv[1])[0]


def keep_uv(g, verts=None, faces=None):
  """重建 Trimesh 时保留纹理坐标——apply_fix/shift 以前只传 vertices/faces，
  UV 被丢掉，导致贴纸导出后是无色白片（数字贴纸就在 MAT_COLOR 图集里）。"""
  m = trimesh.Trimesh(
    vertices=np.asarray(verts if verts is not None else g.vertices),
    faces=np.asarray(faces if faces is not None else g.faces), process=False)
  uv = getattr(getattr(g, "visual", None), "uv", None)
  if uv is not None and len(uv) == len(m.vertices):
    m.visual = trimesh.visual.TextureVisuals(uv=np.asarray(uv))
  return m


def extract_textures(src: str, out_dir: str) -> dict:
  """把 GLB 内嵌的 PNG 图集导出成文件，MuJoCo 的 2d 纹理要从磁盘读。"""
  raw = open(src, "rb").read()
  G = json.loads(_glb_json_bytes(src))
  off, bin_chunk = 12, None
  while off < len(raw):
    ln, ty = struct.unpack_from("<II", raw, off)
    if ty == 0x004E4942:
      bin_chunk = raw[off + 8:off + 8 + ln]
      break
    off += 8 + ln
  if bin_chunk is None:
    return {}
  tex_dir = os.path.join(out_dir, "tex")
  os.makedirs(tex_dir, exist_ok=True)
  out = {}
  for i, im in enumerate(G.get("images", [])):
    bv = G["bufferViews"][im["bufferView"]]
    st = bv.get("byteOffset", 0)
    data = bin_chunk[st:st + bv["byteLength"]]
    name = (im.get("name") or f"tex{i}").replace("/", "_") + ".png"
    with open(os.path.join(tex_dir, name), "wb") as f:
      f.write(data)
    out[i] = f"tex/{name}"
    print(f"  导出纹理 {name}  {len(data)/1024:.0f} KiB")
  return out


def _is_embossing(node_name: str) -> bool:
  """装甲板上的浮雕号码件：名字正好以 _ARMOR 结尾（排除 _ARMOR_BASE/_ARMOR_CENTER）。"""
  return node_name.lower().endswith("_armor")


def part_index(node_name: str) -> int:
  """装甲板编号：1_/2_/3_/4_/_ 前缀。"""
  head = node_name.split("_", 1)[0]
  return int(head) if head.isdigit() else 0


def main() -> int:
  ap = argparse.ArgumentParser(description="vehicle.glb → MuJoCo 转换器")
  ap.add_argument("--src", default=os.path.expanduser(
    "~/rm-bevy/bevy_robomaster_simulator-master/assets/vehicle.glb"))
  ap.add_argument("--out-dir", default=DEFAULT_ASSETS)
  ap.add_argument("--up-axis", choices=("y", "z"), default="y",
                  help="源模型的上轴（glTF 惯例是 y；MuJoCo 需要 z）")
  ap.add_argument("--no-preview", action="store_true")
  ap.add_argument("--flatten-plates", action="store_true",
                  help="装甲板只保留朝外的面（去掉内壳），消除近距离斜角下的双层观感")
  ap.add_argument("--keep-red-led", action="store_true",
                  help="保留源模型藏在装甲板内部的红色被击中指示灯条（默认删除）")
  ap.add_argument("--texture", dest="texture", action="store_true", default=False,
                  help="导出 GLB 内嵌图集并给材质挂 UV；vehicle.glb 的图集是纯白，默认关闭")
  ap.add_argument("--sticker", default="3", choices=list(STICKER_LABELS),
                  help="装甲板号码槽位（对应 SMALL_ARMOR_STICKER_SLOTS：2-5 步兵/英雄，B 基地，G 哨兵，O 前哨）")
  a = ap.parse_args()

  obj_dir = os.path.join(a.out_dir, "obj")
  os.makedirs(obj_dir, exist_ok=True)
  print(f"  读取 {a.src}")
  scene = trimesh.load(a.src, process=False)
  tex_files = extract_textures(a.src, a.out_dir) if a.texture else {}
  tex_main = tex_files.get(0)          # MAT_COLOR：baseColor 图集（数字贴纸在里面）

  # ---------- 1) 按类别汇总几何（套上完整的世界变换）----------
  # 分类依据是**树结构**（BASE / GIMBAL 分支），而不是节点名——名字匹配会误判
  # （例如 GIMBAL 下的"激光检测总装"节点会被错分到底盘）。
  raw_nodes = json.loads(_glb_json_bytes(a.src))["nodes"]
  parent: dict[int, int] = {}
  for i, nd in enumerate(raw_nodes):
    for c in nd.get("children", []):
      parent[c] = i
  name_of = {i: (nd.get("name") or "") for i, nd in enumerate(raw_nodes)}
  geo_node = {}                       # 几何体名 -> 节点名（用于取世界变换）
  for nd_i, nd in enumerate(raw_nodes):
    if nd.get("mesh") is not None:
      geo_node.setdefault(nd_i, nd)

  def branch_of(idx: int) -> str | None:
    while idx in parent:
      idx = parent[idx]
      if name_of.get(idx) in ("BASE", "GIMBAL"):
        return name_of[idx]
    return None

  buckets: dict[str, list] = {"chassis": [], "gimbal": [], "led_red": [], "led_blue": [], "sticker": []}
  armor_parts: dict[int, list] = {}
  bucket_colors: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
  bucket_mats: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)

  def accum(key: str, g) -> None:
    c = mat_color(g)
    if c is not None:
      bucket_colors[key][c] += max(len(g.faces), 1)
    m = getattr(getattr(g, "visual", None), "material", None)
    mname = getattr(m, "name", None) or "None"
    bucket_mats[key][(mname, tuple(np.round(np.asarray(c if c else (0, 0, 0)), 3)))] += max(len(g.faces), 1)
  outliers: list[tuple[str, list]] = []
  for node in scene.graph.nodes_geometry:
    T, gname = scene.graph.get(node)
    g = scene.geometry[gname].copy()
    g.apply_transform(T)
    kind = classify(node)
    if kind == "skip":
      continue
    # 按树分支纠正分类
    idxs = [i for i, nm in name_of.items() if nm == node]
    br = branch_of(idxs[0]) if idxs else None
    if kind in ("led_red", "led_blue"):
      # 红色灯条是源模型的"被击中"指示条，装配在装甲板内部（每块板里各 2,376 个顶点），
      # 导出后会在板边露出零星红块，看起来像"多出来的模型穿模"。默认丢掉。
      if kind == "led_red" and not a.keep_red_led:
        continue
    elif br == "GIMBAL":
      kind = "gimbal"
    elif br == "BASE" and kind == "gimbal":
      kind = "chassis"
    if kind == "armor":
      _pi = part_index(node)
      armor_parts.setdefault(_pi, []).append(g)
      accum(f"armor_{_pi}", g)
    elif kind == "sticker":
      # 贴纸网格命名 <装甲板><槽位>，槽位就是 SMALL_ARMOR_STICKER_SLOTS 的 7 个
      # 型号标签 (B/G/O/2/3/4/5)：步兵/英雄显示 2~5 号，基地 B、哨兵 G、前哨 O。
      # 7 套图案在同一位置互相重叠，只能取一套——其余必须丢弃。
      # 注意不能写成 `elif ... and node[1]==label: ... else: buckets[kind].append(g)`：
      # 那样不匹配的槽位会掉进 else 又按 kind="sticker" 塞回同一个桶，过滤形同虚设。
      if node == a.sticker or (len(node) >= 2 and node[1] == a.sticker):
        buckets["sticker"].append(g)
        accum("sticker", g)
    else:
      buckets[kind].append(g)
      accum(kind, g)
      if kind == "chassis":
        outliers.append((node, (g.bounds[1] - g.bounds[0]).round(3).tolist(),
                         g.bounds[0].round(3).tolist(), g.bounds[1].round(3).tolist()))
  print_buckets(bucket_colors, bucket_mats)
  if outliers:
    print("  · 底盘桶内的几何（原始坐标系）:")
    for nm, sz, lo_, hi_ in outliers[:12]:
      print(f"      {nm:26s} 尺寸 {sz}  z范围[{lo_[2]:+.3f},{hi_[2]:+.3f}]")

  # 出膛参考帧（SHOT_DIRECTION 只是空节点，从 glTF 原始数据里取）
  raw = json.loads(_glb_json_bytes(a.src))
  yy = np.array([0.0, 1.0, 0.0, 0.0])   # glTF Y-up 的上轴（齐次）
  for nd in raw["nodes"]:
    if nd.get("name") == "SHOT_DIRECTION":
      muzzle = np.array(nd.get("translation", [0, 0, 0]), dtype=float)
  if muzzle is None:
    print("  ! 未找到 SHOT_DIRECTION，使用云台网格中心作为出膛点")
    muzzle = np.zeros(3)

  # ---------- 2) 上轴修正：源 → MuJoCo(Z-up) ----------
  fix = rot_x(90.0) if a.up_axis == "y" else np.eye(4)   # (x,y,z)->(x,-z,y)：glTF Y-up 转 MuJoCo Z-up
  fix3 = fix[:3, :3]
  muzzle = (fix3 @ np.array([muzzle[0], muzzle[1], muzzle[2]])) if a.up_axis == "y" else muzzle

  def apply_fix(mesh_list):
    out = []
    for g in mesh_list:
      h = np.hstack([g.vertices, np.ones((len(g.vertices), 1))]) @ fix.T
      out.append(keep_uv(g, verts=h[:, :3]))
    return out

  buckets = {k: apply_fix(v) for k, v in buckets.items()}
  armor_parts = {k: apply_fix(v) for k, v in armor_parts.items()}

  # ---------- 2b) 炮管朝向：以源模型 SHOT_DIRECTION 的局部 +Y 为准 ----------
  # 依据（bevy_robomaster_simulator 源码）：
  #   projectile.rs:  direction = (gimbal.rot * launch_offset.rot) * Vec3::Y
  #   setup.rs:192:   InfantryLaunchOffset 挂在 GLB 的 "SHOT_DIRECTION" 节点上
  # 所以射向 = SHOT_DIRECTION 的局部 +Y 轴。实测该轴在 glTF 里是 (0, 0.423, -0.906)，
  # 转成 MuJoCo 是 (0, 0.906, 0.423)：朝 +Y、上仰 25°——正好对上那根 15 mm 细管。
  # （我最初按节点的 +X 做，那一头是 4.7x6.4 cm 的粗方块，是错的。）
  shot_dir = None
  for nd in raw_nodes:
    if (nd.get("name") or "") == "SHOT_DIRECTION":
      m4 = node_matrix(nd)
      shot_dir = fix3 @ (m4[:3, :3] @ np.array([0.0, 1.0, 0.0]))
  if shot_dir is not None and np.linalg.norm(shot_dir[:2]) > 1e-6:
    shot_dir = shot_dir / np.linalg.norm(shot_dir)
    az = math.atan2(float(shot_dir[1]), float(shot_dir[0]))     # 水平投影方位
    ang = -az                                                   # 转到 +X 所需的 z 旋转
    ca, sa = math.cos(ang), math.sin(ang)
    Rz = np.array([[ca, -sa, 0.0], [sa, ca, 0.0], [0.0, 0.0, 1.0]])

    def rot_z(meshes):
      return [keep_uv(g, verts=g.vertices @ Rz.T) for g in meshes]

    buckets = {k: rot_z(v) for k, v in buckets.items()}
    armor_parts = {k: rot_z(v) for k, v in armor_parts.items()}
    d3 = Rz @ shot_dir
    barrel_elev = math.degrees(math.asin(float(np.clip(d3[2], -1.0, 1.0))))
    # 管口：旋转后沿射轴最远、且离轴 3 cm 内的云台顶点
    GV0 = np.vstack([g.vertices for g in buckets["gimbal"]]) if buckets["gimbal"] else np.zeros((0, 3))
    if len(GV0):
      dd = d3 / np.linalg.norm(d3)
      proj = GV0 @ dd
      # 炮管是偏置安装的：它的轴线不经过云台原点。先取远端顶点（真正在炮管上的
      # 那批）求出横向偏置，再沿"偏置后的轴线"取管口，否则会把枢轴附近的
      # 任意顶点当成管口（实测会算成只伸出 5 cm）。
      far = proj > max(0.10, 0.7 * float(proj.max()))
      band = GV0[far] if int(far.sum()) >= 20 else GV0[proj > 0.05]
      perp_vec = band - np.outer(band @ dd, dd)
      off = np.median(perp_vec, axis=0)                 # 管轴相对原点的平移
      dist = np.linalg.norm(GV0 - np.outer(proj, dd) - off, axis=1)
      ok = dist < 0.025
      muzzle = GV0[ok][int(np.argmax(proj[ok]))] if ok.any() else GV0[int(np.argmax(proj))]
      print(f"  管轴偏置 {np.round(off, 4)}，管口伸出 {float(proj[ok].max()) if ok.any() else 0:.3f} m")
      print(f"  炮管：源 SHOT_DIRECTION 局部 +Y → 方位 {math.degrees(az):+.1f}°、"
            f"上仰 {barrel_elev:+.1f}°")
      print(f"  → 整车绕 z 旋转 {math.degrees(ang):+.1f}° 使炮管水平投影对齐 +X；"
            f"管口 {np.round(muzzle, 4)}")
  else:
    barrel_elev = 0.0
    print("  ! 未找到 SHOT_DIRECTION，炮管沿用转向前的坐标系")

  # 调试：打印各部件包围盒，找出离群件
  for k, v in list(buckets.items()) + [(f"armor_{i}", v) for i, v in armor_parts.items()]:
    if not v:
      continue
    mm = trimesh.util.concatenate(v)
    lo_, hi_ = mm.bounds
    print(f"    [{k:10s}] 面 {len(mm.faces):>7,}  尺寸 {np.round(hi_-lo_,3)}  中心 {np.round((hi_+lo_)/2,3)}")

  # ---------- 3) 确定底盘原点：底面 z=0、中心 x=y=0 ----------
  chassis_all = trimesh.util.concatenate(buckets["chassis"] + [m for v in armor_parts.values() for m in v])
  lo, hi = chassis_all.bounds
  origin = np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]])
  print(f"  底盘包围盒 {np.round(hi - lo, 3)} m，原点设在 {np.round(origin, 3)}")

  def shift(meshes, by):
    return [keep_uv(g, verts=g.vertices - by) for g in meshes]

  buckets = {k: shift(v, origin) for k, v in buckets.items()}
  armor_parts = {k: shift(v, origin) for k, v in armor_parts.items()}
  muzzle = muzzle - origin

  # ---------- 4) 云台转轴 & 炮口：全部由网格几何反推 ----------
  # GLB 里的 SHOT_DIRECTION / CAM_DIRECTION / 激光检测总装 都带 0.001 缩放，
  # 直接拿它们的 translation 当枢轴会把俯仰轴抬到云台顶面之上（实测 0.559 m），
  # 因此位置一律从网格算。
  gim_all = trimesh.util.concatenate(buckets["gimbal"])
  glo, ghi = gim_all.bounds
  GV = np.asarray(gim_all.vertices)

  # yaw 竖直轴：取云台底部 25% 高度那一层（回转底座）的重心 = 真正的回转中心
  band = GV[:, 2] <= glo[2] + 0.25 * (ghi[2] - glo[2])
  yaw_xy = GV[band].mean(0)[:2] if band.sum() > 32 else (glo[:2] + ghi[:2]) / 2
  z_yaw = float(glo[2])                       # yaw 关节置于云台底面（与底盘顶面齐平）

  # 炮口：用 2b) 里几何识别出的细管管口（整车已转向，炮管在 +X）
  muzzle_w = np.asarray(muzzle, dtype=float)

  # pitch 轴高度 = 炮口高度：真实 RM 云台的耳轴就在炮管高度上，
  # 这样静止位炮管天然水平，与游戏自身"射向 +X 且水平"一致，且无需旋转网格。
  z_pitch = float(muzzle_w[2])
  yaw_pivot = np.array([yaw_xy[0], yaw_xy[1], z_yaw])
  pitch_pivot = np.array([0.0, 0.0, z_pitch - z_yaw])   # 相对 yaw body 的偏移（高度不重复叠加）

  gim_shifted = shift(buckets["gimbal"], yaw_pivot)     # 相对 yaw body
  gim_pitch = shift(gim_shifted, pitch_pivot)           # 相对 pitch body

  # 管口在 pitch body 系里的位置（静止位 z≈0 → 炮管水平），再留 1.5 cm 余量
  muzzle_local = (muzzle_w - yaw_pivot) - pitch_pivot + np.array([0.015, 0.0, 0.0])
  print(f"  炮口（识别出的管口）: 底盘系 {np.round(muzzle_w, 4)}")
  print(f"  pitch 轴高度 {z_pitch:.4f} m（= 炮口高度，相对 yaw body {pitch_pivot[2]:.4f}）")

  # ---------- 4b) 贴纸：用源模型自带的图案 ----------
  # 实车观感 = 深色装甲板 + 白色号码 + 两侧蓝色 LED 灯条，而源模型的材质本来就是这个
  #   MAT_1.00x = 0.008 近黑装甲板 / WHITE.00x = 0.8 白字 / BLUE emissive=[0,0.14,1] 灯条
  # 所以绝不能用程序生成的贴纸去替换它。这里只修两个纯几何问题：
  #   ① 源贴纸有朝内装配的（实测 4 片全朝内），从车外看不见 → 沿板中面镜像翻到外侧；
  #   ② 贴纸整片陷在装甲板里 → 沿板法向外移到板面之外。
  plate_xy, plate_hn = {}, {}
  for _i, _ms in armor_parts.items():
    _lo, _hi = trimesh.util.concatenate(_ms).bounds
    _c = (_lo + _hi) / 2
    plate_xy[_i] = _c[:2]
    _nn = _c[:2] / (float(np.linalg.norm(_c[:2])) + 1e-9)
    plate_hn[_i] = abs(float(_nn @ ((_hi - _lo)[:2] / 2)))      # 沿板法向的半厚

  def outward_xy(pt):
    _j = min(plate_xy, key=lambda k: float(np.linalg.norm(pt - plate_xy[k])))
    _n = plate_xy[_j] / (float(np.linalg.norm(plate_xy[_j])) + 1e-9)
    return _j, _n

  def decal_normal(g):
    fn = np.asarray(getattr(g, "face_normals", np.zeros((0, 3))), dtype=float)
    if len(fn) == 0:
      return None
    w = np.asarray(getattr(g, "area_faces", np.ones(len(fn))), dtype=float).reshape(-1, 1)
    avg = (fn * w).sum(0)
    nz = float(np.linalg.norm(avg))
    return avg / nz if nz > 1e-12 else None

  fixed, flipped = [], 0
  for g in buckets["sticker"]:
    _j, _n = outward_xy(g.bounds.mean(0)[:2])
    _n3 = np.array([_n[0], _n[1], 0.0])
    _fn = decal_normal(g)
    if _fn is not None and float(_fn[:2] @ _n) < 0.0:
      # 源贴纸的装配位置本来就是对的（贴在板面内侧 2.4 mm 浮雕内），
      # 唯一的问题是**面法线朝内**：MuJoCo 双面渲染所以不会消失，
      # 但光照按背面算，整片贴纸渲染成黑的。所以只反转绕序，不动几何——
      # 早先我按"沿板中面镜像翻到外侧"处理，反而把贴纸推到装甲板后面去了。
      g = keep_uv(g, verts=g.vertices, faces=np.asarray(g.faces)[:, ::-1])
      flipped += 1
    fixed.append(g)

  # 第 4 块板：源模型没给它配号码贴纸（它自带一套白色浮雕图案），但用户要求四块一致，
  # 所以把正对面那块（板 3）的号码贴纸补过来。
  #   ① 变换必须用"关于机体质心对称面的反射"：用绕 z 轴 180° 旋转会让图案左右翻转
  #      （渲染出来是镜像的 3）；反射 + 从对面看才可读，且反射要同步反转绕序。
  #   ② 位置要抬到该装甲板**自身图案之外**：板 3 的贴纸前面在离板心 22.8 mm，
  #      而板 0 的装甲外缘在 25.4 mm，直接搬过去会被板自己的浮雕遮住（实测确实如此）。
  # 沿板法向整体外移：源贴纸自带约 2.4 mm 浮雕，装配位置本来就是对的，
  # 只补 0.2 mm 防 z-fighting。不要按"整片推到板面外"来算——那会让贴纸浮出装甲板 2.7 cm。
  lifted = []
  for g in fixed:
    _j, _n = outward_xy(g.bounds.mean(0)[:2])
    _n3 = np.array([_n[0], _n[1], 0.0])
    lifted.append(keep_uv(g, verts=g.vertices + _n3 * 0.0002))
  buckets["sticker"] = lifted
  _cst = dominant_color(bucket_colors.get("sticker"), (0.8, 0.8, 0.8))
  print(f"  · 贴纸：沿用源模型自带图案 {len(lifted)} 片（修正朝内法线 {flipped} 片），"
        f"材质 {np.round(_cst, 3)} 白字 + 近黑装甲板 + 蓝色 LED")

  # ---------- 4c) 可选：装甲板压平（去掉内壳）----------
  if a.flatten_plates:
    for _i, _ms in armor_parts.items():
      _n = plate_xy[_i] / (float(np.linalg.norm(plate_xy[_i])) + 1e-9)
      _c2 = np.asarray(plate_xy[_i], dtype=float)
      _out = []
      for g in _ms:
        if len(g.faces) == 0:
          continue
        F = np.asarray(g.faces)
        # 用面中心的法向深度判内外：装甲板网格绕序是反转的，按 face_normal 筛会
        # 反过来把内表面留下（viser 不渲染背面 → 板子看着发灰，其实是透出了白色车体）。
        _depth = (np.asarray(g.triangles_center)[:, :2] - _c2) @ _n
        _keep = _depth > 0.0
        if not _keep.any():
          _out.append(g)
          continue
        F = F[_keep]
        # 留下的面统一反向，让法线真正指向车外
        _flip = (np.asarray(g.face_normals)[_keep][:, :2] @ _n) < 0.0
        F = np.where(_flip[:, None], F[:, ::-1], F)
        _out.append(keep_uv(g, verts=g.vertices, faces=F))
      armor_parts[_i] = _out
    print("  · 装甲板已压平（只保留车外侧的面，并统一法线朝外）")

  # ---------- 5) 导出 OBJ ----------
  def dump(name, meshes):
    if not meshes:
      return None
    m = trimesh.util.concatenate(meshes) if len(meshes) > 1 else meshes[0]
    p = os.path.join(obj_dir, name)
    m.export(p)
    return m, p

  files = {}
  for key, name in (("chassis", "chassis.obj"), ("led_red", "led_red.obj"),
                    ("led_blue", "led_blue.obj"), ("sticker", "sticker.obj")):
    r = dump(name, buckets[key])
    if r:
      files[key] = r[1]
  r = dump("gimbal.obj", gim_pitch)
  if r:
    files["gimbal"] = r[1]
  armor_boxes = {}
  for idx, parts in sorted(armor_parts.items()):
    r = dump(f"armor_{idx}.obj", parts)
    if not r:
      continue
    files[f"armor_{idx}"] = r[1]
    lo2, hi2 = r[0].bounds
    armor_boxes[idx] = {"center": ((lo2 + hi2) / 2).round(4).tolist(),
                        "half": ((hi2 - lo2) / 2).round(4).tolist()}
  print(f"  导出部件: {', '.join(sorted(files))}")
  print(f"  yaw 轴 {np.round(yaw_pivot,3)}   pitch 轴高度 {z_pitch:.4f}（相对 yaw body {pitch_pivot[2]:.4f}）")

  # ---------- 6) 生成 MJCF ----------
  P = REAL_PARAMS
  rel = os.path.relpath(obj_dir, a.out_dir) if False else "./obj"
  mesh_xml = "".join(f'    <mesh name="{k}" file="{rel}/{os.path.basename(v)}"/>\n'
                     for k, v in sorted(files.items()))
  armor_geoms = ""
  for idx, box in sorted(armor_boxes.items()):
    cx, cy, cz = box["center"]
    hx, hy, hz = box["half"]
    armor_geoms += (
      f'      <geom name="armor{idx}_vis" type="mesh" mesh="armor_{idx}" material="mat_armor" group="2" contype="0" conaffinity="0" mass="0"/>\n'
      f'      <geom name="armor{idx}" type="box" group="3" size="{hx} {hy} {hz}" pos="{cx} {cy} {cz}"\n'
      f'            mass="0.3" condim="3" friction="0.6 0.005 0.0001" rgba="0 0 0 0"/>\n'
    )
  # 底盘碰撞：用包围盒近似（真实网格 42 万面不适合做碰撞）
  # 有图集时把 MAT_COLOR 挂到 body / gimbal / sticker 材质上，UV 由 OBJ 的 vt 提供
  # 颜色：取各桶内出现最多（按面数加权）的源材质 baseColorFactor
  c_body = dominant_color(bucket_colors.get("chassis"), (0.03, 0.03, 0.035))
  c_gimbal = dominant_color(bucket_colors.get("gimbal"), (0.03, 0.03, 0.035))
  c_sticker = dominant_color(bucket_colors.get("sticker"), (0.8, 0.8, 0.8))
  c_armor = dominant_color(bucket_colors.get("armor_0"), (1.0, 1.0, 1.0))
  print(f"  材质取色: 机体{np.round(c_body,3)} 云台{np.round(c_gimbal,3)} "
        f"贴纸{np.round(c_sticker,3)} 装甲板{np.round(c_armor,3)}")
  if tex_main:
    tex_xml = f'    <texture name="tex_color" type="2d" file="{tex_main}"/>\n'
    body_mat = 'texture="tex_color" rgba="1 1 1 1"'
    sticker_mat = 'texture="tex_color" rgba="1 1 1 1"'
    gimbal_mat = 'texture="tex_color" rgba="1 1 1 1"'
  else:
    tex_xml = ""
    body_mat = f'rgba="{c_body[0]} {c_body[1]} {c_body[2]} 1" specular="0.25" shininess="0.25"'
    sticker_mat = f'rgba="{c_sticker[0]} {c_sticker[1]} {c_sticker[2]} 1" specular="0.15" shininess="0.15"'
    gimbal_mat = f'rgba="{c_gimbal[0]} {c_gimbal[1]} {c_gimbal[2]} 1" specular="0.3" shininess="0.3"'
  armor_mat = (f'    <material name="mat_armor" rgba="{c_armor[0]} {c_armor[1]} {c_armor[2]} 1"'
               ' specular="0.2" shininess="0.2"/>\n')
  mat_xml = tex_xml + (
    f'    <material name="mat_body"    {body_mat}/>\n'
    f'    <material name="mat_gimbal"  {gimbal_mat}/>\n'
    '    <!-- 自发光灯条：MuJoCo 的 emission 是标量强度，颜色由 rgba 决定 -->\n'
    '    <material name="led_red"  rgba="1.0 0.07 0.0 1" emission="1"/>\n'
    '    <material name="led_blue" rgba="0.0 0.14 1.0 1" emission="1"/>\n'
    f'    <material name="mat_sticker" {sticker_mat}/>\n' + armor_mat)
  # 红色灯条默认不导出（源模型里它藏在装甲板内部的被击中指示灯条）。
  # 网格与 geom 必须同时去掉，否则 MuJoCo 会报 mesh 'led_red' not found。
  led_red_geom = (
    '      <geom name="led_red_vis"  type="mesh" mesh="led_red"  material="led_red"'
    '  contype="0" conaffinity="0" mass="0"/>\n'
    if "led_red" in files else "")
  elev_rad = math.radians(barrel_elev)
  # 绕 +y 转 -elev：让 site 的 +x 对齐"上仰 elev"的炮管轴。
  # 注意 MuJoCo XML 的四元数是 (w,x,y,z)，glTF 是 (x,y,z,w)——顺序写反会变成
  # 绕错轴的 180° 旋转，炮管直接翻过来。
  _h = -elev_rad * 0.5
  mq = np.array([math.cos(_h), 0.0, math.sin(_h), 0.0])
  print(f"  安装仰角 {barrel_elev:+.1f}° 已写入 muzzle site 的 quat，"
        f"pitch 限位平移为 [{PITCH_UP + elev_rad:+.3f}, {PITCH_DOWN + elev_rad:+.3f}]")
  cl, ch = trimesh.util.concatenate(buckets["chassis"]).bounds
  chalf = ((ch - cl) / 2).round(4)
  ccenter = ((ch + cl) / 2).round(4)
  mjcf = f"""<mujoco model="rm_vehicle_real">
  <!-- 由 vehicle.glb 自动转换生成；参数取自 bevy_robomaster_simulator/config.toml -->
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.001" integrator="Euler" ccd_iterations="50" iterations="20" ls_iterations="10"/>
  <size njmax="3000" nconmax="500"/>
  <visual>
    <headlight diffuse="0.65 0.65 0.65" ambient="0.32 0.32 0.32"/>
    <global azimuth="130" elevation="-18" offwidth="1600" offheight="1000"/>
  </visual>
  <asset>
{mesh_xml}{mat_xml}  </asset>

  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" directional="true"/>

    <!-- ===== 底盘：自由关节 + 4 块装甲板（视觉网格 + 图元碰撞）===== -->
    <body name="chassis" pos="0 0 0.12">
      <freejoint name="chassis_free"/>
      <geom name="chassis_vis" type="mesh" mesh="chassis" material="mat_body" group="0" contype="0" conaffinity="0" mass="0"/>
{led_red_geom}      <geom name="led_blue_vis" type="mesh" mesh="led_blue" material="led_blue" group="2" contype="0" conaffinity="0" mass="0"/>
      <geom name="sticker_vis" type="mesh" mesh="sticker" material="mat_sticker" group="2" contype="0" conaffinity="0" mass="0"/>
      <geom name="chassis" type="box" group="3" size="{chalf[0]} {chalf[1]} {chalf[2]}"
            pos="{ccenter[0]} {ccenter[1]} {ccenter[2]}" mass="7"
            condim="3" friction="0.8 0.005 0.0001" rgba="0 0 0 0"/>
{armor_geoms}
    </body>

    <!-- ===== 云台：yaw(绕 z) → pitch(绕 y) → 炮管 +X + muzzle site ===== -->
    <!-- 炮管：源模型 SHOT_DIRECTION 的局部 +Y 就是射向（bevy 源码 projectile.rs
         用 (gimbal.rot * launch_offset.rot) * Vec3::Y，launch_offset 即该节点）。
         转换时已把整车绕 z 转到"炮管水平投影 = +X"，并用 site 的 quat 保留
         模型自带的 +25° 安装仰角。沿 +X 的炮管只能绕 y 俯仰；正角低头、负角抬。 -->
    <body name="gimbal_yaw" pos="{yaw_pivot[0]:.4f} {yaw_pivot[1]:.4f} {yaw_pivot[2]:.4f}">
      <inertial pos="0 0 0" mass="0.6" diaginertia="0.002 0.002 0.002"/>
      <joint name="g_yaw" type="hinge" axis="0 0 1" damping="0.15" armature="0.02"/>
      <body name="gimbal_pitch" pos="{pitch_pivot[0]:.4f} {pitch_pivot[1]:.4f} {pitch_pivot[2]:.4f}">
        <inertial pos="0 0 0" mass="1.2" diaginertia="0.006 0.006 0.006"/>
        <joint name="g_pitch" type="hinge" axis="0 1 0" damping="0.15" armature="0.01"
               range="{PITCH_UP + elev_rad:.4f} {PITCH_DOWN + elev_rad:.4f}" limited="true"/>
        <geom name="gimbal_vis" type="mesh" mesh="gimbal" material="mat_gimbal" group="1" contype="0" conaffinity="0" mass="0"/>
        <!-- 出膛点：site 的局部 +x 就是炮管实际指向。
             源模型炮管自带 {barrel_elev:+.1f}° 安装仰角，这里用 quat 让 site 跟着倾斜，
             否则弹丸会沿水平飞出、和看到的炮管对不上。 -->
        <site name="muzzle" pos="{muzzle_local[0]:.4f} {muzzle_local[1]:.4f} {muzzle_local[2]:.4f}"
              quat="{mq[0]:.6f} {mq[1]:.6f} {mq[2]:.6f} {mq[3]:.6f}"
              size="0.01" rgba="1 0.5 0 1"/>
      </body>
    </body>
  </worldbody>

  <actuator>
    <position name="yaw_servo"   joint="g_yaw"   kp="50" kv="6" ctrlrange="-3.1416 3.1416"/>
    <position name="pitch_servo" joint="g_pitch" kp="50" kv="6"
              ctrlrange="{PITCH_UP + elev_rad:.4f} {PITCH_DOWN + elev_rad:.4f}"/>
  </actuator>
</mujoco>
"""
  xml_path = os.path.join(a.out_dir, "rm_vehicle_real.xml")
  open(xml_path, "w", encoding="utf-8").write(mjcf)
  manifest = {"parts": {k: os.path.basename(v) for k, v in files.items()},
              "barrel_elev_deg": round(float(barrel_elev), 3),
              "armor_boxes": armor_boxes, "muzzle_local": muzzle_local.round(4).tolist(),
              "pitch_pivot": pitch_pivot.round(4).tolist(), "yaw_pivot": yaw_pivot.round(4).tolist(),
              "params": REAL_PARAMS, "up_axis_fixed": a.up_axis == "y"}
  json.dump(manifest, open(os.path.join(a.out_dir, "manifest.json"), "w", encoding="utf-8"),
            indent=2, ensure_ascii=False)
  print(f"  ✓ 生成 {xml_path}")
  print(f"  ✓ 生成 {os.path.join(a.out_dir, 'manifest.json')}")

  # ---------- 7) 预览渲染（验证朝向/尺寸是否正确）----------
  if not a.no_preview:
    try:
      import mujoco
      from PIL import Image
      m = mujoco.MjModel.from_xml_path(xml_path)
      d = mujoco.MjData(m)
      mujoco.mj_forward(m, d)
      r = mujoco.Renderer(m, 900, 1500)
      cam = mujoco.MjvCamera()
      mujoco.mjv_defaultCamera(cam)
      cam.lookat[:] = (0, 0, 0.16)
      cam.distance, cam.azimuth, cam.elevation = 1.35, 135, -12
      r.update_scene(d, camera=cam, scene_option=_scene_opt(mujoco))
      out = os.path.join(a.out_dir, "preview.png")
      Image.fromarray(r.render()).save(out)
      print(f"  ✓ 预览图 {out}")
    except Exception as e:
      print(f"  ! 预览渲染失败: {type(e).__name__}: {str(e)[:120]}")
  return 0


def _scene_opt(mujoco):
  o = mujoco.MjvOption()
  mujoco.mjv_defaultOption(o)
  return o


def _glb_json_bytes(path: str) -> bytes:
  b = open(path, "rb").read()
  off = 12
  while off < len(b):
    ln, ty = struct.unpack_from("<II", b, off)
    if ty == 0x4E4F534A:
      return b[off + 8:off + 8 + ln]
    off += 8 + ln
  raise ValueError("no JSON chunk")


if __name__ == "__main__":
  sys.exit(main())
