#!/usr/bin/env python3
"""rm_sim 外观调整：把仿真渲染往"真实赛场画面"靠。

为什么需要它
------------
sp_vision 自带的目标检测模型是在**真实赛场画面**上训练的。直接拿它跑当前仿真图，
实测召回只有 3%~18%（精确率倒是 0.99）—— 认出来的都对，但绝大多数认不出来。

根因是外观失配，最大的一条是：源车模是**纯白**的（mat_body = 1,1,1），
再叠上 headlight diffuse 0.65 + ambient 0.32，车体直接过曝到 255；
而真实 RM 车体是深色的，画面上最亮的应该是**装甲板灯条**。
本模块把车体压暗、灯条提亮、场地打亮，让亮度层级和真实画面对齐。

用法
----
    from rm_appearance import apply_appearance
    xml = apply_appearance(xml, preset="realistic")

也可以单独覆盖某一项（值都是"最终要写进 MJCF 的数"）：

    xml = apply_appearance(xml, body=(0.13, 0.13, 0.15), led_emission=2.0)

实现方式是对已经生成好的 MJCF 字符串做重写，和 apply_terrain() 同一套路，
所以不改转换器产出的资产文件，随时可以切回 original。
"""

from __future__ import annotations

import re

# 预设：original = 保持原样（对照用）；realistic = 往真实赛场画面靠
PRESETS = {
  "original": dict(
    body=(1.0, 1.0, 1.0),
    gimbal=(1.0, 1.0, 1.0),
    led_color=(0.0, 0.14, 1.0),    # 原来的纯蓝（luma 低，检测器不敏感）
    led_emission=1.0,
    ball=(1.0, 0.85, 0.20),        # 原来的黄色弹丸
    hide_chassis=False,            # 对照用，保留完整车体
    ambient=0.32,
    diffuse=0.65,
    floor=(0.34, 0.37, 0.42),
    sticker=(0.8, 0.8, 0.8),
    armor=(0.0078, 0.0078, 0.0078),
  ),
  # 车体不那么黑，看是否对召回有帮助
  "realistic_lite": dict(
    body=(0.24, 0.24, 0.27),
    gimbal=(0.26, 0.26, 0.29),
    led_color=(0.55, 0.78, 1.0),
    led_emission=2.2,
    ambient=0.42,
    diffuse=0.80,
    floor=(0.52, 0.55, 0.60),
    sticker=(1.0, 1.0, 1.0),
    armor=(0.02, 0.02, 0.02),
  ),
  # 深色车体 + 高亮灯条 + 打亮的场地（实测最优组合，见 sim/README.md §6）
  #
  # 关于灯条亮度：MuJoCo 的 emission 是标量且**被钳在 [0,1]**，所以 2.2 和 3.4
  # 实际没差别。真正决定"看起来多亮"的是灯的 rgba 颜色 ——
  # 人眼/检测器看的是 luma(0.299R+0.587G+0.114B)，而纯蓝 (0,0.14,1.0) 的
  # 蓝色通道只贡献 0.114，算出来 luma 才 ~126，灰度上只算中灰。
  # 真实赛场灯条是接近过曝的白芯，所以要把 R/G 一起抬上去。
  "realistic": dict(
    body=(0.13, 0.13, 0.15),
    gimbal=(0.16, 0.16, 0.18),
    # 灯条颜色：**保持饱和蓝**。曾试过调白 (0.55,0.78,1.0) 来提 luma，
    # 结果蓝色特征被抹掉、车整体发灰，识别率崩掉 —— 蓝色本身就是识别线索，
    # 不能为了亮度牺牲它。真要调亮请用 --led-color 自己试，并复测识别率。
    led_color=(0.0, 0.14, 1.0),
    led_emission=3.4,
    ball=(0.22, 1.0, 0.08),        # 荧光绿 (57,255,20)，调试时一眼能挑出来
    hide_chassis=True,             # 只留装甲板/贴纸/灯条，去掉 55 万顶点的车体
    ambient=0.42,
    diffuse=0.80,
    floor=(0.52, 0.55, 0.60),
    sticker=(1.0, 1.0, 1.0),
    armor=(0.02, 0.02, 0.02),
  ),
}


def _fmt(rgb, alpha: float = 1.0) -> str:
  return "%.4f %.4f %.4f %.2f" % (rgb[0], rgb[1], rgb[2], alpha)


def _set_material(xml: str, name: str, rgba: str, emission: float | None = None) -> str:
  """重写 <material name="..."/> 这条自闭合标签。"""
  pat = re.compile(r'<material\s+name="' + re.escape(name) + r'"[^>]*?/>')

  def repl(_m):
    s = '<material name="%s" rgba="%s"' % (name, rgba)
    if emission is not None:
      s += ' emission="%.3f"' % emission
    return s + "/>"

  new, n = pat.subn(repl, xml)
  if n == 0:
    # 材质不存在就补一条进 <asset>（不同模型资产名可能不同）
    new = xml.replace("</asset>", '  <material name="%s" rgba="%s"/>\n  </asset>' % (name, rgba), 1)
  return new


def _set_headlight(xml: str, ambient: float, diffuse: float) -> str:
  pat = re.compile(r"<headlight[^>]*/>")
  repl = '<headlight diffuse="%.3f %.3f %.3f" ambient="%.3f %.3f %.3f" specular="0.2 0.2 0.2"/>' % (
    diffuse, diffuse, diffuse, ambient, ambient, ambient)
  if pat.search(xml):
    return pat.sub(repl, xml, count=1)
  return xml.replace("<visual>", "<visual>\n    " + repl, 1)


def _set_ball(xml: str, rgb) -> str:
  """弹丸的 rgba 是直接写在 geom 上的（<geom name="bg0" ... rgba="..."/>），不走材质。

  改亮绿色是为了调试时一眼能从画面里挑出弹丸 —— 原来跟场地/灯光色调接近，
  在 debug 窗口和浏览器全局视角里都不显眼。
  """
  pat = re.compile(r'(<geom\s+name="bg\d+"[^>]*?rgba=")[^"]*(")')
  return pat.sub(lambda m: m.group(1) + _fmt(rgb) + m.group(2), xml)


def _hide_chassis_vis(xml: str) -> str:
  """删掉靶车的车体视觉网格（chassis_vis）。

  为什么：车体网格有 **554,488 个顶点**，而它被 3 台车各引一份 ->
  光这一个网格每次渲染就要 166 万顶点，占全部顶点量的 **87%**
  （装甲板+贴纸+灯条加起来才 18 万）。而识别只需要装甲板、贴纸和灯条 ——
  车体对检测器没有贡献，纯属白烧。

  注意只删 **视觉** geom（_vis 后缀）。碰撞用的 <geom name="chassis"/> 必须留着，
  否则靶车就没有碰撞体、会被打飞或者掉下去。灯条(led_blue_vis)、
  贴纸(sticker_vis)、装甲板(armor*_vis) 全部保留。
  """
  # 必须在 duplicate_cars() 之前调用 —— 那样 3 台车的副本都不带车体网格
  pat = re.compile(r'\s*<geom\s+name="chassis(_\d+)?_vis"[^>]*/>')
  return pat.sub("", xml)


def _set_floor(xml: str, rgb) -> str:
  """地面 geom 的 rgba 是写在 geom 上的，不是材质。"""
  pat = re.compile(r'(<geom\s+name="floor(_far)?"[^>]*?rgba=")[^"]*(")')
  return pat.sub(lambda m: m.group(1) + _fmt(rgb) + m.group(3), xml)


def apply_appearance(xml: str, preset: str = "original", **kw) -> str:
  p = dict(PRESETS.get(preset, PRESETS["original"]))
  p.update({k: v for k, v in kw.items() if v is not None})

  xml = _set_material(xml, "mat_body", _fmt(p["body"]))
  xml = _set_material(xml, "mat_gimbal", _fmt(p["gimbal"]))
  xml = _set_material(xml, "mat_sticker", _fmt(p["sticker"]))
  xml = _set_material(xml, "mat_armor", _fmt(p["armor"]))
  # 灯条：颜色保持车队色，靠 emission 拉亮度
  led = p.get("led_color", (0.0, 0.14, 1.0))
  xml = _set_material(xml, "led_blue", _fmt(led), emission=p["led_emission"])
  # 红灯同比例抬亮（红本身 luma 系数 0.299，比蓝高，所以抬得少一点）
  xml = _set_material(
    xml, "led_red", _fmt((1.0, led[1] * 0.9, led[2] * 0.35)), emission=p["led_emission"])
  xml = _set_headlight(xml, p["ambient"], p["diffuse"])
  xml = _set_floor(xml, p["floor"])
  xml = _set_ball(xml, p.get("ball", (1.0, 0.85, 0.20)))
  if p.get("hide_chassis", False):
    xml = _hide_chassis_vis(xml)
  return xml


# ---------------------------------------------------------------------------
# 命令行自测：打印重写后的关键行，确认改动生效
# ---------------------------------------------------------------------------
if __name__ == "__main__":
  import sys
  src = sys.argv[1] if len(sys.argv) > 1 else "assets/rm_demo_merged.xml"
  with open(src, encoding="utf-8") as f:
    xml = f.read()
  for name in ("original", "realistic"):
    out = apply_appearance(xml, preset=name)
    print("=== %s ===" % name)
    for line in out.splitlines():
      if "material name=" in line or "headlight" in line or 'name="floor"' in line:
        print("  " + line.strip()[:110])