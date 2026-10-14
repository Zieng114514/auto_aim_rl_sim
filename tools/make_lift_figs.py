#!/usr/bin/env python3
"""生成轮腿升降的两张核对图（文档用，输出到 rm_sim/preview/）：
  32_轮腿升降_渲染核对.png —— 从渲染画面量出来的装甲板高度 vs 指令升降
  33_轮腿升降_行程与占空比.png —— 行程 0~+300mm、活动/静止窗口、限幅曲线
"""
import json, math, os, sys
import numpy as np
from PIL import Image, ImageDraw
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "Noto Sans CJK JP", "Droid Sans Fallback"]
plt.rcParams["axes.unicode_minus"] = False

# 路径**相对脚本自身**：发布到 GitHub 后别人 clone 到哪都能跑（不能写死 /home/zieng/...）
WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(WS, "rm_sim"))
from rm_lift import BodyLift, LiftParams   # noqa: E402

FR = os.environ.get("LIFT_FRAMES", os.path.join(WS, "scratch/wl_frames2"))
PREV = os.path.join(WS, "rm_sim/preview")
os.makedirs(PREV, exist_ok=True)

# ------------------------------------------------ 图 33：行程 + 占空比 + 限幅
DT = 1.0 / 500.0
def trace(p, seconds=12.0, seed=0):
  b = BodyLift(p, np.random.default_rng(seed)); out = []
  for _ in range(int(seconds / DT)):
    pv = b.vel; z = b.step(DT)
    out.append((b.t, z * 1000, b.vel * 1000, (b.vel - pv) / DT * 1000, 1.0 if b.active else 0.0))
  return b, np.array(out[5:])

base = dict(mode="periodic", amplitude_mm=150, period_s=2.0,
            accel_limit_mps2=2.0, vel_limit_mps=0.5)
b_def, d_def = trace(LiftParams(**base))
b_no, d_no = trace(LiftParams(**{**base, "pause_s": ()}))
b_ti, d_ti = trace(LiftParams(**{**base, "accel_limit_mps2": 1.0}))

fig, ax = plt.subplots(4, 1, figsize=(9.6, 9.2), sharex=True,
                       gridspec_kw={"height_ratios": [2.2, 1, 1, 0.7]})
ax[0].plot(d_no[:, 0], d_no[:, 1], color="#bbbbbb", lw=1.0, label="无窗口档（全程都在动，兼容旧行为）")
ax[0].plot(d_def[:, 0], d_def[:, 1], color="#1f77b4", lw=1.8, label="默认档（活动窗口 + 静止窗口）")
ax[0].plot(d_ti[:, 0], d_ti[:, 1], color="#d62728", lw=1.2, alpha=0.85, label="加速度上限调 1.0（轨迹被削平）")
ax[0].axhline(300, ls="--", c="gray", lw=0.9); ax[0].axhline(0, ls="-", c="#444", lw=1.0)
ax[0].text(0.05, 306, "行程上沿 +300mm（= bottom + 2A）", fontsize=8, color="#444")
ax[0].text(0.05, -26, "行程下沿 0 = 贴地姿态（再往下就是嵌地 —— 负责人实测发现的问题）",
           fontsize=8, color="#a00")
ax[0].set_ylabel("车体高度偏移 (mm)"); ax[0].set_ylim(-45, 345)
ax[0].legend(loc="upper right", fontsize=8.5)
ax[0].set_title("轮腿升降：行程 0~+300mm、约 40% 时间在动（负责人 2026-10-14 两条要求）")
ax[1].plot(d_def[:, 0], d_def[:, 2], color="#1f77b4", lw=1.0)
ax[1].axhline(500, ls="--", c="gray", lw=0.9); ax[1].axhline(-500, ls="--", c="gray", lw=0.9)
ax[1].set_ylabel("速度 (mm/s)")
ax[2].plot(d_def[:, 0], d_def[:, 3], color="#1f77b4", lw=0.9)
ax[2].plot(d_ti[:, 0], d_ti[:, 3], color="#d62728", lw=0.9, alpha=0.85)
ax[2].axhline(2000, ls="--", c="gray", lw=0.9); ax[2].axhline(-2000, ls="--", c="gray", lw=0.9)
ax[2].axhline(1000, ls="--", c="gray", lw=0.9); ax[2].axhline(-1000, ls="--", c="gray", lw=0.9)
ax[2].set_ylabel("加速度 (mm/s²)"); ax[2].set_ylim(-2600, 2600)
ax[3].fill_between(d_def[:, 0], 0, d_def[:, 4], color="#2ca02c", alpha=0.5, step="post")
ax[3].set_ylabel("活动窗口"); ax[3].set_yticks([0, 1]); ax[3].set_yticklabels(["停", "动"])
ax[3].set_xlabel("仿真时间 (s)")
ax[3].text(0.05, 1.35, "绿=活动窗口（1~2 个完整升降），空白=静止窗口（2.5~5s，停在贴地姿态）",
           fontsize=8, color="#2a6")
ax[2].annotate("默认档峰 1.48 < 上限 2.0（不介入）", xy=(1.5, 1480), xytext=(3.2, 2150),
               fontsize=8, color="#1f77b4", arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=0.8))
fig.tight_layout(); fig.savefig(os.path.join(PREV, "33_轮腿升降_行程与占空比.png"), dpi=110)
print("写好 33_轮腿升降_行程与占空比.png（默认档占空比 %.2f）" % b_def.duty)

# ------------------------------------------------ 图 32：渲染核对
CAM = np.array([-2.6, -2.2, 2.3])
X = np.array([0.431455, -0.902134, 0.0]); X /= np.linalg.norm(X)
Y = np.array([0.350501, 0.167631, 0.921438]); Y /= np.linalg.norm(Y)
Z = np.cross(X, Y); F = (720 / 2) / math.tan(math.radians(45.0) / 2)

def project(p):
  d = np.asarray(p, float) - CAM
  return np.array([640 + F * (d @ X) / -(d @ Z), 360 - F * (d @ Y) / -(d @ Z)]), float(-(d @ Z))

idx = json.load(open(os.path.join(FR, "index.json")))
samples = []
for r in idx:
  cp = np.array(r["car_pos"], float); off = r["offset_mm"]
  uv_b, depth = project(cp - np.array([0, 0, off / 1000.0]))
  mmpp = depth / F * 1000.0
  half = 750.0 / mmpp
  x0, y0 = int(uv_b[0] - half), int(uv_b[1] - half)
  im = np.asarray(Image.open(os.path.join(FR, r["file"])).convert("L"), float)
  win = im[max(0, y0):int(uv_b[1] + half), max(0, x0):int(uv_b[0] + half)]
  rows = (win < 95).sum(axis=1)
  rr = np.nonzero(rows >= 6)[0]
  if rr.size == 0:
    continue
  top = rr.min() + max(0, y0)
  samples.append(dict(file=r["file"], off=off, t=r["t"], uv_b=uv_b, mmpp=mmpp,
                      top_mm=(top - uv_b[1]) * mmpp, depth=depth, half=half))
zero = [s["top_mm"] for s in samples if abs(s["off"]) < 30]
H0 = float(np.median(zero))
print("零升降基准高度 H0 = %.0f mm（%d 帧）" % (H0, len(zero)))

TILE, EXTENT_MM = 300, 2000.0
CANON = EXTENT_MM / TILE
sheet = Image.new("RGB", (TILE * 3, TILE + 30), (255, 255, 255))
gaps = []
for i, tg in enumerate((300, 150, 0)):
  s = min(samples, key=lambda q: abs(q["off"] - tg))
  im = Image.open(os.path.join(FR, s["file"])).convert("RGB")
  box = (int(s["uv_b"][0] - s["half"]), int(s["uv_b"][1] - s["half"]),
         int(s["uv_b"][0] + s["half"]), int(s["uv_b"][1] + s["half"]))
  crop = im.crop(box).resize((TILE, TILE), Image.LANCZOS)
  d = ImageDraw.Draw(crop); c = TILE / 2
  ref = c + H0 / CANON
  meas = c + s["top_mm"] / CANON
  d.line([(0, ref), (TILE, ref)], fill=(220, 30, 30), width=2)
  d.line([(0, meas), (TILE, meas)], fill=(0, 190, 0), width=2)
  d.text((5, 3), "指令 %+.0f mm   %.2f s   d=%.1fm" % (s["off"], s["t"], s["depth"]), fill=(0, 0, 0))
  d.text((5, TILE - 16), "红线=不升降参考  绿线=实测（差 %+.0f mm）" % (s["top_mm"] - H0), fill=(0, 0, 0))
  sheet.paste(crop, (i * TILE, 0))
  gaps.append((s["off"], s["top_mm"] - H0))
sheet.save(os.path.join(PREV, "32_轮腿升降_渲染核对.png"))
print("写好 32_轮腿升降_渲染核对.png")
for off, gap in gaps:
  print("  指令 %+6.0f mm -> 画面量到 %+6.0f mm" % (off, gap))
a = np.array([[s["off"], s["top_mm"]] for s in samples])
k = np.polyfit(a[:, 0], a[:, 1], 1)
print("全 %d 帧回归：斜率 %.3f（理想 -1.00，完全不跟随为 0），残差 RMS %.0f mm"
      % (len(a), k[0], float(np.sqrt(((a[:, 1] - np.polyval(k, a[:, 0])) ** 2).mean()))))
