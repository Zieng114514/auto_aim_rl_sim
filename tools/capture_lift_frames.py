#!/usr/bin/env python3
"""抓 overview 帧 + 同步真值，用于"渲染到底跟不跟随升降"的回归。

产物给 tools/make_lift_figs.py 生成 rm_sim/preview/32_轮腿升降_渲染核对.png。
用私有端口、只 terminate 自己起的 pid —— **绝不 pkill 进程名**
（踩过：收尾 pkill -9 -f rm_link 把负责人正在跑的联调一起杀了）。
"""
import json, os, subprocess, sys, time, urllib.request

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8094
WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.environ.get("LIFT_FRAMES", os.path.join(WS, "scratch/wl_frames2"))
URL = "http://127.0.0.1:%d" % PORT
os.makedirs(OUT, exist_ok=True)
for f in os.listdir(OUT):
  os.remove(os.path.join(OUT, f))

def jget(p):
  with urllib.request.urlopen(URL + p, timeout=3) as r:
    return json.loads(r.read())

def bget(p):
  with urllib.request.urlopen(URL + p, timeout=3) as r:
    return r.read()

def busy():
  try:
    jget("/state.json"); return True
  except Exception:
    return False

if busy():
  print("端口被占，退出"); sys.exit(2)
sim = subprocess.Popen(
  [os.path.expanduser("~/mjlab-env/venv/bin/python"), "-u", "rm_link.py", "--model", "real",
   "--port", str(PORT), "--seed", "11", "--car-lift", "periodic", "--car-script-dwell", "600"],
  cwd=os.path.join(WS, "rm_sim"), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
try:
  for _ in range(60):
    if busy():
      break
    time.sleep(0.5)
  idx, n, t0 = [], 0, time.time()
  while time.time() - t0 < 26.0 and n < 400:
    try:
      bget("/overview.jpg")            # 先 touch（惰性渲染）
      st = jget("/state.json")
      jpg = bget("/overview.jpg")
    except Exception:
      time.sleep(0.05); continue
    lf = st.get("car_lift") or {}
    rec = {"file": "f%03d.jpg" % n, "t": st["t"], "offset_mm": lf.get("offset_mm"),
           "active": lf.get("active"), "car_pos": st.get("car_pos")}
    open(os.path.join(OUT, rec["file"]), "wb").write(jpg)
    idx.append(rec); n += 1
    time.sleep(0.08)
  json.dump(idx, open(os.path.join(OUT, "index.json"), "w"), indent=1)
  off = [r["offset_mm"] for r in idx if r["offset_mm"] is not None]
  print("抓了 %d 帧（sim %.1f~%.1f s），升降 %.1f ~ %.1f mm，其中在动的帧 %.0f%%"
        % (n, idx[0]["t"], idx[-1]["t"], min(off), max(off),
           100.0 * sum(1 for r in idx if r["active"]) / max(n, 1)))
finally:
  sim.terminate()
  try:
    sim.wait(timeout=10)
  except Exception:
    sim.kill()
  print("（只终止自己起的 pid=%d）" % sim.pid)
