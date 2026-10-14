#!/usr/bin/env python3
"""RoboMaster 云台仿真：给 YOLO 用的视频流 + 标注真值服务。

云台 pitch 体上挂一个朝炮管方向(+x)的相机，无头跑物理 + 离屏渲染，通过 HTTP 提供：

  GET /stream.mjpg    MJPEG 流（OpenCV / 浏览器都能直接读）
  GET /frame.jpg      最新单帧
  GET /state.json     当前状态 + 装甲板检测框真值（像素坐标）

用 OpenCV 取流：

    import cv2, requests
    cap = cv2.VideoCapture("http://127.0.0.1:8099/stream.mjpg")
    while True:
        ok, frame = cap.read()            # frame: BGR ndarray
        gt = requests.get("http://127.0.0.1:8099/state.json").json()
        ...

跑起来：

    ~/mjlab-env/venv/bin/python rm_stream.py --model real --port 8099
    ~/mjlab-env/venv/bin/python rm_stream.py --model real --save-dir ~/rm-dataset  # 同时落盘
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ASSETS = os.path.join(_HERE, "assets")   # 转换产物；整目录可随意搬移

import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mujoco  # noqa: E402

import rm_gimbal_demo as R  # noqa: E402


# ---------------------------------------------------------------------------
# 相机：挂在云台 pitch 体上，朝炮管 +x 方向
# ---------------------------------------------------------------------------
# MuJoCo 相机沿自身 -z 看、+x 为右、+y 为上。要看向机体的 +x：
#   z_cam = -x_body, y_cam = +z_body, x_cam = y_cam × z_cam = -y_body
# 即 xyaxes = "0 -1 0   0 0 1"。
def camera_xyaxes(elev_deg: float) -> str:
  """相机朝向 = 炮管轴（含安装仰角）。

  相机固定在俯仰体上，而源模型炮管自带安装仰角：俯仰关节为了把炮管摆平会转
  同样的角度，若相机仍朝机体 +x，画面就会整体低头、目标跑到画面顶部。
  这里把相机对齐到炮管轴：观察方向 d=(cos e,0,sin e)，
  z_cam=-d，y_cam=+z 垂直于 d 的分量，x_cam=y_cam×z_cam。
  """
  e = math.radians(elev_deg)
  d = np.array([math.cos(e), 0.0, math.sin(e)])
  zc = -d
  up = np.array([0.0, 0.0, 1.0])
  yc = up - float(up @ d) * d
  yc = yc / (np.linalg.norm(yc) + 1e-12)
  xc = np.cross(yc, zc)
  return " ".join(f"{v:.6f}" for v in np.concatenate([xc, yc]))


def inject_camera(xml: str, pos=(0.26, -0.012, 0.035), fovy: float = 60.0,
                  elev_deg: float = 0.0) -> str:
  """把取流相机插进 MJCF 的 muzzle site 之后（site 是 pitch 体的最后一个子元素）。

  相机放在云台前方、车体之外，不会被自身几何遮挡。
  """
  anchor = "<site name=\"muzzle\""
  i = xml.index(anchor)
  j = xml.index("/>", i) + 2
  cam = ('\n        <camera name="eye" mode="fixed" pos="{px:.4f} {py:.4f} {pz:.4f}" '
         'xyaxes="{ax}" fovy="{fv}"/>').format(px=pos[0], py=pos[1], pz=pos[2],
                                               ax=camera_xyaxes(elev_deg), fv=fovy)
  return xml[:j] + cam + xml[j:]


def read_barrel_elev(assets_dir: str) -> float:
  """从转换器产物里读炮管安装仰角，好让相机与炮管对齐。"""
  try:
    with open(os.path.join(assets_dir, "manifest.json"), encoding="utf-8") as f:
      return float(json.load(f).get("barrel_elev_deg", 0.0))
  except Exception:
    return 0.0


# ---------------------------------------------------------------------------
# 投影：world → 像素
# ---------------------------------------------------------------------------
def project(pts_w: np.ndarray, cam_pos: np.ndarray, cam_mat: np.ndarray,
            fovy_deg: float, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
  """MuJoCo 相机为针孔模型、沿 -z 观察。返回 (像素坐标 [N,2], 是否在相机前方)。"""
  rel = pts_w - cam_pos
  cam = rel @ cam_mat                      # 世界 → 相机系（cam_mat 的列是相机三轴）
  z = -cam[:, 2]                           # 前方为正
  f = 1.0 / math.tan(math.radians(fovy_deg) * 0.5)
  aspect = width / max(height, 1)
  x = cam[:, 0] * f / aspect
  y = cam[:, 1] * f
  with np.errstate(divide="ignore", invalid="ignore"):
    u = (x / z) * (width * 0.5) + width * 0.5
    v = height * 0.5 - (y / z) * (height * 0.5)
  return np.stack([u, v], 1), z > 1e-6


# ---------------------------------------------------------------------------
# 仿真 + 渲染线程
# ---------------------------------------------------------------------------
class StreamSim:
  def __init__(self, args) -> None:
    self.args = args
    self.lock = threading.Lock()
    self.jpeg: bytes | None = None
    self.jpeg_id = 0
    self.state: dict = {"t": 0.0, "detections": []}
    self.frame_idx = 0
    self.stop = False

    # ---- 建模型：复用演示的 Demo，只在加载前插一个取流相机 ----
    elev = read_barrel_elev(args.assets_dir) if args.model == "real" else 0.0
    self.barrel_elev = elev
    self.demo = R.Demo(
      dt=args.dt,
      real_assets=(args.assets_dir if args.model == "real" else None),
      xml_hook=lambda xml: inject_camera(xml, fovy=args.fovy, elev_deg=elev),
    )
    self.model, self.data = self.demo.model, self.demo.data
    self.cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "eye")
    if self.cam_id < 0:
      raise RuntimeError("取流相机没插进去：MJCF 里没有 eye 相机")
    self.plate_geoms = [
      g for g in range(self.model.ngeom)
      if (mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, g) or "").startswith("armor")
    ]
    print(f"  取流相机 eye (id={self.cam_id}, fovy={args.fovy}°, 对齐炮管 +{self.barrel_elev:.1f}°)，"
          f"分辨率 {args.width}x{args.height}，装甲板几何 {len(self.plate_geoms)} 块")

  # ------------------------------------------------------------------ 每帧
  def tick(self) -> None:
    d, m = self.demo, self.model
    # 云台持续跟踪靶车中心：相机挂在云台上，不跟踪就会把目标挤出画面。
    # 这里不做重力补偿——相机要看向目标本身，补偿是给弹道用的。
    car = np.array(self.data.xpos[self.demo.car_bid])
    car = car + np.array([0.0, 0.0, self.demo.armor_dz])      # 瞄装甲板中心
    yaw, pitch = self.demo.aim_to_point(car, gravity_comp=False)
    self.demo.cmd.aim_point = None
    self.demo.cmd.yaw, self.demo.cmd.pitch = yaw, pitch
    for _ in range(max(1, int(round(1.0 / (self.args.fps * m.opt.timestep))))):
      self.demo.step(m, self.data)

    self.renderer.update_scene(self.data, camera="eye")
    rgb = self.renderer.render()
    from PIL import Image
    im = Image.fromarray(rgb)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=self.args.quality)
    dets = self.detections()

    with self.lock:
      self.jpeg = buf.getvalue()
      self.jpeg_id += 1
      self.state = {
        "t": float(self.data.time),
        "frame": self.frame_idx,
        "gimbal": {
          "yaw": float(self.data.qpos[self.demo.yaw_qadr]),
          "pitch": float(self.data.qpos[self.demo.pitch_qadr]),
        },
        "car_pos": [float(v) for v in self.data.xpos[self.demo.car_bid]],
        "detections": dets,
      }
    self.frame_idx += 1

    if self.args.save_dir and self.frame_idx % max(1, self.args.save_every) == 0:
      self._save(im, dets)

  def _save(self, im, dets) -> None:
    os.makedirs(self.args.save_dir, exist_ok=True)
    stem = f"frame_{self.frame_idx:06d}"
    im.save(os.path.join(self.args.save_dir, stem + ".jpg"), quality=92)
    with open(os.path.join(self.args.save_dir, stem + ".json"), "w") as f:
      json.dump({"frame": stem, "detections": dets}, f, ensure_ascii=False)

  # ------------------------------------------------------- 装甲板真值框
  def detections(self) -> list[dict]:
    """把每块装甲板的 box 八个角投影到像素，取外接矩形当 YOLO 标注。"""
    cam_pos = np.array(self.data.cam_xpos[self.cam_id])
    cam_mat = np.array(self.data.cam_xmat[self.cam_id]).reshape(3, 3)
    w, h = self.args.width, self.args.height
    out = []
    for gid in self.plate_geoms:
      name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, gid) or f"g{gid}"
      if name.endswith("_vis"):
        continue                                    # 视觉网格不算，只要碰撞 box
      size = self.model.geom_size[gid].copy()
      # 8 个角（geom 局部系）
      signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float)
      local = signs * size
      Rg = np.array(self.data.geom_xmat[gid]).reshape(3, 3)
      world = local @ Rg.T + np.array(self.data.geom_xpos[gid])
      uv, front = project(world, cam_pos, cam_mat, self.model.cam_fovy[self.cam_id], w, h)
      if not front.all():
        continue
      x0, y0 = uv.min(0)
      x1, y1 = uv.max(0)
      # 完全在画面外就丢掉
      if x1 < 0 or y1 < 0 or x0 > w or y0 > h:
        continue
      x0, y0 = max(0.0, x0), max(0.0, y0)
      x1, y1 = min(float(w), x1), min(float(h), y1)
      if x1 - x0 < 2 or y1 - y0 < 2:
        continue
      out.append({
        "cls": self.args.label,                     # 贴纸号码，就是 YOLO 的类别
        "plate": name,
        "bbox_xyxy": [round(float(x0), 1), round(float(y0), 1),
                      round(float(x1), 1), round(float(y1), 1)],
        "bbox_xywh": [round(float(x0), 1), round(float(y0), 1),
                      round(float(x1 - x0), 1), round(float(y1 - y0), 1)],
        "center": [round(float((x0 + x1) / 2), 1), round(float((y0 + y1) / 2), 1)],
      })
    return out

  def run(self) -> None:
    # MuJoCo 的 GL 后端（GLX）上下文是**按线程**的：在主线程创建的 Renderer
    # 拿到仿真线程里用会 BadAccess（X_GLXMakeCurrent）。必须在用它的线程里建。
    self.renderer = mujoco.Renderer(self.model, height=self.args.height, width=self.args.width)
    period = 1.0 / max(self.args.fps, 1.0)
    n = 0
    t0 = time.time()
    while not self.stop:
      s = time.time()
      self.tick()
      n += 1
      if n % 60 == 0:
        el = time.time() - t0
        print(f"  {n} 帧 / {el:.1f}s = {n / el:.1f} FPS  "
              f"（仿真 {self.data.time:.1f}s，检测框 {len(self.state['detections'])} 个）")
      sleep = period - (time.time() - s)
      if sleep > 0:
        time.sleep(sleep)


# ---------------------------------------------------------------------------
# HTTP 服务
# ---------------------------------------------------------------------------
BOUNDARY = b"--framer"


def make_handler(sim: StreamSim):
  class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):        # 静音访问日志
      pass

    def _send(self, body: bytes, ctype: str, code: int = 200) -> None:
      self.send_response(code)
      self.send_header("Content-Type", ctype)
      self.send_header("Content-Length", str(len(body)))
      self.send_header("Cache-Control", "no-store")
      self.send_header("Access-Control-Allow-Origin", "*")
      self.end_headers()
      self.wfile.write(body)

    def do_GET(self):  # noqa: N802
      path = self.path.split("?")[0]
      if path in ("/", "/index.html"):
        self._send(
          ("<html><body style='background:#111;color:#ddd;font-family:monospace'>"
           "<h3>RoboMaster sim stream</h3>"
           "<img src='/stream.mjpg' style='width:70%'>"
           "<p><a href='/frame.jpg'>/frame.jpg</a> · "
           "<a href='/state.json'>/state.json</a></p></body></html>").encode(),
          "text/html; charset=utf-8")
      elif path == "/state.json":
        with sim.lock:
          body = json.dumps(sim.state, ensure_ascii=False).encode()
        self._send(body, "application/json")
      elif path == "/frame.jpg":
        with sim.lock:
          body = sim.jpeg
        self._send(body or b"", "image/jpeg")
      elif path == "/stream.mjpg":
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=framer")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        last = -1
        try:
          while not sim.stop:
            with sim.lock:
              jpeg, idx = sim.jpeg, sim.jpeg_id
            if jpeg is None or idx == last:
              time.sleep(0.003)
              continue
            last = idx
            self.wfile.write(BOUNDARY + b"\r\nContent-Type: image/jpeg\r\n"
                             + f"Content-Length: {len(jpeg)}\r\n\r\n".encode()
                             + jpeg + b"\r\n")
        except (BrokenPipeError, ConnectionResetError):
          pass
      else:
        self._send(b"not found", "text/plain", 404)

  return Handler


# ---------------------------------------------------------------------------
def main() -> int:
  ap = argparse.ArgumentParser(description="RM 云台仿真视频流（给 YOLO 用）")
  ap.add_argument("--model", choices=("simple", "real"), default="real")
  ap.add_argument("--assets-dir", default=DEFAULT_ASSETS)
  ap.add_argument("--port", type=int, default=8099)
  ap.add_argument("--width", type=int, default=640)
  ap.add_argument("--height", type=int, default=480)
  ap.add_argument("--fps", type=float, default=30.0)
  ap.add_argument("--quality", type=int, default=85, help="JPEG 质量")
  ap.add_argument("--fovy", type=float, default=60.0, help="相机垂直视场角")
  ap.add_argument("--dt", type=float, default=0.001)
  ap.add_argument("--label", default="3", help="装甲板类别名（= 贴纸号码）")
  ap.add_argument("--save-dir", default="", help="同时把帧+标注落盘到目录")
  ap.add_argument("--save-every", type=int, default=2, help="每 N 帧存一次")
  ap.add_argument("--seconds", type=float, default=0.0, help="跑多少秒后退出（0=一直跑）")
  a = ap.parse_args()

  sim = StreamSim(a)
  srv = ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(sim))
  threading.Thread(target=srv.serve_forever, daemon=True).start()
  print(f"  MJPEG:  http://127.0.0.1:{a.port}/stream.mjpg")
  print(f"  单帧:   http://127.0.0.1:{a.port}/frame.jpg")
  print(f"  标注:   http://127.0.0.1:{a.port}/state.json")

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
  print("\n  退出")
  return 0


if __name__ == "__main__":
  sys.exit(main())
