#!/usr/bin/env python3
"""多靶车 + 血量机制。

给 rm_sim 加上：
  * **血量**：每台靶车 600 HP，每发弹丸 20 伤害（= 30 发打死）
  * **死亡刷新**：血空后立刻换一个新位置满血复活
  * **多车同场**：每次刷新随机决定场上有 1~3 台同时运动

实现方式
--------
MJCF 里恒建 3 台靶车（复制 chassis 子树，各自独立 freejoint/geom 名），
用不到的台数**不是删掉而是挪到远处停放**，这样不用重建模型。

车体运动由本模块统一驱动（`cfg.car_move` 置 False，关掉 Demo 自带的单车逻辑），
所以三台车行为完全一致；命中判定也在这里按车维护血量。
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

import numpy as np

# 血量 / 伤害
CAR_HP_MAX = 600
BULLET_DAMAGE = 20

# 活动范围与防撞（与 rm_gimbal_demo 保持一致的口径）
CAR_BOX = (1.7, 3.6, -1.6, 1.6)
CAR_KEEPOUT = 1.45
PARK_XY = (60.0, 60.0)


def duplicate_cars(xml: str, n: int = 3) -> str:
  """把 <body name="chassis">...</body> 复制 n 份，各自改名。

  第 0 台保持原名（armor0..3），避免动到 rm_gimbal_demo 里已经建好的
  armor_gid2name 映射；第 k 台用 armor_{k}_{j}。
  """
  m = re.search(r'<body name="chassis"[^>]*>.*?</body>', xml, re.S)
  if not m:
    return xml
  block = m.group(0)
  out = xml
  for k in range(1, n):
    nb = block
    nb = nb.replace('<body name="chassis"', '<body name="chassis_%d"' % k, 1)
    nb = nb.replace('<freejoint name="chassis_free"', '<freejoint name="chassis_free_%d"' % k)
    # 剩下的 name="chassis" 就是碰撞盒 geom
    nb = nb.replace('name="chassis"', 'name="chassis_%d"' % k)
    nb = nb.replace('name="chassis_vis"', 'name="chassis_%d_vis"' % k)
    nb = nb.replace('name="led_blue_vis"', 'name="led_blue_%d_vis"' % k)
    nb = nb.replace('name="sticker_vis"', 'name="sticker_%d_vis"' % k)
    nb = re.sub(r'name="armor(\d)_vis"', lambda mm, k=k: 'name="armor_%d_%s_vis"' % (k, mm.group(1)), nb)
    nb = re.sub(r'name="armor(\d)"', lambda mm, k=k: 'name="armor_%d_%s"' % (k, mm.group(1)), nb)
    out = out[:m.end()] + "\n    " + nb + out[m.end():]
  return out


class Car:
  """一台靶车的运行时状态。"""

  def __init__(self, idx, model, mujoco):
    self.idx = idx
    self.body = "chassis" if idx == 0 else "chassis_%d" % idx
    self.joint = "chassis_free" if idx == 0 else "chassis_free_%d" % idx
    self.bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, self.body)
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, self.joint)
    self.qadr = int(model.jnt_qposadr[jid])
    self.dadr = int(model.jnt_dofadr[jid])
    # 装甲板 geom：第 0 台叫 armor0..3，第 k 台叫 armor_{k}_{j}
    names = (["armor%d" % j for j in range(4)] if idx == 0
             else ["armor_%d_%d" % (idx, j) for j in range(4)])
    self.armor_gids = {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n): n for n in names}
    self.armor_gids = {g: n for g, n in self.armor_gids.items() if g >= 0}

    self.gyro = None          # 小陀螺状态机（car_motion=gyro 时懒建）
    self.hp = CAR_HP_MAX
    self.alive = True
    # 击杀时间统计（TTK）：
    #   spawn_t      这条命的出生时刻（仿真秒）
    #   first_hit_t  这条命第一次中弹的时刻（None = 还没中过）——用来区分
    #                "找到目标 + 瞄准"的耗时 和 "持续输出"的耗时
    #   spawn_shots  出生时的全场总发射数，死亡时相减 = 干掉这条命用了多少发
    self.spawn_t = 0.0
    self.first_hit_t = None
    self.spawn_shots = 0
    self.omega = 0.0
    self.spin_phase = 0.0
    self.spin_period = 1.5
    self.dir = np.array([1.0, 0.0])
    self.v = 0.0
    self.vel = np.zeros(3)
    self.target = np.array([2.6, 0.0])
    self.timer = 0.0
    self.dash = True
    self.kills = 0
    self.tilt_w = (0.0, 0.0)

  def active_armor_gids(self):
    return list(self.armor_gids.keys())


class CarField:
  """管理场上所有靶车：刷新、运动、命中扣血、死亡重生。"""

  def __init__(
    self, demo, model, data, mujoco, rng, car_move_cfg, spawn_count: int = 1,
    random_spawn: bool = False
  ):
    self.demo = demo
    self.model = model
    self.data = data
    self.mj = mujoco
    self.rng = rng
    self.cfg = car_move_cfg          # rm_gimbal_demo.Cmd 实例，借它的运动参数
    self.cars = [Car(i, model, mujoco) for i in range(3)]
    # 默认只上 1 台。
    # 多车暂时不可用的原因：四块装甲板的贴纸号码是**同一个**（--label，默认 "3"），
    # 所以场上有几台车，视觉侧就看到几组一模一样的 "3"，tracker 无法区分是哪一台，
    # 瞄准会在几台之间乱跳。真实比赛里步兵是 3 号 / 4 号这样各自不同的号码。
    # 想回到"随机 1~3 台"就 --cars-random。
    self.base_spawn_count = max(1, min(3, int(spawn_count)))
    self.random_spawn = bool(random_spawn)
    # 只有弹丸能造成伤害。装甲板之间互撞（多车同场时会互相碰）不算命中 ——
    # 不加这个过滤会把车与车的接触也当成中弹，血量掉得飞快。
    self.ball_gids = {
      g for g in range(model.ngeom)
      if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or "").startswith("bg")
    }
    # 弹丸 geom -> 它自由关节的 qpos/dof 地址，用于命中后立刻把弹丸挪走。
    # 不挪的话弹丸会带着残骸飞完/躺完整个 TTL(4s)，既白渲染又白算物理。
    self.ball_adr = {}
    for g in self.ball_gids:
      b = int(model.geom_bodyid[g])
      j = int(model.body_jntadr[b])
      if j >= 0 and model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
        self.ball_adr[g] = (int(model.jnt_qposadr[j]), int(model.jnt_dofadr[j]))

    self.contacting: set = set()     # 边沿触发用：(ball_gid, armor_gid)
    self._alive_sig = None           # _score 的 gid→车 映射缓存（只依赖存活集合）
    self._gid2car: dict = {}
    self.spawn_count = 1
    self.respawns = 0
    self.deaths = 0
    self.hits = 0
    # 每次击杀一条记录：出生/首中/死亡时刻 + 干掉它用了多少发。
    # 只看命中率有致命缺陷（把门限放松、把时间拉长，命中率自然好看，实战却更差），
    # 击杀时间（TTK）才是"多长时间打死一台车"的硬指标。
    self.kill_log: list[dict] = []
    self.park_all()
    self.respawn_round(first=True)

  # ---------------------------------------------------------------- 停放/刷新
  def _park(self, car):
    d = self.data
    d.qpos[car.qadr:car.qadr + 3] = (PARK_XY[0] + 1.2 * car.idx, PARK_XY[1], 0.05)
    d.qpos[car.qadr + 3:car.qadr + 7] = (1.0, 0.0, 0.0, 0.0)
    d.qvel[car.dadr:car.dadr + 6] = 0.0
    car.alive = False

  def park_all(self):
    for c in self.cars:
      self._park(c)

  def _random_xy(self):
    yoke = np.asarray(self.data.xpos[self.demo.yoke_bid][:2], dtype=float)
    for _ in range(40):
      p = np.array([self.rng.uniform(CAR_BOX[0], CAR_BOX[1]),
                    self.rng.uniform(CAR_BOX[2], CAR_BOX[3])])
      if float(np.linalg.norm(p - yoke)) >= CAR_KEEPOUT:
        return p
    return np.array([2.6, 0.0])

  def respawn_round(self, first: bool = False) -> int:
    """来一轮新的：按设定的台数（默认 1 台）全部满血、随机位置。"""
    if self.random_spawn:
      self.spawn_count = int(self.rng.integers(1, 4))    # 1..3，仅 --cars-random 时
    else:
      self.spawn_count = self.base_spawn_count
    self.park_all()
    for c in self.cars[:self.spawn_count]:
      p = self._random_xy()
      self.data.qpos[c.qadr:c.qadr + 3] = (p[0], p[1], 0.12)
      self.data.qpos[c.qadr + 3:c.qadr + 7] = (1.0, 0.0, 0.0, 0.0)
      self.data.qvel[c.dadr:c.dadr + 6] = 0.0
      c.hp = CAR_HP_MAX
      c.alive = True
      c.gyro = None
      c.spawn_t = float(self.demo._sim_time)
      c.first_hit_t = None
      c.spawn_shots = int(self.demo.shots)
      c.kills = 0
      c.v = 0.0
      c.dir = self.rng.normal(size=2)
      c.dir = c.dir / (np.linalg.norm(c.dir) + 1e-9)
      c.target = self._random_xy()
      c.timer = 0.0
      c.dash = True
      c.spin_phase = float(self.rng.uniform(0, 6.28))
      c.spin_period = float(self.rng.uniform(1.2, 2.0))
      if c.idx == 0:
        # 第 0 台由 Demo 驱动，重生后要把它的目标点也同步过去，
        # 否则它会照旧往老位置跑回去
        self.demo._car_target = np.array(c.target, dtype=float)
        self.demo._car_vel = np.zeros(3)
        self.demo._car_v = 0.0
    return self.spawn_count

  # ---------------------------------------------------------------- 每步运动
  def _tilt_rates(self, car):
    """姿态伺服：目标姿态 = 机体 z 轴对齐当地地形法向，yaw 保持自转方向。

    **这一步不能省**。自由关节的车体如果不做姿态伺服就是纯自由刚体：
    7 rad/s 自转 + 地形起伏 + 碰撞冲量会直接把它掀翻 ——
    漏掉这个的时候，多出来的两台车全都"倒插在地面上"。
    （第 0 台由 rm_gimbal_demo 自己做同一件事，见那里的 car_tilt 分支。）
    """
    d = self.data
    pos = np.asarray(d.xpos[car.bid], dtype=float)
    n = self.demo.terrain.normal(float(pos[0]), float(pos[1]))
    mat = np.zeros(9)
    self.mj.mju_quat2Mat(mat, d.qpos[car.qadr + 3:car.qadr + 7])
    R = mat.reshape(3, 3)
    xd = R[:, 0] - float(R[:, 0] @ n) * n          # 保持当前朝向的切向分量
    nx = float(np.linalg.norm(xd))
    xd = xd / nx if nx > 1e-6 else np.array([1.0, 0.0, 0.0])
    yd = np.cross(n, xd)
    Rd = np.column_stack([xd, yd, n])
    qc = np.zeros(4)
    qd = np.zeros(4)
    self.mj.mju_mat2Quat(qc, np.ascontiguousarray(R).ravel())
    self.mj.mju_mat2Quat(qd, np.ascontiguousarray(Rd).ravel())
    err = np.zeros(3)
    self.mj.mju_subQuat(err, qd, qc)               # 当前 -> 期望，机体局部系
    w = np.clip(self.demo._tilt_kp * err, -4.0, 4.0)
    return float(w[0]), float(w[1])

  def _drive(self, car, dt: float):
    c = self.cfg
    d = self.data
    if not car.alive:
      return
    # --- 小陀螺状态机（与第 0 台车同一套 GyroPlan；car_motion=classic 时不建）---
    gyro_active = False
    gyro_spin = 0.0
    if getattr(c, "car_motion", "classic") == "gyro":
      if car.gyro is None:
        car.gyro = GyroPlan(GyroParams.from_cmd(c), self.rng)
      gyro_active = car.gyro.spinning

    # --- 自转：正弦调制，每个周期重抽周期，相位累加保证连续 ---
    car.spin_phase += dt * 2 * math.pi / max(car.spin_period, 0.2)
    if car.spin_phase > 2 * math.pi:
      car.spin_phase -= 2 * math.pi
      car.spin_period = float(self.rng.uniform(1.2, 2.0))
    q = getattr(c, "car_spin_shape", 0.45)
    shape = (0.5 * (1 - math.cos(car.spin_phase))) ** q
    car.omega = (c.car_spin_base - c.car_spin_amp) + 2 * c.car_spin_amp * shape
    if not c.car_spin:
      car.omega = 0.0
    if car.gyro is not None:
      car.omega = car.gyro.step(dt, car.omega)   # 模式混合 + 角加速度限幅

    pos = np.asarray(d.xpos[car.bid], dtype=float)
    # --- 急停/冲刺状态机 ---
    if c.car_dash:
      car.timer -= dt
      if car.timer <= 0.0:
        if car.dash:
          car.dash = False
          car.timer = float(self.rng.uniform(*c.car_halt_time))
        else:
          car.dash = True
          car.timer = float(self.rng.uniform(*c.car_dash_time))
          car.target = self._random_xy()
      goal = 1.0 if car.dash else 0.0
      speed = c.car_dash_speed_max if car.dash else 0.0
      if car.dash:
        speed = float(self.rng.uniform(c.car_dash_speed_min, c.car_dash_speed_max))
    else:
      goal, speed = 1.0, 3.08   # 与 demo 的冲刺速度同步 +40%（原 2.2）
    if gyro_active:      # 小陀螺期间按 move_frac 缩放线速度（0 = 停住）
      frac = car.gyro.p.move_frac
      goal, speed = goal * frac, speed * frac

    d_xy = car.target - pos[:2]
    nd = float(np.linalg.norm(d_xy))
    if nd > 1e-6:
      want = math.atan2(float(d_xy[1]), float(d_xy[0]))
      cur = math.atan2(float(car.dir[1]), float(car.dir[0]))
      da = (want - cur + math.pi) % (2 * math.pi) - math.pi
      step = max(-c.car_turn_rate * dt, min(c.car_turn_rate * dt, da))
      cur += step
      car.dir = np.array([math.cos(cur), math.sin(cur)])
    v_des = goal * speed
    if nd < 0.35:
      v_des *= max(nd / 0.35, 0.0)
    dv = max(-c.car_accel * dt, min(c.car_accel * dt, v_des - car.v))
    car.v += dv
    car.vel[:2] = car.dir * car.v

    # --- 硬防撞 ---
    yoke = np.asarray(d.xpos[self.demo.yoke_bid][:2], dtype=float)
    to_g = pos[:2] - yoke
    dg = float(np.linalg.norm(to_g))
    if dg < CAR_KEEPOUT:
      out = to_g / dg if dg > 1e-6 else np.array([1.0, 0.0])
      car.vel[:2] = out * max(c.car_dash_speed_min, 1.2)
      car.dir = out
      car.v = float(np.linalg.norm(car.vel[:2]))
      car.target = yoke + out * (CAR_KEEPOUT + 0.8)

    d.qvel[car.dadr:car.dadr + 2] = car.vel[:2]
    # 角速度三个分量都要写：yaw 是自转，roll/pitch 是姿态伺服（防止翻车）
    car.tilt_w = self._tilt_rates(car)
    d.qvel[car.dadr + 3] = car.tilt_w[0]
    d.qvel[car.dadr + 4] = car.tilt_w[1]
    d.qvel[car.dadr + 5] = car.omega

  def pre_step(self, dt: float):
    """Demo.step() 之前：把各车速度和姿态目标写进 qvel。

    第 0 台跳过：它由 rm_gimbal_demo 自带的运动逻辑驱动
    （car_move=False 时那段代码会把 qvel 清零，会和这里打架），
    本模块只负责它的**血量与命中**。运动参数两边用的是同一套（CAR_BOX / keepout）。
    """
    for car in self.cars:
      if car.idx == 0:
        continue
      self._drive(car, dt)

  def post_step(self, dt: float):
    """Demo.step() 之后：贴地、重新断言速度、命中判定。"""
    d = self.data
    for car in self.cars:
      if not car.alive or car.idx == 0:
        continue
      d.qvel[car.dadr:car.dadr + 2] = car.vel[:2]
      # 姿态角速度也要重新断言：接触冲量会在 mj_step 里改写它，
      # 只写线速度的话姿态伺服压不住（第 0 台在 demo 里也是这么补的）
      d.qvel[car.dadr + 3] = car.tilt_w[0]
      d.qvel[car.dadr + 4] = car.tilt_w[1]
      d.qvel[car.dadr + 5] = car.omega
      px, py = float(d.xpos[car.bid][0]), float(d.xpos[car.bid][1])
      # 贴地滑行（竖直方向运动学跟随地形，与 Demo 对第 0 台的处理一致）
      d.qpos[car.qadr + 2] = self.demo.terrain.height(px, py) + self.demo._ride_z
      d.qvel[car.dadr + 2] = 0.0
    self._score()

  def _park_ball(self, gid: int) -> None:
    """把弹丸挪到远处的停放点并清零速度（等效于回收）。"""
    adr = self.ball_adr.get(gid)
    if adr is None:
      return
    qa, da = adr
    self.data.qpos[qa:qa + 3] = (PARK_XY[0], PARK_XY[1], 0.05)
    self.data.qpos[qa + 3:qa + 7] = (1.0, 0.0, 0.0, 0.0)
    self.data.qvel[da:da + 6] = 0.0

  # ---------------------------------------------------------------- 命中扣血
  def _score(self):
    """弹丸打到装甲板：边沿触发扣 20 血，血空就换地方重生。"""
    d = self.data
    # gid→车 映射原来**每个物理步**重建一次（40 颗弹丸 + 每次接触两次查表 → 实测
    # 单步里 500+ 次 dict.get）。这个映射只依赖"哪些车活着"，签名没变就直接复用。
    sig = tuple(c.alive for c in self.cars)
    if sig != self._alive_sig:
      self._alive_sig = sig
      gid2car = {}
      for car in self.cars:
        if car.alive:
          for g in car.armor_gids:
            gid2car[g] = car
      self._gid2car = gid2car
    gid2car = self._gid2car
    now = set()
    ncon = d.ncon
    if ncon:
      # **用数组视图而不是 d.contact[k]**：后者每个接触点都要构造一个 MjContact
      # 对象（实测 2.3 µs/个，而场上常驻 ~94 个接触 —— 40 颗停放弹丸压在地面上），
      # 一步就是 200 µs 的纯开销。d.contact.geom 是 (ncon,2) 视图，取数 0.5 µs/个。
      cg = d.contact.geom
      ball_gids = self.ball_gids
      for k in range(ncon):
        ga, gb = int(cg[k, 0]), int(cg[k, 1])
        for a, b in ((ga, gb), (gb, ga)):
          car = gid2car.get(b)
          if car is None or a not in ball_gids:
            continue
          key = (a, b)
          now.add(key)
          if key not in self.contacting:      # 边沿：持续接触只算一次
            car.hp -= BULLET_DAMAGE
            self.hits += 1
            if car.first_hit_t is None:
              car.first_hit_t = float(self.demo._sim_time)
            # 命中即回收：弹丸打中之后就没有意义了，直接挪到停放点，
            # 省掉后续几秒的渲染和物理（40 个弹丸每个都是一次独立绘制）
            self._park_ball(a)
            if car.hp <= 0:
              car.alive = False
              self.deaths += 1
              t = float(self.demo._sim_time)
              self.kill_log.append({
                "idx": int(car.idx),
                "spawn_t": round(car.spawn_t, 3),
                "first_hit_t": None if car.first_hit_t is None else round(car.first_hit_t, 3),
                "death_t": round(t, 3),
                # 从出生到死亡（含搜索/瞄准期）与从首中到死亡（纯输出期）
                "ttk_spawn": round(t - car.spawn_t, 3),
                "ttk_hit": None if car.first_hit_t is None else round(t - car.first_hit_t, 3),
                "shots": int(self.demo.shots - car.spawn_shots),
                "hits": int(CAR_HP_MAX // BULLET_DAMAGE),
              })
    self.contacting = now
    # 有车被打死 -> 立刻来一轮新的（重新随机 1~3 台 + 新位置）
    if any((not c.alive) for c in self.cars[:self.spawn_count]):
      self.respawns += 1
      # 把刚死的车挪走，避免它的碰撞体还留在原地干扰
      for c in self.cars:
        if not c.alive:
          self._park(c)
      self.respawn_round()

  # ---------------------------------------------------------------- 真值导出
  def armor_geoms(self):
    """[(car, geom_id, name)]，只含活着的车。"""
    out = []
    for car in self.cars:
      if car.alive:
        for g, n in car.armor_gids.items():
          out.append((car, g, n))
    return out

  def status(self):
    return {
      "spawn_count": self.spawn_count,
      "deaths": self.deaths,
      "respawns": self.respawns,
      "hits": self.hits,
      "kill_log": list(self.kill_log),
      "cars": [
        {"idx": c.idx, "alive": bool(c.alive), "hp": int(c.hp),
         "pos": [round(float(v), 3) for v in self.data.xpos[c.bid]]}
        for c in self.cars
      ],
    }

# ---------------------------------------------------------------------------
# 靶车「小陀螺」机动
# ---------------------------------------------------------------------------
# 真机上的移动靶不会老老实实匀速巡逻：它会**偶尔停下来原地快速甩头**
# （自研小陀螺算法），转速比巡逻时快、正弦周期更短，而且车体是静止的
# —— 对自瞄来说这是最难的一类机动：目标不走了，但装甲板在高速环绕，
# 提前量模型基本失效，tracker 还要处理装甲板"转到背面对着你"的断续。
#
# 做成**可开关**的状态机（car_motion: classic | gyro）：
#   * classic = 原来的匀速巡逻 + 正弦调制的连续自转（零行为变化，用于 A/B）；
#   * gyro    = 巡逻与小陀螺交替（时长各自随机），小陀螺期间**线速度归零**，
#               自转角速度按正弦调制，基准/幅值/周期都比巡逻时更激进。
@dataclass
class GyroParams:
  """小陀螺机动的参数（全部可被 CLI 覆盖）。"""

  patrol_min: float = 2.0      # 巡逻（移动）持续时长下限 s
  patrol_max: float = 5.0      # 上限
  dwell_min: float = 1.0       # 小陀螺（静止甩头）持续时长下限 s
  dwell_max: float = 3.0       # 上限
  spin_base: float = 7.0       # 甩头角速度基准 rad/s
  spin_amp: float = 5.0        # 正弦调制幅值 rad/s  -> ω ∈ [2.0, 12.0]
  period_min: float = 0.4      # 正弦周期下限 s（巡逻自转是 1.2~2.0，"间隔更小"）
  period_max: float = 0.8      # 上限
  shape: float = 0.45          # 波形指数：越小低谷越短、回升越陡
  enter_prob: float = 1.0      # 巡逻结束后进入小陀螺的概率
  # 角加速度限幅 rad/s²。**必须加**：不加的话进出小陀螺时角速度会一步跳变
  # （实测单步跳 12 rad/s），既不物理，也会让 tracker 在模式切换瞬间炸掉。
  # 120 的量级：正弦本身的斜率峰值约 amp·2π/T ≈ 52，不会被削；
  # 而 0 -> 12 rad/s 的起步只需要 0.1 s。
  accel: float = 120.0
  # 小陀螺期间保留多少线速度：0 = 完全停住（真机"静止甩头"，默认）；
  # 1 = 保持正常巡逻速度 = **边走边甩头**（真机上更狠的一档，见 sim/README.md §10.4）。
  move_frac: float = 0.0

  @classmethod
  def from_cmd(cls, c) -> "GyroParams":
    """从 Cmd 取值，这样 CLI 调参不用改代码。"""
    return cls(
      patrol_min=float(getattr(c, "gyro_patrol_min", 2.0)),
      patrol_max=float(getattr(c, "gyro_patrol_max", 5.0)),
      dwell_min=float(getattr(c, "gyro_dwell_min", 1.0)),
      dwell_max=float(getattr(c, "gyro_dwell_max", 3.0)),
      spin_base=float(getattr(c, "gyro_spin_base", 7.0)),
      spin_amp=float(getattr(c, "gyro_spin_amp", 5.0)),
      period_min=float(getattr(c, "gyro_period_min", 0.4)),
      period_max=float(getattr(c, "gyro_period_max", 0.8)),
      shape=float(getattr(c, "gyro_shape", 0.45)),
      enter_prob=float(getattr(c, "gyro_enter_prob", 1.0)),
      accel=float(getattr(c, "gyro_accel", 120.0)),
      move_frac=float(getattr(c, "gyro_move_frac", 0.0)),
    )


class GyroPlan:
  """每台车一份：在「巡逻」与「静止小陀螺」之间切换，并给出当前自转角速度。"""

  def __init__(self, params: GyroParams, rng, hold: bool = False):
    """hold=True：全程保持小陀螺（不做 patrol/gyro 交替）—— 基准脚本里
    的「原地随机甩头」段就是这么用的，整段 10 s 都在甩。"""
    self.p = params
    self.rng = rng
    self.hold = bool(hold)
    self.mode = "gyro" if self.hold else "patrol"
    self.timer = float(rng.uniform(params.patrol_min, params.patrol_max))
    self.phase = 0.0
    self.period = float(rng.uniform(params.period_min, params.period_max))
    self.omega = 0.0

  @property
  def spinning(self) -> bool:
    """当前是否处于小陀螺（静止甩头）状态 —— 线速度要归零。"""
    return self.mode == "gyro"

  def release(self) -> None:
    """退出「全程甩头」（hold=True）：回到 patrol 且**不再随机进入** gyro。

    为什么不是直接丢掉这个对象：ω 要经过**角加速度限幅**平滑地交回巡逻自转，
    直接销毁会让自转角速度一步跳变。timer 拉长 = 之后只是把巡逻自转原样透传。
    （踩过：不 release 的话，基准脚本里第 2 段的 inplace_spin 之后，
      后面每一段都还在按小陀螺转 —— /state.json 的 car_motion_mode 一直是 gyro。）
    """
    if not self.hold:
      return
    self.hold = False
    self.mode = "patrol"
    self.timer = 1e9

  def step(self, dt: float, patrol_omega: float = 0.0) -> float:
    """推进状态机与相位，返回**最终**自转角速度 ω（rad/s）。

    patrol_omega = 巡逻态本来自转模型给出的角速度。返回值是"目标值"经过
    **角加速度限幅**后的结果，所以进出小陀螺时不会出现一步跳变；
    调用方无论处于哪种模式，都直接把这个值写进 qvel[5] 即可。
    """
    self.timer -= dt
    if self.hold:
      self.mode = "gyro"          # 全程甩头，不切换
    elif self.timer <= 0.0:
      if self.mode == "patrol":
        if self.rng.random() < self.p.enter_prob:
          self.mode = "gyro"
          self.timer = float(self.rng.uniform(self.p.dwell_min, self.p.dwell_max))
          self.period = float(self.rng.uniform(self.p.period_min, self.p.period_max))
          self.phase = 0.0
        else:
          self.timer = float(self.rng.uniform(self.p.patrol_min, self.p.patrol_max))
      else:
        self.mode = "patrol"
        self.timer = float(self.rng.uniform(self.p.patrol_min, self.p.patrol_max))

    if self.mode != "gyro":
      target = float(patrol_omega)
    else:
      # 相位累加：周期只在过零时更换，保证角速度连续（不会突跳）
      self.phase += 2.0 * math.pi * dt / max(self.period, 1e-3)
      if self.phase >= 2.0 * math.pi:
        self.phase -= 2.0 * math.pi
        self.period = float(self.rng.uniform(self.p.period_min, self.p.period_max))
      g = 0.5 * (1.0 - math.cos(self.phase))
      p = self.p
      target = (p.spin_base - p.spin_amp) + 2.0 * p.spin_amp * (g ** max(p.shape, 1e-3))

    # 角加速度限幅：真机不可能一步把自转从 0 拉到 12 rad/s
    max_step = max(self.p.accel, 1e-6) * dt
    d = target - self.omega
    self.omega += max(-max_step, min(max_step, d))
    return self.omega
