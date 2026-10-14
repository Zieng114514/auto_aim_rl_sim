#!/usr/bin/env python3
"""配置层：把 rm_sim/configs/ 下的 YAML 读成 Python 对象。

为什么要这一层
--------------
赛季规则每年变（血量、热量上限/冷却、伤害、弹丸、靶型…），把这些数值散在代码里，
换赛季就得改一堆 .py。现在统一成"一份 YAML = 一个赛季"，代码只认接口。

设计约束（很重要）
------------------
1. **默认必须与"没有配置层之前"逐项一致**：默认规则集 legacy_sim 的数值就是
   历史默认值（热量 100/10、锁枪只拒单发、靶车 600 血 / 20 伤害…），
   并有黄金轨迹回归（auto_aim_rl/scratch/golden_trace.py 的 qpos hash）兜底。
2. 规则集与物理参数分开：rules/*.yaml 管"裁判系统语义"，projectiles/armor/targets 管物理与几何。
3. 每个值都要能回答"从哪来的"：规则值带 source 页码，拍板值带【负责人指定】，其余【待校准】。

用法
----
    from rm_config import load
    cfg = load()                       # 默认 legacy_sim
    cfg = load(ruleset="rmuc_2026")
    cfg.gun_params()                   # -> GunParams（可直接喂给 Demo）
    cfg.target("infantry_wheel_legged").body_lift
    cfg.preset("benchmark_2026_wl")
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DIR = os.path.join(_HERE, "configs")
DEFAULT_RULESET = "legacy_sim"


class ConfigError(RuntimeError):
  """配置缺字段/取值不合理时抛这个 —— 报错要指名道姓说哪个文件哪个键。"""


def _need(d: dict, key: str, where: str) -> Any:
  if not isinstance(d, dict) or key not in d:
    raise ConfigError("%s 缺字段 %r" % (where, key))
  return d[key]


@dataclass
class BodyLift:
  """靶车升降（轮腿）—— 负责人 2026-10-14：±150 mm、一次约 2 s。"""

  mode: str = "off"                 # off | periodic | random
  amplitude_mm: float = 150.0       # 半幅；行程 = bottom ~ bottom + 2*amplitude
  period_s: float = 2.0
  accel_limit_mps2: float = 2.0     # 【待校准】机械加速度上限
  vel_limit_mps: float = 0.5        # 【待校准】机械速度上限
  rand_hold_s: tuple = (0.35, 0.8)  # random 模式换目标高度的间隔
  # 行程下沿（相对"贴地姿态"）。**默认 0 = 不往地下走**：我们的靶车是单刚体，
  # 贴地姿态就是几何最低点落在地面（ride_z≈-4mm），负值会直接嵌地（负责人实测发现）。
  bottom_mm: float = 0.0
  # 占空比：不期望全程都在上下动（负责人 2026-10-14）。
  #   active_cycles = 每个活动窗口含几个完整升降；pause_s = 窗口之间的静止时长
  active_cycles: tuple = (1, 2)
  pause_s: tuple = (2.5, 5.0)


@dataclass
class Target:
  name: str
  desc: str = ""
  hp: float = 600.0
  armor: str = "small_armor"
  gyro: str = "variable_speed"
  body_lift: BodyLift = field(default_factory=BodyLift)
  raw: dict = field(default_factory=dict)


@dataclass
class Preset:
  name: str
  dwell_s: float = 10.0
  shuffle: bool = True
  modes: tuple = ()
  note: str = ""


class Config:
  """一份装载好的配置。所有查询都走这里，不要在业务代码里直接读 YAML。"""

  def __init__(self, root: str, ruleset: str):
    self.root = root
    self.ruleset_name = ruleset
    self._raw: dict[str, dict] = {}
    self._load_all()
    self._validate()

  # ------------------------------------------------------------------ 装载
  def _yaml(self, rel: str) -> dict:
    path = os.path.join(self.root, rel)
    if not os.path.exists(path):
      raise ConfigError("找不到配置文件 %s" % path)
    with open(path, encoding="utf-8") as f:
      d = yaml.safe_load(f)
    if not isinstance(d, dict):
      raise ConfigError("%s 顶层不是字典" % path)
    return d

  def _load_all(self) -> None:
    self.rules = self._yaml(os.path.join("rules", self.ruleset_name + ".yaml"))
    self.projectiles = self._yaml("projectiles.yaml")
    self.spread = self._yaml("projectile_spread.yaml")
    self.armor = self._yaml("armor.yaml")
    self.targets_cfg = self._yaml("targets.yaml")
    self._raw = {
      "rules": self.rules, "projectiles": self.projectiles, "spread": self.spread,
      "armor": self.armor, "targets": self.targets_cfg,
    }

  def _validate(self) -> None:
    """只做"能不能跑"级别的校验：缺字段、明显非法的取值。
    数值合理性（比如热量上限该多大）不做判断 —— 那是负责人的事。"""
    if self.ruleset_name == "legacy_sim":
      g = _need(self.rules, "gun", "rules/legacy_sim.yaml")
      for k in ("speed_cold", "speed_warm", "heat_max", "heat_per_shot",
                "heat_cool_per_s", "spread_deg_per_axis"):
        _need(g, k, "rules/legacy_sim.yaml:gun")
      tc = _need(self.rules, "target_car", "rules/legacy_sim.yaml")
      for k in ("hp", "damage_per_17mm_hit"):
        _need(tc, k, "rules/legacy_sim.yaml:target_car")
    else:
      for k in ("damage", "heat", "level"):
        _need(self.rules, k, "rules/%s.yaml" % self.ruleset_name)
    for pid in ("p17", "p42"):
      p = _need(self.projectiles, pid, "projectiles.yaml")
      for k in ("diameter_mm", "mass_g", "v0_limit_mps"):
        _need(p, k, "projectiles.yaml:%s" % pid)
    t = _need(self.targets_cfg, "targets", "targets.yaml")
    if not isinstance(t, dict) or not t:
      raise ConfigError("targets.yaml:targets 为空")
    presets = _need(self.targets_cfg, "presets", "targets.yaml")
    default_preset = self.targets_cfg.get("default_preset")
    if default_preset not in presets:
      raise ConfigError("targets.yaml:default_preset=%r 不在 presets 里" % default_preset)

  # ------------------------------------------------------------------ 查询
  def projectile(self, pid: str = "p17") -> dict:
    return self.projectiles[pid]

  def target(self, name: str = "infantry_omni") -> Target:
    d = self.targets_cfg["targets"].get(name)
    if d is None:
      raise ConfigError("targets.yaml 里没有靶车 %r（有 %s）"
                        % (name, list(self.targets_cfg["targets"])))
    bl = d.get("body_lift") or {}
    # YAML 1.1 会把裸 off/on 解析成布尔 —— 这里统一归一化成字符串，
    # 免得"升降模式 = False"这种诡异值一路传下去（踩过：mode: off 变成 False）。
    mode = bl.get("mode", "off")
    mode = {False: "off", True: "periodic"}.get(mode, str(mode))
    return Target(
      name=name, desc=d.get("desc", ""), hp=float(d.get("hp", 600.0)),
      armor=d.get("armor", "small_armor"), gyro=d.get("gyro", "variable_speed"),
      body_lift=BodyLift(
        mode=mode,
        amplitude_mm=float(bl.get("amplitude_mm", 150.0)),
        period_s=float(bl.get("period_s", 2.0)),
        accel_limit_mps2=float(bl.get("accel_limit_mps2", 2.0)),
        vel_limit_mps=float(bl.get("vel_limit_mps", 0.5)),
        rand_hold_s=tuple(bl.get("rand_hold_s", (0.35, 0.8))),
        bottom_mm=float(bl.get("bottom_mm", 0.0)),
        active_cycles=tuple(bl.get("active_cycles", (1, 2))),
        pause_s=tuple(bl.get("pause_s") or ()) if "pause_s" in bl else (2.5, 5.0),
      ),
      raw=d)

  def preset(self, name: str | None = None) -> Preset:
    name = name or self.targets_cfg["default_preset"]
    p = self.targets_cfg["presets"].get(name)
    if p is None:
      raise ConfigError("targets.yaml 里没有预设 %r" % name)
    return Preset(name=name, dwell_s=float(p.get("dwell_s", 10.0)),
                  shuffle=bool(p.get("shuffle", True)),
                  modes=tuple(p.get("modes", ())), note=p.get("note", ""))

  def default_preset_name(self) -> str:
    return self.targets_cfg["default_preset"]

  # ------------------------------------------------------------------ 与现有代码对接
  def gun_params(self):
    """产出 rm_gun.GunParams。**默认路径与历史默认值逐项一致**。"""
    from rm_gun import GunParams  # 延迟导入，避免循环依赖
    if self.ruleset_name == "legacy_sim":
      g = self.rules["gun"]
      return GunParams(
        speed_cold=float(g["speed_cold"]), speed_warm=float(g["speed_warm"]),
        tau_warm=float(g["tau_warm_shots"]), tau_cool=float(g["tau_cool_s"]),
        jitter=float(g["jitter_mps_2sigma"]),
        heat_max=float(g["heat_max"]), heat_per_shot=float(g["heat_per_shot"]),
        heat_cool=float(g["heat_cool_per_s"]), overheat_k=float(g["overheat_damage_k"]),
        self_hp_max=float(g["self_hp_max"]),
        spread_deg=float(g["spread_deg_per_axis"]),
        hard_block=(g.get("lock_policy", "block_single") == "block_single"),
        model_speed=bool(g["model_flags"]["speed"]),
        model_heat=bool(g["model_flags"]["heat"]),
        model_spread=bool(g["model_flags"]["spread"]),
      )
    # rmuc_2026：热量上限/冷却按机器人·等级取；这里先按"默认兵种"给一组
    # （真正接进主回路是 P1 的事，见 configs/README.md 的优先级表）
    raise ConfigError(
      "ruleset=%s 的 gun_params() 还没接：先把 rules/rmuc_2026.yaml 的 level 表接到"
      "兵种选择上（P1）。当前请继续用 legacy_sim 跑仿真。" % self.ruleset_name)

  def car_hp(self) -> float:
    if self.ruleset_name == "legacy_sim":
      return float(self.rules["target_car"]["hp"])
    return float(self.target("infantry_omni").hp)

  def car_damage(self) -> float:
    if self.ruleset_name == "legacy_sim":
      return float(self.rules["target_car"]["damage_per_17mm_hit"])
    return float(self.projectile("p17")["damage"]["robot_armor"])

  def summary(self) -> str:
    """一行摘要，方便日志/自检打印。"""
    t = self.target()
    return ("规则集 %s（%s）| 靶车 %s %g 血 | 17mm 伤害 %g | 热量 %s | 升降 %s"
            % (self.ruleset_name, self.rules.get("version", "?"), t.name, t.hp,
               self.car_damage(), self.rules.get("heat", {}).get("source", "legacy 100/10"),
               t.body_lift.mode))


def load(config_dir: str | None = None, ruleset: str | None = None) -> Config:
  root = config_dir or os.environ.get("RM_SIM_CONFIG_DIR") or DEFAULT_DIR
  name = ruleset or os.environ.get("RM_SIM_RULESET") or DEFAULT_RULESET
  return Config(root, name)
