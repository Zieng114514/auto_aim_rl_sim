# agnet_handover.md —— 交接说明：最近一轮对话的结论与未决问题

> 面向**下一个接手这个项目的 AI**。读完 `agnet.md`（说明书）+ `agnet_experience.md`（经验/差距）
> 之后再读本文：它记录的是**最近一轮和负责人的对话**里问出来的结论、做掉的改动、踩的坑，
> 以及**还没拍板的事**。截至 2026-10-14 中午。

## 0. 30 秒版

* 项目目标：把 RoboMaster 自瞄的「识别 → 跟踪 → 预测 → 开火」整条链路在仿真里跑通并量化（命中率 + TTK）。
* **当前产线**：`sp_vision_25/configs/sim.yaml` 里 `shooter: rl` + `rl_shooter_mode: quality` + λ=0.69。
  这是**阶段 1**（只学开火时机），瞄准点仍由经典 Aimer 给。
* 最近一轮做完的事：轮腿升降的**行程**（防嵌地）与**占空比**、**四种带升降靶型进基准**、
  修掉一个**小陀螺状态机泄漏**的既有 bug（会让分靶型统计失真）。
* **最该先做的事**：小陀螺修复改变了靶车行为 ⇒ **所有 script 模式基线都要重录**（见 §4）。

## 1. 和负责人怎么合作（他的习惯，别踩）

| 事项 | 做法 |
|---|---|
| 分工 | 负责人拍板参数/口径、真机数据、比赛经验、最终验收；AI 负责实现、自测、日志、基准跑数 |
| 参数没数据源时 | 不要停下来等：给一个【待校准】默认值 + 写进配置注释，继续往下做 |
| 他说的数值 | 要标注【负责人指定 日期】；他明确说过的：步兵/哨兵血量先按 600（与手册表不同） |
| 沟通语言 | 中文；他要**数据与结论**，不要形容词；每个结论要能指出复现命令 |
| 他常说的 | 「先不用写任何代码」（= 先讨论方案，别动手）；「顺带…」（= 附带任务，别跑偏主线） |
| 硬性规范 | 每次开发会话都要在 `agnet_logs/` 留日志；临时脚本进 scratch/，有价值的提炼进正式目录 |
| 新规则 | 参数与口径的真相只写在配置/文档里，代码注释里也要写清**单位和坐标系**（这个项目踩过 4 个坐标系 bug） |

## 2. 最近一轮对话：他问了什么、结论是什么

### 2.1 开火信号是电平还是脉冲？（= 持续输出 1 是连发还是只发一发）

**结论：电平触发 = 连发（有热量门限兜着），不是单发脉冲。**四处代码逐层对齐：

| 环节 | 位置 | 行为 |
|---|---|---|
| 原版决策 | `sp_vision_25/tasks/auto_aim/shooter.cpp:31-40` | 每帧重算，条件满足就 `return true`；超差/丢目标立刻变 0（是**条件电平，不是锁存**） |
| 发送 | `sp_vision_25/io/cboard.cpp:53` | 原样写进串口帧，**没有边沿转换** |
| 仿真闭环 | `sim/sim_main.cpp:199` → `rm_sim/rm_link.py:366` | `if d.get("shoot"): pending_fire += 1`，每 tick 排空 ⇒ 收到几次 1 就 `demo.fire()` 几次 |
| 训练环境 | `auto_aim_rl/env/sim_env.py:185` + `env/world.py:109-119` | `shoot = (action==1) and valid`，**与 rm_link 逐行同构** |

补充事实：

* 真正决定射速的是**枪口热量门限**（`rm_sim/rm_gun.py`：`heat + 10 <= 100`，冷却 10/s）⇒ 约 **1 发/仿真秒**；
  被挡下的请求记 `shots_blocked`（不生成弹丸、不进命中率）。
* 请求频率两边不同但都被热量盖住：训练 env 一个控制周期 = 1/30 s（30 次/秒），C++ 闭环 `main_fps: 20`（20 次/秒）。
* `c_block` 默认 0、`c_fire` 只对**真打出去的弹**计费 ⇒ 「一直按住 1」在训练里几乎免费，
  策略的收益只能来自**把有限热量留给好打的时机**（漏桶）。这也解释了打开 `c_block` 会训崩。
* 部署侧同理：`shooter_rl.cpp:392` 的 `fire = (p >= λ || hot) && aim_valid` 也是电平 ⇒
  **策略选的是「哪些时刻允许开火」，实际打哪几发由 p 门限 ∩ 热量门限决定。**

### 2.2 现在只学了开火，还没学输出补偿量？

**对。** 路线图（`agnet.md` §3.0）：

* **阶段 1（当前）**：RL 只输出 {开火 / 不开火}，YOLO/tracker/**经典 Aimer 全部不动**。
* 阶段 2：预测（**监督学习**，不是 RL）—— 轨迹历史 → 飞行时间后的目标位置。
* 阶段 3：RL 输出经典弹道指令上的 **yaw/pitch 修正量** + 开火。

推论（重要）：经典 Aimer 的预测里**没有竖直方向运动学**，所以轮腿升降造成的竖直误差是**系统性偏差**，
一个标定常量救不了（`pitch_offset` 那次修正是同类问题的静态版本，值 +12.5 点命中率）。

### 2.3 真机下位机会不会对连续开火请求节流？

**负责人明确：暂不需要考虑**（不建模）。目前的 `shots_blocked` 只是仿真侧热量门限的计数。

### 2.4 `run_debug.sh` 缺了「装甲板上下变化」这一环？

核实结果：**物理/画面/真值/弹道/命中判定那条本来就同步**（升降写进 `qpos`，一起变），
缺的是三处可视化与默认配置：

| # | 环节 | 状态 |
|---|---|---|
| ① | `tools/run_debug.sh` 默认不开升降、也没有开关；`--car-script` 用的 4 段预设 lift 全是 off | **未做**（负责人先只让做 ③） |
| ② | PlotJuggler 数据源没有升降通道（`sim_link.cpp` 只解析 shots/hits/gun.heat_frac/gun.locked） | **未做** |
| ③ | 浏览器状态条没有升降读数 | **已做**：显示 `车1 升降 +132mm (periodic·动/·停)`，没开升降就不显示；自检 `rm_sim/test_statusbar.py`（用 node 跑真实下发的那段 JS） |

### 2.5 「装甲板会嵌地上，需要注意高度」+「不期望全程都在动」+ 四种靶型进基准

**已全部实现**，细节见 `agnet_logs/20261014_1140_lift_range_duty_modes.md`。要点：

* **嵌地根因（先量几何）**：靶车是单刚体，"贴地姿态"就是几何最低点落在地面（`ride_z ≈ −0.004 m`，
  网格最低点 = 轮子底面），装甲板下沿只有 **+65 mm** ⇒ **任何下行程都在往地下钻**（−150mm 埋 85mm）。
* **改法**：指令 = `bottom + A·(1+sinφ)`，行程 **0 ~ +300 mm**（半幅仍是 150，中位抬到 +150）；
  `bottom_mm` 负值会被夹回 0（自检里有防嵌地回归）。真机能降（腿收拢、轮子留地面），单刚体做不到。
* **占空比**：活动窗口 = 1~2 个完整升降（2~4 s）+ 静止窗口 2.5~5 s（都随机）⇒ 实测占空比 **0.44**，
  窗口切换处 |v| ≤ 6 mm/s。想恢复"全程都在动"：配置里 `pause_s: []`。
* **四种带升降靶型**（负责人点名）：`lift_static` 固定升降 / `lift_linear` 左右移动升降 /
  `lift_spin` 原地旋转升降 / `lift_random` 随机旋转移动升降 ⇒ 预设 **`benchmark_2026_lift`**。
  `MotionSpec.spin` 新增 `none`（不转）。
* **评测入口**：`eval_policy.sh` 会**覆盖** `SIM_ARGS`，所以新增了 `CAR_SCRIPT_PRESET` 变量：
  `CAR_SCRIPT_PRESET=benchmark_2026_lift bash sp_vision_25/sim/eval_policy.sh lift4 300 <seeds>`

### 2.6 GitHub：他说「算了」

本地三个仓库**已经提交好、随时可推**，但**没有推**、`gh` 也**未登录**：

| 仓库 | 分支/提交 | 备注 |
|---|---|---|
| 工作区（建议叫 `rm-vision`） | `main` / `150cbef` | 169 文件 / 71MB；`rm_sim/assets/` 保留了（不含它仿真跑不起来） |
| `auto_aim_rl` | `main` / `1999972` | 18 个文件；`runs/` 被 .gitignore 排除（产线模型只有 8KB，建议入库） |
| `sp_vision_25` | `main` / `491db85` | 领先上游 2 个提交；要推得先 fork（`origin` 是上游只读） |

**已排除、绝不能推**：`远程服务器.md`（⚠️ **有远程训练机明文口令**，已进 `.gitignore`）、`.deps/`、
`datasets/`、`logs/`、`runs/`、`weights/`、`rules/`（官方手册 PDF）、`scratch/`、`*.onnx`。
（注：`.gitignore` **不支持行尾注释**，写成 `"/datasets/ # 注释"` 会让整个模式失效 —— 踩过。）

## 3. 事故记录：不要把负责人的进程杀掉（2026-10-14）

一次 live 探针脚本收尾写了 `pkill -9 -f "rm_[l]ink"`，**把负责人自己 10:58 启动的 `run_debug.sh` 的 rm_sim 杀掉了**
（我误以为那是自己的残留）。已清理残余、释放 8099，教训写进 `agnet_experience.md` §3.5。

**探针/自检脚本的三条铁规矩**：

1. 用**私有端口**，别抢 8099；端口上已经有别人在服务就直接退出。
2. 只 `terminate` 自己 `Popen` 出来的 pid，**永不 pkill 进程名**。
3. 读到的状态要能证明是"自己那个进程"（例如 `/state.json` 的 `t` 必须很小）——
   旧仿真实例会安静地把 `/state.json` 喂给任何客户端，不校验就会拿几分钟前的旧数据下结论。

## 4. 必须重录基线（小陀螺泄漏修复的代价）

修前：`inplace_spin`（原地甩头）段把 `GyroPlan` 建成 `hold=True` 之后**没有任何地方释放**，
下游按 `_gyro is not None` 决定是否混合角速度 ⇒ **从第 2 段起每一段都在按小陀螺转**
（`/state.json` 的 `car_motion_mode` 一直是 `gyro` 是铁证）。分靶型统计因此整体失真。

* 修法：`rm_cars.GyroPlan.release()`（回 patrol、timer 拉长），保留对象是为了让 ω 经角加速度限幅平滑交回。
* **代价**：黄金轨迹 hash `6c974ec8…` → **`c76a72ab…`**；默认 `HeadlessWorld` 就是 script 模式，所以这是**必然**变更。
* **此前所有 script 模式存档都要重录**：`benchmark_2026_legacy` / `benchmark_2026_lift`、
  RL vs sp_vision 验收对照、各次 `eval_policy`、`sim/baselines/*.json`。
  RL 训练用的镜像 env 与 C++ 闭环**共用 `rm_gimbal_demo`**，所以两边行为一直一致，修复后也一致。

## 5. 最近一轮的数据（探索性，别当基线）

`benchmark_2026_lift`，RL λ=0.69，2 种子 × 150 s，仿真帧率 30.3：

| 种子 | 发射 | 命中 | 命中率 | 击杀 |
|---|---|---|---|---|
| 20251012 | 36 | 17 | 47.2% | 0 |
| 777 | 64 | 22 | 34.4% | 0 |

分靶型：固定升降靶 60~78%（好打）· 原地旋转升降靶 0/44% · 随机旋转移动升降靶 23~25% ·
**左右移动升降靶 0~7.7%（最难）**。

读法：四种靶型端到端都通；但 150 s **一台都打不死**（≈20 命中 × 20 伤害 = 400 < 600 血）。
原因叠两层：场景变难 ⇒ λ=0.69 更少放行（36~64 发 vs legacy 159 发）；射击量从"原地甩头（好打）"
分散到平移靶。**被打到的主要是平移/提前量，不是竖直方向** ⇒ 与"阶段 2 先做预测"一致。
这个预设要配 **300 s 以上**才量得到 TTK。

## 6. 未决问题（等负责人拍板）

1. **重录基线**：300 s × 4 种子，跑 `benchmark_2026_legacy` 与 `benchmark_2026_lift`，
   并复验 RL(λ=0.69) vs sp_vision（约 40 min）。
2. `run_debug.sh` 的 `CAR_LIFT` 默认开（缺口①，几行）与 PlotJuggler 升降通道（缺口②，动 C++ 两个文件）。
3. 是否推 GitHub（可见性 / 仓库名 / 服务器 IP 是否脱敏 / 8KB 产线模型是否入库 / 提交作者身份）。
4. 升降的**占空比与行程**还是【待校准】：活动 2~4 s + 静止 2.5~5 s、行程 0~+300mm 都需要真机数据。

## 7. 关键入口（照抄能跑）

```bash
# 自检（6 个，秒级，不需要 MuJoCo 的占多数）
for t in rm_sim/test_*.py; do ~/mjlab-env/venv/bin/python $t; done
# 黄金轨迹（现在应为 c76a72ab6149211b800259e871dad902）
~/mjlab-env/venv/bin/python auto_aim_rl/scratch/golden_trace.py | grep qpos_hash
# 联调（默认不开升降；要看升降得显式给）
SIM_ARGS="--car-lift periodic" bash tools/run_debug.sh
# 四种带升降靶型的评测
CAR_SCRIPT_PRESET=benchmark_2026_lift SHOOTER=rl RL_MODE=quality \
  RL_MODEL=../auto_aim_rl/runs/quality_q.onnx P_THRESH=0.69 \
  bash sp_vision_25/sim/eval_policy.sh lift4 300 20251012 777 42 12345
# 文档引用的两张升降核对图（第 1 条会起一次仿真，约 30 s）
~/mjlab-env/venv/bin/python tools/capture_lift_frames.py 8094
~/mjlab-env/venv/bin/python tools/make_lift_figs.py
```

## 8. 事实速查（避免重新查代码）

* 靶车血量 600 / 每发 20 伤害 ⇒ 30 发一台；小装甲板命中面 141×130 mm。
* 枪：摩擦轮弹速 22→24 m/s（τ_warm=20 发）、热量 100/10 每发/冷却 10 每秒、散布 0.15°/轴。
* 评测协议：主循环限速 `sim.main_fps: 20`；**帧率必须随结果一起记录**（帧率一飘数字就飘）。
* 坐标：`command.pitch = -(elevation)`；仿真关节 pitch = 安装角 25° + command.pitch；
  `R_camera2gimbal` 见 `sp_vision_25/sim/README.md` §3。
* 配置文件：`rm_sim/configs/`（rules / projectiles / armor / targets + presets），默认规则集 `legacy_sim`。
* 轮腿升降参数在 `rm_sim/configs/targets.yaml` 的 `infantry_wheel_legged.body_lift`。