# RM 视觉自瞄：仿真环境 + 学习型开火决策

RoboMaster 自瞄（auto-aim）方向的开发工作区：**MuJoCo 仿真环境** + 与同济
[sp_vision_25](https://github.com/TongjiSuperPower/sp_vision_25) 的闭环联调 +
**学习型开火时机**（RL / 监督学习）。

一句话目标：**在仿真里把「识别 → 跟踪 → 预测 → 开火」整条链路跑通并量化**
（命中率 + 击杀时间 TTK），再上真机。

## 部署：两条命令

前置：Linux（Ubuntu 22.04+ 实测）/ macOS，`python3.10+`，能访问 PyPI。
**两个仓库要并列放在同一个父目录** —— env 直接 import `rm_sim`，脚本也按兄弟目录找对方：

```
ws/
├── vision/           # ← 本仓库（仿真 + 工具 + 文档）
└── auto_aim_rl/      # ← 另一个仓库（学习型开火决策；只跑仿真可以不 clone）
```

```bash
mkdir -p ~/ws && cd ~/ws
git clone <本仓库> vision
git clone <auto_aim_rl 仓库> auto_aim_rl

bash vision/deploy/deploy_rm_sim.sh      # ① 仿真运行环境 + 自检（8 个测试全绿）
bash auto_aim_rl/deploy_env.sh           # ② 训练/评测环境 + 自检（含黄金轨迹）
```

两条脚本都**幂等**（重复跑只补缺的），都装进同一个 venv（默认 `~/mjlab-env/venv`，
`VENV=` 可换），并自动完成：探测渲染后端 → 建 venv → 装依赖 → 跑仓库自带自检 → 打印用法。

三个**不能改的钉子**（脚本已处理，手工装环境时注意）：

| 项 | 版本 | 为什么 |
|---|---|---|
| `mujoco` | **3.11.0**（脚本已钉） | 装 3.16 会让黄金轨迹 hash 从 `c76a72ab…` 变成 `7fa778ba…` |
| `numpy` | Jetson(aarch64) 必须 `<2` | jetson-ai-lab 的 torch 轮子按 NumPy 1.x 编译 |
| `torch` | x86 走 **CPU 轮子索引**；Jetson 走 jetson-ai-lab | x86 上装 PyPI 默认 torch 会拖 2.5 GB CUDA 依赖 |

**无头机器（没有显示器）**：MuJoCo 渲染需要 GL 后端，脚本会自己选 `MUJOCO_GL`
（Jetson→EGL、有核显→GLFW、都没有→软件渲染 osmesa）。软件渲染只有 ~10 FPS，
要实时性就把渲染放到有 GPU 的机器上（见「跨机跑闭环」）。

## 快速开始（装完环境之后）

```bash
V=~/mjlab-env/venv/bin

# 仿真 + 浏览器看画面（上面全局视角、下面云台相机原图）
$V/python rm_sim/rm_link.py --model real --port 8099 --host 0.0.0.0

# 交互演示：拖准星控云台、点击开火
$V/python rm_sim/rm_gimbal_demo.py --model real       # 浏览器 http://localhost:8080

# 自检（不需要 GPU、不需要可视窗口）
for t in rm_sim/test_*.py; do $V/python $t; done
$V/python auto_aim_rl/env/test_env.py
$V/python auto_aim_rl/scratch/golden_trace.py check   # 打印黄金轨迹 hash

# 训练（小步数冒烟）
$V/python auto_aim_rl/train/ppo.py --steps 20000 --envs 4 --out auto_aim_rl/runs/smoke
```

## 仓库里有什么 / 没有什么

| 有 | 说明 |
|---|---|
| `rm_sim/` | MuJoCo 仿真：真实车模型 + 弹道气动 + 靶车机动 + 枪/热量模型 + 评测协议 |
| `deploy/` | **一键部署脚本**（`deploy_rm_sim.sh`；`fetch_real_model.sh` 从源重建车模资产） |
| `rm_sim/assets/` | 车模资产（OBJ + MJCF + 贴图，由上游 GLB 转换而来，见「许可与致谢」） |
| `tools/` | 联调、评测聚合、远程训练、画图等开发工具 |
| `agnet*.md`、`agnet_logs/` | 项目说明书、经验与差距清单、逐次开发日志 |

| 没有（设计如此） | 怎么获得 |
|---|---|
| `sp_vision_25/`（同济自瞄框架） | 团队上游仓库。本仓库只提供**仿真侧接口**（见下），用标准 HTTP，谁都能接自己的视觉栈 |
| `datasets/`、`runs/`、`weights/` | 数据集与训练产物，可重跑得到（`.gitignore` 已排除） |
| `vehicle.glb`（源车模，28 MB，AGPL-3.0） | 转好的资产随仓库发布；想从源重建：`bash deploy/fetch_real_model.sh` |
| 凭据、比赛规则手册 PDF | 前者不许入库，后者请从官网下载 |

## 与 sp_vision 闭环（可选）

仿真是"相机 + 下位机"的替身，走 HTTP 四件事：**取图 + 同帧云台四元数**（`/grab`、
`/state.json`）、**下发云台绝对角与开火**（`POST /command`）、**人看的画面**
（`/stream.mjpg` 云台相机、`/overview.mjpg` 全局视角）。接口细节见
[`rm_sim/README.md`](rm_sim/README.md) §7。

> 无头机器上看"识别框 + 预测瞄点"那块画面：C++ 侧 `sim_main --show_http=8100`，
> 浏览器开 `http://<那台机器>:8100` —— 等价于原来的 OpenCV 调试窗口。

## 跨机跑闭环（可选的部署形态）

旧的单机 Ubuntu（12 线程 + 30HX）已退役。**现行**开发环境是两台机器 + 一台 Mac 当控制端
（机器清单、路径、铁律、运维命令见 [`tools/dual_machine/README.md`](tools/dual_machine/README.md)）。
这样分的原因值得记下来：

| 机器 | 跑什么 | 为什么 |
|---|---|---|
| **Jetson Orin NX** | `rm_sim` 仿真机 | 有 GPU：MuJoCo EGL 渲染 20~24 FPS（纯 CPU 只有 1.1× 实时） |
| **x86 小主机** | 视觉闭环 + 训练 | CPU 强（训练 env 509 步/s @16 envs），但核显无驱动 ⇒ 渲染只有软件 8~10 FPS，达不到 20 FPS 评测口径 |

两者之间传 `/grab`（取图）与 `/command`（指令），**建议网线直连**：Wi-Fi 实测 `/grab`
中位 40 ms、抖动到 80 ms ⇒ 主循环掉到 4.7 FPS；直连亚毫秒。跨机开关见
`tools/dual_machine/README.md`（`SIM_REMOTE`、`SIM_BASE_URL`、`rm_link.py --host`）。

> 只跑仿真 + 训练**不需要**这套跨机形态：一台机器就够（有 GPU 更好）。

## 这个仿真里已经建模了什么

| 项 | 内容 |
|---|---|
| 真实车模型 | 源模型转 MuJoCo（几何/枢轴/炮口/贴纸全部实测校对） |
| 弹道 | 重力 + 弹速模型（摩擦轮 22→24 m/s 升温曲线）+ 枪口散布 |
| 枪口热量 | 漏桶模型：每发 10、冷却 10/s、硬锁枪（决定可持续射频 ≈1 发/s） |
| 靶车机动 | 巡逻/急停冲刺 + 小陀螺（变速甩头）+ **轮腿车体升降** |
| 我方底盘 | 颠簸 / 视轴扰动 / IMU 噪声（可开关） |
| 评测协议 | 固定靶型脚本 + 逐靶型统计、TTK、每击杀弹量（口径写死在文档里） |

细节见 [`rm_sim/README.md`](rm_sim/README.md) 与 [`rm_sim/configs/README.md`](rm_sim/configs/README.md)。

## 文档怎么读

1. [`agnet.md`](agnet.md) —— 现状快照、三阶段路线图、已拍板决策、开发规范。**先看这个。**
2. [`agnet_handover.md`](agnet_handover.md) —— **交接说明**：最近一轮对话的结论、事故记录、未决问题（给接手的人/AI）。
3. [`agnet_experience.md`](agnet_experience.md) —— 参考数据、指标口径、仿真与现实的差距清单、踩坑经验。
4. [`agnet_logs/`](agnet_logs/) —— 每次开发会话的日志（做了什么/为什么/数据/坑/下一步）。
5. [`DEPLOY.md`](DEPLOY.md) —— 环境搭建（OpenVINO / ONNX Runtime / MuJoCo）。

## 许可与致谢

* 自瞄框架来自 [TongjiSuperPower/sp_vision_25](https://github.com/TongjiSuperPower/sp_vision_25)（同济 SuperPower）。
* 仿真环境基于 MuJoCo；规则数值引自 RoboMaster 官方比赛规则手册（PDF 不入库，请从官网下载）。