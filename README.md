# RM 视觉自瞄：仿真环境 + 学习型开火决策

RoboMaster 自瞄（auto-aim）方向的开发工作区：**MuJoCo 仿真环境** + 与同济
[sp_vision_25](https://github.com/TongjiSuperPower/sp_vision_25) 的闭环联调 +
**学习型开火时机**（RL / 监督学习）。

一句话目标：**在仿真里把「识别 → 跟踪 → 预测 → 开火」整条链路跑通并量化**
（命中率 + 击杀时间 TTK），再上真机。

## 仓库布局（三个仓库并列 clone 到同一个父目录）

```
vision/
├── sp_vision_25/      # 同济自瞄框架（我们的 fork：加了 sim/ 闭环联调 + ShooterRL）
├── auto_aim_rl/       # 学习型开火决策（env / 训练 / 离线拟合 / 评测）
└── rm-vision/         # ← 本仓库：仿真环境、工具、文档与实验日志
    ├── rm_sim/        # MuJoCo 仿真：真实车模型 + 弹道 + 靶车机动 + 枪/热量模型
    ├── tools/         # 一键联调、评测聚合、远程训练、画图
    ├── agnet.md       # 项目说明书（现状 / 路线图 / 决策 / 测试规范）
    ├── agnet_experience.md  # 经验与诚实清单（仿真与现实的差距）
    └── agnet_logs/    # 逐次开发日志（每次会话一份）
```

> 脚本里的路径是**相对工作区**的（例如 `rm_sim` 与 `sp_vision_25` 是兄弟目录），
> 所以三者要放在同一个父目录下。

## 现在的部署形态（2026-10-15 起：双机 + Mac 控制端）

旧的单机 Ubuntu（12 线程 + 30HX）已退役。**现行**环境是两台机器，Mac 当控制端。
机器清单、路径、铁律、运维命令的完整说明见
**[`tools/dual_machine/README.md`](tools/dual_machine/README.md)**：

| 机器 | 地址 | 跑什么 |
|---|---|---|
| **Jetson Orin NX**（`ssh jetson`） | 192.168.1.101 | **`rm_sim` 仿真机**：MuJoCo 物理 + GPU 渲染（EGL，20~24 FPS） |
| **x86 小主机**（`ssh x86`） | 192.168.1.236 | **`sp_vision` 视觉闭环**（已编译、CPU 推理）+ 训练候选（env 58 步/s） |
| Mac（本机） | 192.168.1.203 | 代码 / 文档 / 评测的**控制端**（`~/.ssh/config` 有 `x86`、`jetson` 别名） |

为什么这样分：x86 的 Intel Arc 核显在内核 6.8 下没有驱动（`i915`/`xe` 都不加载、无 `/dev/dri`）
⇒ 只能软件渲染，闭环 8~10 FPS，达不到 `sim.main_fps: 20` 的评测口径；而 Jetson 的 CPU 只有
1.1× 实时，只够自己跑物理 + 渲染。两者之间传 `/grab` 与 `/command`，**建议网线直连**
（Wi-Fi 实测 `/grab` 中位 40 ms、抖动到 80 ms ⇒ 主循环掉到 4.7 FPS；直连亚毫秒）。

**跨机跑闭环**（仿真在 Jetson、视觉在 x86）：

```bash
ssh x86 'cd ~/vision && SIM_REMOTE=jetson SIM_BASE_URL=http://192.168.1.101:8099 \
  bash sp_vision_25/sim/bench_ttk.sh 300 20251012'
```

## 快速开始（单机自检）

```bash
git clone <本仓库>            rm-vision
git clone <sp_vision_25 fork> sp_vision_25
git clone <auto_aim_rl>       auto_aim_rl
mv rm-vision vision && cd vision

# 1) 只跑仿真（不需要视觉框架）
~/mjlab-env/venv/bin/python rm_sim/rm_link.py --model real --port 8099 --host 0.0.0.0
#    浏览器打开 http://127.0.0.1:8099 看全局视角 + 状态条（跨机时用 http://<本机IP>:8099）

# 2) 一键联调（仿真 + sp_vision + PlotJuggler + 浏览器）
bash tools/run_debug.sh

# 3) 自检（不需要 MuJoCo，秒级）
for t in rm_sim/test_*.py; do ~/mjlab-env/venv/bin/python $t; done
```

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