> **2026-10-15 起：环境是双机的。** 本文里的 `run_debug.sh` 仍是"本机一键起 rm_sim + sp_vision"，
> 但现行部署里 **rm_sim 跑在 Jetson、sp_vision 跑在 x86**；跨机联调/评测请看
> [`dual_machine/README.md`](dual_machine/README.md)（`SIM_REMOTE` + `SIM_BASE_URL` + 网线直连）。

# tools/ — 联调小工具

## run_debug.sh（一键联调）

一条命令把整套东西拉起来：

```bash
bash tools/run_debug.sh
```

它会依次做四件事：

| 步骤 | 内容 |
|---|---|
| 1 | 起 **rm_sim**（`rm_sim/rm_link.py`，MuJoCo 物理 + 渲染 + `/grab`、`/command` 接口） |
| 2 | 打开**浏览器**到 `http://127.0.0.1:8099` —— 仿真相机原图（MJPEG） |
| 3 | 起 **PlotJuggler** —— UDP 9870 收云台跟踪曲线 |
| 4 | 起 **sp_vision**（`sim_main`）—— **无头机器上**开 `--show_http=8100`，浏览器看 YOLO 识别效果；有 DISPLAY 时才用 `--show` 弹 OpenCV 窗口 |

**退出**：在 `detection` 窗口按 `q`（会顺带把 rm_sim / PlotJuggler 一起收干净）。

### 参数

```bash
bash tools/run_debug.sh --gt           # 用真值识别，隔离检测器（排查坐标系/跟踪问题时用）
bash tools/run_debug.sh --no-plot      # 不起 PlotJuggler
bash tools/run_debug.sh --no-browser   # 不开浏览器
bash tools/run_debug.sh --seconds 30   # 跑 30 秒自动退出（默认一直跑）
bash tools/run_debug.sh --port 8100    # 换端口
```

### 三块画面分别看什么

> **双机部署下看画面的地址、IP 怎么选**见 [`dual_machine/README.md`](dual_machine/README.md) §4.1
> （全局视角 / 云台原图在 Jetson 的 :8099 页面；YOLO 识别效果在 x86 的 :8100）。

* **浏览器（rm_sim 页面）**：上面一块是**全局固定视角**（靶车运动 + 命中情况），
  下面一块是**云台相机原图** —— 也就是 sp_vision 的输入。判断"目标在不在视野里、灯条亮不亮"看这个。
* **YOLO 识别效果（`sim_main --show_http=8100`）**：sp_vision 的调试视图。
  绿框是四点、绿字是 `置信度 类别 大小`，黄框是**预测瞄点**反投影回图像的框（开火时刻该打的那块）。
  判断"识别漏没漏、框准不准、瞄点对不对"看这个。
  它同时支持 MJPEG（`/debug.mjpg`）与单帧（`/debug.jpg`），后者方便 `curl` 抓图留证。
  **有 DISPLAY 的机器**上也可以继续用 `--show` 弹窗口 —— 画的是同一张图。
* **PlotJuggler**：云台跟踪曲线。判断"跟得上跟不上"看这个。

---

## 训练（PPO）——`remote_train.sh`

**2026-10-15 起训练跑在 x86 小主机上**（旧远程 T4 已退役；不要放在 Mac 上跑，
MuJoCo 不释放 GIL、8 worker 就能把机器压死——历史事故整机崩过两次）。

```bash
bash tools/remote_train.sh <run名> [步数] [并行环境数] [额外参数]   # 默认 x86
TRAIN_HOST=jetson bash tools/remote_train.sh stage1_a 200000 8     # 切 Jetson
SYNC=1 bash tools/remote_train.sh stage1_a 200000 16               # 先 rsync 代码过去
EXTRA_ARGS="--c-fire 0.15" bash tools/remote_train.sh ...
```

* 默认并行环境数按机器给：x86 → 12（16 线程），Jetson → 6（8 核）。
* 产物在 `<host>:~/vision/auto_aim_rl/runs/<run>/`：`train.log`、`log.jsonl`、`policy.pt`。
* 实测吞吐：x86 **509 步/s @16 envs**（630 @32）、Jetson 154 步/s @8 envs（饱和）。
  **训练会吃满 x86 的 16 线程 ⇒ 不要和"x86 跑 sp_vision 闭环"同时进行**（要同时就把 envs 压到 4~6）。
* 细节与测法见 `agnet_logs/20261015_1342_train_on_new_machines.md`。

---

## 仿真域检测器流水线（自训 + ONNX Runtime）

现役 `assets/yolov5.xml` 是真实比赛数据训的，喂仿真画面只有 43% 召回。
下面这套东西用 rm_sim 的自动真值训了一个仿真域检测器，召回做到 99.4%，
并且把推理搬到了 30HX（仿真帧率 20.6 -> 25.8 FPS）。
完整背景和实测数据见 `sp_vision_25/sim/README.md` 第 7、8 节。

| 脚本 | 干什么 |
|---|---|
| `tools/sim_detector_pipeline.sh` | **一键全流程**：生成数据集 -> 训练 -> 导出 ONNX -> 同口径对比 |
| `tools/train_sim_detector.sh` | 在 30HX 上微调 yolo11n-pose（长跑建议 nohup） |
| `tools/export_sim_detector.py` | 导出 `assets/sim_armor.onnx` + Python 侧召回评测 |
| `tools/kpt_accuracy.py` | 角点精度诊断（框的 IoU 好 ≠ 角点好，角点才是喂 solvePnP 的） |
| `tools/ort_env.sh` | **必须 source**：把 venv 里的 cublas/cudnn 加进 `LD_LIBRARY_PATH`，否则 CUDA EP 加载失败 |
| `rm_sim/gen_dataset.py` | 从仿真自动标注数据集（云台跟车采集，不吃瞎扫的废帧） |

```bash
# 全流程（数据集 8000 帧 + 60 epoch，约 3 小时）
nohup bash tools/sim_detector_pipeline.sh > /tmp/pipeline.log 2>&1 &

# 只想看检测器对比（需要先编好 build/sim/detect_probe）
source tools/ort_env.sh
bash sp_vision_25/sim/compare_detectors.sh /tmp/eval_frames
```

`configs/sim.yaml` 现在默认就是 `yolo_name: sim_ort` + `device: CUDA`，
所以 `run_debug.sh` 起来跑的已经是自训模型。想换回原路径：
`yolo_name: yolov5` / `device: CPU`。

判断有没有真的跑在 GPU 上，看 sp_vision 日志里这一行（每 300 帧一次）：

    sim_ort 推理 10.98 ms/帧（CUDAExecutionProvider，1500 帧滑动均值）

CUDA 大约 11 ms，CPU 大约 33 ms —— 数量级一眼可辨。

---

## PlotJuggler 配置（已实测跑通）

**已装好**：`sudo snap install plotjuggler` → 3.17.2（snap 发布者就是作者本人）。

**已验证可用**：配好之后 PlotJuggler 的 Timeseries List 会实时刷出 13 个字段
（`err_yaw` / `gimbal_yaw` / `cmd_yaw` / `target_w` …），把字段拖进绘图区就能看曲线。

### 每个会话要做一次（约 4 次点击）

`run_debug.sh` 会把 PlotJuggler 拉起来，但它**不会自动启动 UDP 数据源**（见下方说明），
所以需要手动点一下：

1. 左侧 **Streaming** 那一行的下拉框 → 选 **UDP Server**
2. 点下拉框右边的 **⚙** → 弹出 "UDP Server Settings"
   * **Address**: `127.0.0.1`
   * **Port**: `9870`
   * **Message Protocol**: **json**（默认是 `Influx (Line protocol)`，必须改！）
   * 点 **OK**
3. 点 **Start**
   > ⚠️ 实测 **鼠标点 Start 有时不生效**；先点一下 Start 让它获得焦点，再按 **回车**，就一定会起来。
   > 起来后按钮会变成 **Stop**，日志里出现 `"IPv4 UDP listening on (127.0.0.1, 9870)"`。
4. 把 Timeseries List 里的字段拖到右边绘图区（`err_yaw` 最值得先看）

### 协议必须两边一致（最容易踩的坑）

sp_vision 的遥测默认发 **JSON**，所以 PlotJuggler 里的 **Message Protocol 必须选 `json`**。
两边不一致时 PlotJuggler 不是"没数据"那么客气，而是**直接报错并把 UDP Server 停掉**：

```
Problem parsing the message. UDP Server will be stopped.
[json.exception.parse_error.101] ... invalid literal; last read: 'a'
```

> 看到 `last read: 'a'` 就是收到了 `auto_aim ...` 开头的 **Influx** 行 ——
> 说明发的是 Influx 而协议选的是 json。

反过来也能配：把 `configs/sim.yaml` 的 `sim.plot_format` 改成 `influx`，
遥测就变成 Influx line protocol，匹配 PlotJuggler 的**默认协议**（不用开设置对话框）。
**两种格式不能同时发** —— 另一种会解析失败并让服务器停掉。

### 关于 layout 自动加载

`Layout: ⬆` 图标 → Save As → 存到 `tools/plotjuggler_layout.xml`。
`run_debug.sh` 检测到该文件会用 `plotjuggler -n -l <layout> -p` 启动。

> **实测结论（别抱太大期望）**：这个 layout 能恢复**绘图区布局**，
> 但**不会自动把 UDP 数据源拉起来**（`-p` 也没用）—— PlotJuggler 3.17 只把
> "当前选中的 streamer" 记进 layout，不记它的启动状态。
> 所以**每次还是要手动点一次 Start**。上面那 4 步就是当前能达到的最短路径。

### 曲线字段（`sim_main` 发布的内容）

| 字段 | 含义 |
|---|---|
| `cmd_yaw` / `cmd_pitch` | sp_vision 算出来的**指令角**（世界系绝对角，弧度） |
| `gimbal_yaw` / `gimbal_pitch` | 云台**实际**姿态角（由仿真四元数解出） |
| `err_yaw` | `cmd_yaw - gimbal_yaw`，**跟踪误差**，最该看的一条 |
| `fire` | 开火指令 |
| `control` | 是否在控制云台 |
| `armor_num` / `target_num` | 本帧识别到的装甲板数 / 跟踪到的目标数 |
| `target_w` | 目标自转角速度（rad/s），判断"是不是小陀螺" |
| `target_x` / `target_y` | 目标在云台系下的位置 |
| `t` | 仿真时间 |

**注意**：`cmd_pitch` 和 `gimbal_pitch` **口径不同**，不能直接相减看误差 ——
指令是"世界系仰角"，而云台 pitch 关节的零位是炮管安装角（+25°）。
所以只给了 `err_yaw`；要看 pitch 误差请用 PlotJuggler 的表达式功能算
`cmd_pitch - (gimbal_pitch - 0.4363)`。

---

## 排查思路

| 现象 | 先看哪 |
|---|---|
| 打不中，但识别正常 | PlotJuggler 的 `err_yaw`：持续偏大 = 云台跟不上；只在开火瞬间抖 = 开火门限太松 |
| 识别漏 | `detection` 窗口：是不是板侧对了（本来就看不到），还是灯条太暗（调 `rm_appearance.py`） |
| 目标老丢 | `target_num` 频繁 0/1 跳变 = 相邻帧识别不稳定；配合 `--gt` 对比，如果 `--gt` 下也丢就是 tracker 问题 |
| 瞄准角乱跳 | 用 `--gt` 跑一遍。`--gt` 下正常 = 检测器给的框有问题；`--gt` 下也跳 = 坐标系问题（跑 `sim_geom_check`） |

---

## 相关文件

| 文件 | 说明 |
|---|---|
| `tools/run_debug.sh` | 本工具 |
| `rm_sim/rm_link.py` | 仿真侧双向接口（`/grab` `/state.json` `/command` `/stream.mjpg`） |
| `rm_sim/rm_appearance.py` | 渲染外观预设（`realistic` / `original`） |
| `sp_vision_25/sim/sim_main.cpp` | 闭环主程序（`--show` 开 debug 窗口） |
| `sp_vision_25/sim/sim_geom_check.cpp` | 坐标系自检 |
| `sp_vision_25/sim/detect_probe.cpp` | 识别召回探针 |
| `sp_vision_25/sim/README.md` | 集成细节与坐标系约定（踩过的坑都在那） |