# tools/dual_machine/ —— 双机部署与运维（2026-10-15 起）

从 2026-10-15 起，开发机从"一台 Ubuntu（`/home/zieng`）"换成**两台机器 + Mac 当控制端**。
本目录是这套环境的**唯一入口文档**，脚本都是"在 Mac 上执行、作用于两台机器"。

## 1. 两台机器是谁

| 别名 | 地址 | 用户 | 硬件 | 系统 | 角色（**2026-10-15 负责人拍板 = A 拆分**） |
|---|---|---|---|---|---|
| `x86` | 192.168.1.236 | `clgx` | Intel Core Ultra 9 285H（16 线程）/ 30 GiB / 916 G NVMe / **无可用 GPU** | Ubuntu 22.04.5，内核 6.8.0-40 | **视觉闭环（sp_vision，已编译）+ 训练** |
| `jetson` | 192.168.1.101 | `nvidia` | Jetson Orin NX（8 核 A78AE）/ 15 GiB / 233 G NVMe / **Orin GPU + CUDA 12.6** | L4T R36.4.3（JetPack 6.2），aarch64 | **仿真机（rm_sim，EGL 20~24 FPS）**（也能训练，4 环境 ~89 步/s） |

> 两者之间走**局域网**传 `/grab`（图像+姿态）与 `/command`。**已于 2026-10-15 接通网线直连**：
> 实测 ping **1.13 ms（抖动 0.27）**、`/grab` **1.3~1.65 ms**，跨机闭环主循环 **19.8 FPS**（Large dt 0 次）。
> 对比 Wi-Fi：`/grab` 中位 40 ms、抖动到 80 ms ⇒ 主循环只有 4.7 FPS（数字不可比）。
> 配线流程（**必须先 prepare 再插线**，否则会把两台机器的 Wi-Fi 打掉，见脚本头部说明）：
> ```bash
> bash tools/dual_machine/setup_wired_link.sh prepare   # ① 先写静态配置（禁用自动 DHCP profile）
> #   —— 这时才插线：x86 enp87s0 ↔ jetson enP8p1s0 ——
> bash tools/dual_machine/setup_wired_link.sh status    # ② 查链路/地址（应见 10.10.10.1/24、10.10.10.2/24）
> bash tools/dual_machine/setup_wired_link.sh verify    # ③ ping + /grab 延迟自检
> ```

> Mac（192.168.1.203）是控制端：ssh 别名写在 `~/.ssh/config`，密钥 `~/.ssh/id_ed25519_codex`。
> 账号口令只存在仓库根的 `远程服务器.md`（已 gitignore，**永不入库**）。

## 2. 已装好的环境（现状）

两台都装了：`~/vision/` 代码树 + `~/mjlab-env/venv`（python3.10）。

| 项 | x86 | Jetson |
|---|---|---|
| venv | `~/mjlab-env/venv`（mujoco 3.11.0 / numpy 2.2.6 / cv2 5.0 / viser+mjviser / onnxruntime 1.23） | `~/mjlab-env/venv`（mujoco 3.11.0 / numpy / gymnasium / **torch 2.8.0+cu126** / onnx+onnxruntime / sklearn / matplotlib） |
| 系统依赖 | apt 全套（cmake/opencv4/ceres/eigen/fmt/spdlog/yaml-cpp/json/usb/can-utils + mesa/EGL/osmesa） | 系统自带 OpenCV 4.5.4 + **CUDA 12.6 toolkit + cuDNN 9**（apt 装的） |
| 推理运行时 | **OpenVINO 2024.6.0**（`/opt/intel/openvino_2024.6.0`）+ **ONNX Runtime 1.31.0 CPU**（`~/opt/onnxruntime-linux-x64-1.31.0`） | 无 OpenVINO（官方无 aarch64 Linux 包）；ORT 用 python 侧 |
| C++ | `sp_vision_25` **已编译通过**（`sim_main`/`detect_probe`/`sim_geom_check`） | 未编译（跨到 aarch64 需要移植，见 §5） |

配置 CMake 时 ONNX Runtime 路径与仓库默认不同，需显式给一次（之后进 cache）：
```bash
cmake -B build -DONNXRUNTIME_ROOT=$HOME/opt/onnxruntime-linux-x64-1.31.0
```

## 3. 三条铁律（踩过，别再犯）

1. **`mujoco` 必须钉 `3.11.0`**。装 3.16.0 时黄金轨迹 hash 变成 `7fa778ba…`（错），钉 3.11.0 才是
   `c76a72ab6149211b800259e871dad902`（对）。版本一漂，物理与全部基线都不再可复现。
2. **渲染后端按机器设**：x86 无 `/dev/dri`（Arrow Lake-H 核显在内核 6.8 下无驱动）→ `MUJOCO_GL=osmesa`；
   Jetson → `MUJOCO_GL=egl`（退出时 `EGL...free` 的报错无害，渲染是好的）。
2b. **Jetson 上 `numpy` 必须 `<2`**（用 `numpy==1.26.4`）：jetson-ai-lab 的 torch 轮子是按 NumPy 1.x 编译的，
   配 NumPy 2.x 会 `Failed to initialize NumPy`（torch↔numpy 互转不可靠）。x86 侧 NumPy 2.2.6 已验证可用。
3. **脚本里不要写死用户名/家目录**（`/home/zieng/...` 已全部改成相对脚本自身推算）。
   另：`rsync --exclude 'rules/'` 会**误伤** `rm_sim/configs/rules/`，会撞出
   `找不到配置文件 .../configs/rules/legacy_sim.yaml`。
4. **rsync 的远端路径不要用 `~`**：`rsync -a x/ host:~/vision/x/` 在远端不会展开 `~`
   ⇒ 静默不生效（本次"同步完成"却查不到文件）。写**绝对路径**：`host:/home/<user>/vision/x/`。

## 4. 常用命令

```bash
# 连通性
ssh x86 'hostname'; ssh jetson 'hostname'

# 自检（x86，6 个全绿）+ 黄金轨迹（应打印 c76a72ab6149211b800259e871dad902）
B64=$(base64 < tools/dual_machine/x86_selftest.sh | tr -d '\n')
ssh x86 "echo $B64 | base64 -d > /tmp/x86_selftest.sh; bash /tmp/x86_selftest.sh"

# 起仿真（x86 软件渲染；浏览器看 http://192.168.1.236:8099）
ssh x86 'cd ~/vision/rm_sim && MUJOCO_GL=osmesa ~/mjlab-env/venv/bin/python rm_link.py --model real --port 8099 --host 0.0.0.0'

# 起仿真（Jetson GPU 渲染，20.2 FPS）
ssh jetson 'cd ~/vision/rm_sim && MUJOCO_GL=egl ~/mjlab-env/venv/bin/python rm_link.py --model real --port 8099 --host 0.0.0.0'
```

> `--host 0.0.0.0` 是 2026-10-15 新加的（`rm_sim/rm_link.py`）；默认仍是 `127.0.0.1`，不改变既有行为。

## 5. 本目录脚本

| 脚本 | 用途 |
|---|---|
| `ssh_pw.exp` | 非交互密码 ssh（口令读自 `远程服务器.md`，用于首次装 key） |
| `ssh_sudo.exp` | 需要 sudo 的远端命令（`ssh -tt` + 自动回答口令） |
| `recon.sh` / `recon2.sh` / `recon3.sh` | 三轮侦察：硬件/OS/网络、渲染与构建工具链、CUDA/TRT/apt 源 |
| `x86_setup.sh` | x86：等 apt 完 → 建 venv → 装 python 依赖 → 自检 |
| `jetson_setup.sh` | Jetson：建 venv → torch（jetson-ai-lab cu126）→ 其余依赖 → 自检 |
| `x86_selftest.sh` | x86：跑 `rm_sim/test_*.py` 全量 + 黄金轨迹复验 |
| `x86_intra_test.sh` | 量 `sim_ort_intra_threads` 对单帧推理耗时的影响 |
| `setup_wired_link.sh` | 配 x86 ↔ Jetson 网线直连（静态 IP，幂等；`PERSIST=1` 写进 NetworkManager） |
| `train_sweep.sh` | 训练吞吐扫描（换并行环境数，量 steps/s；**在目标机上执行**，不是 Mac） |

## 5.1 跨机的三个开关（默认行为不变）

| 开关 | 作用 |
|---|---|
| `rm_link.py --host 0.0.0.0` | 仿真监听所有网卡（默认只 127.0.0.1） |
| `SIM_BASE_URL=http://<仿真机IP>:8099` | 环境变量覆盖 `sim.base_url`（`sim_link.cpp`），不用改 yaml |
| `SIM_REMOTE=<ssh 别名>` | `bench_ttk.sh`（以及走它的 `eval_policy.sh`/`acceptance.sh`）用 ssh 把 rm_link 起在远端并自动指向它 |

```bash
# x86 上跑跨机 TTK 基准（需 x86 → jetson 的 ssh 免密，已配好）
ssh x86 'cd ~/vision && SIM_REMOTE=jetson SIM_BASE_URL=http://10.10.10.2:8099 \
  bash sp_vision_25/sim/bench_ttk.sh 300 20251012'
```

投递方式（避免引号地狱）：`B64=$(base64 < 脚本 | tr -d '\n'); ssh <host> "echo $B64 | base64 -d > /tmp/x.sh; bash /tmp/x.sh"`

## 6. 已知硬约束（2026-10-15 实测，决定架构）

| 指标 | x86 | Jetson |
|---|---|---|
| rm_sim 真实 tick 帧率（含渲染） | **10.3 FPS**（osmesa；关阴影 14） | **20.2 FPS**（EGL） |
| 训练 env 步速（无渲染） | **58 步/s（1.9× 实时）** | 32 步/s（1.1× 实时） |
| `sim_ort` 单帧推理 | **19.4 ms**（跨机闭环实测，intra=12） | 未测（CPU 会更慢） |
| 跨机 `/grab`（**有线**，已接通） | **1.3~1.65 ms，零抖动** | 同 |
| 跨机 `/grab`（Wi-Fi） | 25~81 ms（中位 40 ms，抖动大） | 同 |
| 跨机闭环主循环 | **19.8 FPS**（有线）｜4.7 FPS（Wi-Fi） | — |
| CUDA 渲染 640×480 | 无 GPU | 21.3 ms/帧（47 FPS） |

* **x86 单机跑不了 20 FPS 闭环**：软件渲染 8~10 FPS，而 `sim.main_fps: 20` 是评测口径（帧率不控住数字不成立）。
* x86 的 PCIe x8 槽里是 Crucial NVMe 转接卡 + 迷你主机电源 ⇒ 插 30HX 不现实。
* 22.04 仓库里**没有比 6.8 更新的内核**（`-edge` 也是 6.8）⇒ 想点亮 Arc 核显得手工装 mainline 或升 24.04（要重启）。
* `sp_vision_25/sim/sim_link.cpp` 的 HTTP 客户端在跨机分片时会拿到**截断的 body**
  （报 `JSON 解析失败 ... missing closing quote`）⇒ 跨机方案必须先把它改成按 `Content-Length` 收全 + 超时重试。

## 7. 训练（PPO）在哪跑

**2026-10-15 结论：跑在 x86 小主机上**（旧远程 T4 退役）。

```bash
bash tools/remote_train.sh <run名> [步数] [环境数] [额外参数]     # 默认 TRAIN_HOST=x86
TRAIN_HOST=jetson bash tools/remote_train.sh stage1_a 200000 8
SYNC=1 bash tools/remote_train.sh stage1_a 200000 16             # 先把 Mac 上的代码 rsync 过去
```

实测（`--steps 8192 --rollout 256 --episode-s 90 --randomize`，步速含 PPO 更新）：

| 并行环境数 | x86（16 线程） | Jetson（8 核） |
|---|---|---|
| 4 | 198 步/s | 95 步/s |
| 8 | 392 步/s | **154 步/s**（饱和） |
| 12 | 554 步/s | 149 步/s |
| 16 | 509 步/s | 148 步/s |
| 32 | **630 步/s** | — |

* x86 是 Jetson 的 **约 4 倍**，也是旧远程 T4（cgroup 8 核、230 步/s）的 **2.7 倍**。
* **互斥约束**：训练吃满 16 线程 ⇒ 别和"x86 上跑闭环评测"同时进行；
  必须在同一台同时跑时，把 envs 压到 4~6。
* 两台都要有训练依赖（`gymnasium` + torch；x86 用 CPU 版 torch，见 `x86_setup.sh`）。
* 完整记录：`agnet_logs/20261015_1342_train_on_new_machines.md`。
