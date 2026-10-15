# 2026-10-15 12:50 双机环境部署（x86 小主机 + Jetson Orin NX）

> 目标：把开发环境从旧的 Ubuntu（`/home/zieng`，12 线程 + 30HX）迁到 **x86 小主机 + Jetson Orin NX**，
> 由 Mac 当控制端。本文件记录这一轮的落地内容、实测数据、踩的坑与**未决的架构问题**。

## 1. 做了什么

### 1.1 通道与迁移
* 配好 Mac → 两台的 **SSH 免密**（`~/.ssh/config` 加 `x86` / `jetson` 别名，密钥 `id_ed25519_codex`；
  首次用 `expect` 传口令装 key）。口令仍只在 `远程服务器.md`（gitignore）。
* 代码迁移：`rsync -a` Mac → 两台（排除 `build/`、`logs/`、`__pycache__`、`sp_vision_25/{assets,io,.git}`(Jetson) 等）。
  x86 侧 937 MB（含 `datasets/`、`.deps/`、`sp_vision_25`），Jetson 侧 119 MB（只需 `rm_sim` + `auto_aim_rl`）。
* 两台都建了 `~/mjlab-env/venv`（路径沿用旧命名，脚本不用改）。

### 1.2 x86（192.168.1.236，用户 clgx）
* apt：`build-essential cmake pkg-config libopencv-dev libeigen3-dev libfmt-dev libspdlog-dev
  libyaml-cpp-dev nlohmann-json3-dev libusb-1.0-0-dev libceres-dev can-utils python3-venv python3-pip
  libgl1 libegl1 libosmesa6 libglvnd-dev libglfw3-dev mesa-utils screen curl wget rsync`（sudo 免密，脚本化安装）。
* **OpenVINO 2024.6.0**：用仓库自带的 `.deps/openvino_2024.6.0.tgz` 解到 `/opt/intel/`，再把解出来的
  `l_openvino_toolkit_ubuntu22_2024.6.0.17404.4c0f47d2335_x86_64` **重命名**成 CMake 写死的
  `/opt/intel/openvino_2024.6.0`（`deploy_deps.sh` 里本来就是这个做法）。
* **ONNX Runtime 1.31.0**：仓库 CMake 默认找 `$HOME/opt/onnxruntime-linux-x64-gpu_cuda12-1.31.0`，
  但 x86 没有 NVIDIA GPU ⇒ 装 **CPU 版**到 `~/opt/onnxruntime-linux-x64-1.31.0`，configure 时显式
  `-DONNXRUNTIME_ROOT=$HOME/opt/onnxruntime-linux-x64-1.31.0`（进 cache，之后 `build.sh` 照旧）。
  取包走 `api.github.com`（`github.com` 在境内不可达，但 api 与 `objects.githubusercontent.com` 可达）。
* venv：mujoco 3.11.0 / numpy 2.2.6 / PyYAML / matplotlib / Pillow / trimesh / viser / mjviser /
  opencv-python 5.0 / requests / onnxruntime 1.23 / pytest。
* **`sp_vision_25` 编译成功**（`BUILD_RC=0`，170 个编译单元；`build/sim/{sim_main,detect_probe,sim_geom_check}` 齐）。
  `ldd` 确认同时挂上 `libopenvino.so.2460` 与 `libonnxruntime.so.1`。

### 1.3 Jetson（192.168.1.101，用户 nvidia）
* apt：`python3-pip python3-venv curl` → 再装 **CUDA 12.6 toolkit（`cuda-toolkit-12-6`，含 nvcc）
  与 cuDNN 9（`libcudnn9-cuda-12`）**（L4T 源 `repo.download.nvidia.cn/jetson`）。
* venv：mujoco 3.11.0 / gymnasium / numpy / PyYAML / onnx / onnxruntime / scikit-learn / matplotlib +
  **torch 2.8.0（jetson-ai-lab `jp6/cu126` 的 aarch64 CUDA 轮子，226 MB）**。
* 冒烟：`auto_aim_rl/env/test_env.py` 通过（607 步 / 18.6 s，obs 无 NaN，变化率三维是活特征）。

### 1.4 代码修复（都是迁移必需、最小、可回退）
* 硬编码路径 → 相对脚本自身：`sp_vision_25/sim/{run_closed_loop,compare_render,compare_detectors}.sh`、`deploy_deps.sh`。
* `sp_vision_25/configs/sim.yaml`：`device: CUDA` → **`CPU`**（x86 无 NVIDIA GPU），并写清原因。
* `rm_sim/rm_link.py`：新增 **`--host`**（默认 `127.0.0.1`，跨机联调写 `0.0.0.0`；打印的 URL 跟着变）。
  这是跨机取流的前提，默认行为不变。

## 2. 结果数据（实测）

| 验收项 | 结果 |
|---|---|
| `rm_sim/test_*.py`（6 个） | **6/6 PASS** |
| 黄金轨迹 qpos_hash | mujoco **3.11.0 → `c76a72ab6149211b800259e871dad902`**（与 `agnet_handover.md` 记录**逐位一致**）；3.16.0 → `7fa778ba…`（错） |
| `sp_vision_25` 编译 | BUILD_RC=0；sim_ort 生效（`-- sim_ort: 使用 ONNX Runtime /home/clgx/opt/...`） |
| Jetson rm_sim 真实帧率（EGL 渲染） | **20.0~20.2 FPS**（旧机器 20.6，口径相当） |
| x86 rm_sim 真实帧率（osmesa 渲染） | **10.3 FPS**（关阴影 14 FPS） |
| 训练 env 步速 | x86 **58 步/s（1.9× 实时）**；Jetson 32 步/s（1.1× 实时） |
| `sim_ort` 单帧推理（x86 CPU） | 39 ms（intra=4）/ 34 ms（intra=12） |
| Jetson CUDA 渲染 640×480 | 21.3 ms/帧 = 47 FPS |
| 跨机 Wi-Fi `/grab` | 25~81 ms（中位 ~40 ms）；`/state.json` 10~14 ms |
| 跨机闭环试跑（x86 sim_main → Jetson 仿真） | 主循环仅 6.0 FPS，`Large dt` 160 次，1 次 JSON 截断 |

## 3. 踩的坑（都记下来，别再重走）

1. **`mujoco` 版本必须钉 3.11.0** —— pip 默认装到 3.16.0，物理与黄金 hash 全变。
2. **`rsync --exclude 'rules/'` 误伤 `rm_sim/configs/rules/`** —— Jetson 上 `rm_link` 直接报
   `找不到配置文件 .../configs/rules/legacy_sim.yaml`（与 `.gitignore` 行尾注释同类的模式坑）。
3. **`pkill -f 'rm_link.py'` 会杀掉自己的 ssh 会话** —— 因为同一条命令行里也含 `rm_link.py`。
   要杀就用 `pkill -f "[r]m_link"`，且**别把启动命令放在同一条命令里**。
4. **`setsid` 在 macOS 上不存在**（exit 127）——后台任务用 `(nohup ... &)`。
5. **远端 sudo 的口令管道**：`printf "$PW" | ssh host 'sudo -S ...'` 里**不能有裸 `sudo -k`**（它会先吃掉 stdin）。
   可靠做法是 `ssh -tt` + expect 回口令（`tools/dual_machine/ssh_sudo.exp`）。
6. **expect 的正则里 `[:：]` 会被 Tcl 当命令替换** —— 方括号必须转义，或干脆避开。
7. **`github.com` 两台机器都不可达**（`api.github.com`、`objects.githubusercontent.com`、`pypi.org` 可达）⇒
   取 GitHub Release 资产要走 api 的 `assets/<id>` + `Accept: application/octet-stream`。
8. **x86 的 Intel Arc 核显（Arrow Lake-H, 7d51）在内核 6.8 下完全没驱动**（`i915`/`xe` 都未加载，
   dmesg 连探测行都没有），22.04 仓库里也**没有 6.8 以上的内核**（`-edge` 同样是 6.8）⇒ 只能软件渲染。
   `lspci -tv` 还显示那个 x8 槽被 Crucial NVMe 转接卡占着，且是迷你主机电源 ⇒ 插 30HX 不现实。
9. **Jetson 的 `MUJOCO_GL=egl` 退出时会抛 `EGL...free` 异常**（`Exception ignored in ... __del__`）——
   渲染本身是好的（47 FPS），别被吓到；`MUJOCO_GL=glfw` 反而不可用（无 X 授权）。
10. **`sim_link.cpp` 跨机会拿到截断 body**（`JSON 解析失败 ... missing closing quote`）——
    它虽然循环 `recv` 到连接关闭，但没按 `Content-Length` 收全、也没有超时重试。
11. **rsync 的远端路径不能用 `~`**：`rsync -a tools/ x86:~/vision/tools/` 在远端不会展开 `~`
    ⇒ **静默不生效**（脚本报"同步完成"，机器上查不到文件；同一类问题还有 `--exclude 'rules/'`）。
    一律写绝对路径 `x86:/home/clgx/vision/tools/`。已写进 `tools/dual_machine/README.md` §3。
12. **macOS 没有 `timeout`**（`remote_train.sh` 在 Mac 上跑）：写了 `timeout 60 ssh ...` 会
    `command not found` ⇒ 训练静默没起来。已改成探测 `timeout`/`gtimeout`。

## 4. 结论：原定架构被硬件卡住，需要负责人拍板

负责人原本规划 **x86 跑 rm_sim、Jetson 跑 auto_aim_rl（视频流走局域网）**。实测结论：
**x86 没有可用 GPU ⇒ 软件渲染只有 8~10 FPS，跑不了 `sim.main_fps: 20` 的评测口径**（帧率不控住数字不成立），
而 Jetson 的 GPU 渲染能稳稳 20.2 FPS；反过来，**x86 的 CPU（1.9× 实时、58 步/s）比 Jetson（1.1×、32 步/s）更适合训练**。

四条可选路线（细节见 `tools/dual_machine/README.md` §6）：

| 方案 | 仿真/闭环放哪 | 需要做的工作 | 风险 |
|---|---|---|---|
| **A. 拆分（倾向）** | **Jetson 跑 rm_sim**（20.2 FPS ✓）+ **x86 跑 sp_vision 闭环**（编译产物已就绪；检测 39 ms） | ① 修 `sim_link` 按 Content-Length 收全 + 重试；② `sim.yaml` 的 `base_url` 指向 Jetson；③ **两台插网线**（Wi-Fi `/grab` 中位 40 ms + 抖动） | 网络延迟进控制环（需用实测耗时做延迟补偿，机制已有） |
| **B. 全放 Jetson** | Jetson 同时跑 rm_sim + sp_vision + 检测 | sp_vision **aarch64 移植**：CMake 里把 OpenVINO 变可选、只留 sim_ort；检测要落 Orin GPU（ORT-CUDA/TensorRT）才够快 | 工作量最大；Jetson CPU 余量只有 ~1.1× 实时 |
| **C. 维持原计划** | 都在 x86 | 接受 8~10 FPS；把渲染设置（关阴影）固定、**复核/重训检测器在该渲染域**的召回、**所有基线按新口径重录** | 与历史数字不可比；且仍达不到 20 FPS 协议 |
| **D. 点亮核显** | 都在 x86 | 手工装 mainline 内核或升 Ubuntu 24.04（要重启；这台还跑着别人的 docker 负载） | 改系统，风险自负；成功则可回到"单机全包" |

**下一步（等负责人拍板）**：
1. 选 A / B / C（D 是否愿意试）；
2. 若选 A：两台之间**插一根网线**（x86 `enp87s0`、Jetson `enP8p1s0` 都空着，可直连或过交换机），
   我再补静态 IP + 修 `sim_link` + 跑通 20 FPS 闭环并重录基线；
3. 训练侧环境已就绪：`x86` 适合跑 PPO（58 步/s），待确认是否把 `tools/remote_train.sh` 的目标从旧 T4 改成 x86 本机；
4. 文档同步：`README.md` / `DEPLOY.md` 仍在描述旧的单机 Ubuntu 环境，待架构定稿后一并改写。

## 5. 本轮新增/改动文件

| 文件 | 说明 |
|---|---|
| `tools/dual_machine/README.md` | 双机环境说明书（角色/路径/铁律/命令/硬约束） |
| `tools/dual_machine/*.{sh,exp}` | 侦察、安装、自检、投递用脚本（从 /tmp 提炼固化） |
| `sp_vision_25/sim/{run_closed_loop,compare_render,compare_detectors}.sh` | 工作区路径改成相对脚本自身 |
| `deploy_deps.sh` | 同上 |
| `sp_vision_25/configs/sim.yaml` | `device: CUDA` → `CPU`（并注明原因） |
| `rm_sim/rm_link.py` | 新增 `--host`（默认 127.0.0.1，跨机写 0.0.0.0） |
| 两台机器上的 `~/vision`、`~/mjlab-env/venv`、`/opt/intel/openvino_2024.6.0`、`~/opt/onnxruntime-*`、CUDA/cuDNN | 环境产物（不入库） |

## 6. 收尾追加验证

| 项 | 结果 |
|---|---|
| Jetson：torch + CUDA | ✅ `torch 2.8.0`（jetson-ai-lab `jp6/cu126`）+ CUDA 12.6 toolkit + cuDNN 9 ⇒ `cuda_avail=True, device=Orin` |
| Jetson：`numpy` 必须 `<2` | torch 轮子是 NumPy 1.x 构建的 ⇒ 钉 `numpy==1.26.4`（否则 `Failed to initialize NumPy`，torch↔numpy 互转不可靠） |
| Jetson：PPO 端到端冒烟 | ✅ `train/ppo.py --steps 3000 --envs 4 --episode-s 30` → 2048 步 / 23 s ≈ **89 步/s**（4 环境），`policy.pt` 落盘 |
| x86：torch | 后台装 CPU 版（`torch 2.14.1`，554 MB）——把"谁负责训练"这个选择留着不锁死 |
| 清理 | 两台 8099 端口的残留仿真都已停掉（端口空闲） |

## 7. A 方案（拆分）落地：跨机闭环已跑通（负责人 2026-10-15 拍板）

**决定**：**Jetson 跑 rm_sim（GPU 渲染 20~24 FPS）+ x86 跑 sp_vision 闭环**，两者走局域网传 `/grab` 与 `/command`。

### 7.1 代码改动

| 文件 | 改动 | 为什么 |
|---|---|---|
| `sp_vision_25/sim/sim_link.cpp` | ① 按 `Content-Length` 收全（超时最多再等 3 个周期）；② body 残缺直接报失败；③ `grab()` 外层失败**重试一次**；④ 新增 `SIM_BASE_URL` 环境变量覆盖 `sim.base_url`，日志里标明地址来源 | 跨机分片会拿到半截 JSON；跨机不该靠改 yaml |
| `sp_vision_25/sim/sim_link.hpp` | 声明 `grab_once`（重试层拆出来） | 同上 |
| `sp_vision_25/sim/bench_ttk.sh` | 新增 `SIM_REMOTE`（ssh 远起仿真，`setsid` + 本地 `timeout` 兜底）、`SIM_MUJOCO_GL`（无 `/dev/dri` 自动 osmesa）、readiness/取状态走 `SIM_BASE_URL`、起不来时给排查提示 | 让 `eval_policy.sh` / `acceptance.sh` 全套评测**零改动**支持跨机 |
| `tools/dual_machine/setup_wired_link.sh` | 新增：网线直连静态 IP（x86 10.10.10.1 / jetson 10.10.10.2，幂等，可选 `PERSIST=1` 持久化） | Wi-Fi 抖动撑不住 20 FPS |

### 7.2 首次跨机实测（Wi-Fi，60 s，种子 20251012）

* **JSON 截断 0 次** —— `Content-Length` 收全的修复生效；`/grab` 重试路径还真的命中过 1 次并成功
* 结果：发射 47 / 命中 21 = **44.7%**、仿真帧率 24.4、实弹速 23.7、过热被拒 107
* **但 C++ 主循环只有 4.7 FPS**（251 次 `Large dt`）⇒ Wi-Fi 下不可用：
  `/grab` 中位 40 ms（抖动到 80 ms）+ 检测 39 ms 已经超了 20 FPS 的 50 ms 预算
* ⇒ **必须网线直连**（两台千兆口都空着：`x86/enp87s0`、`jetson/enP8p1s0`，当前 `carrier=0` 即未插）

### 7.3 待办（等网线插好）

1. 跑 `bash tools/dual_machine/setup_wired_link.sh` 配好直连，复测
   `SIM_BASE_URL=http://10.10.10.2:8099`，目标主循环 ≈20 FPS（帧率随结果一起记录）
2. `sim_ort_intra_threads` 4 → 12（实测 39 ms → 34 ms，给 20 FPS 留余量）
3. **重录基线**：`benchmark_2026_legacy` / `benchmark_2026_lift`（300 s × 4 种子）+ RL vs sp_vision 验收对照

## 8. 收尾项（负责人 2026-10-15：先调线程 + 更新文档，暂不插网线）

* **`sim_ort_intra_threads` 默认 4 → 12**（写进 `configs/sim.yaml` 并注明依据）：
  实测单帧推理 **39 ms → 28.6 ms**，启动日志出现「线程 12」（20 FPS 预算 50 ms，留余量）。
  **顺手修掉一个隐患**：该键原本已存在于 yaml（值 4），我第一次插入注释块时造成了**重复键**
  （yaml-cpp 取第一个，行为不确定）—— 已合并成唯一一条并复验解析值 = 12。
* **文档更新到双机口径**：
  * `README.md`：新增"现在的部署形态（双机 + Mac 控制端）"（机器表 + 为什么这样分 + 跨机命令），
    并把原"快速开始"改成"快速开始（单机自检）"。
  * `DEPLOY.md`：顶部加历史文档标注，新增 **§8 双机部署（现行）**（机器表 / mujoco 必须钉 3.11.0 /
    ORT CPU 版与 `-DONNXRUNTIME_ROOT` / `MUJOCO_GL` 按机器设 / 跨机闭环用法 / 网线建议）。
  * `tools/README.md`：顶部指向 `dual_machine/`。
* **仍未做（等网线）**：直连复测 20 FPS → 重录基线。




