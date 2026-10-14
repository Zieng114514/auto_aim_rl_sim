# sp_vision_25 环境部署报告

> 部署日期：2026-10-12
> 目标：让 `sp_vision_25`（同济 SuperPower 25 赛季自瞄框架）在本机可配置、可编译、可运行。

---

## 1. 结论

**环境已部署完成并通过端到端验证。** 39 个目标全部编译成功，32 个可执行文件全部通过动态库解析检查，
自瞄核心链路（YOLO 推理 → 解算 → 跟踪 → 决策）已在官方 demo 视频上实际跑通。

**没有修改任何上游源码。** `sp_vision_25/` 仍是对上游仓库的干净克隆，只新增了两个辅助脚本。

---

## 2. 主机信息

| 项 | 值 |
|---|---|
| 操作系统 | Ubuntu 22.04.5 LTS (jammy), x86_64 |
| 内核 | 6.8.0-138-generic |
| CPU | 12 线程 |
| 内存 / 磁盘 | 15 GiB / 150 GiB 可用 |
| 显卡 | Intel 核显 (a7a8) + NVIDIA CMP 30HX |

---

## 3. 已安装依赖

### 3.1 系统包（apt，182 个）

```
build-essential  cmake 3.22.1  pkg-config 0.29.2
libopencv-dev 4.5.4     libeigen3-dev 3.4.0    libfmt-dev 8.1.1
libspdlog-dev 1.9.2     libyaml-cpp-dev 0.7.0  nlohmann-json3-dev 3.10.5
libusb-1.0-0-dev        libceres-dev 2.0.0     can-utils  screen
```

> `libceres-dev` 是**必需**的，不是可选项：`tasks/auto_buff/CMakeLists.txt` 里有
> `find_package(Ceres REQUIRED)`，缺了会导致整个 CMake 配置失败。

### 3.2 OpenVINO 2024.6.0

- 归档：`.deps/openvino_2024.6.0.tgz`（51 MB，来自 storage.openvinotoolkit.org）
- 安装位置：**`/opt/intel/openvino_2024.6.0`**
- 为什么是这个路径：`CMakeLists.txt` 与 `tasks/auto_aim/CMakeLists.txt` 里**硬编码**了
  `set(OpenVINO_DIR "/opt/intel/openvino_2024.6.0/runtime/cmake")`。
  装到该路径即可开箱即用，**无需打补丁**。
- 自带插件：`libopenvino_intel_cpu_plugin.so`、`libopenvino_intel_gpu_plugin.so`、
  ONNX 前端（`tiny_resnet.onnx` 分类器要用）等，共 29 个 .so。

---

## 4. 验证结果（实测证据）

| 验证项 | 命令 / 方式 | 结果 |
|---|---|---|
| CMake 配置 | `cmake -B build` | ✅ OpenCV 4.5.4 / Ceres 2.0.0 / Eigen 3.4.0 全部找到 |
| 完整编译 | `make -C build -j12` | ✅ 39 个目标，100%，无 error |
| 重复编译 | 再次 `make` | ✅ exit 0，幂等 |
| 动态库解析 | 对全部 32 个可执行文件跑 `ldd` | ✅ 无 "not found" |
| **端到端自瞄链路** | `./build/auto_aim_test`（demo 视频） | ✅ **687 帧跑通**，YOLO 单帧均值 **9.88 ms（≈101 FPS）** |
| 规划器 (TinyMPC) | `./build/planner_test_offline configs/demo.yaml` | ✅ 10 s 内约 100 次 MPC 求解，无异常 |
| 主程序启动 | `./build/standard` | ✅ 正常启动（仅因无 CAN 硬件而等待，符合预期） |
| OpenVINO 设备 | 自建 `ov_devices` 探测程序 | ✅ CPU + Intel 核显**均可用**（见 §6.1） |
| CPU / GPU 推理对比 | 同一视频各跑 687 帧 | ✅ CPU 9.88 ms **优于** 核显 14.34 ms |

**关键点：不需要设置 `LD_LIBRARY_PATH`。** 编译产物已带 RUNPATH：

```
RUNPATH: .../io/hikrobot/lib/amd64:.../io/mindvision/lib/amd64:/opt/intel/openvino_2024.6.0/runtime/lib/intel64
```

---

## 5. 日常使用

### 编译

```bash
bash /home/zieng/Desktop/vision/sp_vision_25/build.sh        # 自动用 nproc 并行
bash /home/zieng/Desktop/vision/sp_vision_25/build.sh -j4    # 指定并行度
```

等价于上游 README 的：

```bash
cd /home/zieng/Desktop/vision/sp_vision_25
cmake -B build && make -C build -j`nproc`
```

> ⚠️ 程序里的配置路径、`assets/` 路径都是**相对当前工作目录**的，
> 所以必须在 `sp_vision_25/` 目录下运行 `./build/xxx`。

### 跑离线自瞄 demo

```bash
cd /home/zieng/Desktop/vision/sp_vision_25
./build/auto_aim_test          # 用默认 configs/demo.yaml + assets/demo/demo.avi
```

### 重新部署依赖 / 换机器

```bash
sudo bash /home/zieng/Desktop/vision/deploy_deps.sh
```

脚本幂等，可重复执行。OpenVINO 归档已在 `.deps/` 里，重装无需重新下载。

---

## 6. 已知限制（都不是故障）

| 项 | 说明 | 影响 |
|---|---|---|
| **ROS2 未安装** | CMake 检测不到 `rclcpp`/`sp_msgs`，主动跳过 | `sentry`、`sentry_bp`、`sentry_multithread`、`sentry_debug`、`publish_test`、`subscribe_test`、`topic_loop_test` **这 7 个目标未编译**。自瞄/打符/无人机不受影响。需要哨兵功能再装 ROS2 Humble。 |
| **GPU 值得注意** | 见 §6.1 的实测结论：**NVIDIA 30HX 完全用不了**（OpenVINO 无 CUDA 后端）；Intel 核显**已装好可用，但比 CPU 更慢**。 | 配置里的 `device: GPU` 建议全部改成 `CPU`。 |
| **无 CAN / 串口硬件** | 没有 `can0`、`/dev/gimbal` | `standard` 等主程序会停在等待下位机姿态，属正常。需要 USB2CAN 或 C 板实机。 |
| **无工业相机** | 海康/迈德威视 SDK 的 .so 已在仓库内并链接成功，但没插相机 | 只能跑离线视频测试，不能跑实时相机。 |
| **相机流对接** | `io/camera.cpp` 只有 hikrobot / mindvision / usbcamera 三种分支 | 仿真流要走 v4l2 或改代码，见 `rm_sim/README.md` 第 7 节。 |

### 6.1 GPU 实测结论（重要）

**① NVIDIA CMP 30HX 无法用于本框架。**

驱动的确装好了（`nvidia-smi` 正常，driver 615.71.09，6 GB），但 **OpenVINO 根本不支持 NVIDIA GPU** ——
它只有 Intel CPU / Intel GPU / Intel NPU 后端，归档里没有任何 CUDA/TensorRT 插件。
要用 30HX，必须把推理后端换成 ONNX Runtime (CUDA) 或 TensorRT，
并在 `tasks/auto_aim/yolos/` 下新写一个后端类 —— 那是**开发任务，不是环境部署**。

**② Intel 核显已装好并可用**（`Intel(R) UHD Graphics (iGPU)`），但**实测比 CPU 慢**：

| 设备 | yolo 推理均值 | 中位 | 最小 | 等效 FPS |
|---|---|---|---|---|
| **CPU**（12 线程） | **9.88 ms** | 9.80 ms | 8.10 ms | **101 FPS** |
| GPU（Intel 核显） | 14.34 ms | 14.30 ms | 10.50 ms | 70 FPS |

测法：同一段 demo 视频各跑 687 帧，`configs/demo.yaml` 只改 `device` 一行，取除首帧外的均值。

**结论：保持 `device: CPU`。** 这台机器的 CPU（Raptor Lake-P，12 线程）跑 yolov5 已经到 ~101 FPS，
核显反而慢 45%（EU 数量少 + 共享内存带宽 + 首帧内核编译开销）。
所以 `standard3/4.yaml`、`sentry.yaml`、`mvs.yaml`、`ascento.yaml` 里的 `device: GPU` **建议改成 CPU**。

**③ 如果确实要装 Intel 核显驱动**（例如换台 CPU 弱的机器），已跑通的配方是：

```bash
# Intel 官方源（level-zero 不在 Ubuntu 官方源里）
curl -fsSL https://repositories.intel.com/gpu/intel-graphics.key \
  | sudo gpg --dearmor --yes -o /etc/apt/keyrings/intel-graphics.gpg
echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/intel-graphics.gpg] https://repositories.intel.com/gpu/ubuntu jammy unified" \
  | sudo tee /etc/apt/sources.list.d/intel-gpu-jammy.list

sudo apt-get update
# 关键：显式点名 libigdgmm12，否则 apt 不会自动升级它，依赖会卡住
sudo apt-get install -y intel-level-zero-gpu level-zero libze1 libigdgmm12
# 关键：必须锁到 v1 版！最新版 intel-opencl-icd 依赖 libigc2，与 L0 驱动要的 libigc1 冲突
sudo apt-get install -y intel-opencl-icd=24.26.30049.10-950~22.04
```

两个坑（都实际踩过）：
1. `intel-opencl-icd`（新版，libigc2）与 `intel-level-zero-gpu`（libigc1）**互斥**，
   Intel 仓库当前状态同时装会报 `libigc2 : 冲突: libigc1`。必须把 ICD 锁到 24.39 以前的 v1 版本。
2. `apt` 不会自动升级 `libigdgmm12`（Ubuntu 版 22.1.2 与 Intel 版 22.7.2 同为 priority 500），
   必须显式写进安装列表。

装完可用 `.deps/diag/ov_devices` 确认设备列表。

---

## 7. 本次新增的文件

| 文件 | 说明 |
|---|---|
| `deploy_deps.sh` | 依赖部署脚本（需 root，幂等） |
| `sp_vision_25/build.sh` | 编译辅助脚本（无需 root） |
| `DEPLOY.md` | 本报告 |
| `.deps/openvino_2024.6.0.tgz` | OpenVINO 归档，供重装使用 |
| `.deps/diag/ov_devices` | OpenVINO 设备探测小工具（源码同目录） |

除此之外 `sp_vision_25/` 未做任何改动（`git status` 只多出未跟踪的 `build.sh` 与 `build/`）。
