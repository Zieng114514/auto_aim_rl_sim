#!/usr/bin/env bash
# =============================================================================
#  sp_vision_25 依赖部署脚本
#
#  用法（需要 root，因为要写入 /usr 和 /opt）：
#      sudo bash deploy_deps.sh
#
#  可选：同时安装 OpenVINO 的 Intel GPU 运行时（配置里 device: GPU 时需要）
#      sudo INSTALL_GPU=1 bash deploy_deps.sh
#
#  说明：
#    * OpenVINO 归档已提前下载到 .deps/openvino_2024.6.0.tgz
#    * 解压后放到 /opt/intel/openvino_2024.6.0，与 CMakeLists.txt 里
#      set(OpenVINO_DIR "/opt/intel/openvino_2024.6.0/runtime/cmake") 完全一致，
#      因此**无需修改任何源码**。
# =============================================================================
set -euo pipefail

WS="/home/zieng/Desktop/vision"
OV_TGZ="$WS/.deps/openvino_2024.6.0.tgz"
OV_EXTRACTED="l_openvino_toolkit_ubuntu22_2024.6.0.17404.4c0f47d2335_x86_64"
OV_PREFIX="/opt/intel/openvino_2024.6.0"

if [[ ${EUID} -ne 0 ]]; then
  echo "错误：本脚本需要 root 权限。请用： sudo bash $0" >&2
  exit 1
fi

export DEBIAN_FRONTEND=noninteractive

echo "==> [1/3] 更新 apt 索引"
apt-get update

echo
echo "==> [2/3] 安装编译依赖（182 个包，约数百 MB）"
apt-get install -y \
  build-essential cmake pkg-config \
  libopencv-dev libeigen3-dev libfmt-dev libspdlog-dev \
  libyaml-cpp-dev nlohmann-json3-dev libusb-1.0-0-dev libceres-dev \
  can-utils screen

echo
echo "==> [3/3] 安装 OpenVINO 2024.6.0 -> $OV_PREFIX"
if [[ ! -f "$OV_TGZ" ]]; then
  echo "错误：找不到 OpenVINO 归档 $OV_TGZ" >&2
  exit 1
fi
mkdir -p /opt/intel
rm -rf "$OV_PREFIX" "/opt/intel/$OV_EXTRACTED"
tar -xzf "$OV_TGZ" -C /opt/intel
mv "/opt/intel/$OV_EXTRACTED" "$OV_PREFIX"

if [[ "${INSTALL_GPU:-0}" == "1" ]]; then
  echo
  echo "==> [可选] 安装 Intel 核显 (OpenCL + level-zero) 运行时"
  echo "    注意：本机实测 **核显比 CPU 慢**（9.88ms -> 14.34ms），一般不需要装。"
  # level-zero / intel-level-zero-gpu 不在 Ubuntu 官方源里，必须用 Intel 官方 GPU 源
  mkdir -p /etc/apt/keyrings
  curl -fsSL https://repositories.intel.com/gpu/intel-graphics.key \
    | gpg --dearmor --yes -o /etc/apt/keyrings/intel-graphics.gpg
  echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/intel-graphics.gpg] https://repositories.intel.com/gpu/ubuntu jammy unified" \
    > /etc/apt/sources.list.d/intel-gpu-jammy.list
  apt-get update
  # 坑 1：必须显式点名 libigdgmm12，否则 apt 不升级它，依赖解析失败
  apt-get install -y intel-level-zero-gpu level-zero libze1 libigdgmm12
  # 坑 2：必须锁 v1 版！最新 intel-opencl-icd 依赖 libigc2，与 L0 驱动要的 libigc1 冲突
  apt-get install -y intel-opencl-icd=24.26.30049.10-950~22.04
  echo "Intel 核显运行时已安装。用 .deps/diag/ov_devices 确认设备列表。"
fi

echo
echo "===================== 校验 ====================="
printf '%-28s' "cmake:";        cmake --version | head -1
printf '%-28s' "g++:";          g++ --version | head -1
printf '%-28s' "pkg-config:";   pkg-config --version
printf '%-28s' "OpenVINO CMake:"; ls -d "$OV_PREFIX/runtime/cmake/OpenVINOConfig.cmake"
printf '%-28s' "OpenVINO CPU plugin:"; ls "$OV_PREFIX"/runtime/lib/intel64/libopenvino_intel_cpu_plugin.so >/dev/null && echo OK || echo "缺失"
for p in opencv4 eigen3 fmt spdlog yaml-cpp; do
  printf '%-28s' "$p:"; pkg-config --modversion "$p" 2>/dev/null || echo "(无 .pc，CMake config 方式)"
done
echo "==============================================="
echo "依赖部署完成。"
