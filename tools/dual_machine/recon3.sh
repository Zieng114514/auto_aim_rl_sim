#!/usr/bin/env bash
# =============================================================================
#  第三轮侦察：CUDA / TensorRT / GPU 设备 / apt 源 / 网络与包管理可用性
# =============================================================================
export LC_ALL=C
echo "### ARCH"; uname -m
echo "### /usr/local"; ls /usr/local/ 2>/dev/null
echo "### CUDA-LIBS"
ls /usr/lib/aarch64-linux-gnu/ 2>/dev/null | grep -E '^libcudart|^libcudnn|^libnvinfer|^libcublas|^libcufft' | head -8
ls /usr/lib/x86_64-linux-gnu/ 2>/dev/null | grep -E '^libcudart|^libcudnn|^libnvinfer|^libcublas|^libcufft' | head -8
echo "### NV-PKGS"; dpkg -l 2>/dev/null | grep -E 'nvidia|cuda|nvinfer' | awk '{print $2}' | head -20
echo "### APT-SRC"; ls /etc/apt/sources.list.d/ 2>/dev/null
echo "### APT-CUDA"; apt-cache policy cuda-toolkit-12-6 2>/dev/null | head -3
apt-cache policy libopencv-dev 2>/dev/null | head -2
echo "### DRM-DRIVER"
lsmod 2>/dev/null | grep -E 'i915|xe|nvidia|nvgpu|simpledrm' | head -5 || echo "(none of i915/xe/nvidia/nvgpu loaded)"
sudo -n dmesg 2>/dev/null | grep -iE 'i915|xe |drm|nvidia' | tail -6
echo "### DEV"; ls -l /dev/dri 2>/dev/null || echo "(no /dev/dri)"; ls -l /dev/nvidiactl /dev/nvhost-gpu 2>/dev/null | head -3
echo "### TOOLS2"; which wget curl pip3 python3-pip 2>/dev/null; python3 -m pip --version 2>&1 | head -1
echo "### NET2"
timeout 12 wget -qS -O /dev/null https://pypi.org/simple/ 2>&1 | grep -E 'HTTP/' | head -2
timeout 12 wget -qS -O /dev/null https://pypi.jetson-ai-lab.dev/ 2>&1 | grep -E 'HTTP/' | head -2
timeout 12 wget -qS -O /dev/null https://github.com 2>&1 | grep -E 'HTTP/' | head -2
echo "### DISK2"; df -h /home | tail -1
echo "### DONE"
