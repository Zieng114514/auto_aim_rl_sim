#!/usr/bin/env bash
# =============================================================================
#  第二轮侦察：渲染/显示、构建工具链、推理运行时、sudo 与 apt 能力
# =============================================================================
export LC_ALL=C
echo "### SUDO"; sudo -n true 2>&1 | head -1; echo "sudo_rc=$?"
echo "### DISPLAY"; echo "DISPLAY=${DISPLAY:-<empty>}"; ls /tmp/.X11-unix 2>/dev/null || echo "(no X11 sockets)"
ls /usr/share/xsessions 2>/dev/null | head -3; loginctl list-sessions 2>/dev/null | head -4
echo "### BUILD"; which cmake make ninja pkg-config; cmake --version 2>/dev/null | head -1
echo "### OPENCV"; pkg-config --modversion opencv4 2>/dev/null || echo "(no opencv4.pc)"
echo "### OPENVINO"; ls -d /opt/intel/openvino* 2>/dev/null || echo "(none)"
echo "### ORT"; ls -d /opt/onnxruntime* /usr/local/onnxruntime* /opt/onnxruntime-linux-* 2>/dev/null; ldconfig -p 2>/dev/null | grep -i onnxruntime | head -3 || echo "(no libonnxruntime in ldconfig)"
echo "### CUDA"; ls -d /usr/local/cuda* 2>/dev/null | head -3; ls /usr/local/cuda/bin 2>/dev/null | head -4
python3 -c "import tensorrt; print('tensorrt', tensorrt.__version__)" 2>&1 | tail -1
echo "### DRM"; ls -l /dev/dri 2>/dev/null || echo "(no /dev/dri)"
echo "### APT"; grep -c . /var/lib/apt/lists/*Packages 2>/dev/null | head -2; apt-cache policy libceres-dev 2>/dev/null | head -2
echo "### PIP"; pip3 -V 2>&1 | head -1; python3 -m venv --help >/dev/null 2>&1 && echo "venv=OK" || echo "venv=MISSING"
echo "### DOCKER"; docker ps 2>&1 | head -2
echo "### NET"; ip -br addr 2>/dev/null | head -6; ip route 2>/dev/null | head -4
echo "### DONE"
