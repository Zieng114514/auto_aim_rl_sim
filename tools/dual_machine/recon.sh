#!/usr/bin/env bash
# =============================================================================
#  双机部署侦察脚本（在目标机器上执行，输出纯文本报告）
#  用法： base64 < recon.sh | tr -d '\n'  →  ssh ... "echo <B64> | base64 -d | bash"
# =============================================================================
export LC_ALL=C
echo "### HOST"; hostname; echo
echo "### OS"; head -2 /etc/os-release; uname -m; uname -r
echo "### CPU"; lscpu | grep -E 'Model name|^CPU\(s\)|Thread|Architecture' || true
echo "### MEM"; free -h | head -2
echo "### DISK"; df -h / | tail -1
echo "### PCI"; lspci 2>/dev/null | grep -iE 'vga|3d|display|ethernet|network' | head -8
echo "### GL"; ls /usr/lib/*/libEGL* /usr/lib/*/libOSMesa* 2>/dev/null | head -5
echo "### INET"
timeout 10 curl -sI https://pypi.org/simple/ 2>&1 | head -1
timeout 10 curl -sI https://archive.ubuntu.com 2>&1 | head -1
echo "### PY-MODULES"
python3 -V 2>&1
python3 -c "import numpy;print('numpy',numpy.__version__)" 2>&1 | tail -1
python3 -c "import mujoco;print('mujoco',mujoco.__version__)" 2>&1 | tail -1
python3 -c "import torch;print('torch',torch.__version__,torch.cuda.is_available())" 2>&1 | tail -1
python3 -c "import cv2;print('cv2',cv2.__version__)" 2>&1 | tail -1
echo "### TOOLS"; which git cmake gcc g++ rsync docker pip3 nvcc 2>/dev/null; ls /usr/local/cuda*/bin/nvcc 2>/dev/null
echo "### JETSON"; cat /etc/nv_tegra_release 2>/dev/null; /usr/local/cuda/bin/nvcc --version 2>/dev/null | tail -2
echo "### TRT"; dpkg -l 2>/dev/null | grep -icE 'nvinfer|tensorrt'
echo "### APT"; apt-cache policy python3-venv 2>/dev/null | head -3
echo "### PROJECT"; ls -d ~/vision ~/mjlab-env 2>/dev/null || echo "(no ~/vision, no ~/mjlab-env)"
echo "### LAN"
ping -c 3 -q 192.168.1.101 2>&1 | tail -2
ping -c 3 -q 192.168.1.236 2>&1 | tail -2
ping -c 3 -q 192.168.1.203 2>&1 | tail -2
echo "### DONE"
