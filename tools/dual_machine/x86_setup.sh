#!/usr/bin/env bash
# =============================================================================
#  x86 小主机：rm_sim 运行环境（~/mjlab-env/venv，python3.10）
#  无独显、无 /dev/dri ⇒ 渲染走软件（osmesa/llvmpipe），检测走 ONNX Runtime CPU
# =============================================================================
set -x
set -o pipefail
V="$HOME/mjlab-env/venv/bin"

# 1) 等 apt 把依赖装完（dpkg 锁释放）
for i in $(seq 1 240); do
  if sudo -n fuser /var/lib/dpkg/lock-frontend >/dev/null 2>&1; then sleep 5; else break; fi
done
dpkg -l cmake libopencv-dev libceres-dev >/dev/null 2>&1 && echo "APT_DEPS_OK" || echo "APT_DEPS_MISSING(继续装 python 侧)"

# 2) venv + python 依赖（清华镜像；失败则回落 PyPI）
python3 -m venv "$HOME/mjlab-env/venv" || exit 1
"$V/pip" install -U pip setuptools wheel
PIP_ARGS="-i https://pypi.tuna.tsinghua.edu.cn/simple"
"$V/pip" install $PIP_ARGS mujoco numpy PyYAML matplotlib Pillow trimesh viser mjviser opencv-python requests onnxruntime pytest \
  || "$V/pip" install mujoco numpy PyYAML matplotlib Pillow trimesh viser mjviser opencv-python requests onnxruntime pytest
# 训练侧依赖（2026-10-15 起 x86 也能当训练机）：gymnasium + CPU 版 torch + 离线拟合要用的 sklearn/onnx。
# torch 必须用 CPU 索引：PyPI 的默认 torch 是 CUDA 版，会拖 2.5 GB 的 nvidia-* 依赖，而这台没有 N 卡。
"$V/pip" install $PIP_ARGS gymnasium scikit-learn onnx
"$V/pip" install --index-url https://download.pytorch.org/whl/cpu torch

echo "=== 自检 ==="
"$V/python" - <<'PY'
import numpy, mujoco, yaml, matplotlib, PIL, trimesh, cv2, onnxruntime
print("numpy", numpy.__version__)
print("mujoco", mujoco.__version__)
print("cv2", cv2.__version__)
print("onnxruntime", onnxruntime.__version__)
import viser, mjviser
print("viser ok, mjviser ok")
print("ALL_IMPORTS_OK")
PY
echo "X86_SETUP_DONE"
