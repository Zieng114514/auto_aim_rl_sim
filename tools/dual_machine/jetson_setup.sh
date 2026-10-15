#!/usr/bin/env bash
# =============================================================================
#  Jetson Orin NX：auto_aim_rl 训练环境（~/mjlab-env/venv，python3.10）
#  要点：torch 走 jetson-ai-lab（CUDA/aarch64），其余走 PyPI；mujoco 用于无渲染物理
# =============================================================================
set -x
set -o pipefail
V="$HOME/mjlab-env/venv/bin"

python3 -m venv "$HOME/mjlab-env/venv" || exit 1
"$V/pip" install -U pip setuptools wheel

# torch：aarch64 CUDA 轮子（jp6/cu126 = JetPack 6.2 / CUDA 12.6）
"$V/pip" install --index-url https://pypi.jetson-ai-lab.io/jp6/cu126 \
                 --extra-index-url https://pypi.org/simple \
                 "torch==2.8.0"
"$V/python" -c "import torch;print('TORCH', torch.__version__, 'cuda=', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

# 其余依赖（PyPI）
"$V/pip" install mujoco gymnasium numpy PyYAML onnx onnxruntime scikit-learn matplotlib

echo "=== 自检 ==="
"$V/python" - <<'PY'
import numpy, mujoco, yaml, gymnasium, torch, onnx, onnxruntime, sklearn, matplotlib
print("numpy", numpy.__version__)
print("mujoco", mujoco.__version__)
print("gymnasium", gymnasium.__version__)
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("onnxruntime", onnxruntime.__version__)
print("ALL_IMPORTS_OK")
PY
echo "JETSON_SETUP_DONE"
