#!/usr/bin/env bash
# =============================================================================
#  ⚠️ 历史脚本（2026-10-13 ~ 2026-10-15）：面向**旧的远程 T4 训练机**
#  （root@183.239.162.232:30046，/root/gpufree-data）。该机器已不再使用。
#
#  现在的训练环境是**双机本地**（x86 小主机 / Jetson Orin NX），环境搭建脚本见：
#      tools/dual_machine/x86_setup.sh      # x86：venv + mujoco/cv2/viser…
#      tools/dual_machine/jetson_setup.sh   # Jetson：venv + torch(cu126)/mujoco…
#  起训练用 tools/remote_train.sh（TRAIN_HOST=x86|jetson）。
#  保留本文件只为记录当时的部署做法（uv 装 python、CPU 版 torch 的取舍）。
# =============================================================================
#  远程训练服务器（Tesla T4）环境部署 —— 在服务器上执行，不在本地跑
#
#  为什么训练要搬到远程：本地（12 线程 / 无独显余量）跑 PPO 时
#  MuJoCo 8 个 worker 把机器压死过（实测直接整机崩了两次）。
#  这里 80 核 / 376G 内存，可以开 40+ 个并行环境，吞吐是本地的一个数量级。
#
#  用法（本地）:
#      scp -P 30046 tools/remote_setup.sh root@183.239.162.232:/root/
#      ssh -p 30046 root@183.239.162.232 'bash /root/remote_setup.sh'
#
#  目录约定（都放在大盘 /root/gpufree-data 上，根分区只有 8.7G）：
#      /root/gpufree-data/vision      代码（rsync 同步过去）
#      /root/gpufree-data/venv        python 3.10 虚拟环境
#      /root/gpufree-data/runs        训练产物
# =============================================================================
set -euo pipefail
ROOT=/root/gpufree-data
WS=$ROOT/vision
VENV=$ROOT/venv
export UV_PYTHON_INSTALL_DIR=$ROOT/uv-python
export UV_CACHE_DIR=$ROOT/uv-cache
export PATH=$ROOT/bin:$HOME/.local/bin:$PATH
mkdir -p "$ROOT" "$ROOT/runs"

echo "==> 1/4 安装 uv（自带 python 版本管理，不碰系统 3.8）"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH=$HOME/.local/bin:$PATH
fi
uv --version

echo "==> 2/4 python 3.10 + venv"
uv python install 3.10
[ -d "$VENV" ] || uv venv "$VENV" --python 3.10
"$VENV/bin/python" -V

echo "==> 3/4 依赖：mujoco / numpy / gymnasium / pillow + CPU 版 torch"
# 策略是 21->64->64->2 的 MLP，GPU 完全用不上（瓶颈是 MuJoCo 单步物理），
# 所以装 CPU 版 torch：体积小 10 倍，也不会和别人的卡抢显存。
uv pip install --python "$VENV/bin/python" numpy mujoco gymnasium pillow onnx
uv pip install --python "$VENV/bin/python" torch --index-url https://download.pytorch.org/whl/cpu

echo "==> 4/4 自检：导入 + 建一个无渲染世界"
"$VENV/bin/python" - <<'PY'
import sys
sys.path.insert(0, "/root/gpufree-data/vision/auto_aim_rl/env")
import numpy, mujoco, gymnasium, torch, PIL
print("numpy %s | mujoco %s | gymnasium %s | torch %s" %
      (numpy.__version__, mujoco.__version__, gymnasium.__version__, torch.__version__))
from world import HeadlessWorld
w = HeadlessWorld(seed=0)
print("世界构建 OK：nbody=%d nq=%d fps=%.1f" % (w.model.nbody, w.model.nq, w.fps))
PY
echo "==> 部署完成：$VENV/bin/python"
