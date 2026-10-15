#!/usr/bin/env bash
# =============================================================================
#  一键部署 rm_sim 运行环境（venv + 依赖 + 真模型资产 + 自检）
#
#  用法：
#      bash deploy/deploy_rm_sim.sh                      # 装到 ~/mjlab-env/venv
#      VENV=~/venvs/sim bash deploy/deploy_rm_sim.sh     # 换 venv 路径
#      bash deploy/deploy_rm_sim.sh --no-assets          # 不准备真模型（只用免网格 simple 模型）
#      bash deploy/deploy_rm_sim.sh --no-selftest        # 跳过自检
#      bash deploy/deploy_rm_sim.sh --glb /path/vehicle.glb   # 用本地 GLB 转出真模型资产
#
#  装完的状态：
#      * venv 里 mujoco / numpy / cv2 / trimesh / viser 都能 import（版本见下）
#      * `rm_sim/test_*.py` 全绿（8 个，**不需要 GPU、不需要可视窗口**）
#      * 真模型资产（`rm_sim/assets/obj/`）就位 ⇒ `--model real` 可用；
#        没有也能跑：`--model simple` 是免网格模型（默认台架自检都用它）
#
#  ---- 两个必须钉住的版本（踩过，别改）----
#  1) **mujoco 必须 3.11.0**：装 3.16 会让黄金轨迹 hash 从
#     c76a72ab6149211b800259e871dad902 变成 7fa778ba…（见 agnet.md 决策表）
#  2) **Jetson(aarch64) 上 numpy < 2**：jetson-ai-lab 的 torch 轮子按 NumPy 1.x 编译
#
#  ---- 无头机器（没有显示器）----
#  MuJoCo 渲染需要一个 GL 后端：装好内核驱动的机器用 EGL，普通 x86 无核显驱动时用
#  **软件渲染 osmesa**。脚本会自动探测并导出 MUJOCO_GL，自检时就按这个跑。
# =============================================================================
set -uo pipefail

WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${VENV:-$HOME/mjlab-env/venv}"
PIP_MIRROR="${PIP_MIRROR:-https://pypi.tuna.tsinghua.edu.cn/simple}"
MUJOCO_PIN="${MUJOCO_PIN:-mujoco==3.11.0}"
WITH_ASSETS=1
SELFTEST=1
GLB=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-assets)     WITH_ASSETS=0; shift ;;
    --no-selftest)   SELFTEST=0; shift ;;
    --glb)           GLB="$2"; shift 2 ;;
    -h|--help)       sed -n '2,25p' "$0"; exit 0 ;;
    *) echo "未知参数: $1（--help 看用法）" >&2; exit 2 ;;
  esac
done

ok()   { echo "    ✓ $*"; }
bad()  { echo "    ✗ $*"; }
say()  { echo "==> $*"; }

say "[1/5] 环境检查"
[[ -d "$WS/rm_sim" ]] || { bad "找不到 $WS/rm_sim（脚本要放在工作区根的 deploy/ 下）"; exit 1; }
command -v python3 >/dev/null || { bad "没有 python3"; exit 1; }
PY3="$(command -v python3)"
PYVER="$("$PY3" -c 'import sys;print("%d.%d"%sys.version_info[:2])')"
ARCH="$(uname -m)"
ok "python3 $PYVER ($ARCH)，venv 目标 $VENV"

# ---- 渲染后端自动探测（只影响无头机器的自检/运行，不改任何数值）----
if [[ -z "${MUJOCO_GL:-}" ]]; then
  if [[ "$ARCH" == "aarch64" ]] && compgen -c tegrastats >/dev/null 2>&1; then
    export MUJOCO_GL=egl                      # Jetson：有 GPU，EGL 才快
  elif [[ -e /dev/dri/renderD128 ]] || [[ -n "${DISPLAY:-}" ]]; then
    export MUJOCO_GL=glfw                     # 有显示/有核显驱动
  else
    export MUJOCO_GL=osmesa                   # 无头且无驱动 ⇒ 软件渲染
  fi
  ok "自动选渲染后端 MUJOCO_GL=$MUJOCO_GL（可自己 export 覆盖）"
else
  ok "沿用已有 MUJOCO_GL=$MUJOCO_GL"
fi

say "[2/5] 建 venv"
if [[ -x "$VENV/bin/python" ]]; then
  ok "venv 已存在：$VENV"
else
  "$PY3" -m venv "$VENV" || { bad "建 venv 失败（缺 python${PYVER}-venv 包？）"; exit 1; }
  ok "已创建 $VENV"
fi
V="$VENV/bin"
"$V/pip" install -q -U pip setuptools wheel || { bad "升级 pip 失败（网络？）"; exit 1; }

say "[3/5] 安装依赖（清单与 tools/dual_machine/*_setup.sh 同一套，已实测）"
NUMPY_PIN="numpy"
[[ "$ARCH" == "aarch64" ]] && NUMPY_PIN="numpy<2"
# 清华镜像先试，失败回落 PyPI —— 有些网络环境对镜像不通
if ! "$V/pip" install -q -i "$PIP_MIRROR" \
      "$MUJOCO_PIN" "$NUMPY_PIN" PyYAML matplotlib Pillow trimesh viser mjviser \
      opencv-python requests pytest; then
  bad "镜像源装失败，回落 PyPI 再试一次"
  "$V/pip" install -q "$MUJOCO_PIN" "$NUMPY_PIN" PyYAML matplotlib Pillow trimesh viser mjviser \
      opencv-python requests pytest || { bad "依赖装不上，检查网络/代理"; exit 1; }
fi
ok "依赖装好"

say "[4/5] 真模型资产（vehicle.glb → assets/obj/*.obj）"
if [[ "$WITH_ASSETS" == 0 ]]; then
  ok "跳过（--no-assets）：只能用 --model simple"
elif [[ -f "$WS/rm_sim/assets/obj/chassis.obj" ]]; then
  ok "已就位：$(ls "$WS/rm_sim/assets/obj" | wc -l | tr -d ' ') 个网格文件"
else
  if [[ -n "$GLB" ]]; then
    bash "$WS/deploy/fetch_real_model.sh" --glb "$GLB" || bad "转换失败（见上面输出；只用 simple 模型也照样能跑）"
  else
    bash "$WS/deploy/fetch_real_model.sh" || {
      bad "没拿到 vehicle.glb ⇒ 真模型暂不可用（不影响 simple 模型与自检）"
      bad "补做：把 vehicle.glb 放到本机后跑  bash deploy/fetch_real_model.sh --glb <路径>"
    }
  fi
fi

if [[ "$SELFTEST" == 0 ]]; then
  say "[5/5] 跳过自检（--no-selftest）"
  exit 0
fi

say "[5/5] 自检"
export MUJOCO_GL
"$V/python" - <<'PY' || exit 1
import sys
import mujoco, numpy, yaml, PIL, trimesh, cv2
print("    python     ", sys.version.split()[0])
print("    mujoco     ", mujoco.__version__)
print("    numpy      ", numpy.__version__)
print("    opencv     ", cv2.__version__)
print("    trimesh    ", trimesh.__version__)
if mujoco.__version__ != "3.11.0":
    print("    ⚠ mujoco 不是 3.11.0 —— 黄金轨迹 hash 会变（见 agnet.md 决策表）")
    print("      修正： pip install 'mujoco==3.11.0'")
PY
ok "依赖 import 通过"

# 单元自检：不需要 GPU/显窗，跑的就是仓库里那 8 个（含弹道、自旋、枪/热量、轮腿、状态条）
FAILED=""
for t in "$WS"/rm_sim/test_*.py; do
  printf "    %-34s " "$(basename "$t")"
  if (cd "$WS/rm_sim" && timeout 600 "$V/python" "$t" > /tmp/deploy_rm_sim_test.log 2>&1); then
    echo "PASS"
  else
    echo "FAIL"; tail -5 /tmp/deploy_rm_sim_test.log | sed 's/^/        /'; FAILED="$FAILED $(basename "$t")"
  fi
done
[[ -z "$FAILED" ]] && ok "rm_sim 自检全绿" || { bad "有失败：$FAILED"; exit 1; }

# 端到端冒烟：真起一次仿真，确认渲染 + HTTP 接口都活着（自检里唯一碰渲染的一步）
if command -v curl >/dev/null 2>&1; then
  SMOKE_PORT="${SMOKE_PORT:-8097}"
  MODEL=real; [[ -f "$WS/rm_sim/assets/obj/chassis.obj" ]] || MODEL=simple
  ( cd "$WS/rm_sim" && timeout 22 "$V/python" -u rm_link.py --model "$MODEL" --port "$SMOKE_PORT" \
      > /tmp/deploy_rm_sim_smoke.log 2>&1 & )
  STATE=""
  for _ in $(seq 1 25); do
    STATE="$(curl -s -m 3 "http://127.0.0.1:$SMOKE_PORT/state.json" 2>/dev/null)"
    [[ -n "$STATE" ]] && break
    sleep 1
  done
  if [[ -n "$STATE" ]]; then
    ok "仿真冒烟通过（--model $MODEL，/state.json 有响应）"
  else
    bad "仿真冒烟失败：/state.json 无响应，日志："
    tail -12 /tmp/deploy_rm_sim_smoke.log | sed 's/^/        /'
    exit 1
  fi
  pkill -f "rm_link.py --model $MODEL --port $SMOKE_PORT" 2>/dev/null
else
  ok "没有 curl，跳过仿真冒烟"
fi

cat <<TIP

==> 装好了。接下来：

  1) 跑仿真（浏览器看画面）：
       $V/python rm_sim/rm_link.py --model real --port 8099 --host 0.0.0.0
     浏览器打开 http://127.0.0.1:8099 （跨机时用 http://<本机IP>:8099）
     → 上面是全局视角，下面是云台相机原图

  2) 交互演示（拖准星控云台、点击开火）：
       $V/python rm_sim/rm_gimbal_demo.py --model real
     浏览器打开 http://localhost:8080

  3) 换实时渲染后端（无头 x86 默认 osmesa 只有 ~10 FPS；有核显/GPU 时）：
       export MUJOCO_GL=egl     # 或 glfw

  4) 要跑学习型开火决策（auto_aim_rl）： bash auto_aim_rl/deploy_env.sh
     要跟 sp_vision 闭环联调：见 README「与 sp_vision 闭环」（需要另外的 C++ 框架）
TIP

