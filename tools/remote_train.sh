#!/usr/bin/env bash
# =============================================================================
#  在目标机上起一次 PPO 训练（nohup 后台，日志落盘）
#
#  2026-10-15 起：训练**不再依赖远程 T4**，改在双机上跑（见 tools/dual_machine/README.md）：
#      x86 小主机（16 线程，默认）  /  Jetson Orin NX（8 核 + Orin GPU）
#  两台都已有 ~/mjlab-env/venv（**mujoco 必须钉 3.11.0**；Jetson 侧 numpy<2）。
#  旧流程（root@183.239.162.232:30046 + /root/gpufree-data）已废弃，只留在 git 历史里。
#
#  用法（在 Mac 上）:
#      bash tools/remote_train.sh stage1_a 200000 12          # 默认 x86，12 个并行环境
#      TRAIN_HOST=jetson bash tools/remote_train.sh stage1_a 200000 6
#      SYNC=1 bash tools/remote_train.sh stage1_a 200000 12   # 先 rsync 代码到目标机
#      EXTRA_ARGS="--c-fire 0.15" bash tools/remote_train.sh ...
#  参数: 1=run 名  2=总步数  3=并行环境数  4=额外参数（不传则看 $EXTRA_ARGS）
#  产物: <host>:~/vision/auto_aim_rl/runs/<run>/{train.log, log.jsonl, policy.pt}
# =============================================================================
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"

HOST="${TRAIN_HOST:-x86}"
RUN="${1:-stage1_a}"
STEPS="${2:-200000}"
ENVS="${3:-}"
EXTRA="${4:-${EXTRA_ARGS:-}}"

WS="${TRAIN_WS:-~/vision}"                       # 目标机上的工作区
VENV="${TRAIN_PY:-~/mjlab-env/venv/bin/python}"  # 目标机上的 venv
ROLLOUT="${TRAIN_ROLLOUT:-512}"
EPISODE_S="${TRAIN_EPISODE_S:-90}"
SYNC="${SYNC:-0}"

# 并行环境数默认按机器给：x86 = 16 线程 → 12；Jetson = 8 核 → 6
if [ -z "$ENVS" ]; then
  case "$HOST" in
    x86*)    ENVS=12 ;;
    jetson*) ENVS=6 ;;
    *)       ENVS=8 ;;
  esac
fi
OUT="$WS/auto_aim_rl/runs/$RUN"

# --- 可选：把代码从 Mac 同步过去（Mac 是控制端，代码在这里编辑）---
if [ "$SYNC" = "1" ]; then
  echo "==> rsync 代码到 $HOST"
  rsync -a --exclude '__pycache__/' --exclude 'runs/' \
    "$HERE/../auto_aim_rl/" "$HOST:$WS/auto_aim_rl/"
  rsync -a --exclude '__pycache__/' \
    "$HERE/../rm_sim/" "$HOST:$WS/rm_sim/"
fi

echo "==> $HOST 上起训练：$RUN"
echo "    $STEPS 步 / $ENVS 环境 / rollout $ROLLOUT / episode ${EPISODE_S}s / randomize + seed 0"
# setsid + </dev/null + 本地 timeout：远端进程脱离 ssh 会话；本地 ssh 在这种"后台化+重定向"
# 写法下偶尔不返回，用 timeout 兜底（踩过，见 agnet_logs/20261015_1250_dual_machine_deploy.md §7）。
REMOTE_CMD="cd $WS && mkdir -p $OUT && OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 setsid nohup $VENV -u \
  auto_aim_rl/train/ppo.py --steps $STEPS --envs $ENVS --rollout $ROLLOUT --episode-s $EPISODE_S \
  --randomize --seed 0 --out $OUT $EXTRA > $OUT/train.log 2>&1 < /dev/null & echo TRAIN_STARTED"
# 注意：本脚本在 **Mac** 上跑，而 macOS 默认没有 `timeout`（要 coreutils 的 gtimeout）——
# 所以这里探测一下，都没有就退化成"不加超时"（宁可可能卡住，也不要静默不启动）。
SSH_TMO=""
if command -v timeout >/dev/null 2>&1; then
  SSH_TMO="timeout 60"
elif command -v gtimeout >/dev/null 2>&1; then
  SSH_TMO="gtimeout 60"
fi
# shellcheck disable=SC2086
$SSH_TMO ssh -n -o BatchMode=yes "$HOST" "$REMOTE_CMD" || true

echo
echo "日志：   ssh $HOST 'tail -f $OUT/train.log'"
echo "进度：   ssh $HOST 'tail -3 $OUT/log.jsonl'"
echo "停止：   ssh $HOST 'pkill -f \"[p]po.py\"'"
