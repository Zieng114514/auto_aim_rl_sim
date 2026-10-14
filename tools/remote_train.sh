#!/usr/bin/env bash
# =============================================================================
#  在远程训练服务器上起一次 PPO 训练（nohup 后台，日志落盘）
#
#  用法（本地）:
#      RSYNC 代码后： bash tools/remote_train.sh stage1_a 400000 40
#  参数: 1=run 名  2=总步数  3=并行环境数  4=额外参数（可选，原样透传给 ppo.py）
# =============================================================================
set -uo pipefail
HOST="${T4_HOST:-root@183.239.162.232}"
PORT="${T4_PORT:-30046}"
RUN="${1:-stage1_a}"
STEPS="${2:-400000}"
ENVS="${3:-40}"
EXTRA="${4:-}"
ROOT=/root/gpufree-data
WS=$ROOT/vision

ssh -p "$PORT" "$HOST" "mkdir -p $ROOT/runs/$RUN && cd $WS && \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nohup $ROOT/venv/bin/python auto_aim_rl/train/ppo.py \
    --steps $STEPS --envs $ENVS --rollout 512 --episode-s 90 --randomize --seed 0 \
    --out $ROOT/runs/$RUN $EXTRA > $ROOT/runs/$RUN/train.log 2>&1 & echo started pid=\$!"
echo "日志： ssh -p $PORT $HOST 'tail -f $ROOT/runs/$RUN/train.log'"
