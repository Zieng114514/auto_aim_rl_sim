#!/usr/bin/env bash
# =============================================================================
#  训练吞吐扫描：同一配置下换并行环境数，量 steps/s（口径 = wall_s，含 PPO 更新）
#
#  **在目标机上执行**（不是 Mac 上）：
#      B64=$(base64 < tools/dual_machine/train_sweep.sh | tr -d '\n')
#      ssh x86 "echo $B64 | base64 -d > /tmp/train_sweep.sh; bash /tmp/train_sweep.sh x86 4 8 16 24 32"
#
#  实测（2026-10-15）：x86 4/8/12/16/24/32 → 198/392/554/509/602/630 步/s；
#                      jetson 2/4/8/12/16 → 55/95/154/149/148 步/s（8 核饱和）。
#  详见 agnet_logs/20261015_1342_train_on_new_machines.md 与 tools/dual_machine/README.md §7。
# =============================================================================
set -uo pipefail
TAG="${1:?用法: train_sweep.sh <tag> <envs...>}"; shift
V="$HOME/mjlab-env/venv/bin/python"
cd "$HOME/vision" || exit 1

echo "=== $TAG 吞吐扫描（steps=8192 rollout=256 episode-s=90 randomize）==="
for N in "$@"; do
  OUT="auto_aim_rl/runs/bench_${TAG}_e${N}"
  rm -rf "$OUT"; mkdir -p "$OUT"
  start=$(date +%s)
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 timeout 1800 "$V" -u auto_aim_rl/train/ppo.py \
    --steps 8192 --envs "$N" --rollout 256 --episode-s 90 --randomize --seed 0 \
    --out "$OUT" > "$OUT/train.log" 2>&1
  rc=$?
  "$V" - "$OUT" "$N" "$TAG" "$rc" "$(( $(date +%s) - start ))" <<'PY'
import json, os, sys
out, n, tag, rc, total = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]
p = os.path.join(out, "log.jsonl")
last = None
if os.path.exists(p):
    for line in open(p):
        line = line.strip()
        if line:
            last = json.loads(line)
if last is None:
    print("  %-7s envs=%-3s  FAILED(rc=%s, 总耗时 %ss) —— 看 %s/train.log" % (tag, n, rc, total, out))
    raise SystemExit(0)
rate = last["steps"] / max(last["wall_s"], 1e-9)
print("  %-7s envs=%-3s  步数=%-6d 训练耗时=%6.1fs -> %6.1f 步/s（含 PPO 更新；进程总耗 %ss）"
      % (tag, n, last["steps"], last["wall_s"], rate, total))
PY
done
echo "SWEEP_${TAG}_DONE"
