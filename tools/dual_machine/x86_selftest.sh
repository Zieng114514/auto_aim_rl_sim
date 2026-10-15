#!/usr/bin/env bash
# x86 上跑 rm_sim 全部自检，打印 PASS/FAIL 汇总
V="$HOME/mjlab-env/venv/bin"
cd "$HOME/vision/rm_sim" || exit 1
pass=0; fail=0
for t in test_*.py; do
  if out=$("$V/python" "$t" 2>&1); then
    printf '%-26s PASS\n' "$t"; pass=$((pass + 1))
  else
    printf '%-26s FAIL\n' "$t"; printf '%s\n' "$out" | tail -8; fail=$((fail + 1))
  fi
done
echo "==== 汇总: PASS=$pass FAIL=$fail ===="
echo "=== 黄金轨迹复验 ==="
cd "$HOME/vision" && "$V/python" auto_aim_rl/scratch/golden_trace.py check 2>&1 | grep -E 'qpos_hash|shots'
echo "=== 文件数 ==="
find "$HOME/vision" -type f | wc -l
