#!/usr/bin/env bash
# 测 sim_ort 在不同 intra_threads 下的单帧推理耗时（x86 本地闭环，检测器耗时与渲染无关）
V="$HOME/mjlab-env/venv/bin"
cd "$HOME/vision/rm_sim" || exit 1
MUJOCO_GL=osmesa nohup "$V/python" -u rm_link.py --model real --port 8099 --seconds 200 --seed 20251012 \
  > /tmp/x86_local.log 2>&1 &
sleep 12
cd "$HOME/vision/sp_vision_25" || exit 1
for T in 4 12 16; do
  cp configs/sim.yaml "/tmp/sim_t$T.yaml"
  if grep -q '^sim_ort_intra_threads:' "/tmp/sim_t$T.yaml"; then
    sed -i "s/^sim_ort_intra_threads:.*/sim_ort_intra_threads: $T/" "/tmp/sim_t$T.yaml"
  else
    sed -i "s|^sim_ort_input_size:|sim_ort_intra_threads: $T\nsim_ort_input_size:|" "/tmp/sim_t$T.yaml"
  fi
  timeout -s INT 20 ./build/sim/sim_main "/tmp/sim_t$T.yaml" > "/tmp/sv_t$T.log" 2>&1
  echo "intra=$T | $(grep -oE 'sim_ort 推理 [0-9.]+ ms/帧' "/tmp/sv_t$T.log" | tail -1) | $(grep -oE '主循环平均 [0-9.]+ FPS' "/tmp/sv_t$T.log" | tail -1) | 识别 $(grep -oE '识别 [0-9]+/[0-9]+' "/tmp/sv_t$T.log" | tail -1)"
done
echo "=== 仿真侧帧率 ==="
grep -E "FPS" /tmp/x86_local.log | tail -2
