#!/usr/bin/env bash
# =============================================================================
#  一键跑完整条"仿真域检测器"流水线
#
#      生成数据集 -> 在 30HX 上训练 -> 导出 ONNX -> 同口径对比现役检测器
#
#  长跑建议这样用（训练要几个小时）：
#      nohup tools/sim_detector_pipeline.sh > /tmp/pipeline.log 2>&1 &
#
#  用法:
#      tools/sim_detector_pipeline.sh                 # 全流程，默认 8000 帧 / 60 epoch
#      FRAMES=2000 EPOCHS=20 tools/sim_detector_pipeline.sh   # 快速试跑
#      SKIP_DATA=1 tools/sim_detector_pipeline.sh     # 复用已有数据集
#      SKIP_TRAIN=1 tools/sim_detector_pipeline.sh    # 只导出+对比已有权重
# =============================================================================
set -uo pipefail
WS="$(cd "$(dirname "$0")/.." && pwd)"
PY="$HOME/mjlab-env/venv/bin/python"
FRAMES="${FRAMES:-8000}"
EPOCHS="${EPOCHS:-60}"
DATA="$WS/datasets/sim_armor"

step() { echo; echo "==================== $* ===================="; }

if [ -z "${SKIP_DATA:-}" ]; then
  step "1/4 生成数据集（$FRAMES 帧）"
  ( cd "$WS/rm_sim" && env DISPLAY="${DISPLAY:-:1}" "$PY" -u gen_dataset.py \
      --out datasets/sim_armor --frames "$FRAMES" --cars 2 --seed 7 ) || exit 1
else
  step "1/4 跳过数据集生成"
fi

LATEST="$(ls -d "$WS"/runs/sim_armor_* 2>/dev/null | sort | tail -1)"
if [ -z "${SKIP_TRAIN:-}" ]; then
  step "2/4 训练（$EPOCHS epoch，30HX）"
  bash "$WS/tools/train_sim_detector.sh" "$DATA/data.yaml" "$EPOCHS" || exit 1
  LATEST="$(ls -d "$WS"/runs/sim_armor_* 2>/dev/null | sort | tail -1)"
else
  step "2/4 跳过训练，用最近一次：$LATEST"
fi

[ -n "$LATEST" ] && [ -f "$LATEST/weights/best.pt" ] || { echo "✗ 找不到 $LATEST/weights/best.pt"; exit 1; }

step "3/4 导出 ONNX + Python 侧召回评测"
"$PY" "$WS/tools/export_sim_detector.py" "$LATEST/weights/best.pt" \
    --out "$WS/sp_vision_25/assets/sim_armor.onnx" --frames /tmp/eval_frames || exit 1

step "4/4 C++ 侧同口径对比"
bash "$WS/sp_vision_25/sim/compare_detectors.sh" /tmp/eval_frames

echo
echo "全部完成。模型：$WS/sp_vision_25/assets/sim_armor.onnx"
echo "把它用起来： 把 configs/sim.yaml 的 yolo_name 改成 sim_ort"
