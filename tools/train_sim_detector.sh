#!/usr/bin/env bash
# =============================================================================
#  在 30HX 上微调仿真域装甲板检测器
#
#  为什么需要它：现役 assets/yolov5.xml 是真实比赛数据训的，喂仿真画面只有约
#  69% 召回（域差）。rm_sim/gen_dataset.py 能自动标注任意多张仿真图，这个脚本
#  用那些数据训一个 YOLO-pose 装甲板检测器（1 类 + 4 个角点关键点），导出 ONNX
#  后交给 sp_vision 的 sim_ort 后端（ONNX Runtime，CPU 或 30HX 都行）。
#
#  用法:
#      tools/train_sim_detector.sh                 # 默认 120 epoch
#      tools/train_sim_detector.sh "" 60           # 只训 60 epoch
#      EPOCHS=200 tools/train_sim_detector.sh
# =============================================================================
set -uo pipefail
WS="$(cd "$(dirname "$0")/.." && pwd)"
PY="$HOME/mjlab-env/venv/bin/python"
YOLO="$HOME/mjlab-env/venv/bin/yolo"

DATA="${1:-$WS/datasets/sim_armor/data.yaml}"
EPOCHS="${2:-${EPOCHS:-120}}"
BATCH="${BATCH:-16}"
IMGSZ="${IMGSZ:-640}"
# 预训练权重：pose 版比 detect 版多一套关键点头，冷启动收敛快很多
MODEL="${MODEL:-$WS/.deps/weights/yolo11n-pose.pt}"
NAME="${NAME:-sim_armor_$(date +%m%d_%H%M)}"

[ -f "$DATA" ] || { echo "✗ 找不到数据集 $DATA（先跑 rm_sim/gen_dataset.py）"; exit 1; }
[ -f "$MODEL" ] || { echo "✗ 找不到权重 $MODEL"; exit 1; }
[ -x "$YOLO" ] || { echo "✗ 找不到 ultralytics CLI（pip install ultralytics）"; exit 1; }

echo "===================== 训练配置 ====================="
echo "  数据    : $DATA"
echo "  权重    : $MODEL"
echo "  epochs  : $EPOCHS   batch: $BATCH   imgsz: $IMGSZ"
echo "  设备    : 30HX (cuda:0)"
"$PY" - <<'PY'
import torch
print("  torch   : %s | cuda=%s | %s" % (
    torch.__version__, torch.cuda.is_available(),
    torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-"))
PY
echo "==================================================="

cd "$WS"
"$YOLO" pose train \
  model="$MODEL" \
  data="$DATA" \
  imgsz="$IMGSZ" \
  epochs="$EPOCHS" \
  batch="$BATCH" \
  device=0 \
  workers=6 \
  project="$WS/runs" \
  name="$NAME" \
  exist_ok=True \
  plots=True \
  val=True \
  seed=0 \
  amp=False \
  pretrained=True \
  cos_lr=True \
  close_mosaic=10 \
  patience=40
rc=$?
echo
if [ $rc -ne 0 ]; then echo "✗ 训练失败（退出码 $rc）"; exit $rc; fi
BEST="$WS/runs/$NAME/weights/best.pt"
ls -la "$BEST" 2>/dev/null || { echo "✗ 没找到 best.pt"; exit 1; }
echo "✓ 训练完成：$BEST"
echo "  下一步：$PY $WS/tools/export_sim_detector.py $BEST"
