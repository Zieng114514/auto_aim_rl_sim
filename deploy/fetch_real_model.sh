#!/usr/bin/env bash
# =============================================================================
#  从上游 GLB **重建**真模型资产（vehicle.glb → rm_sim/assets/obj/*.obj + MJCF）
#
#  什么时候需要它：
#    * 仓库里已带的网格资产被删掉了（比如按发布策略精简了仓库）；
#    * 你想换一份源模型，或想复现"资产是怎么来的"。
#
#  用法：
#      bash deploy/fetch_real_model.sh                    # 自动去找/下 vehicle.glb
#      bash deploy/fetch_real_model.sh --glb <path>       # 用本地已有的 GLB（推荐）
#      GLB_URL=<url> bash deploy/fetch_real_model.sh      # 换下载地址
#      VENV=~/venvs/sim bash deploy/fetch_real_model.sh   # 指定 venv（要有 numpy/trimesh）
#
#  源模型**不属于本仓库**：vehicle.glb（28 MB，AGPL-3.0）来自
#      https://github.com/Blackjack200/bevy_robomaster_simulator  (assets/vehicle.glb)
#  许可与出处见 rm_sim/README.md 的「许可与致谢」。
#
#  ⚠ **重建后请复验黄金轨迹**：`python auto_aim_rl/scratch/golden_trace.py x` 必须打印
#     qpos_hash=c76a72ab6149211b800259e871dad902。转换参数（上轴/压平/贴纸槽位）会影响
#     碰撞几何 ⇒ 影响物理；换了流程就要重新标定基线，别直接混用旧数字。
# =============================================================================
set -uo pipefail

WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${VENV:-$HOME/mjlab-env/venv}"
PY="$VENV/bin/python"
ASSETS="${ASSETS:-$WS/rm_sim/assets}"
GLB=""
FORCE=0
# 上游地址（优先用能通的；raw.githubusercontent 在国内网络经常不通，所以给了 codeload 备选）
GLB_URL="${GLB_URL:-https://raw.githubusercontent.com/Blackjack200/bevy_robomaster_simulator/master/assets/vehicle.glb}"
GLB_URL_FALLBACK="https://codeload.github.com/Blackjack200/bevy_robomaster_simulator/zip/refs/heads/master"
DEFAULT_LOCAL="$HOME/rm-bevy/bevy_robomaster_simulator-master/assets/vehicle.glb"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --glb)   GLB="$2"; shift 2 ;;
    --force) FORCE=1; shift ;;
    -h|--help) sed -n '2,26p' "$0"; exit 0 ;;
    *) echo "未知参数: $1（--help 看用法）" >&2; exit 2 ;;
  esac
done

ok()  { echo "    ✓ $*"; }
bad() { echo "    ✗ $*" >&2; }
say() { echo "==> $*"; }

[[ -x "$PY" ]] || { bad "找不到 $PY —— 先跑 bash deploy/deploy_rm_sim.sh"; exit 1; }
"$PY" -c "import trimesh, numpy" 2>/dev/null || { bad "$VENV 里缺 numpy/trimesh"; exit 1; }

if [[ "$FORCE" == 0 && -f "$ASSETS/obj/chassis.obj" ]]; then
  ok "资产已在位（$ASSETS/obj，$(ls "$ASSETS/obj" | wc -l | tr -d ' ') 个文件）；要强制重建加 --force"
  exit 0
fi

say "[1/3] 找 vehicle.glb"
if [[ -n "$GLB" ]]; then
  [[ -f "$GLB" ]] || { bad "--glb 指向的文件不存在：$GLB"; exit 1; }
  ok "用指定的 $GLB"
elif [[ -f "$DEFAULT_LOCAL" ]]; then
  GLB="$DEFAULT_LOCAL"; ok "用本机已有的 $GLB"
else
  TMP="$(mktemp -d)"
  GLB="$TMP/vehicle.glb"
  say "    本机没有，尝试下载（约 28 MB）"
  if curl -fL --retry 2 -m 300 -o "$GLB" "$GLB_URL" 2>/dev/null && [[ -s "$GLB" ]]; then
    ok "下载成功：$GLB_URL"
  else
    bad "直接下载失败，改试整包 zip（codeload）"
    if curl -fL --retry 2 -m 600 -o "$TMP/src.zip" "$GLB_URL_FALLBACK" 2>/dev/null; then
      ( cd "$TMP" && unzip -q src.zip ) || { bad "解压失败"; exit 1; }
      GLB="$(find "$TMP" -name vehicle.glb | head -1)"
      [[ -n "$GLB" ]] || { bad "zip 里没找到 vehicle.glb"; exit 1; }
      ok "从 zip 里取到 $GLB"
    else
      cat >&2 <<'TIP'
    下载不通（网络/代理/上游不可达）。三个办法任选：
      1) 手动下 vehicle.glb 后： bash deploy/fetch_real_model.sh --glb <路径>
      2) 从已有部署的机器上拷： scp <机器>:~/rm-bevy/*/assets/vehicle.glb .
      3) 先用免网格模型跑： rm_link.py --model simple（其余功能不受影响）
TIP
      exit 1
    fi
  fi
fi

say "[2/3] 转换（rm_glb_to_mjcf.py）"
# 参数与本仓库当前资产一致；换参数会改碰撞几何 ⇒ 物理变 ⇒ 黄金轨迹 hash 变（见文件头警告）
( cd "$WS/rm_sim" && "$PY" rm_glb_to_mjcf.py --src "$GLB" --out-dir "$ASSETS" ) || {
  bad "转换失败"; exit 1; }
ok "OBJ + MJCF 已写入 $ASSETS"

say "[3/3] 校验（真起一次物理，确认模型能加载）"
( cd "$WS/rm_sim" && "$PY" - <<'PY'
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath("rm_gimbal_demo.py")))
import rm_gimbal_demo as R
merged = R.merge_real_assets(os.path.join(os.getcwd(), "assets"), dt=0.001)
import mujoco
m = mujoco.MjModel.from_xml_path(merged)
print("    ✓ 模型加载成功：ngeom=%d nbody=%d mesh=%d" % (m.ngeom, m.nbody, m.nmesh))
PY
) || { bad "模型校验失败"; exit 1; }

echo
ok "重建完成。别忘了复验黄金轨迹（见文件头）："
echo "    $PY auto_aim_rl/scratch/golden_trace.py rebuilt"
