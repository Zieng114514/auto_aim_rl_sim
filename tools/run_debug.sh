#!/usr/bin/env bash
# =============================================================================
#  一键联调：rm_sim + 浏览器 + PlotJuggler + sp_vision debug 画面
#
#  用法：
#      bash tools/run_debug.sh                # 全套
#      bash tools/run_debug.sh --gt           # 用真值识别（隔离检测器）
#      bash tools/run_debug.sh --no-plot      # 不起 PlotJuggler
#      bash tools/run_debug.sh --no-browser   # 不开浏览器
#      bash tools/run_debug.sh --seconds 30   # 跑 30 秒自动退出
#
#  退出：在 sp_vision debug 窗口按 q；**无 DISPLAY 的机器上用 Ctrl-C**
#        （那种情况下画面走 MJPEG，见下面 SHOW_HTTP_PORT 的说明）。
#
#  无头机器（当前部署的 x86 小主机就是）：没有 Xorg/DISPLAY，cv::imshow 根本弹不出窗口，
#  所以脚本会自动把 sp_vision 的 debug 画面改成 MJPEG 出口 ——
#  浏览器打开 http://<本机IP>:<SHOW_HTTP_PORT>（默认 8100）即可，图像与窗口里那张**完全一样**。
#  想改端口： SHOW_HTTP_PORT=8110 bash tools/run_debug.sh
#
#  设计要点（上一版的问题是"某一步静默失败，用户不知道哪步没起来"）：
#    * 每一步都**校验**并明确打印 ✓/✗，日志留在 /tmp 下
#    * PlotJuggler 要等窗口真的出现才继续（snap，起得慢，约 20s）
#    * PlotJuggler 的 UDP 数据源交给 pj_udp_autostart.sh 自动配好并 Start
# =============================================================================
set -uo pipefail

WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PY="$HOME/mjlab-env/venv/bin/python"
PORT=8099
# sp_vision 的 debug 画面（MJPEG）端口：无 DISPLAY 的机器上用它替代 OpenCV 窗口
SHOW_HTTP_PORT="${SHOW_HTTP_PORT:-8100}"
SECONDS_RUN=0
USE_GT=0
OPEN_BROWSER=1
OPEN_PLOT=1
CAR_SCRIPT=0        # --car-script：启用运动基准脚本（评测 RL 用）

ok()  { echo "    ✓ $*"; }
bad() { echo "    ✗ $*"; }
say() { echo "==> $*"; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gt)         USE_GT=1; shift ;;
    --no-browser) OPEN_BROWSER=0; shift ;;
    --no-plot)    OPEN_PLOT=0; shift ;;
    --port)       PORT="$2"; shift 2 ;;
    --seconds)    SECONDS_RUN="$2"; shift 2 ;;
    --car-script) CAR_SCRIPT=1; shift ;;   # 靶车运动基准脚本（固定 10s 一段、四种靶型乱序）
    -h|--help)    sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "未知参数: $1"; exit 1 ;;
  esac
done

DETECT_SRC=$([ "$USE_GT" = 1 ] && echo "gt" || echo "yolo")

fail() { echo "✗ $*" >&2; exit 1; }
[[ -f "$WS/rm_sim/rm_link.py" ]] || fail "找不到 rm_sim/rm_link.py"
[[ -x "$VENV_PY" ]] || fail "找不到 $VENV_PY"
[[ -x "$WS/sp_vision_25/build/sim/sim_main" ]] || \
  fail "还没编译 sim_main： cd $WS/sp_vision_25 && bash build.sh"

PIDS=()
cleanup() {
  echo
  say "收尾"
  for p in "${PIDS[@]:-}"; do kill "$p" 2>/dev/null; done
  pkill -9 -f "rm_[l]ink" 2>/dev/null
  # PlotJuggler 必须单独 pkill：它是 snap，/snap/bin/plotjuggler 只是个壳，
  # 真正的进程是壳 fork 出来的，kill $! 根本收不到它。
  # （漏掉这一步的后果：残留实例一直占着 9870，下次启动就报
  #   "Couldn't bind to IPv4 UDP server at (127.0.0.1, 9870)"）
  pkill -9 -f "plotju[g]" 2>/dev/null
  sleep 1
  echo "==> 已退出"
}
trap cleanup EXIT INT TERM

# 清端口时要**等它真的空出来**再往下走。
# 只 pkill + sleep 2 是不够的：旧进程可能还没退干净，
# 新实例会和它抢同一个端口，表现为"rm_sim 起不来"但日志是空的。
wait_port_free() {
  local p="$1" proto="$2" i
  for i in $(seq 1 30); do
    if [[ "$proto" = tcp ]]; then
      ss -ltn 2>/dev/null | grep -q ":$p " || return 0
    else
      ss -lun 2>/dev/null | grep -q ":$p " || return 0
    fi
    sleep 0.5
  done
  return 1
}

if ss -ltn 2>/dev/null | grep -q ":$PORT "; then
  say "端口 $PORT 被占用，先清掉旧进程"
  pkill -9 -f "rm_[l]ink" 2>/dev/null
  wait_port_free "$PORT" tcp && ok "端口 $PORT 已释放" || bad "端口 $PORT 一直没释放"
fi

# PlotJuggler 的 9870 也检查一下：上次没退干净的话，这次会报
# "Couldn't bind to IPv4 UDP server at (127.0.0.1, 9870)"
if ss -lun 2>/dev/null | grep -q ":9870"; then
  say "9870 被占用（多半是上次的 PlotJuggler 没退干净），先清掉"
  pkill -9 -f "plotju[g]" 2>/dev/null
  wait_port_free 9870 udp && ok "9870 已释放" || bad "9870 一直没释放"
fi

# ---------------------------------------------------------------- 1/4 rm_sim
say "[1/4] 启动 rm_sim（MuJoCo + 渲染 + /grab /command）"
# -u：Python 重定向到文件时是块缓冲的，进程被 kill 时缓冲区会丢，
# 结果 /tmp/rm_sim_debug.log 里永远看不到启动横幅（云台参数、靶车台数那些）。
# 绑核：MuJoCo 的物理+渲染和 sp_vision 的推理都在 CPU 上抢核心（12 核）。
# 实测不绑核时仿真被拖到 12.4 FPS，绑核后回到 15.9（只跑 rm_sim 是 16.9）。
HALF=$(( $(nproc) / 2 ))
SIM_CPUS="0-$((HALF - 1))"
SV_CPUS="$HALF-$(( $(nproc) - 1 ))"
SIM_PIN=""
SV_PIN=""
if command -v taskset > /dev/null 2>&1; then
  SIM_PIN="taskset -c $SIM_CPUS"
  SV_PIN="taskset -c $SV_CPUS"
  echo "    （绑核：rm_sim -> CPU $SIM_CPUS，sp_vision -> CPU $SV_CPUS）"
fi

# 底盘扰动：**联调默认打开**（人看的场景要真实：镜头会随底盘颠簸/漂移），
# 想回到"干净相机"用 CHASSIS_DISTURB=off。
# 注意：rm_link 自身的默认是 off（保证基准脚本可复现），只有这里显式打开。
DISTURB_ARGS=()
if [[ "${CHASSIS_DISTURB:-on}" != "off" ]]; then
  DISTURB_ARGS=(--chassis-disturb on)
fi
# 靶车运动基准脚本：固定 10s 一段、四种靶型乱序（默认关闭 = 随机旋转移动靶）
CAR_ARGS=()
if [[ "$CAR_SCRIPT" = 1 ]]; then
  CAR_ARGS=(--car-motion script --car-script-dwell "${CAR_SCRIPT_DWELL:-10}")
fi
( cd "$WS/rm_sim" && exec $SIM_PIN "$VENV_PY" -u rm_link.py --model real --port "$PORT" \
    "${DISTURB_ARGS[@]}" "${CAR_ARGS[@]}" ${SIM_ARGS:-} ) \
  > /tmp/rm_sim_debug.log 2>&1 &
PIDS+=("$!")
SIM_OK=0
for _ in $(seq 1 60); do
  curl -s -m 2 "http://127.0.0.1:$PORT/state.json" > /dev/null 2>&1 && { SIM_OK=1; break; }
  sleep 0.5
done
if [[ "$SIM_OK" = 1 ]]; then
  ok "rm_sim 就绪（日志 /tmp/rm_sim_debug.log）"
else
  bad "rm_sim 起不来，最后 20 行日志："
  tail -20 /tmp/rm_sim_debug.log
  exit 1
fi

# ---------------------------------------------------------------- 2/4 浏览器
if [[ "$OPEN_BROWSER" = 1 ]]; then
  say "[2/4] 打开浏览器 http://127.0.0.1:$PORT"
  URL="http://127.0.0.1:$PORT"
  if command -v xdg-open > /dev/null 2>&1; then
    setsid xdg-open "$URL" > /tmp/browser.log 2>&1 &
  else
    setsid "$VENV_PY" -c "import webbrowser,sys;webbrowser.open(sys.argv[1])" "$URL" \
      > /tmp/browser.log 2>&1 &
  fi
  sleep 3
  if pgrep -f "chrome|chromium|firefox" > /dev/null 2>&1; then
    ok "浏览器已启动"
  else
    bad "没检测到浏览器进程 —— 手动打开 $URL"
  fi
else
  say "[2/4] 跳过浏览器"
fi

# ---------------------------------------------------------------- 3/4 PlotJuggler
if [[ "$OPEN_PLOT" = 1 ]]; then
  say "[3/4] 启动 PlotJuggler"
  if ! command -v plotjuggler > /dev/null 2>&1; then
    bad "没装 plotjuggler（sudo snap install plotjuggler），跳过"
  else
    LAYOUT="$WS/tools/plotjuggler_layout.xml"
    if [[ -f "$LAYOUT" ]]; then
      setsid plotjuggler -n -l "$LAYOUT" -p > /tmp/plotjuggler.log 2>&1 &
    else
      setsid plotjuggler -n > /tmp/plotjuggler.log 2>&1 &
    fi
    WID=""
    for _ in $(seq 1 90); do
      WID="$(xdotool search --name '^PlotJuggler$' 2>/dev/null | head -1)"
      [[ -n "$WID" ]] && break
      sleep 0.5
    done
    if [[ -z "$WID" ]]; then
      bad "等 45s 没等到 PlotJuggler 窗口（日志 /tmp/plotjuggler.log）"
    else
      ok "PlotJuggler 窗口已出现"
      # 窗口出现 ≠ 界面画完。xdotool 是按坐标点控件的，界面没画完就会点空。
      # 实测这一步**经常性失败**（同一台机器两次跑一次成一次不成），所以重试 3 轮：
      # 每轮之间多等一会儿让 Qt 把控件画出来，比让用户手动点 4 下划算得多。
      PLOT_OK=0
      for attempt in 1 2 3; do
        sleep $((3 + attempt * 2))
        if bash "$WS/tools/pj_udp_autostart.sh"; then
          PLOT_OK=1
          break
        fi
        [[ $attempt -lt 3 ]] && echo "    …第 $attempt 次自动配置没成，重试"
      done
      if [[ "$PLOT_OK" = 1 ]]; then
        ok "UDP 数据源已自动启动，9870 正在监听 —— 曲线直接能看"
      else
        bad "自动配置没成功（重试 3 次），手动来（约 4 次点击）："
        cat <<'TIP'
            1. 左侧 Streaming 下拉框 → 选 "UDP Server"
            2. 点右边齿轮 → Port 9870、Message Protocol 选 json → OK
            3. 点 Start（若没反应，再按一下回车）
            4. 把 Timeseries List 里的 err_yaw 等拖到绘图区
TIP
      fi
    fi
  fi
else
  say "[3/4] 跳过 PlotJuggler"
fi

# ---------------------------------------------------------------- 4/4 sp_vision
CFG=/tmp/sim_debug.yaml
sed "s/^  detect_source:.*/  detect_source: \"$DETECT_SRC\"/" \
    "$WS/sp_vision_25/configs/sim.yaml" > "$CFG"

say "[4/4] 启动 sp_vision（识别来源 = $DETECT_SRC）"
# 无头机器没有 DISPLAY：cv::imshow 起来就报错，所以自动改走 MJPEG 出口。
# 两条路径是**同一份绘制代码**，所以图像内容不会有差异。
SHOW_ARGS=(--show)
if [[ -z "${DISPLAY:-}" ]]; then
  SHOW_ARGS=("--show_http=$SHOW_HTTP_PORT")
  echo "    本机没有 DISPLAY ⇒ 不开 OpenCV 窗口，debug 画面走 MJPEG："
  echo "      http://<本机IP>:$SHOW_HTTP_PORT   ← 浏览器打开（绿框=检测，黄框=预测瞄点）"
  echo "    退出：Ctrl-C"
else
  echo "    debug 窗口：绿框 = 检测到的装甲板　黄框 = 预测框（瞄准点反投影）"
  echo "    退出：在 debug 窗口按 q"
fi
echo

# sim_ort 后端如果用 device: CUDA，ONNX Runtime 需要能找到 cublas/cudnn 这些
# 动态库（它们躺在 python 的 site-packages 里）。ort_env.sh 就是干这个的，
# 在这里 source 一下，省得每次手动 export。
if [[ -f "$WS/tools/ort_env.sh" ]]; then
  # shellcheck disable=SC1091
  source "$WS/tools/ort_env.sh" >/dev/null 2>&1 || true
fi

cd "$WS/sp_vision_25"
if [[ "$SECONDS_RUN" -gt 0 ]]; then
  timeout -s INT "$SECONDS_RUN" $SV_PIN ./build/sim/sim_main "${SHOW_ARGS[@]}" "$CFG"
else
  $SV_PIN ./build/sim/sim_main "${SHOW_ARGS[@]}" "$CFG"
fi
exit 0
