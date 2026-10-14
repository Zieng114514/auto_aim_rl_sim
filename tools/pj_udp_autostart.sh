#!/usr/bin/env bash
# =============================================================================
#  自动把 PlotJuggler 的 UDP 数据源配好并启动（best-effort）
#
#  为什么需要它：PlotJuggler 3.17 的 layout **不保存数据源的启动状态**，
#  每次开都要手动选 UDP Server + Start；而鼠标点 Start 实测经常不生效
#  （要先让它获得焦点，再按回车）。这里用 xdotool 把这几步做掉。
#
#  返回码：0=已监听 9870；1=失败（调用方应回退到手动说明）；2=没装 xdotool
#
#  ---- 实测踩过的坑 ----
#  1) 选下拉项**不能用 Up**。
#     原写法假设"默认停在 WebSocket Server，往上一个是 UDP Server" —— 那是**全新
#     启动**的情况。layout 会把"当前选中的 streamer"存下来（只是不存启动状态），
#     只要这个 layout 是在选中 UDP Server 时保存的，载入后选中项就已经是 UDP Server，
#     再按 Up 就变成 Serial Port，Start 起来的是串口，9870 永远不监听。
#     症状很好认：同一天第一次跑成功、之后每次都失败。
#     => 改成 Home + Down：拉到第一项再往下一项，必定是 UDP Server，与当前选中项无关。
#
#  2) windowactivate 在 mutter + snap 下**不可靠**（返回 0，但 getactivewindow 是空的）。
#     键盘事件于是可能落到别的窗口上。所以 windowraise + windowactivate 都做，
#     而且整个序列**重试两轮**，靠 ss 检查 9870 判定成败 —— 不猜，只看结果。
#
#  3) 不要试图"检查光标下的窗口是不是 PlotJuggler"来提前判断：中文输入法的候选窗
#     会跟着光标出现（窗口名就是候选词），永远压在光标位置，这个检查会 100% 失败，
#     反而把本来能成的序列挡掉。踩过，别再写。
# =============================================================================
set -uo pipefail
export DISPLAY="${DISPLAY:-:1}"

command -v xdotool > /dev/null 2>&1 || exit 2
ss -lun 2>/dev/null | grep -q ":9870" && exit 0        # 已经在跑了

WID="$(xdotool search --name '^PlotJuggler$' 2>/dev/null | head -1)"
[ -n "$WID" ] || exit 1

# 坐标是**相对窗口左上角**的固定偏移（左侧面板位置与窗口大小无关）
DROPDOWN_X=129; DROPDOWN_Y=165     # Streaming 那个下拉框
START_X=32;     START_Y=197        # Start 按钮

listening() { ss -lun 2>/dev/null | grep -q ":9870"; }

eval "$(xdotool getwindowgeometry --shell "$WID" 2>/dev/null)"
[ -n "${X:-}" ] || exit 1

for attempt in 1 2; do
  xdotool windowraise "$WID" 2>/dev/null
  xdotool windowactivate --sync "$WID" 2>/dev/null
  sleep 1

  # 1) 展开 Streaming 下拉框
  xdotool mousemove $((X+DROPDOWN_X)) $((Y+DROPDOWN_Y)) click 1
  sleep 1
  # 2) Home 到第一项，Down 到第二项 = UDP Server（列表顺序固定：
  #    Serial Port / UDP Server / WebSocket Server / ZMQ Subscriber）
  xdotool key Home
  sleep 0.3
  xdotool key Down
  sleep 0.4
  xdotool key Return
  sleep 1
  # 3) 点 Start 拿焦点，再回车（纯鼠标点经常不生效）
  xdotool mousemove $((X+START_X)) $((Y+START_Y)) click 1
  sleep 0.6
  xdotool key Return

  # 4) 校验：snap 版起得慢，点完到真正 bind 有可观延迟，给足 12s
  for _ in $(seq 1 24); do
    listening && exit 0
    sleep 0.5
  done
done

exit 1
