#!/usr/bin/env bash
# =============================================================================
#  配 x86 <-> Jetson 的**网线直连**（静态 IP，不走 DHCP / 网关）
#
#  为什么需要：拆分架构下 x86 每秒要拉 20 帧、每帧 ~52 KB 的 /grab。
#  实测 Wi-Fi：`/grab` 中位 40 ms、抖动到 80 ms ⇒ C++ 主循环掉到 6 FPS；
#  直连千兆网线可以做到亚毫秒、几乎零抖动（两台都有空闲千兆口）。
#
#  用法（在 Mac 上执行）：
#      bash tools/dual_machine/setup_wired_link.sh             # 配置 + 自检
#      PERSIST=1 bash tools/dual_machine/setup_wired_link.sh   # 额外写进 NetworkManager（重启后仍在）
#      bash tools/dual_machine/setup_wired_link.sh status      # 只看状态，不改动
#
#  地址约定（避开家宽 192.168.1.0/24）：
#      x86    enp87s0    10.10.10.1/24
#      jetson enP8p1s0   10.10.10.2/24
#  两台都要 sudo：x86 免密；Jetson 走 ssh_sudo.exp 回口令（口令读自 远程服务器.md）。
# =============================================================================
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
MODE="${1:-apply}"

X86="${X86:-x86}";             JETSON="${JETSON:-jetson}"
X86_IF="${X86_IF:-enp87s0}";   JETSON_IF="${JETSON_IF:-enP8p1s0}"
X86_IP="${X86_IP:-10.10.10.1}"; JETSON_IP="${JETSON_IP:-10.10.10.2}"
PFX="${PFX:-24}"
PERSIST="${PERSIST:-0}"

say() { printf '  %s\n' "$*"; }

# 网线插了没：/sys/class/net/<if>/carrier = 1 表示链路已建立
link_state() { ssh -o BatchMode=yes "$1" "cat /sys/class/net/$2/carrier 2>/dev/null || echo 0"; }

show_status() {
  for pair in "$X86 $X86_IF" "$JETSON $JETSON_IF"; do
    set -- $pair
    say "$1/$2: carrier=$(link_state "$1" "$2")  $(ssh -o BatchMode=yes "$1" "ip -4 -o addr show dev $2 | awk '{print \$4}' | head -1" 2>/dev/null)"
  done
}

if [ "$MODE" = "status" ]; then
  echo "=== 直连状态 ==="; show_status; exit 0
fi

# --- 1) 先确认链路 ---
echo "=== 1) 链路状态 ==="
show_status
c1=$(link_state "$X86" "$X86_IF"); c2=$(link_state "$JETSON" "$JETSON_IF")
if [ "$c1" != "1" ] || [ "$c2" != "1" ]; then
  say "⚠️ 至少一端 carrier=0 ⇒ 网线还没插好（或被写进别的网络配置）。"
  say "   请把那根网线插到 x86 的 $X86_IF 与 Jetson 的 $JETSON_IF（直连或过交换机都行），再跑一次。"
  exit 1
fi

# --- 2) 配地址（幂等：ip addr replace）---
echo "=== 2) 配置静态地址 ==="
say "$X86  $X86_IF  -> $X86_IP/$PFX"
ssh -o BatchMode=yes "$X86" "sudo -n ip addr replace $X86_IP/$PFX dev $X86_IF && sudo -n ip link set $X86_IF up" \
  && say "OK" || { say "失败（x86 sudo 免密可用？）"; exit 1; }

say "$JETSON  $JETSON_IF -> $JETSON_IP/$PFX"
expect -f "$HERE/ssh_sudo.exp" "$JETSON" nvidia \
  "sudo ip addr replace $JETSON_IP/$PFX dev $JETSON_IF && sudo ip link set $JETSON_IF up" 2>&1 | tail -2

# --- 3) 自检 ---
echo "=== 3) 自检（ping + HTTP）==="
ssh -o BatchMode=yes "$X86" "ping -c 3 -q $JETSON_IP 2>&1 | tail -2"
ssh -o BatchMode=yes "$X86" "printf '/state.json over wired: '; curl -s -o /dev/null -w '%{http_code} %{time_total}s\n' --max-time 5 http://$JETSON_IP:8099/state.json || true"
ssh -o BatchMode=yes "$X86" "for i in 1 2 3 4 5; do curl -s -o /dev/null -w '%{time_total}s ' --max-time 5 http://$JETSON_IP:8099/grab; done; echo"

# --- 4) 可选持久化（NetworkManager）---
if [ "$PERSIST" = "1" ]; then
  echo "=== 4) 写进 NetworkManager（重启后仍在）==="
  ssh -o BatchMode=yes "$X86" "sudo -n nmcli con add type ethernet ifname $X86_IF con-name rm-link ip4 $X86_IP/$PFX 2>&1 | tail -1; sudo -n nmcli con mod rm-link ipv4.method manual ipv4.never-default yes 2>&1 | tail -1" || true
  expect -f "$HERE/ssh_sudo.exp" "$JETSON" nvidia \
    "nmcli con add type ethernet ifname $JETSON_IF con-name rm-link ip4 $JETSON_IP/$PFX; nmcli con mod rm-link ipv4.method manual ipv4.never-default yes" 2>&1 | tail -2
else
  say "（提示：加 PERSIST=1 可把配置写进 NetworkManager，否则重启后失效；本次是临时配置）"
fi
say "完成。跨机跑闭环时用：SIM_BASE_URL=http://$JETSON_IP:8099"
