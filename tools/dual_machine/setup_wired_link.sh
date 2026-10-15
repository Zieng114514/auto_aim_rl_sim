#!/usr/bin/env bash
# =============================================================================
#  配 x86 <-> Jetson 的**网线直连**（静态 IP，不走 DHCP、不抢默认路由）
#
#  为什么需要：拆分架构下 x86 每秒要拉 20 帧、每帧 ~52 KB 的 /grab。
#  实测 Wi-Fi：`/grab` 中位 40 ms、抖动到 80 ms ⇒ C++ 主循环掉到 4.7 FPS；
#  直连千兆可以做到亚毫秒、几乎零抖动（两台都有空闲千兆口）。
#
#  ⚠️ 顺序与元凶（2026-10-15 踩过）：**先把配置建好、再插线**。
#  两台机器都有一个 NetworkManager **自动生成的有线 profile（DHCP）**；直接插线时
#  NM 会激活它 —— 做 DHCP、可能抢默认路由/重启连接，实测**两台机器的 Wi-Fi 当场掉线**
#  （机器从 192.168.1.x 消失，整段 LAN 延迟从 3 ms 飙到 40~80 ms）。
#  所以 `prepare` 做两件事：① 关掉那个自动 profile 的 autoconnect；② 建我们的静态 profile
#  （`ipv4.never-default yes` + 低优先级 + 大 metric ⇒ 永不抢默认路由）。全过程不碰当前连通性。
#
#  用法（在 Mac 上）：
#      bash tools/dual_machine/setup_wired_link.sh prepare   # ① 插线前（安全，可反复跑）
#      #   —— 这时才去插网线：x86 enp87s0 ↔ jetson enP8p1s0（直连或过交换机都行）——
#      bash tools/dual_machine/setup_wired_link.sh status    # ② 查链路/地址
#      bash tools/dual_machine/setup_wired_link.sh verify    # ③ 自检 ping + HTTP（要亚毫秒）
#      bash tools/dual_machine/setup_wired_link.sh remove    # 需要时拆掉
#
#  地址约定（避开家宽 192.168.1.0/24 与 172.17/19 docker 段）：
#      x86    enp87s0    10.10.10.1/24
#      jetson enP8p1s0   10.10.10.2/24
#  两台都要 sudo：x86 免密；Jetson 走 ssh_sudo.exp 回口令（口令读自 远程服务器.md）。
# =============================================================================
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
MODE="${1:-prepare}"

X86="${X86:-x86}";              JETSON="${JETSON:-jetson}"
X86_IF="${X86_IF:-enp87s0}";    JETSON_IF="${JETSON_IF:-enP8p1s0}"
X86_IP="${X86_IP:-10.10.10.1}"; JETSON_IP="${JETSON_IP:-10.10.10.2}"
PFX="${PFX:-24}"
CON="${CON:-rm-link}"

say() { printf '  %s\n' "$*"; }
jid() { expect -f "$HERE/ssh_sudo.exp" "$JETSON" nvidia "$1" 2>&1 | tail -2; }

# 生成"目标机上的准备脚本"：禁用自动有线 profile + 建静态 profile
gen_prep() {  # $1 = iface, $2 = ip
  cat <<EOF
# ① 关掉 NetworkManager 自动生成的有线 profile（DHCP，插线时会抢路由/触发重配）。
#    注意：这类 profile 的 DEVICE 字段是空的（未绑定设备），必须按 TYPE 找；
#    而且名字里可能有空格（"有线连接 1" / "Wired connection 1"），要按行读、不能 for 拆词。
nmcli -t -f NAME,TYPE con show | awk -F: '\$2=="802-3-ethernet" {print \$1}' | while IFS= read -r p; do
  [ "\$p" = '$CON' ] && continue
  nmcli con mod "\$p" connection.autoconnect no >/dev/null && echo "  已禁用自动 profile: \$p"
done
# ② 建静态 profile：never-default + 大 metric + 低优先级 ⇒ 永不抢 Wi-Fi 的默认路由
nmcli con delete '$CON' >/dev/null 2>&1 || true
nmcli con add type ethernet ifname '$1' con-name '$CON' ipv4.method manual \\
  ipv4.addresses $2/$PFX ipv4.never-default yes ipv6.method disabled autoconnect yes | tail -1
nmcli con mod '$CON' connection.autoconnect-priority -10 ipv4.route-metric 700
nmcli -t -f NAME,DEVICE,AUTOCONNECT con show | grep '^$CON:'
EOF
}

show_status() {
  for pair in "$X86 $X86_IF" "$JETSON $JETSON_IF"; do
    set -- $pair
    printf '  %s/%s: carrier=%s  addr=%s\n' "$1" "$2" \
      "$(ssh -o BatchMode=yes -o ConnectTimeout=6 "$1" "cat /sys/class/net/$2/carrier 2>/dev/null || echo '?'" 2>/dev/null)" \
      "$(ssh -o BatchMode=yes -o ConnectTimeout=6 "$1" "ip -4 -o addr show dev $2 2>/dev/null | awk '{print \$4}' | head -1" 2>/dev/null)"
  done
}

case "$MODE" in
status)
  echo "=== 直连状态 ==="; show_status; exit 0 ;;
remove)
  echo "=== 拆掉 $CON 配置（自动 profile 的 autoconnect 需自行恢复）==="
  say "x86"; ssh -o BatchMode=yes "$X86" "sudo -n nmcli con delete $CON 2>&1 | tail -1" || true
  say "jetson"; jid "nmcli con delete $CON 2>&1 | tail -1"
  exit 0 ;;
verify)
  echo "=== 链路自检（x86 → jetson 有线口）==="
  ssh -o BatchMode=yes -o ConnectTimeout=8 "$X86" "ping -c 5 -q $JETSON_IP 2>&1 | tail -2"
  ssh -o BatchMode=yes "$X86" "printf '  /grab x5: '; for i in 1 2 3 4 5; do curl -s -o /dev/null -w '%{time_total}s ' --max-time 5 http://$JETSON_IP:8099/grab; done; echo"
  exit 0 ;;
prepare)
  echo "=== ① 写配置（禁用自动有线 profile + 建静态 never-default profile）==="
  say "x86  $X86_IF -> $X86_IP/$PFX"
  B=$(gen_prep "$X86_IF" "$X86_IP" | base64 | tr -d '\n')
  ssh -o BatchMode=yes "$X86" "echo $B | base64 -d > /tmp/prep_wired.sh && sudo -n bash /tmp/prep_wired.sh"
  say "jetson $JETSON_IF -> $JETSON_IP/$PFX"
  B2=$(gen_prep "$JETSON_IF" "$JETSON_IP" | base64 | tr -d '\n')
  ssh -o BatchMode=yes "$JETSON" "echo $B2 | base64 -d > /tmp/prep_wired.sh"
  jid "sudo bash /tmp/prep_wired.sh"
  echo
  say "配置就绪。现在去插网线（x86 ${X86_IF} <=> jetson ${JETSON_IF}），插完跑："
  say "  bash tools/dual_machine/setup_wired_link.sh status && bash tools/dual_machine/setup_wired_link.sh verify"
  exit 0 ;;
*)
  echo "用法: $0 {prepare|status|verify|remove}"; exit 2 ;;
esac
