#!/usr/bin/env python3
"""两个策略的评测对照表（同一套口径、同一批种子）。

为什么单独写一个：验收标准是"**命中率与击杀时间同时不差**"，光看一个数会选错 ——
放松门限能刷命中率但 TTK 变慢，拉长时长能刷击杀数但每秒输出更低。
（口径定义见 agnet_experience.md §4，基准数据见 §1.13 / §1.14。）

用法：
    python3 tools/compare_policies.py sp_vision_v2 rl_stage1
    python3 tools/compare_policies.py sp_vision_v2 rl_stage1 --seeds 20251012 777 42 12345
"""
import argparse
import json
import os
import sys

WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(WS, "sp_vision_25", "sim", "baselines")

MODE_ORDER = ["inplace_spin", "dash_halt", "move_spin_rand", "move_spin_const"]

# 判定用的指标：越大越好 / 越小越好
ROWS = [
    ("命中率 %", "rate", True, "%.1f"),
    ("击杀台数", "kills", True, "%d"),
    ("每击杀弹量", "shots_per_kill", False, "%.0f"),
    ("TTK 首中→死亡 s", "ttk_hit", False, "%.1f"),
    ("持续时间 s", "dur", True, "%.0f"),
]


def load(label):
    p = os.path.join(BASE, label + ".json")
    if not os.path.exists(p):
        sys.exit("找不到存档 %s（先跑 eval_policy.sh %s）" % (p, label))
    d = json.load(open(p))
    runs = {str(r["seed"]): r for r in d["runs"]}
    return d, runs


def agg(runs, keys=None):
    rs = [r for k, r in runs.items() if keys is None or k in keys]
    shots = sum(r["shots"] for r in rs)
    hits = sum(r["hits"] for r in rs)
    kills = sum(r["kills"] for r in rs)
    ttk = [r["ttk_hit"] for r in rs if r.get("ttk_hit")]
    ttk = sorted(ttk)[len(ttk) // 2] if ttk else None
    dur = sum(r.get("dur", 0) for r in rs)
    return {
        "rate": 100.0 * hits / max(shots, 1), "shots": shots, "hits": hits,
        "kills": kills, "shots_per_kill": (shots / kills) if kills else None,
        "ttk_hit": ttk, "dur": dur, "blocked": sum(r.get("blocked", 0) for r in rs),
    }


def fmt(v, f):
    return "-" if v is None else (f % v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--seeds", nargs="*", default=None, help="只比这批种子（默认取两者交集）")
    args = ap.parse_args()

    da, ra = load(args.a)
    db, rb = load(args.b)
    seeds = args.seeds or sorted(set(ra) & set(rb))
    if not seeds:
        sys.exit("两个存档没有共同种子，无法成对比较")

    A, B = agg(ra, set(seeds)), agg(rb, set(seeds))
    print("=" * 72)
    print("策略对照（%d 个共同种子：%s）" % (len(seeds), " ".join(seeds)))
    print("=" * 72)
    print("  %-18s %14s %14s   %s" % ("指标", args.a, args.b, "胜者"))
    for name, key, bigger_better, f in ROWS:
        va, vb = A.get(key), B.get(key)
        win = "-"
        if va is not None and vb is not None and abs(va - vb) > 1e-9:
            awin = (va > vb) if bigger_better else (va < vb)
            win = args.a if awin else args.b
        print("  %-18s %14s %14s   %s" % (name, fmt(va, f), fmt(vb, f), win))
    print("  %-18s %14d %14d   （越大越差：热量约束下的浪费）" % (
        "过热被拒次数", A["blocked"], B["blocked"]))

    print("\n  分靶型命中率：")
    print("    %-20s %10s %10s   %s" % ("靶型", args.a, args.b, "胜者"))
    for m in MODE_ORDER:
        ma = da["modes"].get(m, {})
        mb = db["modes"].get(m, {})
        va, vb = ma.get("rate"), mb.get("rate")
        if va is None or vb is None:
            continue
        win = args.a if va > vb else (args.b if vb > va else "-")
        print("    %-20s %9.1f%% %9.1f%%   %s" % (
            ma.get("desc", m), va, vb, win))

    # 验收：命中率不低**且** TTK 不长（把前者当"待验收策略"，后者当基线）
    ok = (A["rate"] >= B["rate"]) and (A["ttk_hit"] or 1e9) <= (B["ttk_hit"] or 1e9)
    print("\n  验收（命中率不低 且 TTK 不长）：%s vs %s -> %s" % (
        args.a, args.b, "**达标**" if ok else "未达标（继续迭代）"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
