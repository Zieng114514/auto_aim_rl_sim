#!/usr/bin/env python3
"""把多个种子的评测结果汇总成"总体 + 分靶型"的表 —— sp_vision 与 RL 用同一套口径。

用法：
    python3 tools/aggregate_bench.py <label> /tmp/eval_<label>_*.json
输出：
    * 屏幕上的对照表（总体 + 分靶型，均值 ± 极差）
    * sp_vision_25/sim/baselines/<label>.json（供以后对比/回归）
"""
import glob
import json
import os
import sys

WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(WS, "sp_vision_25", "sim", "baselines")

MODE_ORDER = ["inplace_spin", "dash_halt", "move_spin_rand", "move_spin_const"]
MODE_DESC = {
    "inplace_spin": "原地随机甩头",
    "dash_halt": "急停冲刺靶",
    "move_spin_rand": "随机旋转移动靶",
    "move_spin_const": "固定周期旋转移动靶",
}


def mean_range(vals):
    if not vals:
        return "-", "-"
    return "%.1f" % (sum(vals) / len(vals)), "%.1f~%.1f" % (min(vals), max(vals))


def main():
    label = sys.argv[1]
    files = sorted(glob.glob(sys.argv[2])) if len(sys.argv) > 2 else []
    if not files:
        print("没有找到结果文件：", sys.argv[2] if len(sys.argv) > 2 else "(缺参数)")
        return 1

    runs = []
    modes = {}
    for f in files:
        d = json.load(open(f))
        shots, hits = d.get("shots", 0), d.get("hits", 0)
        cf = d.get("carfield", {})
        kl = [x for x in cf.get("kill_log", []) if x.get("ttk_hit") is not None]
        runs.append({
            "seed": os.path.basename(f).rsplit("_", 1)[-1].replace(".json", ""),
            "shots": shots, "hits": hits,
            "rate": 100.0 * hits / max(shots, 1),
            "kills": cf.get("deaths", 0),
            "ttk_hit": sorted(x["ttk_hit"] for x in kl)[len(kl) // 2] if kl else None,
            "shots_per_kill": shots / cf["deaths"] if cf.get("deaths") else None,
            "blocked": (d.get("gun") or {}).get("shots_blocked", 0),
            # 帧率：闭环结果对机器负载敏感（瞄准补偿用实测耗时），必须一起存档，
            # 否则"这次怎么变差了"根本无从判断。
            # **不能写死 300**：本脚本也用于 150 s 的快速对照，写死会把帧率算成一半
            # （实测 150 s、真实 30.3 FPS 被算成 14.7，触发「与对照帧率相差 >10%」的假警报，
            #   差点让一组本来可比的数据被当成不可比）。
            # 用快照里的仿真时间 t（= 这一轮真实跑了多久），没有才退回 300。
            "fps": d.get("frame", 0) / (float(d.get("t") or 0.0) if float(d.get("t") or 0.0) > 1.0 else 300.0),
        })
        for s in (d.get("car_script") or {}).get("segments", []):
            if s.get("t1") is None and s.get("dur", 0) < 1.0:
                continue
            m = modes.setdefault(s["mode"], {"shots": 0, "hits": 0, "kills": 0, "dur": 0.0, "n": 0})
            m["shots"] += s["shots"]; m["hits"] += s["hits"]
            m["kills"] += s["kills"]; m["dur"] += s["dur"]; m["n"] += 1

    print("=" * 78)
    print("策略评测：%s   （%d 个种子）" % (label, len(runs)))
    print("=" * 78)
    print("  %-10s %7s %7s %8s %7s %10s %12s" %
          ("种子", "发射", "命中", "命中率", "击杀", "TTK(首中)", "每击杀弹量"))
    for r in runs:
        print("  %-10s %7d %7d %7.1f%% %7d %9s %12s" % (
            r["seed"], r["shots"], r["hits"], r["rate"], r["kills"],
            ("%.1fs" % r["ttk_hit"]) if r["ttk_hit"] else "-",
            ("%.0f 发" % r["shots_per_kill"]) if r["shots_per_kill"] else "-"))
    mr, rr = mean_range([r["rate"] for r in runs])
    print("  %-10s %7s %7s %7s%% %7s %9s %12s" % ("均值", "-", "-", mr, "-", "-", "-"))
    print("  %-10s %7s %7s %7s  %7s %9s %12s" % ("极差", "-", "-", rr, "-", "-", "-"))

    # 帧率是**可比性前提**：闭环的瞄准补偿用实测耗时，机器一忙帧率就掉、命中率跟着掉
    # （实测 22.1 FPS → 22.7%，28.3 FPS → 34.5%）。所以对比两个策略时帧率必须接近，
    # 相差 10% 以上就该重跑，而不是硬比。
    fps_list = [r["fps"] for r in runs]
    print("  帧率：%s（均值 %.1f FPS，跨度 %.1f）—— 与对照实验的帧率相差 >10%% 时不可直接比"
          % (" ".join("%.1f" % f for f in fps_list), sum(fps_list) / len(fps_list),
             max(fps_list) - min(fps_list)))
    tot_s = sum(r["shots"] for r in runs)
    tot_h = sum(r["hits"] for r in runs)
    tot_k = sum(r["kills"] for r in runs)
    print("\n  合计：%d 发 / %d 中 = %.1f%% 命中，击杀 %d 台，平均每击杀 %.0f 发"
          % (tot_s, tot_h, 100.0 * tot_h / max(tot_s, 1), tot_k, tot_s / max(tot_k, 1)))

    print("\n  分靶型：")
    print("    %-20s %5s %8s %8s %8s %8s" % ("靶型", "段数", "发射", "命中", "命中率", "击杀"))
    summary = {"label": label, "runs": runs, "modes": {}}
    for m in MODE_ORDER:
        if m not in modes:
            continue
        a = modes[m]
        rate = 100.0 * a["hits"] / max(a["shots"], 1)
        print("    %-20s %5d %8d %8d %7.1f%% %8d" %
              (MODE_DESC.get(m, m), a["n"], a["shots"], a["hits"], rate, a["kills"]))
        summary["modes"][m] = {"desc": MODE_DESC.get(m, m), "segments": a["n"],
                               "shots": a["shots"], "hits": a["hits"],
                               "rate": round(rate, 2), "kills": a["kills"]}
    summary["total"] = {"shots": tot_s, "hits": tot_h,
                        "rate": round(100.0 * tot_h / max(tot_s, 1), 2), "kills": tot_k}

    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "%s.json" % label)
    with open(out, "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("\n  已存档：%s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
