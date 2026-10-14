#!/usr/bin/env python3
"""浏览器状态条的自检：把 rm_link.py 里**真正下发的那段 JS** 抠出来，用 node 跑一遍。

为什么值得单独测：状态条那段 JS 是**嵌在 Python 字符串里**的，既不过编译器也不进任何
测试，写错了（少个括号、字段名打错）只有人打开浏览器才发现 —— 而"打开浏览器才发现"
正是本项目反复踩的一类坑（改了参数没生效、曲线没数据）。这里用 node 把同一段 JS
喂真实口径的 /state.json 载荷，断言它渲染出来的那行字。

node 不在就跳过（不阻塞其它自检）。
"""
import ast
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
FAILED = []


def check(name: str, ok: bool, detail: str = "") -> None:
  print("  %s %s%s" % ("✓" if ok else "✗", name, ("  " + detail) if detail else ""))
  if not ok:
    FAILED.append(name)


def extract_statusbar_js() -> str:
  """从 rm_link.py 源码里抠出 <script> ... </script> 之间的 JS。

  那段 HTML 是在 do_GET 里就地拼的字符串字面量，所以这里按"字面量"还原：
  取 "<script>" 与 "</script>" 两个标记之间的所有 Python 字符串字面量再拼起来。
  改动了那段代码的写法（比如换成 f-string / 三引号）时这里会报错 —— 这是**故意的**，
  宁可自检报错也不要静默跳过。
  """
  src = open(os.path.join(HERE, "rm_link.py"), encoding="utf-8").read()
  m = re.search(r'"<script>"(.*?)"</script>"', src, re.S)
  if not m:
    raise RuntimeError("在 rm_link.py 里找不到 '<script>' ... '</script>' 这对标记 —— "
                       "状态条改写法了？请同步更新本自检的提取逻辑")
  lits = re.findall(r'"(?:[^"\\]|\\.)*"', m.group(1))
  return "".join(ast.literal_eval(x) for x in lits)


HARNESS = """
const fs = require('fs');
const js = fs.readFileSync(process.argv[2], 'utf8');
const data = JSON.parse(fs.readFileSync(process.argv[3], 'utf8'));
let out = null;
global.document = {getElementById: function () { return {set textContent(v) { out = v; }}; }};
global.fetch = function () { return Promise.resolve({json: function () { return Promise.resolve(data); }}); };
global.setInterval = function () {};
eval(js);
setTimeout(function () { console.log(out === null ? '<null>' : out); }, 50);
"""


def render(js: str, payload: dict, tmp: str) -> str:
  p_js = os.path.join(tmp, "bar.js")
  p_jsn = os.path.join(tmp, "state.json")
  p_run = os.path.join(tmp, "run.js")
  open(p_js, "w", encoding="utf-8").write(js)
  json.dump(payload, open(p_jsn, "w", encoding="utf-8"))
  open(p_run, "w", encoding="utf-8").write(HARNESS)
  out = subprocess.run(["node", p_run, p_js, p_jsn], capture_output=True, text=True, timeout=30)
  if out.returncode != 0:
    raise RuntimeError("node 跑状态条 JS 失败：\n" + out.stderr)
  return out.stdout.strip()


def payload(**kw) -> dict:
  base = {"shots": 10, "hits": 4, "t": 12.34,
          "carfield": {"spawn_count": 1, "hits": 4, "deaths": 0,
                       "cars": [{"alive": True, "idx": 0, "hp": 580}]}}
  base.update(kw)
  return base


if not shutil.which("node"):
  print("跳过：没装 node（状态条是浏览器 JS，只能用 node 验）")
  sys.exit(0)

js = extract_statusbar_js()
print("== 1. 从源码抠出来的 JS 是完整的一段 ==")
check("包含 tick() 且以 tick() 调用收尾", "function tick()" in js and js.rstrip().endswith("tick();"),
      "长度 %d 字符" % len(js))

with tempfile.TemporaryDirectory() as tmp:
  print("\n== 2. 开着升降时要显示读数 ==")
  s = render(js, payload(car_lift={"mode": "periodic", "offset_mm": 132.4, "active": True}), tmp)
  check("显示『车1 升降 +132mm (periodic·动)』", "车1 升降 +132mm (periodic·动)" in s, s)
  s = render(js, payload(car_lift={"mode": "periodic", "offset_mm": 0.0, "active": False}), tmp)
  check("静止窗口显示『(periodic·停)』（一眼看出不是全程都在动）",
        "车1 升降 +0mm (periodic·停)" in s, s)
  s = render(js, payload(car_lift={"mode": "random", "offset_mm": 149.6, "active": True}), tmp)
  check("正值带 + 号、四舍五入到整数", "车1 升降 +150mm (random·动)" in s, s)

  print("\n== 3. 没开升降时不能污染状态条 ==")
  s = render(js, payload(car_lift={"mode": "off", "offset_mm": 0.0}), tmp)
  check("mode=off 时不显示升降", "升降" not in s, s)
  s = render(js, payload(), tmp)          # 老服务端没有 car_lift 字段
  check("car_lift 缺失时不崩、也不显示", "升降" not in s and "仿真 12.3s" in s, s)

  print("\n== 4. 原有读数一个都不能少 ==")
  s = render(js, payload(car_lift={"mode": "periodic", "offset_mm": 12.0}), tmp)
  for want in ("场上 1 台", "发射 10", "命中 4", "命中率 40.0%", "靶车阵亡 0",
               "车0 HP 580/600", "仿真 12.3s"):
    check("保留『%s』" % want, want in s, s)

if FAILED:
  print("\n✗ %d 项未通过：%s" % (len(FAILED), "、".join(FAILED)))
  sys.exit(1)
print("\n✓ 全部通过")
