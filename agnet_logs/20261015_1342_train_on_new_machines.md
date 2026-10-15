# 2026-10-15 13:45 训练流程搬到新机器（x86 / Jetson）+ 吞吐对比

> 承接 `20261015_1250_dual_machine_deploy.md`。负责人拍板 A 方案后，训练的落点要重建：
> 旧流程是"Mac 上 `tools/remote_train.sh` → 远程 T4 机（root@183.239.162.232:30046，
> `/root/gpufree-data`）"，那台机器已退役。本节记录新流程、实测吞吐与操作约束。

## 1. 做了什么

| 文件 | 改动 |
|---|---|
| `tools/remote_train.sh` | **重写**：默认 `TRAIN_HOST=x86`，可 `TRAIN_HOST=jetson`；环境数按机器给默认值（x86→12 / jetson→6）；`SYNC=1` 从 Mac rsync 代码过去；远端用 `setsid` + `< /dev/null` 起、本地 `timeout` 兜底（跨机踩过的坑）；产物落在 `<host>:~/vision/auto_aim_rl/runs/<run>/` |
| `tools/remote_setup.sh` | 标注为**历史脚本**（T4 专用），指向 `tools/dual_machine/{x86,jetson}_setup.sh` |
| `tools/dual_machine/x86_setup.sh` | 补上训练侧依赖：**gymnasium / scikit-learn / onnx / CPU 版 torch**（原来只装了仿真侧，导致 `ppo.py` 直接 `ModuleNotFoundError: gymnasium`） |
| `tools/README.md` / `tools/dual_machine/README.md` | 训练入口与"能不能同时跑训练和评测"的约束写清楚 |

## 2. 吞吐实测（口径：`--steps 8192 --rollout 256 --episode-s 90 --randomize --seed 0`）

`wall_s` 取自 `runs/<run>/log.jsonl`（**含 PPO 网络更新**；环境构建不计入），两台机器并行扫：

| 并行环境数 | **x86 小主机（16 线程）** | **Jetson Orin NX（8 核 + Orin GPU）** |
|---|---|---|
| 2 | — | 54.5 步/s |
| 4 | 198.4 步/s | 94.6 步/s |
| 8 | 392.0 步/s | **154.3 步/s** |
| 12 | 553.5 步/s | 149.1 步/s |
| 16 | 508.8 步/s | 147.9 步/s（饱和） |
| 24 | 602.4 步/s | — |
| 32 | **630.2 步/s** | — |

结论：

* **x86 是 Jetson 的约 4 倍**（630 vs 154），也是旧远程 T4（cgroup 限 8 核，230 步/s）的 **2.7 倍**。
* x86 到 32 个环境（2× 超订）仍在涨；**12~16 个环境已有峰值的 80~87%**，日常用 16 就够。
* Jetson 在 **8 个环境**就饱和（8 个物理核），再多只有噪声级差异；GPU 对这个小 MLP 基本无感
  （瓶颈始终是 MuJoCo 单步物理）。
* 默认并行环境数已按此改：`remote_train.sh` 里 x86→12、Jetson→6。

## 3. 操作约束（重要）

* **训练（16+ envs）会吃满 x86 的 16 线程** ⇒ 与"x86 跑 sp_vision 闭环"**互斥**：
  评测/联调时不要同时训练；要同时跑就把 envs 压到 4~6。
* 大强度训练**不要放在 Jetson**：它要负责 20~24 FPS 的仿真，抢 CPU 会直接拉低闭环帧率
  （而帧率是要随结果一起记录的指标）。
* 两台都用 `~/mjlab-env/venv`；**mujoco 必须钉 3.11.0**（Jetson 侧还要 `numpy<2`）。

## 4. 复现命令

```bash
# 起一轮训练（默认 x86，12 环境；SYNC=1 先把 Mac 上的代码同步过去）
bash tools/remote_train.sh stage1_a 200000 16
TRAIN_HOST=jetson bash tools/remote_train.sh stage1_a 200000 8
# 看进度/日志
ssh x86 'tail -3 ~/vision/auto_aim_rl/runs/stage1_a/log.jsonl'
# 吞吐复现（就是上面那张表）
ssh x86 'bash /tmp/train_sweep.sh x86 4 8 16 24 32'      # 脚本见下
```
> 吞吐扫描脚本 `train_sweep.sh` 目前是临时脚本（在 /tmp）——它只是把
> `ppo.py` 跑几个 envs 值再读 `log.jsonl` 的 `steps/wall_s`，需要固化时再进 `tools/`。

## 5. 下一步

1. 用新流程**重跑一轮正式训练**（对照旧 T4 的 stage1~5 存档）——建议 x86 + 16 envs，200k~500k 步
   （按 508 步/s 算，500k 步 ≈ 16 分钟，比旧 T4 的 ~36 分钟快一倍多）。
2. 网线到位后：跨机闭环 20 FPS 复测 → 重录基线（`benchmark_2026_legacy` / `lift`）。
3. 吞吐扫描脚本已固化进 `tools/dual_machine/train_sweep.sh`（在目标机上跑）。

## 6. 本轮踩的坑（新增两条）

1. **macOS 没有 `timeout`**：`remote_train.sh` 是在 **Mac** 上跑的，第一版直接写 `timeout 60 ssh ...` ⇒
   `command not found`，**训练静默没起来**（脚本还把"日志/进度"提示打印得好好的，很容易被骗过去）。
   已改成探测 `timeout` / `gtimeout`，都没有就退化成不加超时。
   **教训**：跨机脚本必须在**实际执行的 shell** 上真跑一遍才算完成（`bash -n` 查不出这类问题）。
2. **x86 漏装 `gymnasium`**：`x86_setup.sh` 原先只装仿真侧依赖 ⇒ `ppo.py` 立刻
   `ModuleNotFoundError: No module named 'gymnasium'`（扫描三个 envs 值全 1~2 s 就"失败"）。
   已在 `x86_setup.sh` 补 `gymnasium / scikit-learn / onnx + CPU 版 torch`。

## 7. 验证记录

* `bash tools/remote_train.sh smoke_x86 8192 8` ⇒ `TRAIN_STARTED`，
  `upd 2/2 步数 8192 21.5s`（**381 步/s**，与扫描的 8 envs 392 步/s 一致），
  产物 `runs/smoke_x86/{train.log, log.jsonl, policy.pt}` 齐全，进程正常退出。
