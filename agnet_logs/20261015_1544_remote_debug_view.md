# 2026-10-15 15:44 恢复"看 YOLO/云台画面"：无头机器上窗口换 MJPEG

## 0. 背景（负责人反馈）

> "之前是通过网页查看全局视角，电脑会开一个窗口显示 YOLO 的画面数据；
> 现在只能看到全局，看不到云台视角的画面。"

双机部署（Jetson 跑 rm_sim、x86 跑 sp_vision）之后，**看画面这件事坏了两处**，都不在算法链路里。

## 1. 根因（逐条查证）

| # | 现象 | 证据 | 原因 |
|---|---|---|---|
| 1 | YOLO 画面窗口不见了 | x86 上 `DISPLAY` 为空、无 `/tmp/.X11-unix`、无 Xorg 进程、`who` 无图形会话 | 该窗口是 `sp_vision_25/sim/sim_main.cpp` 的 `cv::imshow("sp_vision debug")`，**必须有 DISPLAY**；有桌面的旧机器已退役 ⇒ 无头机器上无处可弹 |
| 2 | 网页只剩全局视角 | `rm_sim/rm_link.py` 的 `/` 处理里写着"页面上**只放全局固定视角**……云台那一路接口保留" | `/stream.mjpg`（云台相机原图）一直在，只是**没摆到页面上**；`cam_block` 是个空串 |
| 3 | 从 Mac 打不开 `10.10.10.2:8099` | `nc -z 10.10.10.2 8099` 超时；`ping` 却通 | `10.10.10.x` 是 **x86↔Jetson 网线直连**（专用给 `/grab`、`/command`）；从 Mac 要走**同网段 Wi-Fi**：Jetson `192.168.1.101`、x86 `192.168.1.236`。ping 通是路由器代答，会骗人 |

## 2. 做法

### 2.1 `sim_main --show_http=<端口>`（新增，默认关）

* **同一张图**：debug 图的绘制代码一行没动，只是把结果**先 `publish` 再 `imshow`** ⇒
  网页与窗口（如果有）看到的永远一致，不存在"两边各画一份慢慢漂移"。
* **出口**：`/`（页面）、`/debug.mjpg`（MJPEG 流）、`/debug.jpg`（单帧，给 `curl` 抓图）。
  尺寸沿用 `show_scale`（默认 640×480，与原来的窗口一样）。
* **实现**：`sim/DebugStream` 裸 socket（与文件里给 PlotJuggler 发 UDP 的风格一致），
  零新依赖；accept 线程用 `select` 200 ms 超时以便退出，每条连接一个 detach 线程。
* **默认关**：不传 `--show_http` / yaml `show_http: 0` ⇒ 行为与以前完全一致。

### 2.2 rm_link 网页补回云台相机原图

页面现在是两块：**全局固定视角**（上）+ **云台相机原图**（下，sp_vision 的输入，无检测框）。

### 2.3 脚本层

* `tools/run_debug.sh`：**自动检测 DISPLAY** —— 没有就切 `--show_http=$SHOW_HTTP_PORT`（默认 8100）
  并打印地址（原来无头机器上它会直接报错）。
* `sp_vision_25/sim/bench_ttk.sh`：`SHOW_HTTP=8100 bash bench_ttk.sh …` 透传（跑基准时也能看画面）。

## 3. 踩的坑（都写进 `agnet_experience.md` §1.19）

1. **`cv::CommandLineParser` 对非布尔参数只认 `--name=value`**：
   `--show_http 8100` 时 `8100` 被当成 **config-path** →
   `terminate called after throwing an instance of 'YAML::BadFile' what(): bad file: 8100`
   （报错把人往"YAML 坏了"带）。现在 `sim_main` 构造 parser 前把空格写法规范成 `=` 写法。
2. **`SIGPIPE`**：客户端断开后写 socket 的默认动作是**杀进程** ⇒ 浏览器刷新一下闭环就没了。
   `start()` 里 `signal(SIGPIPE, SIG_IGN)`，`send` 用 `MSG_NOSIGNAL`。
3. **IP 用错**（见上表 #3）。
4. **工艺教训**：把"同步+编译"与"跑测试"放在**同一次并行调用**里，测试跑的是**旧二进制** ——
   明明改了源码行为却没变，白排查了一轮。有依赖关系的命令必须串行。

## 4. 验证证据（全部实测）

| 检查 | 结果 |
|---|---|
| `--help` 里出现 `--show_http` | ✓ |
| `GET /` | 返回页面 HTML ✓ |
| `GET /debug.jpg` | **640×480 有效 JPEG**（15~18 KB；`cv2.imread` → `(480,640,3)`）✓ |
| `GET /debug.mjpg` | 标准 multipart 头（`--framer` + `Content-Type: image/jpeg`）✓ |
| 客户端断开后 | `sim_main` 仍存活（SIGPIPE 处理生效）✓ |
| 未知路径 | `404` ✓ |
| **从 Mac** `http://192.168.1.236:8100/` | **HTTP=200** ✓（真·端到端，走 Wi-Fi） |
| 图的内容 | 顶部信息条 `armors 2 targets 1 cmd yaw +17.7 pitch +8.6 gimbal yaw +19.9 FIRE` + 绿框与类别/置信度标签 ✓（与原窗口同款） |
| Jetson 网页 | 同时含 `全局固定视角` 与 `云台相机原图` 两块 ✓ |
| 空格/`=` 两种写法 | 都能启动 ✓ |

## 5. 日常怎么用（三块画面）

| 看什么 | 地址 |
|---|---|
| 全局视角（靶车运动 + 命中） | `http://192.168.1.101:8099/` 第一块 |
| 云台相机原图（sp_vision 的输入） | 同上，第二块 |
| **YOLO 检测框 + 预测瞄点** | `http://192.168.1.236:8100/`（x86 起 `--show_http=8100`） |

```bash
# 仿真（Jetson）
ssh jetson 'cd ~/vision/rm_sim && setsid env MUJOCO_GL=egl ~/mjlab-env/venv/bin/python -u \
  rm_link.py --model real --port 8099 --host 0.0.0.0 > /tmp/rm_link.log 2>&1 < /dev/null &'
# 视觉（x86）：跨机闭环 + 开 debug 画面出口
ssh x86 'cd ~/vision/sp_vision_25 && SIM_BASE_URL=http://10.10.10.2:8099 \
  ./build/sim/sim_main --show_http=8100 configs/sim.yaml'
```

> 注意：`sim_main` 走 ssh 起时建议加 `ssh -f` 或 `setsid` + 重定向，
> 否则 ssh 会一直挂着等后台进程（本次就撞了几次）。
