# ball_detection（中文文档）

轮椅网球机器人实时网球检测、三维定位与弹道预测模块。

两套实现并存，按硬件选择：

| | ZED（生产） | D455（独立） |
|---|---|---|
| **硬件** | ZED 2 双目相机 + Jetson Nano | Intel RealSense D455 |
| **语言** | C++（CUDA 加速） | Python |
| **入口** | `src/ball_detection.cpp` | `ball_detection_d455.py` |
| **输出** | `/ball_detection` ROS 话题 | 终端 + OpenCV 窗口 |
| **依赖** | ZED SDK、OpenCV 4.5.2+、CUDA | pyrealsense2、opencv-python |

---

## ZED / 生产版

在 catkin 工作空间中编译，需要 ZED SDK 和 CUDA：

```bash
catkin_make --pkg ball_detection
roslaunch ball_detection ball_detection.launch
```

配置文件：`config/settings.yaml`（HSV 范围、MOG2 参数、测量协方差多项式）

AprilTag 外参标定见 [ball_calibration](../ball_calibration/README.md)。

---

## D455 / 独立 Python 版

无需 ROS 或 ZED SDK。

### 目录结构

```
ball_detection/
├── config/
│   ├── d455.yaml            — 所有运行时参数（见下方配置说明）
│   └── settings.yaml        — ZED / ROS 参数
├── logs/                    — 录制的轨迹 JSON 文件（已加入 .gitignore）
├── models/                  — YOLO 模型权重（首次运行自动下载）
├── src/
│   └── ball_detection.cpp   — ZED C++ 实现
├── ball_detection_d455.py   — 主程序
├── calibrate_rest.py        — CLI 工具：从录制轨迹拟合恢复系数
├── calibrate_web.py         — 网页交互式恢复系数标定
├── generate_apriltag.py     — 生成 AprilTag PDF 用于地面标定
├── tune_hsv_web.py          — 网页交互式 HSV / MOG2 调参
├── viz3d.py                 — 三维弹道网页查看器（Three.js）
└── webview.py               — 二维相机 + 球场俯视图查看器
```

### 文件说明

| 文件 | 功能 |
|---|---|
| `ball_detection_d455.py` | 主程序：相机、检测、EKF、可视化、内置 MJPEG 服务器 |
| `viz3d.py` | 三维弹道查看器：Three.js 轨迹 + 弹跳点，含恢复系数滑块 |
| `webview.py` | 二维查看器：标注相机画面 + 球场俯视图 |
| `tune_hsv_web.py` | 网页交互 HSV / MOG2 实时调参 |
| `calibrate_rest.py` | CLI 工具：从录制的反弹轨迹拟合 rest_x/y/z |
| `calibrate_web.py` | 网页 UI：交互式选段 + 可视化拟合结果 |
| `generate_apriltag.py` | 生成 A4 AprilTag PDF 贴地面用于坐标系标定 |
| `config/d455.yaml` | 所有运行参数；命令行参数优先级高于此文件 |

### 环境配置

```bash
conda activate catchball   # 需要 pyrealsense2 2.57+、opencv-contrib-python、numpy、scipy、ultralytics
```

Linux 非 root USB 权限（一次性）：

```bash
wget https://raw.githubusercontent.com/IntelRealSense/librealsense/master/config/99-realsense-libusb.rules
sudo cp 99-realsense-libusb.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules && sudo udevadm trigger
```

### 快速开始

```bash
cd software/ball_detection

# 默认：读取 config/d455.yaml 所有设置
python ball_detection_d455.py

# 降低分辨率，提高帧率（最高 60fps）
python ball_detection_d455.py --width 848 --height 480

# 指定检测器
python ball_detection_d455.py --detector yolo
python ball_detection_d455.py --detector both    # 左右分屏对比

# 无界面 / 录制
python ball_detection_d455.py --no-viz
python ball_detection_d455.py --record output.mp4

# 录制弹道用于恢复系数标定
# 传目录路径 → 自动生成 logs/traj_YYYYMMDD_HHMMSS.json
python ball_detection_d455.py --save-traj logs/

# 命令行覆盖 HSV 参数（优先级最高）
python ball_detection_d455.py --h-low 25 --h-high 80 --s-min 80 --v-min 80
```

### 网页查看器

在 `config/d455.yaml` 中启用（或通过命令行传入），然后用浏览器打开：

```yaml
viz3d:   true   # 三维 Three.js 查看器  → http://localhost:5001
webview: true   # 二维相机 + 球场      → http://localhost:5002
```

```bash
# 终端 1 — 主检测进程（同时启动 MJPEG 服务器 + UDP 状态广播）
python ball_detection_d455.py

# 终端 2 — 三维弹道查看器
python viz3d.py          # 浏览器打开 http://localhost:5001

# 终端 3 — 二维相机 + 球场查看器
python webview.py        # 浏览器打开 http://localhost:5002
```

两个查看器均提供实时**恢复系数滑块**（rest X/Y/Z），可在不重启检测进程的情况下即时更新 EKF 弹跳模型。

---

## 配置 — `config/d455.yaml`

所有参数均在文件中有注释说明，命令行参数可覆盖任意值。

### 分辨率 / 帧率
```yaml
width:  1280    # 848×480 可跑 60fps
height: 720
```

### 地面坐标系（冷启动初始值）
```yaml
camera_height: 0.5    # 相机光心到地面高度（米）；0 = 不启用世界坐标系
camera_pitch:  -10.0  # 相机俯仰角（度）；向下为负
```
启动后一旦检测到 AprilTag，这两个值就由 EMA 自动更新，无需精确测量。

### AprilTag 地面标定
```yaml
tag_family: tag36h11         # 与打印的标签一致
tag_ids:    [0, 1, 2, 3]     # 接受的标签 ID；每帧自动选面积最大的
tag_size_m: 0.15             # 黑色方块外缘边长（米）；0 = 不启用
```

### 检测后端
```yaml
detector: hsv    # hsv | yolo | both
```

### HSV 颜色范围
```yaml
h_low:  25    # OpenCV 色调 0–179
h_high: 80
s_min:  100   # 饱和度 0–255
v_min:  100   # 亮度 0–255
```
用 `tune_hsv_web.py` 交互调参，调好后直接粘贴。

### MOG2 运动滤波
```yaml
motion: false   # true = HSV ∩ MOG2（减少静态背景误检）
                # false = 纯 HSV（背景复杂或相机运动时推荐）
```

### HSV 检测阈值
```yaml
mog2_threshold: 50    # MOG2 灵敏度：越大越不敏感
min_radius_px:  3     # 最小检测半径（像素）
circularity:    0.55  # 最小圆度（0–1）；网球约 0.55–0.80
```

### YOLO 参数（detector: yolo 或 both 时有效）
```yaml
yolo_model: models/yolov8n.pt
yolo_imgsz: 480
yolo_conf:  0.3
```

### 物理参数
```yaml
coeff_drag: 0.47   # 空气阻力系数 C_d，网球约 0.47–0.65
rest_x: 0.75       # 水平方向恢复系数（弹跳后速度保留比例）
rest_y: 0.75
rest_z: 0.75       # 竖直方向恢复系数
```

### 可视化 / 录制
```yaml
viz:       true     # 显示 OpenCV 窗口
show_mask: false    # 显示 HSV+MOG2 掩码面板
traj:      true     # 显示预测弹道（红线 + 青色弹跳点）
record:    false    # false | true（自动命名）| "filename.mp4"
```

### 网页查看器端口
```yaml
# 三维查看器 (viz3d.py)
viz3d:      true       # 向 viz3d.py 广播 UDP 状态
viz3d_port: 5565       # viz3d.py 监听的 UDP 端口
ctrl_port:  5566       # 接收滑块反馈的 UDP 端口

# 二维查看器 (webview.py)
webview:      true     # 启用 UDP 状态 + 内置 MJPEG 服务器
webview_port: 5567     # webview.py 监听的 UDP 端口
mjpeg_port:   5568     # 内置全分辨率 MJPEG HTTP 服务器端口
```

相机画面由 ball_detection 进程的**内置 MJPEG 服务器**直接以全分辨率推送（无 UDP 帧传输，无大小限制）。

---

## 内部流程

### 1 · 坐标系

```
RealSense 彩色传感器
    光学坐标系（OpenCV 标准）
        X → 右，Y ↓ 下，Z → 前（深度轴）
        │
        │  optical_to_body()
        ▼
    机体坐标系（EKF 使用）
        X → 前，Y → 左，Z ↑ 上（重力 = −Z）
        │
        │  body_to_world()（仅在地面平面已知时）
        ▼
    世界坐标系（球场地面）
        Z = 0 = 地面，Z ↑ 上（EKF 状态 Z 过零时触发弹跳）
```

- `optical_to_body`：`[X,Y,Z]_body = [Z_opt, −X_opt, −Y_opt]`
- `body_to_world`：绕 Y 轴旋转俯仰角 + 高度偏移

### 2 · AprilTag 地面检测

每帧：
1. `cv2.aruco.detectMarkers` 检测 tag36h11 标签
2. `cv2.solvePnP` 解算标签位姿（光学坐标系下）
3. 从旋转矩阵推算 `camera_height`（相机高度）和 `camera_pitch`（俯仰角）
4. EMA 平滑更新（α=0.3）

状态指示：

| 颜色 | 含义 |
|---|---|
| 绿色 `TAG OK` | 本帧检测到，数值最新 |
| 橙色 `TAG [Nf]` | N 帧前最后看到，使用缓存值 |
| 红色 `NO TAG` | 超过 30 帧未见，使用 yaml 冷启动值 |

### 3 · 球检测

**HSV 后端**（默认，快速）：
1. MOG2 背景减除 → 运动掩码（可选）
2. `cv2.inRange(HSV)` → 颜色掩码
3. 两者取交集（或仅颜色掩码）
4. `findContours` → 按圆度和面积筛选

**YOLO 后端**：YOLOv8n 目标检测，类别 32（sports ball）。

**both 模式**：两者并行运行，各维护独立 EKF 状态。

### 4 · 深度融合

```
视觉深度：depth_vis = fx × BALL_RADIUS / r_px
传感器深度：D455 深度帧 5×5 中位数 + BALL_RADIUS 偏移
融合：比值在 0.5–2.0 内则加权平均（50/50），否则优先视觉深度
```

### 5 · 物理 EKF

**状态**：`[px, py, pz, vx, vy, vz]`

**过程模型**（二次空气阻力 + 重力）：
```
a_drag = C_d · ½ρ · πR² · v² / m
ax = −sign(vx) · a_drag · |vx|/|v|
ay = −sign(vy) · a_drag · |vy|/|v|
az =  g − sign(vz) · a_drag · |vz|/|v|   (g = −9.795 m/s²)
```

**物理常数**：

| 参数 | 数值 | 来源 |
|---|---|---|
| 重力加速度 | −9.79528 m/s² | Atlanta；按场地调整 |
| 空气密度 | 1.225 kg/m³ | |
| 球质量 | 0.0575 kg | ITF 标准 |
| 球半径 | 0.0335 m | ITF 标准 |
| C_d | 0.47 | 可调 |
| 恢复系数 | rest_x/y/z ≈ 0.75 | 可标定 |

### 6 · 弹道预测与弹跳

`PhysicsEKF.rollout()` 以 10 ms 步长向前 Euler 积分 1 秒。

弹跳检测：当积分步跨越 Z=0 时，二次方程精确求解落地时刻，施加恢复系数：
```
vx *= rest_x
vy *= rest_y
vz  = rest_z × |vz|   （方向取反）
```

仅在**世界坐标系激活**时生效（需要 AprilTag 检测到）。

可视化：红色点/线 = 预测轨迹，青色点 = 预测弹跳点。

---

## 工具

### `viz3d.py` — 三维弹道查看器

```bash
python viz3d.py          # 打开 http://localhost:5001
python viz3d.py --port 5001 --udp-port 5565 --ctrl-port 5566
```

| 元素 | 说明 |
|---|---|
| 橙色渐变线 | 历史轨迹（越亮越新） |
| 红色线 | 预测弹道（未来 1 秒） |
| 青色点 | 预测弹跳点 |
| 右侧滑块 | 实时调整 rest X/Y/Z，立即更新 EKF |
| 连接指示灯 | 绿 = 在线，橙 = 数据陈旧，红 = 断连 |

---

### `webview.py` — 二维相机 + 球场查看器

```bash
python webview.py          # 打开 http://localhost:5002
python webview.py --port 5002 --udp-port 5567 --mjpeg-port 5568
```

左侧：全分辨率标注相机画面（MJPEG 直连，无中转）  
右侧：球场俯视图 + 信息面板（速度、坐标、EKF 状态）  
顶部：HSV / YOLO / Both 检测器切换按钮（实时生效）

---

### `tune_hsv_web.py` — 交互式 HSV 调参

```bash
python tune_hsv_web.py                          # 1280×720
python tune_hsv_web.py --width 848 --height 480
# 打开 http://localhost:5000
```

四宫格 MJPEG 画面：原图 + HSV 掩码 + MOG2 运动掩码 + 合并掩码。  
浏览器滑块：`H low/high`、`S min`、`V min`、`MOG2 threshold`、`Min radius`、`Circularity`。  
"Copy YAML" 按钮：一键复制调好的参数，粘贴到 `config/d455.yaml`。

---

### `generate_apriltag.py` — 生成 AprilTag 标定板

```bash
python generate_apriltag.py --family tag36h11 --tag-id 1 --tag-size-mm 150
# 输出：apriltag_tag36h11_id1_150mm_a4.pdf
```

以 100% 比例打印（不要"缩放以适合页面"），过塑后贴在球场地面。  
`tag_size_m` 填**黑色方块外缘**边长，打印后用尺子量实际尺寸确认。

---

### `calibrate_rest.py` — CLI 恢复系数标定

从录制的反弹轨迹自动拟合 `rest_x`、`rest_y`、`rest_z`。

```bash
# 1. 录制：确保 AprilTag 在视野内，将球从约 0.8m 高处丢下，连续弹跳 5+ 次
python ball_detection_d455.py --save-traj logs/

# 2. 查看 Z(t) 总览图，确认检测到的弹跳点
python calibrate_rest.py logs/traj_20260511_120000.json --plot

# 3. 排除走过相机/球滚动的时间段，重新拟合
python calibrate_rest.py logs/traj_20260511_120000.json --segments "2.5-8.0,14.0-21.5" --plot
```

**两阶段拟合算法：**
- **阶段 1 — `rest_z`**：相邻反弹峰值高度比 `√(h₂/h₁)` 的中位数，完全不依赖 EKF 速度估计
- **阶段 2 — `rest_x/y`**：以固定的 `rest_z` 优化水平恢复系数

正常室内硬地参考值：`rest_z ≈ 0.70–0.75`，`rest_x/y ≈ 0.45–0.65`，RMSE < 5 cm。

---

### `calibrate_web.py` — 网页交互式标定

与 `calibrate_rest.py` 使用相同算法，但提供可视化 UI。

```bash
python calibrate_web.py logs/traj_20260511_120000.json
# 自动打开 http://localhost:5010
```

操作流程：
1. 在 Z(t) 图上**水平拖动**选择干净的弹跳时间段（多次拖动累积选择）
2. 用 checkbox 精细**勾选/取消**具体弹跳编号
3. 点击 **Run Optimization**（耗时 5–20 秒）
4. 查看每个弹跳的 Z 拟合对比图（蓝色实测 vs 橙色虚线预测）
5. 点击 YAML 区域**一键复制**，粘贴到 `config/d455.yaml`

`rest_z` 下方会显示每对相邻弹跳的高度比数值，便于判断数据一致性。

**需要排除的情况：**
- 球在地面滚动（实测 Z 全程贴地）
- 极弱弹起（峰值 < 0.05 m）
- 弹跳中途球离开相机视野（轨迹被截断）
- 一次事件包含两次反弹（实测 Z 中途回到 0 再弹起）
