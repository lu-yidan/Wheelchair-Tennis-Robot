# ball_detection（中文文档）

轮椅网球机器人实时网球检测、三维定位与弹道预测模块。

两套实现并存，按硬件选择：

| | ZED（生产） | D455（独立） |
|---|---|---|
| **硬件** | ZED 2 双目相机 + Jetson Nano | Intel RealSense D455（支持单机双路） |
| **语言** | C++（CUDA 加速） | Python |
| **入口** | `src/ball_detection.cpp` | `ball_detection.py` |
| **输出** | `/ball_detection` ROS 话题 | MJPEG 流 + ZMQ 轨迹发布 |
| **依赖** | ZED SDK、OpenCV 4.5.2+、CUDA | pyrealsense2、opencv-python、pyzmq |

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

无需 ROS 或 ZED SDK。支持单相机和双相机融合两种模式。

### 环境配置

```bash
conda activate catchball
pip install pyzmq   # 仅首次需要
```

Linux 非 root USB 权限（一次性）：

```bash
wget https://raw.githubusercontent.com/IntelRealSense/librealsense/master/config/99-realsense-libusb.rules
sudo cp 99-realsense-libusb.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules && sudo udevadm trigger
```

---

### 双 D455 融合（推荐）

两台相机同时运行，EKF 融合两路测量，向局域网发布预测弹道。

```bash
cd software/ball_detection
./run_dual_d455.sh
```

启动后：

| 地址 | 内容 |
|---|---|
| `http://localhost:8080/monitor.html` | 融合监控页：双路 MJPEG + 球场俯视图 + 侧视图 |
| `http://localhost:5568/main` | D455 A 原始 MJPEG 流 |
| `http://localhost:5668/main` | D455 B 原始 MJPEG 流 |
| `tcp://<本机IP>:5580` | ZMQ PUB，30 Hz 轨迹发布（见下方） |

日志：
```bash
tail -f /tmp/fusion-logs/fusion.log   # 融合 EKF 状态
tail -f /tmp/fusion-logs/d455_a.log   # 相机 A 检测日志
tail -f /tmp/fusion-logs/d455_b.log   # 相机 B 检测日志
```

**相机配置：**

| 文件 | 序列号 | 端口偏移 |
|---|---|---|
| `config/d455_a.yaml` | 260722302887 | 0（默认端口） |
| `config/d455_b.yaml` | 152522251463 | +100 |

---

### 方法对比测试

在两台相机上分别跑不同的检测/深度组合：

```bash
./run_compare.sh hsv-fused yolo-fused     # 对比 HSV vs YOLO（都用深度融合）
./run_compare.sh hsv-vis   hsv-sensor     # 对比纯视觉深度 vs 纯传感器深度
```

可用方法名：`hsv-vis` `hsv-sensor` `hsv-fused` `yolo-vis` `yolo-sensor` `yolo-fused` `both-vis` `both-fused`

---

### 单相机模式

```bash
# 读取 config/d455_a.yaml 所有设置
python ball_detection.py --config config/d455_a.yaml

# 指定检测器
python ball_detection.py --config config/d455_a.yaml --detector yolo
python ball_detection.py --config config/d455_a.yaml --detector both   # 左右分屏对比

# 深度权重（0=纯传感器，1=纯视觉，默认0=传感器）
python ball_detection.py --config config/d455_a.yaml --vis-weight 0.0

# 录制弹道用于恢复系数标定
python ball_detection.py --config config/d455_a.yaml --save-traj logs/
```

---

## ZMQ 轨迹订阅

`fusion.py` 在 **5580 端口**以 30 Hz 发布融合后的预测弹道，供局域网内其他机器人订阅。

### 消息格式

```json
{
  "stamp":    1749562345.123,
  "t_obs":    1749562345.089,
  "detected": true,
  "pos":  [x, y, z],
  "vel":  [vx, vy, vz],
  "traj": [[x, y, z, t], ...]
}
```

| 字段 | 说明 |
|---|---|
| `stamp` | 发布时刻（Unix 秒） |
| `t_obs` | EKF 最后一次收到相机测量的时刻 |
| `detected` | `true` = 200ms 内有新测量；`false` = EKF 惯性预测 |
| `pos` | 当前球位置（米，AprilTag 坐标系，Z 向上） |
| `vel` | 当前球速度（米/秒） |
| `traj` | 预测轨迹，约 200 点覆盖 2 秒；`t` = 距 `stamp` 的秒数 |

**坐标系：** 原点 = AprilTag 中心，Z+ 向上，单位米/秒。

### 订阅示例

```python
import zmq, json

sock = zmq.Context().socket(zmq.SUB)
sock.connect("tcp://192.168.1.100:5580")   # 替换为发布方 IP
sock.setsockopt_string(zmq.SUBSCRIBE, "")
sock.setsockopt(zmq.RCVHWM, 2)            # 只保留最新帧

while True:
    msg = json.loads(sock.recv())
    if not msg["detected"]:
        continue   # EKF 惯性预测，可选择忽略
    traj = msg["traj"]   # [[x,y,z,t], ...]
```

完整最小订阅脚本见 `subscribe_traj.py`（无超时退出，无球时定期打印等待提示）。

### 自测

```bash
# 终端1：./run_dual_d455.sh
# 终端2：
python test_traj_pub.py           # 本机
python test_traj_pub.py --host 192.168.1.100   # 局域网
```

测试脚本会打印每帧延迟、实际发布频率、detected 率，并校验消息格式。

---

## 配置参数

所有参数均可在 yaml 文件中设置，命令行参数优先级更高。

### AprilTag 地面标定
```yaml
tag_family: tag36h11
tag_ids:    [0, 1, 2, 3]
tag_size_m: 0.25          # 黑色方块外缘边长（米）；打印后用尺子量实际值
```
> **注意**：`tag_size_m` 必须准确，否则世界坐标系整体比例错误。

### 检测后端
```yaml
detector:  hsv       # hsv | yolo | both
vis_weight: 0.0      # 深度权重：0=纯传感器，1=纯视觉
```

### HSV 颜色范围
```yaml
h_low:  25    # OpenCV 色调 0–179（网球黄绿约 25–80）
h_high: 80
s_min:  75
v_min:  140
```
用 `tune_hsv_web.py` 交互调参，调好后点"Copy YAML"直接粘贴。

### MOG2 运动滤波
```yaml
motion:         true    # true = HSV ∩ MOG2，减少静态背景误检
mog2_threshold: 50
min_radius_px:  3
circularity:    0.55
```

### 物理参数
```yaml
coeff_drag: 0.47
rest_x: 0.75
rest_y: 0.75
rest_z: 0.65
```

### 可视化
```yaml
viz:       false   # OpenCV 本地窗口（服务器模式下关闭）
show_mask: false   # 显示 HSV+MOG2 掩码
traj:      true    # 显示预测弹道（红线 + 青色弹跳点）
```

### 多相机融合端口
```yaml
cam_id:      d455_a
fusion_port: 5570    # fusion.py 监听端口
port_offset: 0       # 相机 B 用 100，避免端口冲突
```

---

## fusion.py — 多相机 EKF 融合

接收来自多个 `ball_detection.py` 实例的 UDP 测量值，运行单个物理 EKF 融合。

```bash
python fusion.py --port 5570 --webview-port 5571 --traj-pub-port 5580
```

关键参数：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--outlier-m` | 2.0 | 异常值拒绝阈值（米）；新测量距 EKF 状态超过此值则丢弃 |
| `--reset-sec` | 3.0 | 无有效测量超过此秒数则自动重置 EKF |
| `--traj-pub-port` | 0 | ZMQ PUB 端口（0=不发布） |
| `--traj-pub-hz` | 30 | 发布频率（Hz） |
| `--traj-pub-sec` | 2.0 | 预测时长（秒） |
| `--max-tag-age` | 30 | 超过此帧数未见 tag 的包被拒绝 |

---

## 脚本一览

| 脚本 | 功能 |
|---|---|
| `ball_detection.py` | 主检测脚本：相机、HSV/YOLO、EKF、MJPEG 服务器、融合 UDP 发送 |
| `fusion.py` | 多相机 EKF 融合 + ZMQ 轨迹发布 |
| `monitor.py` | HTTP/SSE 中继服务，配合 `monitor.html` |
| `monitor.html` | 融合监控页：双 MJPEG + 球场俯视/侧视 canvas |
| `run_dual_d455.sh` | 一键启动双 D455 + fusion + monitor |
| `run_compare.sh` | A/B 方法对比测试启动器 |
| `subscribe_traj.py` | 轨迹订阅最小示例（无超时，无球时打印等待） |
| `test_traj_pub.py` | ZMQ 发布自测：延迟/频率/格式统计 |
| `tune_hsv_web.py` | 网页交互式 HSV / MOG2 实时调参 |
| `calibrate_rest.py` | CLI：从录制弹跳轨迹拟合 rest_x/y/z |
| `calibrate_web.py` | 网页 UI：交互式选段 + 可视化拟合 |
| `replay.py` | 离线重放原始视频 → 标注视频 + 轨迹 JSON |
| `generate_apriltag.py` | 生成 AprilTag PDF（100% 打印，过塑贴地面） |

---

## 内部流程

### 坐标系

```
RealSense 彩色传感器
    光学坐标系（OpenCV）：X→右，Y↓下，Z→前
        │  optical_to_body()
        ▼
    机体坐标系（EKF）：X→前，Y→左，Z↑上（重力 = −Z）
        │  _to_world()（需要 AprilTag 检测到）
        ▼
    世界坐标系：原点 = AprilTag 中心，Z = 0 = 地面，Z↑上
```

当多台相机共用同一 tag 时，世界坐标系自动对齐（无需 `tag_world_map.yaml`）。
多 tag 场景下配合 `tag_world_map.yaml` 可将不同 tag 的本地坐标统一到同一世界系。

### 球检测

**HSV 后端**：`cv2.inRange(HSV)` → 可选 MOG2 运动掩码取交集 → `findContours` 按圆度和面积筛选

**YOLO 后端**：YOLOv8n，类别 32（sports ball）

**both 模式**：两者并行，各维护独立 EKF，结果左右分屏显示

### 深度估算

```
视觉深度：depth_vis = fx × BALL_RADIUS / r_px
传感器深度：D455 深度帧，球心附近 5×5 像素中位数
融合（vis_weight=0.0 时纯传感器）：
    depth = vis_weight × depth_vis + (1-vis_weight) × depth_sensor
```

D455 传感器深度精度优于视觉估算，**建议 `vis_weight: 0.0`**。

### 物理 EKF

**状态**：`[px, py, pz, vx, vy, vz]`，在世界坐标系中运行（需 AprilTag）

**过程模型**（二次空气阻力 + 重力，g = −9.795 m/s²）：

| 参数 | 数值 |
|---|---|
| 球质量 | 0.0575 kg（ITF） |
| 球半径 | 0.0335 m（ITF） |
| C_d | 0.47（可调） |
| 恢复系数 | rest_x/y/z（可标定） |

### 弹道预测

`PhysicsEKF.rollout()` 以 10 ms 步长向前 Euler 积分（默认 2 秒）。
跨越 Z=0 时精确求解落地时刻，施加恢复系数后继续积分。
仅在**世界坐标系激活**时（AprilTag 检测到）有完整弹跳模型。

---

## 恢复系数标定

```bash
# 1. 录制：AprilTag 在视野内，球从约 0.8m 连续弹跳 5+ 次
python ball_detection.py --config config/d455_a.yaml --save-traj logs/

# 2. CLI 拟合
python calibrate_rest.py logs/traj_YYYYMMDD_HHMMSS.json --plot

# 3. 网页交互式拟合（可视化选段）
python calibrate_web.py logs/traj_YYYYMMDD_HHMMSS.json
# 打开 http://localhost:5010
```

正常室内硬地参考值：`rest_z ≈ 0.70–0.75`，`rest_x/y ≈ 0.45–0.65`，RMSE < 5 cm。
