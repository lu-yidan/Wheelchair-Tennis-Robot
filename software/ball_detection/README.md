# ball_detection

Real-time tennis ball detection, 3-D localisation, and trajectory prediction for the wheelchair tennis robot.

Two implementations coexist — choose based on available hardware:

| | ZED (production) | D455 (standalone) |
|---|---|---|
| **Hardware** | ZED 2 stereo camera + Jetson Nano | Intel RealSense D455 (dual-camera supported) |
| **Language** | C++ (CUDA) | Python |
| **Entry point** | `src/ball_detection.cpp` | `ball_detection.py` |
| **Output** | `/ball_detection` ROS topic | MJPEG stream + ZMQ trajectory publisher |
| **Dependencies** | ZED SDK, OpenCV 4.5.2+, CUDA | pyrealsense2, opencv-python, pyzmq |

---

## ZED / Production

Built inside the catkin workspace. Requires ZED SDK and CUDA.

```bash
catkin_make --pkg ball_detection
roslaunch ball_detection ball_detection.launch
```

Config: `config/settings.yaml`. See [ball_calibration](../ball_calibration/README.md) for AprilTag extrinsic calibration.

---

## D455 / Standalone Python

No ROS or ZED SDK required. Supports single-camera and dual-camera fusion modes.

### Setup

```bash
conda activate catchball
pip install pyzmq   # first time only
```

Linux udev (one-off, grants non-root USB access):

```bash
wget https://raw.githubusercontent.com/IntelRealSense/librealsense/master/config/99-realsense-libusb.rules
sudo cp 99-realsense-libusb.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules && sudo udevadm trigger
```

---

### Dual D455 fusion (recommended)

```bash
cd software/ball_detection
./run_dual_d455.sh
```

| URL / address | Content |
|---|---|
| `http://localhost:8080/monitor.html` | Fusion monitor: dual MJPEG + court top/side views |
| `http://localhost:5568/main` | D455 A raw MJPEG |
| `http://localhost:5668/main` | D455 B raw MJPEG |
| `tcp://<host-IP>:5580` | ZMQ PUB, 30 Hz trajectory (see below) |

Camera configs: `config/d455_a.yaml` (SN 260722302887) and `config/d455_b.yaml` (SN 152522251463, port offset +100).

---

### A/B comparison testing

```bash
./run_compare.sh hsv-fused yolo-fused    # compare HSV vs YOLO, both with sensor depth
./run_compare.sh hsv-vis   hsv-sensor    # compare visual vs sensor depth
```

Available method names: `hsv-vis` `hsv-sensor` `hsv-fused` `yolo-vis` `yolo-sensor` `yolo-fused` `both-vis` `both-fused`

---

### Single-camera mode

```bash
python ball_detection.py --config config/d455_a.yaml
python ball_detection.py --config config/d455_a.yaml --detector yolo
python ball_detection.py --config config/d455_a.yaml --vis-weight 0.0   # sensor depth only
```

---

## ZMQ trajectory subscription

`fusion.py` publishes the fused predicted trajectory at **30 Hz on port 5580** for LAN subscribers.

### Message format

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

| Field | Description |
|---|---|
| `stamp` | Unix timestamp of this publish |
| `t_obs` | Wall time of last accepted camera measurement |
| `detected` | `true` = fresh measurement ≤200 ms ago; `false` = EKF coasting |
| `pos` | Current ball position (m, AprilTag frame, Z-up) |
| `vel` | Current ball velocity (m/s) |
| `traj` | Predicted waypoints (~200 pts, 2 s horizon); `t` = seconds from `stamp` |

**Coordinate frame:** origin = AprilTag centre, Z+ = up, units metres / seconds.

### Minimal subscriber

```python
import zmq, json

sock = zmq.Context().socket(zmq.SUB)
sock.connect("tcp://192.168.1.100:5580")
sock.setsockopt_string(zmq.SUBSCRIBE, "")
sock.setsockopt(zmq.RCVHWM, 2)

while True:
    msg = json.loads(sock.recv())
    if not msg["detected"]:
        continue
    traj = msg["traj"]   # [[x, y, z, t], ...]
```

See `subscribe_traj.py` for a complete example that prints a waiting message when idle.
Run `python test_traj_pub.py` to verify latency, frequency, and message format locally.

---

## Key configuration

### AprilTag
```yaml
tag_family: tag36h11
tag_size_m: 0.25     # outer black-square edge in metres — measure after printing
```

### Detection
```yaml
detector:   hsv      # hsv | yolo | both
vis_weight: 0.0      # depth blend: 0=sensor-only (recommended), 1=visual-only
motion:     true     # enable MOG2 motion mask (reduces static false positives)
traj:       true     # show predicted trajectory overlay
```

### Physics
```yaml
coeff_drag: 0.47
rest_x: 0.75   rest_y: 0.75   rest_z: 0.65
```

---

## fusion.py — multi-camera EKF

Receives UDP measurements from N `ball_detection.py` instances and runs a single physics EKF.

```bash
python fusion.py --port 5570 --webview-port 5571 --traj-pub-port 5580
```

| Flag | Default | Description |
|---|---|---|
| `--outlier-m` | 2.0 | Reject measurements farther than this from current EKF state (m) |
| `--reset-sec` | 3.0 | Reset EKF after this many seconds without a valid measurement |
| `--traj-pub-port` | 0 | ZMQ PUB port (0 = disabled) |
| `--traj-pub-hz` | 30 | Publish rate (Hz) |
| `--traj-pub-sec` | 2.0 | Prediction horizon (seconds) |

---

## Scripts

| Script | Purpose |
|---|---|
| `ball_detection.py` | Main detector: camera, HSV/YOLO, EKF, MJPEG server, fusion UDP |
| `fusion.py` | Multi-camera EKF fusion + ZMQ trajectory publisher |
| `monitor.py` | HTTP/SSE relay for `monitor.html` |
| `monitor.html` | Fusion monitor: dual MJPEG + canvas court views |
| `run_dual_d455.sh` | One-shot launcher: dual D455 + fusion + monitor |
| `run_compare.sh` | A/B method comparison launcher |
| `subscribe_traj.py` | Minimal trajectory subscriber (no timeout, idle status prints) |
| `test_traj_pub.py` | ZMQ self-test: latency / frequency / format validation |
| `tune_hsv_web.py` | Interactive browser-based HSV / MOG2 tuner |
| `calibrate_rest.py` | CLI: fit rest coefficients from recorded bounce trajectory |
| `calibrate_web.py` | Browser UI: interactive segment selection + fit visualisation |
| `replay.py` | Offline reprocess raw video → annotated video + trajectory JSON |
| `generate_apriltag.py` | Generate AprilTag PDF for floor calibration |

---

## Pipeline internals

### Coordinate frames

```
RealSense colour sensor
    optical frame  (X→right, Y↓down, Z→forward)
        │  optical_to_body()
        ▼
    body frame  (X→forward, Y→left, Z↑up, gravity=−Z)
        │  _to_world()  (requires AprilTag)
        ▼
    world frame  (origin = AprilTag centre, Z=0=ground, Z↑up)
```

All cameras sharing the same tag are automatically in the same world frame.
For multi-tag setups, `tag_world_map.yaml` maps each tag to a common world frame.

### Ball detection

**HSV**: `cv2.inRange(HSV)` → optional MOG2 motion mask intersection → `findContours`, filtered by circularity and radius.

**YOLO**: YOLOv8n, class 32 (sports ball).

**both**: both run in parallel with independent EKF states, displayed side-by-side.

### Depth estimation

```
Visual:  depth_vis = fx × BALL_RADIUS / r_px
Sensor:  5×5 median patch on D455 depth frame at projected ball centre
Fused:   vis_weight × depth_vis + (1 − vis_weight) × depth_sensor
```

D455 sensor depth is more accurate than visual estimation. Use `vis_weight: 0.0`.

### Physics EKF

State: `[px, py, pz, vx, vy, vz]` in world frame.

Process model: quadratic aerodynamic drag + gravity (g = −9.795 m/s²).

| Parameter | Value |
|---|---|
| Ball mass | 0.0575 kg (ITF) |
| Ball radius | 0.0335 m (ITF) |
| C_d | 0.47 (tunable) |
| Restitution | rest_x/y/z (calibratable) |

### Trajectory prediction

`PhysicsEKF.rollout()` Euler-integrates 2 seconds ahead at 10 ms steps.
Floor crossings are resolved exactly via quadratic solve; restitution is applied and integration continues.

---

## Rest-coefficient calibration

```bash
# 1. Record: AprilTag in view, drop ball from ~0.8 m, 5+ clear bounces
python ball_detection.py --config config/d455_a.yaml --save-traj logs/

# 2. CLI fit
python calibrate_rest.py logs/traj_YYYYMMDD_HHMMSS.json --plot

# 3. Interactive browser fit
python calibrate_web.py logs/traj_YYYYMMDD_HHMMSS.json
# open http://localhost:5010
```

Expected indoor hard court: `rest_z ≈ 0.70–0.75`, `rest_x/y ≈ 0.45–0.65`, RMSE < 5 cm.
