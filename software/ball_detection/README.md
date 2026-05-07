# ball_detection

Real-time tennis ball detection and 3D position output for the wheelchair tennis robot.

Two implementations coexist — choose based on available hardware:

| | ZED (production) | D455 (standalone testing) |
|---|---|---|
| Hardware | ZED 2 stereo camera + Jetson Nano | Intel RealSense D455 |
| Language | C++ (CUDA) | Python |
| Entry point | `src/ball_detection.cpp` | `ball_detection_d455.py` |
| ROS output | `/ball_detection` topic | terminal + OpenCV window |
| Dependencies | ZED SDK, OpenCV 4.5.2+, CUDA | pyrealsense2, opencv-python |

---

## ZED / Production

Built as part of the catkin workspace. Requires ZED SDK and CUDA.

```bash
catkin_make --pkg ball_detection
roslaunch ball_detection ball_detection.launch
```

Config: `config/settings.yaml` (HSV range, covariance polynomials, MOG2 params)

See [ball_calibration](../ball_calibration/README.md) for extrinsic calibration setup.

---

## D455 / Standalone

Standalone Python script — no ROS, no ZED SDK required. Uses `catchball` conda env.

### Environment

```bash
conda activate catchball   # pyrealsense2 2.57+, opencv, numpy, scipy, ultralytics
```

First-time Linux udev setup (one-off):

```bash
wget https://raw.githubusercontent.com/IntelRealSense/librealsense/master/config/99-realsense-libusb.rules
sudo cp 99-realsense-libusb.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules && sudo udevadm trigger
```

### Running

```bash
cd software/ball_detection

python ball_detection_d455.py                           # 1280×720, HSV+MOG2
python ball_detection_d455.py --width 848 --height 480  # lower res, higher fps
python ball_detection_d455.py --no-viz                  # headless / terminal only
python ball_detection_d455.py --show-mask               # overlay HSV+motion mask
python ball_detection_d455.py --no-motion               # disable MOG2, pure HSV
python ball_detection_d455.py --detector yolo           # YOLO backend
python ball_detection_d455.py --detector both           # side-by-side comparison
python ball_detection_d455.py --record output.mp4       # record annotated video
```

HSV parameter override (from `config/settings.yaml` defaults):

```bash
python ball_detection_d455.py --h-low 10 --h-high 35 --s-min 170 --v-min 170
# fluorescent yellow-green (outdoor / cool light):
python ball_detection_d455.py --h-low 25 --h-high 80 --s-min 80 --v-min 80
```

Ground plane — enable world-frame EKF so Z=0 = court surface:

```bash
# Measure camera centre height above ground (e.g. 1.2 m) and mount pitch angle
python ball_detection_d455.py --camera-height 1.2 --camera-pitch -15

# Without these args the EKF runs in camera body frame and bounce is never triggered
```

Position readout changes from `body` (camera-relative) to `agl` (above ground level).
Bounce points appear as **cyan dots** on the predicted trajectory once the ground is known.

Physics tuning:

```bash
python ball_detection_d455.py --coeff-drag 0.55   # standard tennis ball Cd ≈ 0.47–0.65
python ball_detection_d455.py --rest-z 0.70        # vertical restitution after bounce
```

### Tuning the HSV mask interactively

```bash
python tune_hsv.py                          # 1280×720
python tune_hsv.py --width 848 --height 480
```

Four windows open simultaneously:

| Window | Content |
|---|---|
| **1 Color+detect** | Raw frame — grey = candidates, green = best detection |
| **2 HSV mask** | Yellow tint = pixels passing HSV filter |
| **3 MOG2 motion** | Blue tint = moving pixels |
| **4 Combined** | Green tint = HSV ∩ MOG2 (what drives contour search) |

Trackbars (on window 1): `H_low`, `H_high`, `S_min`, `V_min`, `MOG2_thr`, `Min_r_px`, `Circ100`

Keys: `s` — print tuned params (copy-paste ready for CLI and `settings.yaml`), `r` — reset MOG2 background model, `q` / ESC — quit.

### Architecture (D455 pipeline)

```
D455 (30/60 fps)
  ├── Color frame ──→ MOG2 motion mask ──→ HSV + circularity ──→ (cx, cy, r_px)
  └── Depth frame ──→ 3-step Color→Depth mapping ──→ median patch ──→ depth_sensor
                                                            ↓
                            visual depth (fx·R/r_px) ──→ fusion ──→ depth_fused
                                                            ↓
                                            rs2_deproject → optical frame
                                                            ↓
                                            optical_to_body → body frame (X-fwd, Z-up)
                                                            ↓
                                            PhysicsEKF.update()  ← measurement covariance
                                                            ↓
                                            PhysicsEKF.rollout() → 1 s ahead trajectory
                                                            ↓
                                            visualization (trail + predicted arc + bounce)
```

**Coordinate frame**: camera body (X-forward, Y-left, Z-up). Gravity → −Z.

**Physics constants** (ported from `ball_localization/src/ball_ekf.cpp`):

| Parameter | Value | Notes |
|---|---|---|
| Gravity | −9.79528 m/s² | Atlanta; adjust for venue |
| Air density | 1.225 kg/m³ | |
| Ball mass | 0.0575 kg | ITF standard |
| Ball radius | 0.0335 m | ITF standard |
| C_d | 0.47 | sphere; tunable via `--coeff-drag` |
| Restitution | 0.75 (x, y, z) | tunable via `--rest-*` |
