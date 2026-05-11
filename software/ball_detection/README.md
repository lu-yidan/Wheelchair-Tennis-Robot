# ball_detection

Real-time tennis ball detection, 3-D localisation, and trajectory prediction for the wheelchair tennis robot.

Two implementations coexist — choose based on available hardware:

| | ZED (production) | D455 (standalone) |
|---|---|---|
| **Hardware** | ZED 2 stereo camera + Jetson Nano | Intel RealSense D455 |
| **Language** | C++ (CUDA) | Python |
| **Entry point** | `src/ball_detection.cpp` | `ball_detection_d455.py` |
| **Output** | `/ball_detection` ROS topic | terminal + OpenCV window |
| **Dependencies** | ZED SDK, OpenCV 4.5.2+, CUDA | pyrealsense2, opencv-python |

---

## ZED / Production

Built inside the catkin workspace. Requires ZED SDK and CUDA.

```bash
catkin_make --pkg ball_detection
roslaunch ball_detection ball_detection.launch
```

Config: `config/settings.yaml` (HSV ranges, MOG2 params, measurement covariance polynomials)

See [ball_calibration](../ball_calibration/README.md) for AprilTag-based extrinsic calibration.

---

## D455 / Standalone

Standalone Python script — no ROS, no ZED SDK required.

### Directory structure

```
ball_detection/
├── config/
│   ├── d455.yaml            — all runtime parameters (see Configuration below)
│   └── settings.yaml        — ZED / ROS parameters
├── logs/                    — recorded trajectory JSON files (gitignored)
├── models/                  — YOLO weights (auto-downloaded on first run)
├── src/
│   └── ball_detection.cpp   — ZED C++ implementation
├── ball_detection_d455.py   — main script
├── calibrate_rest.py        — CLI tool: fit rest_x/y/z from recorded trajectory
├── calibrate_web.py         — browser UI: interactive rest calibration
├── generate_apriltag.py     — print AprilTag PDF for ground-plane calibration
├── tune_hsv_web.py          — browser UI: tune HSV / MOG2 parameters (live or --video)
├── replay.py                — offline reprocess raw video → annotated video + trajectory
├── viz3d.py                 — 3-D Three.js trajectory viewer
└── webview.py               — 2-D camera + court viewer
```

### Files

| File | Purpose |
|---|---|
| `ball_detection_d455.py` | Main script — camera, detection, EKF, visualisation, MJPEG server |
| `viz3d.py` | 3-D web viewer — Three.js trajectory + bounce visualisation, restitution sliders |
| `webview.py` | 2-D web viewer — annotated camera feed + court top-down view |
| `tune_hsv_web.py` | Interactive web-based HSV / MOG2 tuner (live camera or `--video` file) |
| `replay.py` | Offline reprocess raw `.mp4` → annotated video + trajectory JSON |
| `calibrate_rest.py` | CLI tool: fit restitution coefficients from a recorded bounce trajectory |
| `calibrate_web.py` | Browser UI: interactive rest calibration with Plotly chart |
| `generate_apriltag.py` | Print an A4 AprilTag PDF for ground-plane calibration |
| `config/d455.yaml` | All runtime parameters; CLI flags override these |

### Environment

```bash
conda activate catchball   # pyrealsense2 2.57+, opencv-contrib-python, numpy, scipy, ultralytics
```

First-time Linux udev setup (one-off, gives non-root USB access):

```bash
wget https://raw.githubusercontent.com/IntelRealSense/librealsense/master/config/99-realsense-libusb.rules
sudo cp 99-realsense-libusb.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules && sudo udevadm trigger
```

### Quick start

```bash
cd software/ball_detection

# Default: read all settings from config/d455.yaml
python ball_detection_d455.py

# Lower resolution, higher fps
python ball_detection_d455.py --width 848 --height 480

# Force specific detector
python ball_detection_d455.py --detector yolo
python ball_detection_d455.py --detector both    # side-by-side comparison

# Headless / recording
python ball_detection_d455.py --no-viz
python ball_detection_d455.py --record output.mp4

# Record ball trajectory for rest-coefficient calibration
# (or set save_traj: true in config/d455.yaml for persistent auto-recording)
python ball_detection_d455.py --save-traj logs/   # → logs/traj_YYYYMMDD_HHMMSS.json

# Override HSV range on command line (CLI always wins over yaml)
python ball_detection_d455.py --h-low 25 --h-high 80 --s-min 80 --v-min 80
```

### Web viewers

Enable in `config/d455.yaml` (or pass CLI flags) then open in a browser:

```yaml
viz3d:   true   # 3-D Three.js viewer  → http://localhost:5001
webview: true   # 2-D camera + court   → http://localhost:5002
```

```bash
# Terminal 1 — main detection (starts MJPEG server + UDP state broadcast)
python ball_detection_d455.py

# Terminal 2 — 3-D trajectory viewer
python viz3d.py          # open http://localhost:5001

# Terminal 3 — 2-D camera + court viewer
python webview.py        # open http://localhost:5002
```

Both web viewers include live **restitution coefficient sliders** (rest X/Y/Z) that update the EKF bounce model in real time without restarting the detector.

---

## Configuration — `config/d455.yaml`

All parameters are documented in the file itself. CLI flags override yaml values at runtime.

### Resolution / frame rate
```yaml
width:  1280    # 848 × 480 can run at 60 fps
height: 720
```

### Ground plane (cold-start values)
```yaml
camera_height: 0.5    # metres from camera centre to floor; 0 = world frame off
camera_pitch:  -10.0  # degrees; negative = looking down toward court
```
These are used on startup. Once an AprilTag is detected they are continuously
updated via EMA and no longer need to be accurate.

### AprilTag ground calibration
```yaml
tag_family: tag36h11         # must match the printed tag
tag_ids:    [0, 1, 2, 3]     # accepted tag IDs; each frame uses the largest visible tag
tag_size_m: 0.15             # outer black-square edge in metres; 0 = disable
```

### Detection backend
```yaml
detector: hsv    # hsv | yolo | both
```

### HSV colour range
```yaml
h_low:  25    # OpenCV hue 0–179
h_high: 80
s_min:  100   # saturation 0–255
v_min:  100   # value (brightness) 0–255
```
Use `tune_hsv_web.py` to find the right values interactively.

### MOG2 motion filter
```yaml
motion: false   # true = HSV ∩ MOG2 (reduces static false positives)
                # false = pure HSV (better for moving camera / complex background)
```

### HSV detection thresholds
```yaml
mog2_threshold: 50    # MOG2 varThreshold — higher = less sensitive to background changes
min_radius_px:  3     # smallest ball radius accepted (px); raise to ignore small noise
circularity:    0.55  # minimum roundness 0–1; tennis ball typically 0.55–0.80
```
These are the same parameters exposed as sliders in `tune_hsv_web.py`.
Copy the tuned values directly from the "Copy YAML" button output.

### YOLO (when detector: yolo or both)
```yaml
yolo_model: models/yolov8n.pt
yolo_imgsz: 480
yolo_conf:  0.3
```

### Physics
```yaml
coeff_drag: 0.47   # tennis ball C_d  ≈ 0.47–0.65
rest_x: 0.75       # horizontal restitution after bounce
rest_y: 0.75
rest_z: 0.75       # vertical restitution
```

### Visualisation / recording
```yaml
viz:       true     # show OpenCV window
show_mask: false    # show HSV+MOG2 mask panel
traj:      true     # show predicted trajectory arc
record:    false    # false | true (auto-name) | "filename.mp4"
```

### Web viewers
```yaml
# 3-D web viewer (viz3d.py)
viz3d:      true       # broadcast state via UDP to viz3d.py
viz3d_port: 5565       # UDP port viz3d.py listens on
ctrl_port:  5566       # UDP port for rest-coefficient feedback from web sliders

# 2-D web viewer (webview.py)
webview:      true     # enable state UDP + internal MJPEG server
webview_port: 5567     # UDP state port webview.py listens on
mjpeg_port:   5568     # HTTP port for built-in full-resolution MJPEG server
```

Video frames are served by ball_detection's **built-in MJPEG HTTP server** on `mjpeg_port`
at full original resolution and quality — no UDP frame transfer, no size limit.
The browser at `localhost:5002` connects directly to `localhost:5568` for the live stream.

---

## Pipeline internals

### 1 · Coordinate frames

```
RealSense colour sensor
    optical frame  (OpenCV standard)
        X → right
        Y ↓ down
        Z → forward (depth axis)
        │
        │  optical_to_body()
        ▼
    body frame  (camera-body, used by EKF)
        X → forward
        Y → left
        Z ↑ up    gravity = −Z
        │
        │  body_to_world()  (only when ground plane is known)
        ▼
    world frame  (court surface)
        Z = 0 = ground
        Z ↑ up    bounce triggers when EKF state Z crosses 0
```

**`optical_to_body`**: `[X, Y, Z]_body = [Z_opt, −X_opt, −Y_opt]`

**`body_to_world`** (pitch rotation around Y axis + height offset):
```
X_world =  cos(pitch)·X_body + sin(pitch)·Z_body
Y_world =  Y_body
Z_world = −sin(pitch)·X_body + cos(pitch)·Z_body + height
```

`body_to_world` / `world_to_body` are inverses.
`pitch` is negative when camera looks down (e.g. −10° for a mounted camera).

---

### 2 · Ground plane detection (AprilTag)

A tag36h11 AprilTag is laminated and taped to the court surface.
The tag's own coordinate frame has Z pointing up — identical to the world frame.

**Per-frame loop** (inside `detection_worker`):

```
colour frame
    │  cv2.aruco.detectMarkers / ArucoDetector
    ▼
detected corners (4 × 2 pixels)
    │  cv2.solvePnP(tag_obj_pts, img_pts, cam_matrix, dist, IPPE_SQUARE)
    ▼
rvec, tvec  — pose of tag in optical frame
    │
    ├─ camera origin in tag/world frame:  t_cam = −R.T @ tvec
    │      camera_height = t_cam[2]
    │
    └─ camera look direction in world:    look = R.T @ [0,0,1]
           camera_pitch  = arcsin(look[2])
    │
    │  EMA smoothing  (α = 0.3 per frame)
    ▼
_pose["height"]    updated
_pose["pitch_rad"] updated
_pose["age"]  = 0
```

**Age indicator** shown on the detection panel:

| Colour | Meaning |
|---|---|
| Green `TAG OK` | Detected this frame — values are fresh |
| Orange `TAG [Nf]` | Last seen N frames ago — using smoothed cache |
| Red `NO TAG` | Not seen for >30 frames — using yaml cold-start values |

If the robot moves and the tag leaves the frame, the last smoothed height and pitch are
held (robot is on a flat court so they remain accurate for many frames).

---

### 3 · Ball detection

**HSV backend** (default, fast):

```
colour frame  →  MOG2 background subtractor (optional, 0.4× scale)
                 │
                 └─ motion mask (dilated)

colour frame  →  cv2.inRange(HSV)  →  morphological close + open
                 │
                 └─ HSV mask

HSV mask ∩ motion mask  (or HSV mask alone when motion: false)
    │  findContours
    ▼
for each contour:
    circularity = 4π·area / perimeter²   (1.0 = perfect circle)
    minEnclosingCircle  →  (cx, cy, r_px)
    keep if  r_px ≥ min_r  and  circularity ≥ threshold
best = highest  area × circularity  score
```

**YOLO backend**: YOLOv8n tracking on a downscaled frame, class 32 (sports ball).

**`detector: both`** runs both in parallel; each has its own EKF state.

---

### 4 · Depth fusion

For each detected ball centre `(cx, cy)`:

```
Visual depth (geometry):
    depth_vis = fx × BALL_RADIUS / r_px
    reliable when ball is round and well-lit

Sensor depth (D455 depth frame):
    3-step colour→depth pixel mapping (compensates stereo baseline)
    median of 5×5 patch around projected centre
    depth_sensor = raw_depth × depth_scale + BALL_RADIUS

Fusion:
    if both valid and ratio 0.5–2.0:  weighted average (50/50)
    else:                             prefer visual depth
```

---

### 5 · Physics EKF

**State**: `x = [px, py, pz, vx, vy, vz]` in either body frame or world frame.

**Process model** (ported from `ball_localization/src/ball_ekf.cpp`):

```
v² = vx² + vy² + vz²
a_drag = C_d · ½ρ · πR² · v² / m      (quadratic drag)

ax = −sign(vx) · a_drag · |vx| / |v|
ay = −sign(vy) · a_drag · |vy| / |v|
az =  g  − sign(vz) · a_drag · |vz| / |v|   (g = −9.795 m/s²)
```

**Measurement**: 3-D ball position from depth fusion.
Measurement noise covariance is a polynomial in depth: `cov(d) = k2·d² + k1·d + k0`.
A 5σ Mahalanobis gate rejects outliers.

**Physics constants**:

| Parameter | Value | Source |
|---|---|---|
| Gravity | −9.79528 m/s² | Atlanta; adjust for venue |
| Air density | 1.225 kg/m³ | |
| Ball mass | 0.0575 kg | ITF standard |
| Ball radius | 0.0335 m | ITF standard |
| C_d | 0.47 | sphere (tunable) |
| Restitution | 0.75 × / 0.75 × / 0.75 z | tunable |

---

### 6 · Trajectory prediction and bounce

`PhysicsEKF.rollout()` Euler-integrates the physics model 1 second forward at 10 ms steps.

**Bounce detection** (per step, from `ball_ekf.cpp` lines 329–347):

```
if Z before step > 0 and Z after step < 0:
    quadratic solve for exact floor-hit time t_f within the step
    propagate state to t_f
    apply restitution:
        vx *= rest_x
        vy *= rest_y
        vz  = rest_z × |vz|   (reverse and attenuate)
    continue remaining step after bounce
```

This only fires when the EKF runs in **world frame** (`_pose["height"] > 0`).
In body frame the EKF clamps Z ≥ 0 but has no physical bounce model.

Trajectory visualisation:
- **Red dots / line**: predicted path
- **Cyan dots**: predicted bounce points (Z crosses 0)

---

## Tools

### `viz3d.py` — 3-D trajectory web viewer

```bash
python viz3d.py          # open http://localhost:5001
python viz3d.py --port 5001 --udp-port 5565 --ctrl-port 5566
```

Three.js visualisation running in the browser — no display server required.

| Element | Description |
|---|---|
| Orange fading line | Ball position history (recent = bright) |
| Red line | Predicted trajectory (1 s ahead) |
| Cyan dots | Predicted bounce points |
| Right panel sliders | Live rest X/Y/Z — updates EKF in ball_detection instantly |
| Connection indicator | Green = online, orange = stale, red = disconnected |

---

### `webview.py` — 2-D camera + court web viewer

```bash
python webview.py          # open http://localhost:5002
python webview.py --port 5002 --udp-port 5567 --mjpeg-port 5568
```

Layout: annotated camera frame (left, full resolution) + court top-down view + info panel (right).

Video is served at **full original resolution** by ball_detection's built-in MJPEG HTTP server
(`mjpeg_port`). The browser connects directly — no intermediate re-encoding or size limit.
Same restitution sliders and ball/velocity readout as viz3d.py.

---

### `tune_hsv_web.py` — Interactive HSV tuner

Works with both live camera and recorded video.

```bash
# Live camera
python tune_hsv_web.py
python tune_hsv_web.py --width 848 --height 480
# open http://localhost:5000

# Recorded video (no camera required — useful for tuning offline)
python tune_hsv_web.py --video recordings/color_video.mp4
python tune_hsv_web.py --video recordings/color_video.mp4 --port 5001
```

Streams a 2×2 MJPEG composite to the browser — no Qt / system fonts required.

| Panel | Content |
|---|---|
| top-left | Raw colour frame + detected circle, ball depth |
| top-right | HSV mask (cyan tint) |
| bottom-left | MOG2 motion mask (orange), red border when disabled |
| bottom-right | Combined mask (green tint) used for contour detection |

Sliders (in browser): `H low/high`, `S min`, `V min`, `MOG2 threshold`, `Min radius`, `Circularity`.
MOG2 toggle switch: compare HSV-only vs HSV∩MOG2 live.
"Copy YAML" button: copies tuned values ready to paste into `config/d455.yaml`.

**Video mode extras** (shown when `--video` is given):
- Frame slider to scrub through the recording
- ◀ / ▶ step buttons (±1 and ±10 frames)
- Play / Pause button for sequential playback at original fps
- MOG2 background model resets automatically on non-sequential seeks

---

### `replay.py` — Offline video reprocessing

Re-runs the full detection pipeline (HSV → AprilTag → EKF) on a raw `.mp4` recorded with `record_raw: true`.

```bash
# Basic: generates recordings/color_video_annotated.mp4
python replay.py recordings/color_video.mp4

# Save annotated video + trajectory JSON
python replay.py recordings/color_video.mp4 --out recordings/annotated.mp4 --save-traj logs/

# Override HSV parameters from a different config
python replay.py recordings/color_video.mp4 --config config/d455.yaml

# Live preview while processing
python replay.py recordings/color_video.mp4 --show
```

The script reads camera intrinsics from `*_intrinsics.json` (auto-saved alongside the raw video when `record_raw: true`). If no sidecar is found it falls back to D455 1280×720 factory defaults.

**AprilTag detection in video**: H.264 compression blurs edges, reducing detection rate to ~40–50%. The script applies image sharpening and relaxed detector thresholds to improve reliability. Pose is held between detections (last-known pose, EMA smoothed).

---

### `generate_apriltag.py` — Print AprilTag for ground calibration

```bash
# Print tag36h11 id=1, 150 mm black square, on A4 paper
python generate_apriltag.py --family tag36h11 --tag-id 1 --tag-size-mm 150

# Output: apriltag_tag36h11_id1_150mm_a4.pdf
# Print at 100% scale (no "fit to page"), laminate, tape to court floor
```

The tag's **outer black-square edge** is the measurement used for `tag_size_m`.
After printing, measure with a ruler and update `config/d455.yaml` if needed.

---

### `calibrate_rest.py` — CLI rest-coefficient calibration

Fits `rest_x`, `rest_y`, `rest_z` from a recorded bounce trajectory.

```bash
# 1. Record: bounce ball ≥5 times clearly in view, ensure AprilTag is detected
python ball_detection_d455.py --save-traj logs/

# 2. Inspect Z(t) overview and identified bounces
python calibrate_rest.py logs/traj_20260511_120000.json --plot

# 3. Exclude bad segments (rolling, walking through frame) by time range
python calibrate_rest.py logs/traj_20260511_120000.json --segments "2.5-8.0,14.0-21.5" --plot
```

**Two-stage fitting algorithm:**
- **Stage 1 — `rest_z`**: computed from peak-height ratios of consecutive bounces (`√(h₂/h₁)`). Does not depend on EKF velocity estimates.
- **Stage 2 — `rest_x/y`**: `differential_evolution` optimises horizontal restitution with `rest_z` held fixed.

Expected results (hard court indoors): `rest_z ≈ 0.70–0.75`, `rest_x/y ≈ 0.45–0.65`, RMSE < 5 cm.

---

### `calibrate_web.py` — Interactive browser calibration

Same algorithm as `calibrate_rest.py` but with a browser UI for selecting clean bounce segments.

```bash
python calibrate_web.py logs/traj_20260511_120000.json
# opens http://localhost:5010
```

Workflow:
1. **Drag** horizontally on the Z(t) chart to add bounces in a time range to the selection
2. **Toggle** individual bounce checkboxes to include/exclude specific bounces
3. Click **Run Optimization** (5–20 s)
4. Inspect the per-bounce Z-fit plots — good fits have predicted ≈ actual
5. Click the YAML block to copy `rest_x/y/z` directly into `config/d455.yaml`

**When to exclude a bounce:**
- Ball rolling on ground (actual Z stays near 0 the whole time)
- Very weak tap (peak height < 0.05 m)
- Ball leaves the camera frame mid-arc (trajectory truncated)
- Two bounces merged into one event (actual Z dips back to 0 mid-arc)

The `rest_z` panel shows the individual `√(h₂/h₁)` values used — if they are inconsistent (spread > 0.10), record cleaner data.
