"""
tune_hsv_web.py

Web-based HSV + MOG2 tuner for D455.
Streams a 2×2 diagnostic panel as MJPEG to the browser.
HTML sliders replace OpenCV trackbars — no Qt / system fonts needed.

Four panels (same as tune_hsv.py):
  top-left  : colour frame + detected circle
  top-right : HSV mask (yellow tint)
  bot-left  : MOG2 motion mask (blue tint)
  bot-right : HSV ∩ MOG2 combined + detected circle (green tint)

Usage:
    # Live camera:
    conda activate catchball
    python tune_hsv_web.py
    python tune_hsv_web.py --port 5000 --width 848 --height 480

    # Recorded video (no camera required):
    python tune_hsv_web.py --video recordings/color_video.mp4
    python tune_hsv_web.py --video recordings/color_video.mp4 --port 5001

    # then open  http://localhost:5000  in any browser
"""

import argparse
import time
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs
import numpy as np
import cv2

try:
    import pyrealsense2 as rs
    _HAS_RS = True
except ImportError:
    rs = None
    _HAS_RS = False

# ── defaults (match d455.yaml) ────────────────────────────────────────────────
DEF = dict(h_low=25, h_high=80, s_min=100, v_min=100,
           mog2_thr=50, min_r=3, circ=55,   # circ is circularity × 100
           motion=True)                      # MOG2 on/off toggle

BALL_RADIUS = 0.0335  # m

# ── shared state ──────────────────────────────────────────────────────────────
_p_lock   = threading.Lock()
_params   = dict(DEF)

_jpg_lock = threading.Lock()
_jpg      = [None]       # latest JPEG bytes of composite frame

_stop     = threading.Event()

# ── video-mode shared state ───────────────────────────────────────────────────
_video_mode = False
_n_frames   = 0            # total frames in video (0 = camera mode)
_seek_lock  = threading.Lock()
_seek_req   = [0]          # current frame index requested by UI
_play       = [False]      # auto-play flag
_fx_video   = 635.7        # default D455 fx for visual depth estimate


def _get():
    with _p_lock:
        return dict(_params)


def _set(key, raw):
    with _p_lock:
        if key not in _params:
            return
        if isinstance(_params[key], bool):
            _params[key] = raw in ('1', 'true', 'True', 'yes')
        else:
            try:
                _params[key] = int(raw)
            except ValueError:
                pass


# ── HTML ──────────────────────────────────────────────────────────────────────

def _html():
    p = _get()
    rows = [
        ('h_low',    'H low',          0,   179),
        ('h_high',   'H high',         0,   179),
        ('s_min',    'S min',          0,   255),
        ('v_min',    'V min',          0,   255),
        ('mog2_thr', 'MOG2 threshold', 1,   200),
        ('min_r',    'Min radius (px)',1,    50),
        ('circ',     'Circularity×100',10,  100),
    ]
    _vm = _video_mode
    _nf = _n_frames
    sliders = ''.join(f'''
      <div class="row">
        <label>{label}</label>
        <input type="range" id="{k}" min="{mn}" max="{mx}" value="{p[k]}"
               oninput="upd(this)">
        <span id="{k}v">{p[k]}</span>
      </div>''' for k, label, mn, mx in rows)

    motion_checked = 'checked' if p['motion'] else ''

    return f'''<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>HSV Tuner — D455</title>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{font-family:monospace;background:#111;color:#ccc;padding:14px;font-size:13px}}
  h2{{color:#fff;margin-bottom:14px;font-size:16px}}
  .layout{{display:flex;gap:20px;flex-wrap:wrap;align-items:flex-start}}
  .ctrl{{width:320px;flex-shrink:0}}
  .row{{display:flex;align-items:center;gap:8px;margin-bottom:10px}}
  .row label{{width:140px;color:#999}}
  .row input[type=range]{{flex:1;accent-color:#4af}}
  .row span{{width:34px;text-align:right;color:#4af;font-weight:bold}}
  .img-wrap img{{max-width:100%;border:1px solid #333;display:block}}
  pre{{background:#0a0a0a;border:1px solid #333;padding:10px;border-radius:4px;
       color:#7c7;margin-top:14px;white-space:pre-wrap;line-height:1.5}}
  button{{margin-top:8px;padding:5px 14px;background:#2a2a2a;color:#ccc;
          border:1px solid #444;border-radius:3px;cursor:pointer;font-family:monospace}}
  button:hover{{background:#3a3a3a}}
  .note{{color:#666;font-size:11px;margin-top:6px}}
  .divider{{border:none;border-top:1px solid #2a2a2a;margin:12px 0}}
  /* toggle switch */
  .toggle-row{{display:flex;align-items:center;gap:12px;margin-bottom:12px}}
  .toggle-row .tlabel{{color:#ccc;font-size:13px}}
  .switch{{position:relative;display:inline-block;width:46px;height:24px}}
  .switch input{{opacity:0;width:0;height:0}}
  .slider-sw{{position:absolute;cursor:pointer;inset:0;background:#333;
              border-radius:24px;transition:.2s}}
  .slider-sw:before{{position:absolute;content:"";height:18px;width:18px;
                     left:3px;bottom:3px;background:#888;border-radius:50%;
                     transition:.2s}}
  input:checked+.slider-sw{{background:#2a6}}
  input:checked+.slider-sw:before{{transform:translateX(22px);background:#fff}}
  .badge{{font-size:11px;padding:2px 7px;border-radius:10px;font-weight:bold}}
  .badge.on{{background:#2a6;color:#fff}}
  .badge.off{{background:#444;color:#888}}
  /* video controls */
  .vid-ctrl{{margin-bottom:12px}}
  .vid-ctrl .row{{margin-bottom:6px}}
  .vid-ctrl input[type=range]{{flex:1;accent-color:#fa0}}
  .vid-ctrl .vid-btns{{display:flex;gap:6px;margin-top:6px}}
  .vid-ctrl .vid-btns button{{flex:1;padding:5px 0;font-size:12px}}
  .vid-badge{{font-size:11px;padding:2px 7px;border-radius:10px;font-weight:bold;
              background:#630;color:#fa0;margin-left:6px}}
</style>
</head>
<body>
<h2>HSV / MOG2 Tuner — {'视频回放' if _vm else 'D455'}</h2>
<div class="layout">
  <div class="ctrl">
{sliders}
    <hr class="divider">
    <div class="toggle-row">
      <label class="switch">
        <input type="checkbox" id="motion" {motion_checked}
               onchange="toggleMotion(this)">
        <span class="slider-sw"></span>
      </label>
      <span class="tlabel">MOG2 运动滤波</span>
      <span class="badge {'on' if p['motion'] else 'off'}" id="motion_badge">
        {'ON' if p['motion'] else 'OFF'}
      </span>
    </div>
    <p class="note" style="margin-bottom:12px">
      开启：HSV ∩ MOG2（减少静止背景误检）<br>
      关闭：纯 HSV（Panel 3 仍显示 MOG2 供参考）
    </p>
    <hr class="divider">
{'<div class="vid-ctrl"><div class="row"><label>帧 <span id="fidx">0</span> / ' + str(_nf-1) + '</label><input type="range" id="fslider" min="0" max="' + str(max(_nf-1,1)) + '" value="0" oninput="seekTo(this.value)"></div><div class="vid-btns"><button onclick="stepFrame(-10)">«10</button><button onclick="stepFrame(-1)">«1</button><button id="playbtn" onclick="togglePlay()">▶ 播放</button><button onclick="stepFrame(1)">1»</button><button onclick="stepFrame(10)">10»</button></div><p class="note" style="margin-top:6px">MOG2 顺序播放时有效；跳帧后自动重置</p></div><hr class="divider">' if _vm else ''}
    <button onclick="copy()">复制 YAML 配置</button>
    <p class="note">粘贴到 config/d455.yaml 即可保存</p>
    <pre id="out">—</pre>
  </div>
  <div class="img-wrap">
    <img id="cam" src="/stream" alt="stream loading...">
  </div>
</div>
<script>
function upd(el){{
  document.getElementById(el.id+'v').textContent = el.value;
  fetch('/set?'+el.id+'='+el.value);
  refresh();
}}
function toggleMotion(el){{
  var on = el.checked;
  fetch('/set?motion='+(on?'1':'0'));
  var badge = document.getElementById('motion_badge');
  badge.textContent = on ? 'ON' : 'OFF';
  badge.className = 'badge '+(on?'on':'off');
  refresh();
}}
function v(id){{return document.getElementById(id).value;}}
function motionOn(){{return document.getElementById('motion').checked;}}
function refresh(){{
  document.getElementById('out').textContent =
    '# 粘贴到 config/d455.yaml\\n'+
    'h_low:  '+v('h_low')+'\\n'+
    'h_high: '+v('h_high')+'\\n'+
    's_min:  '+v('s_min')+'\\n'+
    'v_min:  '+v('v_min')+'\\n'+
    'motion: '+(motionOn()?'true':'false')+'\\n\\n'+
    '# tune_hsv_web 专用参数（d455.yaml 里无对应项）\\n'+
    '# mog2_threshold: '+v('mog2_thr')+'\\n'+
    '# min_radius_px:  '+v('min_r')+'\\n'+
    '# circularity:    '+(v('circ')/100).toFixed(2);
}}
function copy(){{
  navigator.clipboard.writeText(document.getElementById('out').textContent)
    .then(()=>alert('已复制到剪贴板'));
}}
// video controls
var _playing = false;
var _playTimer = null;
function seekTo(f){{
  f = parseInt(f);
  document.getElementById('fidx').textContent = f;
  fetch('/seek?f='+f);
}}
function stepFrame(d){{
  var sl = document.getElementById('fslider');
  if(!sl) return;
  var nv = Math.max(0, Math.min(parseInt(sl.value)+d, parseInt(sl.max)));
  sl.value = nv;
  seekTo(nv);
}}
function togglePlay(){{
  _playing = !_playing;
  document.getElementById('playbtn').textContent = _playing ? '⏸ 暂停' : '▶ 播放';
  fetch('/playpause?play='+(_playing?'1':'0'));
  if(_playing){{
    _playTimer = setInterval(function(){{
      var sl = document.getElementById('fslider');
      if(!sl) return;
      fetch('/getframe').then(r=>r.text()).then(function(t){{
        var fi = parseInt(t);
        if(!isNaN(fi)){{ sl.value=fi; document.getElementById('fidx').textContent=fi; }}
      }});
    }}, 200);
  }} else {{
    if(_playTimer){{ clearInterval(_playTimer); _playTimer=null; }}
  }}
}}
// 断线自动重连
var img = document.getElementById('cam');
img.onerror = function(){{
  setTimeout(function(){{img.src='/stream?t='+Date.now();}}, 1500);
}};
refresh();
</script>
</body>
</html>'''


# ── HTTP handler ──────────────────────────────────────────────────────────────

class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass   # suppress per-request logs

    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path in ('/', '/index.html'):
            body = _html().encode()
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', len(body))
            self.end_headers()
            self.wfile.write(body)

        elif parsed.path == '/stream':
            self.send_response(200)
            self.send_header('Content-Type',
                             'multipart/x-mixed-replace; boundary=--jpgbnd')
            self.send_header('Cache-Control', 'no-cache')
            self.end_headers()
            try:
                while not _stop.is_set():
                    with _jpg_lock:
                        jpg = _jpg[0]
                    if jpg is None:
                        time.sleep(0.05)
                        continue
                    try:
                        self.wfile.write(
                            b'--jpgbnd\r\n'
                            b'Content-Type: image/jpeg\r\n'
                            + f'Content-Length: {len(jpg)}\r\n\r\n'.encode()
                            + jpg + b'\r\n'
                        )
                        self.wfile.flush()
                    except OSError:
                        break
                    time.sleep(0.033)   # ~30 fps to browser
            except (BrokenPipeError, ConnectionResetError):
                pass

        elif parsed.path == '/set':
            for k, vs in parse_qs(parsed.query).items():
                _set(k, vs[0])
            self.send_response(200)
            self.send_header('Content-Type', 'text/plain')
            self.end_headers()
            self.wfile.write(b'ok')

        elif parsed.path == '/seek':
            qs = parse_qs(parsed.query)
            if 'f' in qs:
                try:
                    with _seek_lock:
                        _seek_req[0] = max(0, min(int(qs['f'][0]), _n_frames - 1))
                except ValueError:
                    pass
            self.send_response(200)
            self.send_header('Content-Type', 'text/plain')
            self.end_headers()
            self.wfile.write(b'ok')

        elif parsed.path == '/playpause':
            qs = parse_qs(parsed.query)
            if 'play' in qs:
                _play[0] = qs['play'][0] in ('1', 'true')
            self.send_response(200)
            self.send_header('Content-Type', 'text/plain')
            self.end_headers()
            self.wfile.write(b'ok')

        elif parsed.path == '/getframe':
            with _seek_lock:
                fi = _seek_req[0]
            self.send_response(200)
            self.send_header('Content-Type', 'text/plain')
            self.end_headers()
            self.wfile.write(str(fi).encode())

        else:
            self.send_response(404)
            self.end_headers()


# ── Camera + processing loop ──────────────────────────────────────────────────

def _hw_reset_and_start(width, height):
    print("[INFO] Hardware reset…")
    devs = rs.context().query_devices()
    if not devs:
        raise RuntimeError("No RealSense device found.")
    devs[0].hardware_reset()
    time.sleep(6)
    for fps in (60, 30, 15):
        pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16,  fps)
        try:
            profile = pipeline.start(cfg)
            pipeline.wait_for_frames(timeout_ms=6000)
            print(f"[INFO] RealSense OK ({fps} Hz)")
            return pipeline, profile
        except RuntimeError:
            try: pipeline.stop()
            except Exception: pass
    raise RuntimeError("Could not start RealSense pipeline.")


def _process_frame(color, depth_arr, depth_scale, fx, color_intrin, back_sub, pw, ph):
    """Run HSV + MOG2 detection on one frame; return JPEG bytes of the 2×2 composite."""
    p         = _get()
    h_low     = p['h_low'];    h_high = p['h_high']
    s_min     = p['s_min'];    v_min  = p['v_min']
    mog2_t    = p['mog2_thr']
    min_r     = max(p['min_r'], 1)
    circ      = p['circ'] / 100.0
    use_motion = p['motion']

    # ── HSV mask ──────────────────────────────────────────────────────────────
    hsv_low  = np.array([h_low, s_min, v_min], dtype=np.uint8)
    hsv_high = np.array([h_high, 255,  255  ], dtype=np.uint8)
    hsv      = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
    hsv_mask = cv2.inRange(hsv, hsv_low, hsv_high)
    kern3    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    hsv_mask = cv2.morphologyEx(hsv_mask, cv2.MORPH_CLOSE, kern3, iterations=2)
    hsv_mask = cv2.morphologyEx(hsv_mask, cv2.MORPH_OPEN,  kern3, iterations=1)

    # ── MOG2 motion mask ──────────────────────────────────────────────────────
    BG_RESIZE   = 0.4
    small       = cv2.resize(color, (0, 0), fx=BG_RESIZE, fy=BG_RESIZE,
                             interpolation=cv2.INTER_AREA)
    fgmask      = back_sub.apply(small)
    h_, w_      = color.shape[:2]
    motion_mask = cv2.resize(fgmask, (w_, h_), interpolation=cv2.INTER_NEAREST)
    kern5       = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_DILATE, kern5, iterations=2)

    combined = cv2.bitwise_and(hsv_mask, motion_mask) if use_motion else hsv_mask

    # ── Detect candidates ─────────────────────────────────────────────────────
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    vis_raw = color.copy()
    best    = None
    n_pass  = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < np.pi * min_r ** 2:
            continue
        peri = cv2.arcLength(cnt, True)
        if peri == 0:
            continue
        c = 4 * np.pi * area / peri ** 2
        (cx_f, cy_f), r = cv2.minEnclosingCircle(cnt)
        if r < min_r or r > 200:
            continue
        cv2.circle(vis_raw, (int(cx_f), int(cy_f)), int(r), (80, 80, 80), 1)
        if c >= circ:
            n_pass += 1
            score = area * c
            if best is None or score > best[0]:
                best = (score, int(cx_f), int(cy_f), r, c)

    # ── 3D position ───────────────────────────────────────────────────────────
    depth_m = 0.0
    p3d     = None
    bx = by = br = bc = 0
    if best is not None:
        _, bx, by, br, bc = best
        depth_vis = fx * BALL_RADIUS / br if br > 0 else 0.0
        if depth_arr is not None:
            d_raw      = depth_arr[np.clip(by, 0, depth_arr.shape[0]-1),
                                   np.clip(bx, 0, depth_arr.shape[1]-1)]
            depth_sens = d_raw * depth_scale + BALL_RADIUS if d_raw > 0 else 0.0
            depth_m    = depth_vis if depth_vis > 0 else depth_sens
            if depth_m > 0 and color_intrin is not None and rs is not None:
                p3d = rs.rs2_deproject_pixel_to_point(
                    color_intrin, [float(bx), float(by)], depth_m)
        else:
            depth_m = depth_vis   # video mode: visual depth only

    # ── Build 2×2 composite ───────────────────────────────────────────────────
    hsv_disp       = cv2.cvtColor(hsv_mask, cv2.COLOR_GRAY2BGR)
    hsv_disp[hsv_mask > 0]    = [0, 220, 220]
    mot_disp       = cv2.cvtColor(motion_mask, cv2.COLOR_GRAY2BGR)
    mot_disp[motion_mask > 0] = [200, 80, 0]
    if not use_motion:
        cv2.rectangle(mot_disp, (0, 0), (mot_disp.shape[1], mot_disp.shape[0]), (0, 0, 80), 8)
    comb_disp      = cv2.cvtColor(combined, cv2.COLOR_GRAY2BGR)
    comb_disp[combined > 0]   = [0, 200, 80]
    if best is not None:
        cv2.circle(comb_disp, (bx, by), int(br), (0, 255, 0), 2)

    def _s(img):
        return cv2.resize(img, (pw, ph), interpolation=cv2.INTER_AREA)

    p1 = _s(vis_raw);  p2 = _s(hsv_disp)
    p3 = _s(mot_disp); p4 = _s(comb_disp)

    FS = 0.55; TK = 1; WHT = (255, 255, 255); GRN = (0, 255, 0)

    cv2.putText(p1, "1 Color+detect", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, FS, WHT, TK)
    if best is not None:
        bx_p = round(bx * pw / w_); by_p = round(by * ph / h_)
        br_p = max(round(br * pw / w_), 2)
        cv2.circle(p1, (bx_p, by_p), br_p, GRN, 2)
        cv2.circle(p1, (bx_p, by_p), 3, (0, 0, 255), -1)
        xyz = (f"X={p3d[0]:+.3f} Y={p3d[1]:+.3f} Z={p3d[2]:.3f} m" if p3d is not None
               else f"Z={depth_m:.3f} m (visual)")
        cv2.putText(p1, f"BALL  {xyz}", (8, 42), cv2.FONT_HERSHEY_SIMPLEX, FS, GRN, TK)
        cv2.putText(p1, f"r={br:.0f}px  d={depth_m:.2f}m  c={bc:.2f}",
                    (max(bx_p - br_p, 4), max(by_p - br_p - 6, 60)),
                    cv2.FONT_HERSHEY_SIMPLEX, FS * 0.85, GRN, TK)
    else:
        cv2.putText(p1, "No ball", (8, p1.shape[0] // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, FS * 1.2, (0, 80, 255), 2)
    cv2.putText(p1, f"H=[{h_low},{h_high}] S>={s_min} V>={v_min}  r>={min_r}  circ>={circ:.2f}",
                (8, p1.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, FS * 0.75, (200, 200, 0), TK)
    cv2.putText(p2, f"2 HSV  H=[{h_low},{h_high}] S>={s_min} V>={v_min}",
                (8, 20), cv2.FONT_HERSHEY_SIMPLEX, FS, WHT, TK)
    mot_label = f"3 MOG2  thr={mog2_t}" + ("  [NOT APPLIED]" if not use_motion else "")
    cv2.putText(p3, mot_label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, FS, WHT, TK)
    comb_mode = "HSV+MOG2" if use_motion else "HSV only"
    cv2.putText(p4, f"4 {comb_mode}  cands={len(contours)}  pass={n_pass}",
                (8, 20), cv2.FONT_HERSHEY_SIMPLEX, FS, WHT, TK)

    composite = np.vstack([np.hstack([p1, p2]), np.hstack([p3, p4])])
    max_w = 1280
    if composite.shape[1] > max_w:
        scale     = max_w / composite.shape[1]
        composite = cv2.resize(composite, (max_w, int(composite.shape[0] * scale)),
                               interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode('.jpg', composite, [cv2.IMWRITE_JPEG_QUALITY, 82])
    return buf.tobytes() if ok else None


def _video_loop(video_path, port):
    """Main loop for video-file mode."""
    global _video_mode, _n_frames, _fx_video

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open {video_path}")
        _stop.set()
        return

    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vid_w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vid_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    _n_frames  = total
    _video_mode = True
    print(f"[INFO] Video: {video_path}  {vid_w}×{vid_h}  {vid_fps:.0f}fps  {total} frames")
    print(f"[INFO] Max range ≈ {_fx_video * BALL_RADIUS / DEF['min_r']:.1f} m (visual depth)")
    print(f"\n[INFO] Open  http://localhost:{port}  in your browser")
    print("[INFO] Ctrl+C to quit\n")

    pw = vid_w // 2; ph = vid_h // 2
    back_sub      = cv2.createBackgroundSubtractorMOG2(
        history=100, varThreshold=DEF['mog2_thr'], detectShadows=False)
    prev_mog2_thr = DEF['mog2_thr']
    prev_idx      = -1
    frame         = None

    try:
        while not _stop.is_set():
            with _seek_lock:
                idx = _seek_req[0]

            playing = _play[0]
            if playing:
                next_idx = min(idx + 1, total - 1)
                with _seek_lock:
                    _seek_req[0] = next_idx
                idx = next_idx

            # Load frame only if index changed
            if idx != prev_idx:
                if idx != prev_idx + 1:
                    # Non-sequential seek: reset MOG2 background model
                    p = _get()
                    back_sub = cv2.createBackgroundSubtractorMOG2(
                        history=100, varThreshold=p['mog2_thr'], detectShadows=False)
                    prev_mog2_thr = p['mog2_thr']
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.05)
                    continue
                prev_idx = idx

            if frame is None:
                time.sleep(0.05)
                continue

            # Recreate MOG2 if threshold changed (sequential playback)
            p = _get()
            if p['mog2_thr'] != prev_mog2_thr:
                back_sub = cv2.createBackgroundSubtractorMOG2(
                    history=100, varThreshold=p['mog2_thr'], detectShadows=False)
                prev_mog2_thr = p['mog2_thr']

            color = frame.copy()
            # Add frame counter overlay
            cv2.putText(color, f"frame {idx}/{total-1}",
                        (color.shape[1] - 180, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

            jpg = _process_frame(color, None, 1.0, _fx_video, None, back_sub, pw, ph)
            if jpg is not None:
                with _jpg_lock:
                    _jpg[0] = jpg

            if playing:
                time.sleep(1.0 / vid_fps)
            else:
                time.sleep(0.04)   # ~25 Hz refresh while paused

    except KeyboardInterrupt:
        print("\n[INFO] Stopping…")
    finally:
        cap.release()


def main():
    parser = argparse.ArgumentParser(
        description="Web-based HSV/MOG2 tuner — open browser at http://localhost:PORT")
    parser.add_argument("--video",  default="",
                        help="path to recorded video (skips live camera)")
    parser.add_argument("--width",  type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--port",   type=int, default=5000)
    args = parser.parse_args()

    # ── Start HTTP server ─────────────────────────────────────────────────────
    server = ThreadingHTTPServer(('0.0.0.0', args.port), _Handler)
    srv_thread = threading.Thread(target=server.serve_forever, daemon=True)
    srv_thread.start()

    # ── Video mode ────────────────────────────────────────────────────────────
    if args.video:
        _video_loop(args.video, args.port)
        _stop.set()
        server.shutdown()
        print("[INFO] Done.")
        return

    # ── Camera mode ───────────────────────────────────────────────────────────
    if not _HAS_RS:
        print("[ERROR] pyrealsense2 not installed and no --video given. Exiting.")
        server.shutdown()
        return

    pipeline, profile = _hw_reset_and_start(args.width, args.height)
    color_intrin = (profile.get_stream(rs.stream.color)
                    .as_video_stream_profile().get_intrinsics())
    depth_scale  = profile.get_device().first_depth_sensor().get_depth_scale()
    fx = color_intrin.fx
    print(f"[INFO] fx={fx:.1f}  depth_scale={depth_scale:.5f}")
    print(f"[INFO] Max range ≈ {fx * BALL_RADIUS / DEF['min_r']:.1f} m")
    print(f"\n[INFO] Open  http://localhost:{args.port}  in your browser")
    print("[INFO] Ctrl+C to quit\n")

    back_sub      = cv2.createBackgroundSubtractorMOG2(
        history=100, varThreshold=DEF['mog2_thr'], detectShadows=False)
    prev_mog2_thr = DEF['mog2_thr']
    fps_t0        = time.time()
    fps_smooth    = 0.0
    pw = args.width  // 2
    ph = args.height // 2

    try:
        while not _stop.is_set():
            frames = pipeline.wait_for_frames(timeout_ms=3000)
            cf = frames.get_color_frame()
            df = frames.get_depth_frame()
            if not cf or not df:
                continue

            t_now      = time.time()
            fps_smooth = (0.85 * fps_smooth + 0.15 / max(t_now - fps_t0, 1e-6)
                          if fps_smooth > 0 else 1.0 / max(t_now - fps_t0, 1e-6))
            fps_t0     = t_now

            color     = np.asanyarray(cf.get_data()).copy()
            depth_arr = np.asanyarray(df.get_data())

            p = _get()
            if p['mog2_thr'] != prev_mog2_thr:
                back_sub = cv2.createBackgroundSubtractorMOG2(
                    history=100, varThreshold=p['mog2_thr'], detectShadows=False)
                prev_mog2_thr = p['mog2_thr']

            cv2.putText(color, f"{fps_smooth:.1f} FPS",
                        (color.shape[1] - 90, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

            jpg = _process_frame(color, depth_arr, depth_scale, fx,
                                 color_intrin, back_sub, pw, ph)
            if jpg is not None:
                with _jpg_lock:
                    _jpg[0] = jpg

    except KeyboardInterrupt:
        print("\n[INFO] Stopping…")
    finally:
        _stop.set()
        pipeline.stop()
        server.shutdown()
        print("[INFO] Done.")


if __name__ == "__main__":
    main()
