"""
viz3d.py -- Interactive 3-D ball trajectory viewer  (stdlib only, no Flask)

First run: downloads Three.js (~630 KB) once into the same directory.
After that: fully offline, no CDN needed.

Usage:
    # Terminal 1 -- start detection with UDP broadcast:
    python ball_detection_d455.py --viz3d

    # Terminal 2 -- start viewer:
    python viz3d.py
    # Open  http://localhost:5001  in browser
"""

import argparse
import json
import os
import socket
import socketserver
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer

# ── Local Three.js cache (downloaded once) ────────────────────────────────────
_HERE       = os.path.dirname(os.path.abspath(__file__))
_THREE_PATH = os.path.join(_HERE, "_three.min.js")
_ORBIT_PATH = os.path.join(_HERE, "_orbit.min.js")

_THREE_URL = "https://cdn.jsdelivr.net/npm/three@0.128.0/build/three.min.js"
_ORBIT_URL = ("https://cdn.jsdelivr.net/npm/three@0.128.0"
              "/examples/js/controls/OrbitControls.js")


def _ensure_threejs():
    for path, url, label in [
        (_THREE_PATH, _THREE_URL, "three.min.js"),
        (_ORBIT_PATH, _ORBIT_URL, "OrbitControls.js"),
    ]:
        if not os.path.exists(path):
            print(f"[viz3d] Downloading {label} ...", end=" ", flush=True)
            try:
                urllib.request.urlretrieve(url, path)
                print("OK")
            except Exception as e:
                print(f"FAILED ({e})")
                print(f"[viz3d] Manual download: {url}")
                print(f"[viz3d]   save as: {path}")


# ── HTML template ─────────────────────────────────────────────────────────────
_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Ball Detection - 3-D Viewer</title>
<style>
* { margin:0; padding:0; box-sizing:border-box; }
body { background:#111; color:#ddd; font-family:monospace; overflow:hidden; }
canvas { display:block; }
#panel {
  position:absolute; top:12px; left:12px;
  background:rgba(0,0,0,.82); border:1px solid #333;
  border-radius:8px; padding:10px 14px; font-size:13px;
  min-width:310px; line-height:1.8;
  user-select:none;
}
#panel b { color:#aef; }
#det-bar { display:flex; gap:6px; margin-bottom:6px; }
.det-btn {
  flex:1; padding:3px 0; border-radius:4px; border:1px solid #555;
  background:#222; color:#aaa; font-family:monospace; font-size:12px;
  cursor:pointer;
}
.det-btn.active { background:#1a3a6a; border-color:#4af; color:#fff; }
#ctrl {
  margin-top:8px; border-top:1px solid #333; padding-top:6px;
}
.ctrl-lbl { color:#777; font-size:11px; margin-bottom:2px; }
.ctrl-row { display:flex; align-items:center; gap:6px; margin:4px 0; }
.ctrl-row label { width:50px; color:#888; font-size:11px; }
.ctrl-row input[type=range] { flex:1; accent-color:#fa3; cursor:pointer; }
.ctrl-row span { width:34px; text-align:right; font-size:12px; color:#fa3; }
#hint {
  position:absolute; bottom:10px; left:12px;
  background:rgba(0,0,0,.5); border-radius:4px;
  padding:5px 10px; font-size:11px; color:#555;
}
#legend {
  position:absolute; bottom:10px; right:12px;
  background:rgba(0,0,0,.7); border:1px solid #333; border-radius:6px;
  padding:8px 12px; font-size:11px; line-height:1.9;
}
.lg { display:inline-block; width:14px; height:3px;
      border-radius:2px; vertical-align:middle; margin-right:5px; }
#err {
  position:absolute; top:50%; left:50%; transform:translate(-50%,-50%);
  background:#300; border:1px solid #f44; border-radius:8px;
  padding:20px 28px; font-size:14px; color:#fbb; display:none;
}
.ok  { color:#3e3; }
.old { color:#fa3; }
.bad { color:#e33; }
</style>
</head>
<body>
<div id="err">
  <b>Three.js failed to load.</b><br>
  Run viz3d.py again -- it will re-download the library.
</div>

<div id="panel">
  <b>Ball Detection - 3-D Viewer</b><br>
  <div id="det-bar"></div>
  <span id="s_bd" style="font-size:11px;color:#555">ball_detection: 等待连接…</span><br>
  <span id="s_tag">TAG: --</span><br>
  <span id="s_ball">Ball: --</span><br>
  <span id="s_vel">Vel:  --</span><br>
  <div id="ctrl">
    <div class="ctrl-lbl">Restitution (bounce damping)</div>
    <div class="ctrl-row">
      <label>rest X</label>
      <input type="range" id="r_x" min="0" max="1.5" step="0.05" value="0.75">
      <span id="rv_x">0.75</span>
    </div>
    <div class="ctrl-row">
      <label>rest Y</label>
      <input type="range" id="r_y" min="0" max="1.5" step="0.05" value="0.75">
      <span id="rv_y">0.75</span>
    </div>
    <div class="ctrl-row">
      <label>rest Z</label>
      <input type="range" id="r_z" min="0" max="1.5" step="0.05" value="0.75">
      <span id="rv_z">0.75</span>
    </div>
  </div>
</div>

<div id="hint">Left-drag: rotate &nbsp;|&nbsp; Scroll: zoom &nbsp;|&nbsp; Right-drag: pan</div>

<div id="legend">
  <span class="lg" style="background:#ff5500"></span>Predicted trajectory<br>
  <span class="lg" style="background:#ff9900"></span>Ball history trail<br>
  <span class="lg" style="background:#00ffff; height:8px; border-radius:50%"></span>Bounce point<br>
  <span class="lg" style="background:#22ff55; height:8px; border-radius:50%"></span>Current ball
</div>

<script src="/three.js"></script>
<script src="/orbit.js"></script>
<script>
if (typeof THREE === 'undefined') {
  document.getElementById('err').style.display = 'block';
  throw new Error('THREE not loaded');
}

// ── Renderer ──────────────────────────────────────────────────────────────────
var renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setClearColor(0x111111);
document.body.appendChild(renderer.domElement);
window.addEventListener('resize', function() {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

// ── Camera & controls ─────────────────────────────────────────────────────────
var camera = new THREE.PerspectiveCamera(
  50, window.innerWidth / window.innerHeight, 0.01, 50);
camera.position.set(2, 2.5, 3.5);

var controls = new THREE.OrbitControls(camera, renderer.domElement);
controls.target.set(0, 0.5, 0);
controls.update();

var scene = new THREE.Scene();
scene.add(new THREE.AmbientLight(0xffffff, 0.6));
var dl = new THREE.DirectionalLight(0xffffff, 0.8);
dl.position.set(3, 5, 3);
scene.add(dl);

// ── Coord mapping: world(x,y,z) --> Three.js(x, z_world, -y_world) ───────────
function w2t(x, y, z) { return new THREE.Vector3(x, z, -y); }
function a2t(a) { return w2t(a[0], a[1], a[2]); }

// ── Static scene ──────────────────────────────────────────────────────────────
var ground = new THREE.Mesh(
  new THREE.PlaneGeometry(12, 12),
  new THREE.MeshLambertMaterial({ color: 0x1a2e1a, side: THREE.DoubleSide }));
ground.rotation.x = -Math.PI / 2;
scene.add(ground);
scene.add(new THREE.GridHelper(12, 24, 0x2a3a2a, 0x242424));
scene.add(new THREE.AxesHelper(0.6));

var tagMesh = new THREE.Mesh(
  new THREE.PlaneGeometry(0.15, 0.15),
  new THREE.MeshBasicMaterial({ color: 0xffdd00, side: THREE.DoubleSide }));
tagMesh.rotation.x = -Math.PI / 2;
tagMesh.position.y = 0.003;
scene.add(tagMesh);

// ── Current ball ──────────────────────────────────────────────────────────────
var ballMesh = new THREE.Mesh(
  new THREE.SphereGeometry(0.0335, 20, 16),
  new THREE.MeshLambertMaterial({ color: 0x22ff55 }));
ballMesh.visible = false;
scene.add(ballMesh);

// ── Camera cone ───────────────────────────────────────────────────────────────
var camMesh = new THREE.Mesh(
  new THREE.ConeGeometry(0.06, 0.18, 8),
  new THREE.MeshLambertMaterial({ color: 0xff8c00 }));
camMesh.visible = false;
scene.add(camMesh);

// ── Ball history trail (fading orange line: dim=old, bright=new) ──────────────
var HMAX = 120;
var hPosBuf = new Float32Array(HMAX * 3);
var hColBuf = new Float32Array(HMAX * 3);
var hGeo = new THREE.BufferGeometry();
hGeo.setAttribute('position', new THREE.BufferAttribute(hPosBuf, 3));
hGeo.setAttribute('color',    new THREE.BufferAttribute(hColBuf, 3));
hGeo.setDrawRange(0, 0);
scene.add(new THREE.Line(hGeo,
  new THREE.LineBasicMaterial({ vertexColors: true })));

// ── Predicted trajectory (red line) ──────────────────────────────────────────
var TMAX = 120;
var tBuf = new Float32Array(TMAX * 3);
var tGeo = new THREE.BufferGeometry();
tGeo.setAttribute('position', new THREE.BufferAttribute(tBuf, 3));
tGeo.setDrawRange(0, 0);
scene.add(new THREE.Line(tGeo,
  new THREE.LineBasicMaterial({ color: 0xff3300 })));

// ── Bounce markers (cyan spheres) ─────────────────────────────────────────────
var BMAX = 6;
var bMeshes = [];
for (var bi = 0; bi < BMAX; bi++) {
  var bm = new THREE.Mesh(
    new THREE.SphereGeometry(0.045, 10, 8),
    new THREE.MeshLambertMaterial({ color: 0x00ffff }));
  bm.visible = false;
  scene.add(bm);
  bMeshes.push(bm);
}

// ── Detector toggle ───────────────────────────────────────────────────────────
var currentDet = null;
var knownDets  = [];

function _setDet(det) {
  currentDet = det;
  document.querySelectorAll('.det-btn').forEach(function(b) {
    b.classList.toggle('active', b.dataset.det === det);
  });
}

function _syncDetBar(dets) {
  if (!dets || dets.length === 0) return;
  if (dets.join(',') === knownDets.join(',')) return;
  knownDets = dets.slice();
  var bar = document.getElementById('det-bar');
  bar.innerHTML = '';
  dets.forEach(function(det) {
    var btn = document.createElement('button');
    btn.className = 'det-btn';
    btn.dataset.det = det;
    btn.textContent = det;
    btn.addEventListener('click', function() { _setDet(det); });
    bar.appendChild(btn);
  });
  if (!currentDet || dets.indexOf(currentDet) === -1) _setDet(dets[0]);
  else _setDet(currentDet);
}

// ── Restitution sliders ───────────────────────────────────────────────────────
var _sliderSyncing = false;

function _postRest() {
  if (_sliderSyncing) return;
  var rx = parseFloat(document.getElementById('r_x').value);
  var ry = parseFloat(document.getElementById('r_y').value);
  var rz = parseFloat(document.getElementById('r_z').value);
  document.getElementById('rv_x').textContent = rx.toFixed(2);
  document.getElementById('rv_y').textContent = ry.toFixed(2);
  document.getElementById('rv_z').textContent = rz.toFixed(2);
  fetch('/set_rest', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({rest: [rx, ry, rz]})
  }).catch(function() {});
}

['r_x', 'r_y', 'r_z'].forEach(function(id) {
  document.getElementById(id).addEventListener('input', _postRest);
});

function _syncSliders(rest) {
  _sliderSyncing = true;
  document.getElementById('r_x').value = rest[0];
  document.getElementById('r_y').value = rest[1];
  document.getElementById('r_z').value = rest[2];
  document.getElementById('rv_x').textContent = (+rest[0]).toFixed(2);
  document.getElementById('rv_y').textContent = (+rest[1]).toFixed(2);
  document.getElementById('rv_z').textContent = (+rest[2]).toFixed(2);
  _sliderSyncing = false;
}

// ── ball_detection heartbeat ──────────────────────────────────────────────────
var _bdLastT = null;
setInterval(function() {
  var el = document.getElementById('s_bd');
  if (!el) return;
  if (_bdLastT === null) {
    el.innerHTML = '<span style="color:#555">ball_detection: 等待连接…</span>';
  } else {
    var lag = Date.now() / 1000 - _bdLastT;
    if (lag > 2) {
      el.innerHTML = '<span class="bad">ball_detection: 断开 (' + lag.toFixed(0) + 's)</span>';
    } else {
      el.innerHTML = '<span class="ok">ball_detection: 在线</span>';
    }
  }
}, 500);

// ── SSE state updater ─────────────────────────────────────────────────────────
var sse = new EventSource('/stream');
sse.onmessage = function(ev) {
  var d;
  try { d = JSON.parse(ev.data); } catch(e) { return; }

  if (d.t) _bdLastT = d.t;

  // Shared: TAG status (tag_age=9999 means never seen; treat as NO TAG)
  var age = (d.tag_age !== undefined && d.tag_age < 9000) ? d.tag_age : 9999;
  var cls = age === 0 ? 'ok' : (age < 30 ? 'old' : 'bad');
  var lbl = age === 0 ? 'TAG OK' : (age < 30 ? 'TAG [' + age + 'f]' : 'NO TAG');
  document.getElementById('s_tag').innerHTML = '<span class="' + cls + '">' + lbl + '</span>';

  // Shared: rest values -> sync sliders (suppress re-POST)
  if (d.rest) _syncSliders(d.rest);

  // Shared: camera pose
  if (d.cam && d.cam_look) {
    camMesh.position.copy(a2t(d.cam));
    var look = a2t(d.cam_look).normalize();
    camMesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), look);
    camMesh.visible = true;
  }

  // Detector bar
  _syncDetBar(d.detectors);

  // Per-detector data
  var pd = (currentDet && d[currentDet]) ? d[currentDet] : null;

  // Ball position
  if (pd && pd.ball) {
    document.getElementById('s_ball').textContent =
      'Ball  X=' + pd.ball[0].toFixed(3) +
      '  Y=' + pd.ball[1].toFixed(3) +
      '  Z=' + pd.ball[2].toFixed(3) + ' m';
    ballMesh.position.copy(a2t(pd.ball));
    ballMesh.visible = true;
  } else {
    document.getElementById('s_ball').textContent = 'Ball: not detected';
    ballMesh.visible = false;
  }

  // Velocity
  if (pd && pd.vel) {
    document.getElementById('s_vel').textContent =
      'Vel   ' + pd.vel[0].toFixed(2) + ' / ' +
      pd.vel[1].toFixed(2) + ' / ' + pd.vel[2].toFixed(2) + ' m/s';
  } else {
    document.getElementById('s_vel').textContent = 'Vel:  --';
  }

  // History trail with orange fade
  var hist = pd ? pd.ball_hist : null;
  if (hist && hist.length) {
    var hn = Math.min(hist.length, HMAX);
    for (var i = 0; i < hn; i++) {
      var hp = a2t(hist[i]);
      hPosBuf[i*3] = hp.x; hPosBuf[i*3+1] = hp.y; hPosBuf[i*3+2] = hp.z;
      var t = (hn > 1) ? i / (hn - 1) : 1.0;   // 0=oldest  1=newest
      hColBuf[i*3]   = 0.08 + 0.92 * t;          // R  (orange)
      hColBuf[i*3+1] = 0.03 + 0.35 * t * t;      // G  (slight warmth at bright end)
      hColBuf[i*3+2] = 0.0;                       // B
    }
    hGeo.attributes.position.needsUpdate = true;
    hGeo.attributes.color.needsUpdate    = true;
    hGeo.setDrawRange(0, hn);
  } else {
    hGeo.setDrawRange(0, 0);
  }

  // Predicted trajectory
  var traj = pd ? pd.traj : null;
  if (traj && traj.length) {
    var tn = Math.min(traj.length, TMAX);
    for (var j = 0; j < tn; j++) {
      var tp = a2t(traj[j]);
      tBuf[j*3] = tp.x; tBuf[j*3+1] = tp.y; tBuf[j*3+2] = tp.z;
    }
    tGeo.attributes.position.needsUpdate = true;
    tGeo.setDrawRange(0, tn);
  } else {
    tGeo.setDrawRange(0, 0);
  }

  // Bounce markers
  for (var k = 0; k < BMAX; k++) bMeshes[k].visible = false;
  var bounces = pd ? pd.bounces : null;
  if (bounces) {
    var bn = Math.min(bounces.length, BMAX);
    for (var m = 0; m < bn; m++) {
      bMeshes[m].position.copy(a2t(bounces[m]));
      bMeshes[m].visible = true;
    }
  }
};

sse.onerror = function() {
  document.getElementById('s_tag').textContent = 'SSE: reconnecting...';
};

// ── Render loop ───────────────────────────────────────────────────────────────
(function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
})();
</script>
</body>
</html>
"""

# ── Shared state ──────────────────────────────────────────────────────────────
_lock  = threading.Lock()
_state = {
    "detectors": [],
    "cam": None, "cam_look": None,
    "tag_age": 9999,
    "rest": [0.75, 0.75, 0.75],
}

# ── Control UDP socket (sends rest updates back to ball_detection) ────────────
_ctrl_sock = None   # set in main()
_ctrl_addr = ("127.0.0.1", 5566)


def _udp_receiver(port: int):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", port))
    sock.settimeout(1.0)
    print(f"[viz3d] UDP on 127.0.0.1:{port}  (waiting for ball_detection --viz3d)")
    while True:
        try:
            data, _ = sock.recvfrom(131072)
            with _lock:
                _state.update(json.loads(data))
        except socket.timeout:
            pass
        except Exception:
            pass


# ── HTTP handler ──────────────────────────────────────────────────────────────
_HTML_BYTES = _HTML.encode("utf-8")

def _load_js(path):
    try:
        with open(path, "rb") as f:
            return f.read()
    except FileNotFoundError:
        return None


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, "text/html; charset=utf-8", _HTML_BYTES)

        elif self.path == "/three.js":
            data = _load_js(_THREE_PATH)
            if data:
                self._send(200, "application/javascript", data)
            else:
                self._send(404, "text/plain", b"three.js not found -- restart viz3d.py")

        elif self.path == "/orbit.js":
            data = _load_js(_ORBIT_PATH)
            if data:
                self._send(200, "application/javascript", data)
            else:
                self._send(404, "text/plain", b"orbit.js not found -- restart viz3d.py")

        elif self.path == "/stream":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("X-Accel-Buffering", "no")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            try:
                while True:
                    with _lock:
                        payload = json.dumps(_state)
                    self.wfile.write(f"data: {payload}\n\n".encode())
                    self.wfile.flush()
                    time.sleep(1 / 20)
            except (BrokenPipeError, ConnectionResetError):
                pass

        else:
            self._send(404, "text/plain", b"not found")

    def do_POST(self):
        if self.path == "/set_rest":
            try:
                length = int(self.headers.get("Content-Length", 0))
                body   = self.rfile.read(length)
                pkt    = json.loads(body)
                rest   = pkt.get("rest")
                if isinstance(rest, list) and len(rest) == 3:
                    # Update shared state so SSE reflects it immediately
                    with _lock:
                        _state["rest"] = [round(float(v), 2) for v in rest]
                    # Forward to ball_detection_d455.py via UDP
                    if _ctrl_sock is not None:
                        _ctrl_sock.sendto(
                            json.dumps({"rest": _state["rest"]}).encode(),
                            _ctrl_addr,
                        )
                self._send(200, "application/json", b'{"ok":true}')
            except Exception as e:
                self._send(400, "application/json",
                           json.dumps({"error": str(e)}).encode())
        else:
            self._send(404, "text/plain", b"not found")

    def _send(self, code, ctype, body):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class _ThreadedServer(socketserver.ThreadingMixIn, HTTPServer):
    daemon_threads = True


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    global _ctrl_sock, _ctrl_addr

    parser = argparse.ArgumentParser(description="viz3d -- 3-D ball trajectory viewer")
    parser.add_argument("--port",      type=int, default=5001,
                        help="HTTP port for the web viewer (default 5001)")
    parser.add_argument("--udp-port",  type=int, default=5565,
                        help="UDP port to receive ball_detection state (default 5565)")
    parser.add_argument("--ctrl-port", type=int, default=5566,
                        help="UDP port to send rest control to ball_detection (default 5566)")
    args = parser.parse_args()

    _ctrl_addr = ("127.0.0.1", args.ctrl_port)
    _ctrl_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    _ensure_threejs()

    threading.Thread(
        target=_udp_receiver, args=(args.udp_port,), daemon=True).start()

    print(f"[viz3d] Control UDP → localhost:{args.ctrl_port}  "
          f"(ball_detection must be started with --ctrl-port {args.ctrl_port})")
    print(f"[viz3d] Open  http://localhost:{args.port}  in your browser")
    server = _ThreadedServer(("0.0.0.0", args.port), _Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[viz3d] Done.")


if __name__ == "__main__":
    main()
