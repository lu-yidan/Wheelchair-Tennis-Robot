"""
webview.py -- 2-D web viewer for ball_detection_d455.py

UDP packet from ball_detection (first byte = type):
  0x00 + json  -- state update (ball, velocity, rest, …)

Video frames are served directly by ball_detection's built-in MJPEG server
(mjpeg_port, default 5568) at full resolution — no UDP frame transfer.

Usage:
    # In config/d455.yaml:  webview: true  (and optionally webview_port: 5567)
    python ball_detection_d455.py     # starts camera + MJPEG server

    # In a second terminal:
    python webview.py
    # Open  http://localhost:5002  in browser
"""

import argparse
import json
import os
import socket
import socketserver
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

# ── Shared state ──────────────────────────────────────────────────────────────
_state_lock = threading.Lock()
_state = {
    "detectors": [],
    "cam": None, "cam_look": None,
    "tag_age": 9999,
    "rest": [0.75, 0.75, 0.75],
}

# ── Control socket (forwards rest updates back to ball_detection) ─────────────
_ctrl_sock = None
_ctrl_addr = ("127.0.0.1", 5566)

# ── HTML ──────────────────────────────────────────────────────────────────────
_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Ball Detection - Web Viewer</title>
<style>
* { margin:0; padding:0; box-sizing:border-box; }
html, body { height:100%; background:#0d0d0d; color:#ddd;
             font-family:monospace; overflow:hidden; }

/* ── layout: main camera left, sidebar right ── */
#layout { display:flex; height:100vh; }

#main-wrap {
  flex:1; position:relative; background:#000; overflow:hidden;
}
#main-wrap img { width:100%; height:100%; object-fit:contain; display:block; }
#main-label {
  position:absolute; top:8px; left:8px;
  background:rgba(0,0,0,.6); border-radius:4px;
  padding:2px 8px; font-size:11px; color:#888;
}

#sidebar {
  width:340px; display:flex; flex-direction:column;
  border-left:1px solid #222;
}

#court-wrap {
  flex-shrink:0; height:220px; background:#000; overflow:hidden;
  border-bottom:1px solid #222; position:relative;
}
#court-wrap img { width:100%; height:100%; object-fit:contain; display:block; }
#court-label {
  position:absolute; top:6px; left:8px;
  background:rgba(0,0,0,.6); border-radius:4px;
  padding:2px 8px; font-size:11px; color:#888;
}

#info {
  flex:1; padding:12px 14px; overflow-y:auto; font-size:13px; line-height:1.8;
}
#info b { color:#aef; }
#det-bar { display:flex; gap:6px; margin:6px 0; }
.det-btn {
  flex:1; padding:3px 0; border-radius:4px; border:1px solid #555;
  background:#222; color:#aaa; font-family:monospace; font-size:12px;
  cursor:pointer;
}
.det-btn.active { background:#1a3a6a; border-color:#4af; color:#fff; }

#ctrl { margin-top:10px; border-top:1px solid #333; padding-top:8px; }
.ctrl-lbl { color:#666; font-size:11px; margin-bottom:4px; }
.ctrl-row { display:flex; align-items:center; gap:6px; margin:4px 0; }
.ctrl-row label { width:50px; color:#777; font-size:11px; }
.ctrl-row input[type=range] { flex:1; accent-color:#fa3; cursor:pointer; }
.ctrl-row span { width:34px; text-align:right; font-size:12px; color:#fa3; }

.ok  { color:#3e3; }
.old { color:#fa3; }
.bad { color:#e33; }

#no-stream {
  position:absolute; color:#444; font-size:13px; pointer-events:none;
}
</style>
</head>
<body>
<div id="layout">

  <!-- ── Main camera MJPEG ── -->
  <div id="main-wrap">
    <span id="no-stream">Waiting for stream…</span>
    <img id="main-img">
    <div id="main-label">Camera (annotated)</div>
  </div>

  <!-- ── Sidebar ── -->
  <div id="sidebar">

    <!-- Court 2-D view -->
    <div id="court-wrap">
      <span id="no-court" style="position:absolute;color:#444;font-size:12px">
        Waiting for court view…</span>
      <img id="court-img">
      <div id="court-label">Court top-down</div>
    </div>

    <!-- Info + controls -->
    <div id="info">
      <b>Ball Detection</b>
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
  </div>
</div>

<script>
// ── Detector toggle ───────────────────────────────────────────────────────────
var currentDet = null, knownDets = [];

function _postDet(det) {
  fetch('/set_det', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({det: det})
  }).catch(function(){});
}

function _setDet(det, notify) {
  currentDet = det;
  document.querySelectorAll('.det-btn').forEach(function(b) {
    b.classList.toggle('active', b.dataset.det === det);
  });
  if (notify !== false) _postDet(det);
}

function _syncDetBar(dets) {
  if (!dets || !dets.length) return;
  if (dets.join(',') === knownDets.join(',')) return;
  knownDets = dets.slice();
  var bar = document.getElementById('det-bar');
  bar.innerHTML = '';
  dets.forEach(function(det) {
    var btn = document.createElement('button');
    btn.className = 'det-btn'; btn.dataset.det = det; btn.textContent = det;
    btn.addEventListener('click', function() { _setDet(det); });
    bar.appendChild(btn);
  });
  // auto-select: keep current if still valid, else first; always notify ball_detection
  var sel = (currentDet && dets.indexOf(currentDet) !== -1) ? currentDet : dets[0];
  _setDet(sel);
}

// ── Restitution sliders ───────────────────────────────────────────────────────
var _syncing = false;

function _postRest() {
  if (_syncing) return;
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
  }).catch(function(){});
}

['r_x','r_y','r_z'].forEach(function(id) {
  document.getElementById(id).addEventListener('input', _postRest);
});

function _syncSliders(rest) {
  _syncing = true;
  document.getElementById('r_x').value = rest[0];
  document.getElementById('r_y').value = rest[1];
  document.getElementById('r_z').value = rest[2];
  document.getElementById('rv_x').textContent = (+rest[0]).toFixed(2);
  document.getElementById('rv_y').textContent = (+rest[1]).toFixed(2);
  document.getElementById('rv_z').textContent = (+rest[2]).toFixed(2);
  _syncing = false;
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
      el.innerHTML = '<span style="color:#e33">ball_detection: 断开 (' + lag.toFixed(0) + 's)</span>';
    } else {
      el.innerHTML = '<span style="color:#3e3">ball_detection: 在线</span>';
    }
  }
}, 500);

// ── SSE state updates ─────────────────────────────────────────────────────────
var sse = new EventSource('/stream');
sse.onmessage = function(ev) {
  var d; try { d = JSON.parse(ev.data); } catch(e) { return; }

  if (d.t) {
    _bdLastT = d.t;
    // Hide "waiting" placeholders on first real packet
    var ns = document.getElementById('no-stream');
    var nc = document.getElementById('no-court');
    if (ns) ns.style.display = 'none';
    if (nc) nc.style.display = 'none';
  }

  var age = (d.tag_age !== undefined && d.tag_age < 9000) ? d.tag_age : 9999;
  var cls = age === 0 ? 'ok' : (age < 30 ? 'old' : 'bad');
  var lbl = age === 0 ? 'TAG OK' : (age < 30 ? 'TAG [' + age + 'f]' : 'NO TAG');
  document.getElementById('s_tag').innerHTML = '<span class="' + cls + '">' + lbl + '</span>';

  if (d.rest) _syncSliders(d.rest);
  _syncDetBar(d.detectors);

  var pd = (currentDet && d[currentDet]) ? d[currentDet] : null;

  if (pd && pd.ball) {
    document.getElementById('s_ball').textContent =
      'Ball  X=' + pd.ball[0].toFixed(3) +
      '  Y=' + pd.ball[1].toFixed(3) +
      '  Z=' + pd.ball[2].toFixed(3) + ' m';
  } else {
    document.getElementById('s_ball').textContent = 'Ball: not detected';
  }

  if (pd && pd.vel) {
    document.getElementById('s_vel').textContent =
      'Vel  ' + pd.vel[0].toFixed(2) + ' / ' +
      pd.vel[1].toFixed(2) + ' / ' + pd.vel[2].toFixed(2) + ' m/s';
  } else {
    document.getElementById('s_vel').textContent = 'Vel:  --';
  }
};

sse.onerror = function() {
  document.getElementById('s_tag').innerHTML = '<span class="bad">disconnected</span>';
};

// ── MJPEG stream from ball_detection's built-in server ────────────────────────
var MJPEG_PORT = __MJPEG_PORT__;

function _attachMJPEG(imgId, path) {
  var img = document.getElementById(imgId);
  var url = 'http://' + location.hostname + ':' + MJPEG_PORT + path;
  function load() { img.src = url; }
  img.onerror = function() {
    img.src = '';
    setTimeout(load, 2000);   // retry after 2 s if ball_detection not ready yet
  };
  load();
}

_attachMJPEG('main-img',  '/main');
_attachMJPEG('court-img', '/court');
</script>
</body>
</html>
"""

_HTML_BYTES = _HTML.encode("utf-8")


# ── UDP receiver: state (0x00) only — frames come from ball_detection's MJPEG server ──
def _udp_receiver(port: int):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", port))
    sock.settimeout(1.0)
    print(f"[webview] UDP state on 127.0.0.1:{port}  (waiting for ball_detection --webview)")
    while True:
        try:
            data, _ = sock.recvfrom(65536)
            if len(data) < 2:
                continue
            if data[0] == 0:
                with _state_lock:
                    _state.update(json.loads(data[1:]))
        except socket.timeout:
            pass
        except Exception:
            pass


# ── HTTP handler ──────────────────────────────────────────────────────────────
class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, "text/html; charset=utf-8", _HTML_BYTES)

        elif self.path == "/stream":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("X-Accel-Buffering", "no")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            try:
                while True:
                    with _state_lock:
                        payload = json.dumps(_state)
                    self.wfile.write(f"data: {payload}\n\n".encode())
                    self.wfile.flush()
                    time.sleep(0.1)
            except (BrokenPipeError, ConnectionResetError):
                pass

        else:
            self._send(404, "text/plain", b"not found")

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            body   = self.rfile.read(length)
            pkt    = json.loads(body)
        except Exception as e:
            self._send(400, "application/json",
                       json.dumps({"error": str(e)}).encode())
            return

        if self.path == "/set_rest":
            rest = pkt.get("rest")
            if isinstance(rest, list) and len(rest) == 3:
                with _state_lock:
                    _state["rest"] = [round(float(v), 2) for v in rest]
                if _ctrl_sock is not None:
                    _ctrl_sock.sendto(
                        json.dumps({"rest": _state["rest"]}).encode(),
                        _ctrl_addr,
                    )
            self._send(200, "application/json", b'{"ok":true}')

        elif self.path == "/set_det":
            det = pkt.get("det") or None   # None / "" → composite (both)
            if _ctrl_sock is not None:
                _ctrl_sock.sendto(
                    json.dumps({"det": det}).encode(),
                    _ctrl_addr,
                )
            self._send(200, "application/json", b'{"ok":true}')

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
    global _ctrl_sock, _ctrl_addr, _HTML_BYTES

    parser = argparse.ArgumentParser(
        description="webview -- 2-D web viewer for ball_detection_d455.py")
    parser.add_argument("--port",       type=int, default=5002,
                        help="HTTP port for the web viewer (default 5002)")
    parser.add_argument("--udp-port",   type=int, default=5567,
                        help="UDP port to receive state from ball_detection (default 5567)")
    parser.add_argument("--ctrl-port",  type=int, default=5566,
                        help="UDP port to send rest control to ball_detection (default 5566)")
    parser.add_argument("--mjpeg-port", type=int, default=5568,
                        help="MJPEG port of ball_detection's built-in server (default 5568)")
    args = parser.parse_args()

    # Inject the MJPEG port into the HTML so the browser knows where to connect
    _HTML_BYTES = _HTML.replace("__MJPEG_PORT__", str(args.mjpeg_port)).encode("utf-8")

    _ctrl_addr = ("127.0.0.1", args.ctrl_port)
    _ctrl_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    threading.Thread(
        target=_udp_receiver, args=(args.udp_port,), daemon=True).start()

    print(f"[webview] Open  http://localhost:{args.port}  in your browser")
    print(f"[webview] MJPEG from ball_detection on port {args.mjpeg_port}")
    server = _ThreadedServer(("0.0.0.0", args.port), _Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[webview] Done.")


if __name__ == "__main__":
    main()
