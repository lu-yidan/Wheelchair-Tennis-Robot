"""
calibrate_web.py — interactive browser UI for rest-coefficient calibration.

Usage:
    python calibrate_web.py traj.json
    # opens http://localhost:5010 in browser automatically

    python calibrate_web.py traj.json --port 5011   # custom port
    python calibrate_web.py traj.json --no-browser  # don't auto-open

Workflow:
    1. Run ball_detection_d455.py --save-traj traj.json, bounce ball ≥5 times
    2. Open calibrate_web.py — see Z(t) chart with numbered bounce markers
    3. Drag on chart to select clean bounce segments (cumulative)
       OR click individual checkboxes to toggle bounces
    4. Click "Run Optimization" — waits ~15–40 s
    5. Copy the fitted rest_x/y/z into config/d455.yaml
"""

import argparse
import json
import os
import sys
import threading
import socketserver
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np

# ── Import shared physics/data functions from calibrate_rest.py ───────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from calibrate_rest import _drag_k, _extract_bounces, _objective, _simulate, _load

# ── Global state (set once in main, read-only after) ─────────────────────────
_frames  = []
_events  = []   # list of bounce event dicts from _extract_bounces()
_meta    = {}

_job_lock = threading.Lock()
_job = {"status": "idle", "result": None}

# ══════════════════════════════════════════════════════════════════════════════
#  HTML page
# ══════════════════════════════════════════════════════════════════════════════
_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Rest Calibration</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
* { margin:0; padding:0; box-sizing:border-box; }
html, body { background:#0d0d0d; color:#ddd; font-family:monospace;
             min-height:100vh; }
body { padding:18px; max-width:1120px; margin:0 auto; }

h1  { font-size:15px; color:#aef; margin-bottom:5px; }
.info { font-size:11px; color:#555; margin-bottom:14px; }
.info b { color:#999; }

/* ── overview chart ── */
#chart { width:100%; height:300px; background:#111; border-radius:6px;
         border:1px solid #1e1e1e; margin-bottom:8px; }
.chart-hint { font-size:11px; color:#444; margin-bottom:10px; }

/* ── buttons ── */
.btn { padding:5px 14px; border-radius:4px; border:1px solid #444;
       background:#1e1e1e; color:#bbb; font-family:monospace; font-size:12px;
       cursor:pointer; }
.btn:hover:not(:disabled)  { background:#2a2a2a; border-color:#666; }
.btn:disabled { opacity:0.35; cursor:default; }
.btn-go  { background:#1a3a6a; border-color:#4af; color:#fff; }
.btn-go:hover:not(:disabled) { background:#1e4a80; }

/* ── controls row ── */
.ctl-row { display:flex; align-items:center; gap:8px; flex-wrap:wrap;
           margin-bottom:10px; }

/* ── bounce checkbox grid ── */
#cb-grid { display:flex; flex-wrap:wrap; gap:5px;
           padding:8px; min-height:40px;
           background:#111; border:1px solid #1e1e1e; border-radius:6px;
           margin-bottom:12px; }
.cbl { display:flex; align-items:center; gap:4px;
       padding:3px 8px; border-radius:4px;
       background:#1a1a1a; border:1px solid #2a2a2a;
       cursor:pointer; font-size:11px; color:#888; user-select:none; }
.cbl input { accent-color:#4af; cursor:pointer; }
.cbl.on { border-color:#2a4a2a; color:#3d3; background:#0d1a0d; }

/* ── optimize bar ── */
.opt-row { display:flex; align-items:center; gap:12px; margin-bottom:20px; }
#sel-cnt { font-size:12px; color:#666; }
#spinner { display:none; font-size:11px; color:#888; }
#err-msg { display:none; font-size:11px; color:#e44; }

/* ── results ── */
#results { display:none; border-top:1px solid #1e1e1e; padding-top:18px; }

.res-card { background:#0e1a0e; border:1px solid #1e3a1e; border-radius:6px;
            padding:14px 18px; margin-bottom:16px; }
.rest-row { display:flex; gap:28px; margin-bottom:10px; flex-wrap:wrap; }
.rest-item .lbl { font-size:10px; color:#555; margin-bottom:2px; }
.rest-item .val { font-size:26px; color:#3e3; font-weight:bold; }
.rmse-item .val { font-size:20px; color:#fa3; }
.rmse-item .sub { font-size:10px; color:#555; }

.yaml-box { font-size:12px; color:#afa; background:#0a140a;
            border:1px solid #1a3a1a; border-radius:4px;
            padding:8px 12px; white-space:pre;
            cursor:pointer; margin-top:10px; }
.yaml-box:active { background:#0d1d0d; }
.copy-hint { font-size:10px; color:#555; margin-top:3px; }

.fits-title { font-size:11px; color:#555; margin-bottom:8px; }
#fit-wrap { display:flex; flex-wrap:wrap; gap:10px; }
.fit-div { width:240px; height:180px; }
</style>
</head>
<body>

<h1>Ball Bounce Rest Calibration</h1>
<div class="info" id="info-bar">Loading…</div>

<!-- ── Z(t) overview chart ── -->
<div id="chart"></div>
<div class="chart-hint">
  ← drag horizontally on chart to select bounces in a time range
  (each drag adds to selection)
</div>

<!-- ── Selection controls ── -->
<div class="ctl-row">
  <button class="btn" onclick="selAll()">Select All</button>
  <button class="btn" onclick="clearSel()">Clear</button>
</div>

<div id="cb-grid"></div>

<!-- ── Run button ── -->
<div class="opt-row">
  <span id="sel-cnt">0 bounces selected</span>
  <button class="btn btn-go" id="btn-opt" disabled onclick="runOpt()">
    Run Optimization
  </button>
  <span id="spinner">⏳ optimising (15–40 s)…</span>
  <span id="err-msg"></span>
</div>

<!-- ── Results (hidden until done) ── -->
<div id="results">

  <div class="res-card">
    <div class="rest-row">
      <div class="rest-item"><div class="lbl">rest_x</div><div class="val" id="r_x">—</div></div>
      <div class="rest-item"><div class="lbl">rest_y</div><div class="val" id="r_y">—</div></div>
      <div class="rest-item"><div class="lbl">rest_z</div><div class="val" id="r_z">—</div></div>
      <div class="rest-item rmse-item" style="margin-left:auto">
        <div class="lbl">RMSE</div>
        <div class="val" id="r_rmse">—</div>
        <div class="sub">post-bounce pos. error</div>
        <div class="sub" id="r_nb"></div>
      </div>
    </div>
    <div class="yaml-box" id="yaml-out" onclick="copyYAML()">—</div>
    <div class="copy-hint">↑ click to copy</div>
  </div>

  <div class="fits-title">Per-bounce Z fit — actual (blue) vs predicted (orange dashed)</div>
  <div id="fit-wrap"></div>

</div>

<script>
// ── State ──────────────────────────────────────────────────────────────────
var _bounces = [];
var _selected = new Set();
var _chartOk  = false;
var _pollId   = null;

// ── Load data on page open ─────────────────────────────────────────────────
fetch('/data').then(function(r){ return r.json(); }).then(function(d) {
  var m = d.meta;
  document.getElementById('info-bar').innerHTML =
    '<b>' + m.file + '</b> &nbsp;|&nbsp; ' +
    m.n_frames + ' frames &nbsp;|&nbsp; ' +
    m.duration  + ' s &nbsp;|&nbsp; ' +
    m.n_bounces + ' bounces detected';

  _bounces = d.bounces;
  _initChart(d.frames, d.bounces);
  _renderCBs();
  _updateCnt();
});

// ── Chart ──────────────────────────────────────────────────────────────────
function _shapes() {
  return _bounces.map(function(b, i) {
    var on = _selected.has(i);
    return {
      type:'line', x0:b.t_rel, x1:b.t_rel, y0:0, y1:1, yref:'paper',
      line:{ color: on ? '#3e3' : '#444', width: on ? 1.5 : 1,
             dash:  on ? 'solid' : 'dot' }
    };
  });
}

function _annots() {
  return _bounces.map(function(b, i) {
    var on = _selected.has(i);
    return {
      x: b.t_rel, y: 1.05, yref:'paper',
      text: b.label, showarrow: false,
      xanchor:'center',
      font:{ size:9, color: on ? '#3e3' : '#555' }
    };
  });
}

function _initChart(frames, bounces) {
  var trace = {
    x: frames.map(function(f){ return f.t_rel; }),
    y: frames.map(function(f){ return f.z; }),
    mode:'lines', type:'scatter',
    line:{ color:'#4af', width:1.5 },
    name:'Z (m)',
    hovertemplate:'t=%{x:.2f}s  Z=%{y:.3f}m<extra></extra>'
  };

  var layout = {
    paper_bgcolor:'#111', plot_bgcolor:'#111',
    margin:{ t:30, b:36, l:48, r:10 },
    dragmode:'select', selectdirection:'h',
    xaxis:{
      title:{ text:'time (s)', font:{color:'#555',size:10} },
      color:'#555', gridcolor:'#1a1a1a', zerolinecolor:'#2a2a2a',
      dtick:2,
      minor:{ dtick:0.5, showgrid:true, gridcolor:'#151515' },
      tickfont:{ size:9 }
    },
    yaxis:{
      title:{ text:'Z (m)', font:{color:'#555',size:10} },
      color:'#555', gridcolor:'#1a1a1a', zerolinecolor:'#3a3a3a',
      tickfont:{ size:9 }
    },
    shapes: _shapes(),
    annotations: _annots(),
    font:{ color:'#888', size:10 },
    showlegend: false
  };

  Plotly.newPlot('chart', [trace], layout,
    { displaylogo:false,
      modeBarButtonsToRemove:['toImage','select2d','lasso2d'] });
  _chartOk = true;

  // Drag-to-select: add bounces in range to selection
  document.getElementById('chart').on('plotly_selected', function(ev) {
    if (!ev || !ev.range) return;
    var t0 = ev.range.x[0], t1 = ev.range.x[1];
    _bounces.forEach(function(b, i) {
      if (b.t_rel >= t0 && b.t_rel <= t1) _selected.add(i);
    });
    _refresh();
  });
}

function _refresh() {
  if (_chartOk)
    Plotly.relayout('chart', { shapes:_shapes(), annotations:_annots() });
  _renderCBs();
  _updateCnt();
}

// ── Checkboxes ─────────────────────────────────────────────────────────────
function _renderCBs() {
  var g = document.getElementById('cb-grid');
  g.innerHTML = '';
  _bounces.forEach(function(b, i) {
    var on  = _selected.has(i);
    var lbl = document.createElement('label');
    lbl.className = 'cbl' + (on ? ' on' : '');
    lbl.innerHTML =
      '<input type="checkbox"' + (on ? ' checked' : '') + '>' +
      b.label +
      '<span style="color:#444;margin-left:3px">' + b.t_rel.toFixed(1) + 's</span>';
    lbl.querySelector('input').addEventListener('change', function(e) {
      if (e.target.checked) _selected.add(i); else _selected.delete(i);
      lbl.className = 'cbl' + (_selected.has(i) ? ' on' : '');
      if (_chartOk)
        Plotly.relayout('chart', { shapes:_shapes(), annotations:_annots() });
      _updateCnt();
    });
    g.appendChild(lbl);
  });
}

function _updateCnt() {
  var n = _selected.size, tot = _bounces.length;
  document.getElementById('sel-cnt').textContent =
    n + ' / ' + tot + ' bounces selected';
  document.getElementById('btn-opt').disabled = (n < 1);
}

function selAll()   { _bounces.forEach(function(_,i){ _selected.add(i); }); _refresh(); }
function clearSel() { _selected.clear(); _refresh(); }

// ── Optimization ───────────────────────────────────────────────────────────
function runOpt() {
  document.getElementById('btn-opt').disabled  = true;
  document.getElementById('spinner').style.display = 'inline';
  document.getElementById('err-msg').style.display = 'none';
  document.getElementById('results').style.display = 'none';

  fetch('/optimize', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({ selected: Array.from(_selected) })
  }).then(function() {
    _pollId = setInterval(_poll, 1200);
  }).catch(function(e) { _showErr('Request failed: ' + e); });
}

function _poll() {
  fetch('/result').then(function(r){ return r.json(); }).then(function(job) {
    if (job.status === 'done') {
      clearInterval(_pollId);
      document.getElementById('spinner').style.display = 'none';
      document.getElementById('btn-opt').disabled = false;
      _showResults(job.result);
    } else if (job.status === 'error') {
      clearInterval(_pollId);
      _showErr((job.result && job.result.error) ? job.result.error : 'unknown error');
    }
  });
}

function _showErr(msg) {
  document.getElementById('spinner').style.display = 'none';
  document.getElementById('btn-opt').disabled = false;
  var el = document.getElementById('err-msg');
  el.textContent = '✗ ' + msg;
  el.style.display = 'inline';
}

// ── Results ────────────────────────────────────────────────────────────────
function _showResults(r) {
  document.getElementById('r_x').textContent    = r.rest_x.toFixed(4);
  document.getElementById('r_y').textContent    = r.rest_y.toFixed(4);
  document.getElementById('r_z').textContent    = r.rest_z.toFixed(4);
  document.getElementById('r_rmse').textContent = r.rmse_cm.toFixed(1) + ' cm';
  document.getElementById('r_nb').textContent   = r.n_bounces + ' bounces';

  var yaml =
    'rest_x: ' + r.rest_x.toFixed(2) + '\\n' +
    'rest_y: ' + r.rest_y.toFixed(2) + '\\n' +
    'rest_z: ' + r.rest_z.toFixed(2);
  document.getElementById('yaml-out').textContent = yaml;

  // Per-bounce comparison subplots
  var wrap = document.getElementById('fit-wrap');
  wrap.innerHTML = '';
  r.fits.forEach(function(fit) {
    var div = document.createElement('div');
    div.className = 'fit-div';
    wrap.appendChild(div);
    Plotly.newPlot(div,
      [
        { x: fit.times_post, y: fit.actual_z,
          mode:'lines+markers', marker:{size:3},
          line:{color:'#4af', width:1.5}, name:'actual' },
        { x: fit.times_post, y: fit.predicted_z,
          mode:'lines', line:{color:'#fa3', width:1.5, dash:'dash'},
          name:'predicted' }
      ],
      {
        title:{ text: fit.label, font:{color:'#888',size:11}, y:0.97 },
        paper_bgcolor:'#111', plot_bgcolor:'#111',
        margin:{t:26, b:28, l:36, r:6},
        xaxis:{ title:{text:'t (s)',font:{size:8}}, color:'#555',
                gridcolor:'#1a1a1a', tickfont:{size:8} },
        yaxis:{ title:{text:'Z (m)',font:{size:8}}, color:'#555',
                gridcolor:'#1a1a1a', tickfont:{size:8} },
        legend:{ font:{size:8,color:'#666'}, x:0, y:1.15,
                 orientation:'h', bgcolor:'rgba(0,0,0,0)' },
        font:{ color:'#888' }
      },
      { displayModeBar:false, responsive:true }
    );
  });

  document.getElementById('results').style.display = 'block';
  document.getElementById('results').scrollIntoView({ behavior:'smooth' });
}

function copyYAML() {
  var txt = document.getElementById('yaml-out').textContent;
  navigator.clipboard.writeText(txt).then(function() {
    var box = document.getElementById('yaml-out');
    box.style.background = '#0d1d0d';
    setTimeout(function(){ box.style.background = ''; }, 500);
  });
}
</script>
</body>
</html>
"""

# ══════════════════════════════════════════════════════════════════════════════
#  HTTP handler
# ══════════════════════════════════════════════════════════════════════════════

class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *_): pass

    def _send(self, code, ct, body):
        if isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ct)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj):
        self._send(200, "application/json", json.dumps(obj))

    def do_GET(self):
        path = self.path.split("?")[0]

        if path == "/":
            self._send(200, "text/html; charset=utf-8", _HTML)

        elif path == "/data":
            t0 = _frames[0]["t"] if _frames else 0.0
            self._json({
                "frames": [
                    {"t_rel": round(f["t"] - t0, 3),
                     "z":     round(f["pos"][2], 4)}
                    for f in _frames
                ],
                "bounces": [
                    {"idx":   i,
                     "t_rel": round(ev["t_bounce"] - t0, 3),
                     "label": f"#{i + 1}"}
                    for i, ev in enumerate(_events)
                ],
                "meta": {
                    "file":      _meta.get("file", ""),
                    "n_frames":  len(_frames),
                    "n_bounces": len(_events),
                    "coeff_drag": _meta.get("coeff_drag", 0.47),
                    "duration":  round(_frames[-1]["t"] - t0, 1) if _frames else 0,
                },
            })

        elif path == "/result":
            with _job_lock:
                self._json(dict(_job))

        else:
            self._send(404, "text/plain", b"not found")

    def do_POST(self):
        if self.path != "/optimize":
            self._send(404, "text/plain", b"not found")
            return
        try:
            length  = int(self.headers.get("Content-Length", 0))
            body    = json.loads(self.rfile.read(length))
            indices = [int(i) for i in body.get("selected", [])]
        except Exception as e:
            self._send(400, "application/json",
                       json.dumps({"error": str(e)}).encode())
            return

        with _job_lock:
            _job["status"] = "running"
            _job["result"] = None

        threading.Thread(target=_run_opt, args=(indices,), daemon=True).start()
        self._json({"status": "running"})


# ══════════════════════════════════════════════════════════════════════════════
#  Optimisation worker (background thread)
# ══════════════════════════════════════════════════════════════════════════════

def _run_opt(indices):
    try:
        from scipy.optimize import differential_evolution

        dk     = _drag_k(float(_meta.get("coeff_drag", 0.47)))
        events = [_events[i] for i in sorted(set(indices))
                  if 0 <= i < len(_events)]

        if not events:
            with _job_lock:
                _job.update({"status": "error",
                             "result": {"error": "No bounce events selected"}})
            return

        res = differential_evolution(
            _objective, [(0.05, 1.50)] * 3,
            args=(events, dk),
            seed=42, tol=1e-5, maxiter=1000,
            popsize=15, mutation=(0.5, 1.2), recombination=0.8,
            disp=False,
        )

        rx, ry, rz = res.x
        rmse_cm = float(np.sqrt(res.fun)) * 100.0

        fits = []
        for idx, ev in zip(sorted(set(indices)), events):
            s0 = np.array([
                ev["pre_pos"][0], ev["pre_pos"][1], 0.0,
                ev["pre_vel"][0] * rx,
                ev["pre_vel"][1] * ry,
                abs(ev["pre_vel"][2]) * rz,
            ])
            preds = _simulate(s0, ev["times_post"], dk)
            fits.append({
                "label":       f"#{idx + 1}",
                "times_post":  [round(t, 3) for t in ev["times_post"]],
                "actual_z":    [round(float(p[2]), 4) for p in ev["pos_post"]],
                "predicted_z": [round(float(p[2]), 4) for p in preds],
            })

        with _job_lock:
            _job.update({
                "status": "done",
                "result": {
                    "rest_x":    round(float(rx), 4),
                    "rest_y":    round(float(ry), 4),
                    "rest_z":    round(float(rz), 4),
                    "rmse_cm":   round(rmse_cm, 2),
                    "n_bounces": len(events),
                    "fits":      fits,
                },
            })

    except Exception as e:
        with _job_lock:
            _job.update({"status": "error",
                         "result": {"error": str(e)}})


# ══════════════════════════════════════════════════════════════════════════════
#  Server + entry point
# ══════════════════════════════════════════════════════════════════════════════

class _Server(socketserver.ThreadingMixIn, HTTPServer):
    daemon_threads = True


def main():
    parser = argparse.ArgumentParser(
        description="Interactive browser UI for rest-coefficient calibration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    parser.add_argument("traj_json",
                        help="JSON file from  ball_detection_d455.py --save-traj")
    parser.add_argument("--port",       type=int, default=5010,
                        help="HTTP port (default 5010)")
    parser.add_argument("--no-browser", action="store_true",
                        help="do not auto-open browser")
    args = parser.parse_args()

    global _frames, _events, _meta

    meta, frames = _load(args.traj_json, None)
    if not frames:
        print("[calibrate_web] ERROR: no frames in file — check path")
        sys.exit(1)

    _meta   = dict(meta)
    _meta["file"] = os.path.basename(args.traj_json)
    _frames = frames
    _events = _extract_bounces(frames)

    n_b = len(_events)
    print(f"[calibrate_web] {args.traj_json}")
    print(f"                {len(frames)} frames, "
          f"{frames[-1]['t'] - frames[0]['t']:.1f} s, "
          f"{n_b} bounce events detected")
    if n_b == 0:
        print("[calibrate_web] WARNING: no bounces detected — "
              "ensure AprilTag is visible and ball bounces clearly")
    print(f"[calibrate_web] Open  http://localhost:{args.port}")

    if not args.no_browser:
        threading.Timer(0.6, lambda: webbrowser.open(
            f"http://localhost:{args.port}")).start()

    server = _Server(("0.0.0.0", args.port), _Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[calibrate_web] Done.")


if __name__ == "__main__":
    main()
