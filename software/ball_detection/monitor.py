"""
monitor.py — HTTP + SSE relay so the browser can see fusion.py's live state.

Replaces `python -m http.server 8080` in run_fusion.sh.  Does two things:

  1.  Static file server for monitor.html + any siblings.
  2.  Listens for fusion.py's --webview-port UDP state packets and
      pushes them to subscribed browsers via Server-Sent Events at /events.

UDP wire format (matches fusion.py's webview forwarding):
    b'\\x00' + json_bytes        first byte is a webview-convention type tag
    or plain json_bytes          (we accept both)

JSON shape (one packet per ~100 ms, while EKF is alive):
    {"t": ..., "FUSED": {"ball": [x,y,z], "vel": [vx,vy,vz],
                          "traj": [[x,y,z],...], "bounces":[[x,y,z],...]} }

Usage:
    python monitor.py --http-port 8080 --udp-port 5571
    # then start fusion forwarding state to udp-port:
    python fusion.py --port 5570 --webview-port 5571
"""

import argparse
import http.server
import json
import os
import queue
import socket
import socketserver
import sys
import threading
import time


class _PubSub:
    """Single-publisher, N-subscriber message broker.  Drops on slow sub."""

    def __init__(self):
        self.subs = []                # list[queue.Queue]
        self.last = None              # last published bytes (sent to new subs)
        self.lock = threading.Lock()

    def publish(self, data: bytes):
        with self.lock:
            self.last = data
            for q in list(self.subs):
                try:
                    q.put_nowait(data)
                except queue.Full:
                    pass             # slow subscriber → drop frame

    def subscribe(self) -> "queue.Queue":
        q = queue.Queue(maxsize=32)
        with self.lock:
            if self.last is not None:
                q.put_nowait(self.last)
            self.subs.append(q)
        return q

    def unsubscribe(self, q):
        with self.lock:
            try:
                self.subs.remove(q)
            except ValueError:
                pass


STATE = _PubSub()


def _udp_listener(port: int, stop: threading.Event):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("0.0.0.0", port))
    sock.settimeout(0.5)
    print(f"[monitor] UDP listening on 0.0.0.0:{port} (forward fusion --webview-port here)")
    while not stop.is_set():
        try:
            data, _ = sock.recvfrom(65535)
        except socket.timeout:
            continue
        except OSError:
            break
        # webview convention: 1-byte type prefix (0x00 = state).  Tolerate either.
        if data and data[0:1] == b"\x00":
            data = data[1:]
        try:
            json.loads(data)
        except Exception:
            continue                # bad payload, ignore
        STATE.publish(data)


class _Handler(http.server.SimpleHTTPRequestHandler):
    # Quiet logging (uncomment for debug)
    def log_message(self, *_): pass

    def do_GET(self):                                                # noqa: N802
        if self.path.split("?")[0] == "/events":
            self._serve_sse()
        else:
            super().do_GET()

    def _serve_sse(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        q = STATE.subscribe()
        try:
            while True:
                try:
                    data = q.get(timeout=15.0)
                    self.wfile.write(b"data: ")
                    self.wfile.write(data)
                    self.wfile.write(b"\n\n")
                    self.wfile.flush()
                except queue.Empty:
                    # heartbeat — keeps proxies/browsers from closing idle conns
                    self.wfile.write(b": ping\n\n")
                    self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            STATE.unsubscribe(q)


class _ThreadedHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--http-port", type=int, default=8080,
                    help="static + SSE HTTP port (default 8080)")
    ap.add_argument("--udp-port", type=int, default=5571,
                    help="UDP port fusion.py forwards state to (default 5571)")
    ap.add_argument("--dir", default=None,
                    help="directory to serve (default = this script's dir)")
    args = ap.parse_args()

    serve_dir = args.dir or os.path.dirname(os.path.abspath(__file__))
    os.chdir(serve_dir)
    print(f"[monitor] serving directory: {serve_dir}")

    stop = threading.Event()
    threading.Thread(target=_udp_listener, args=(args.udp_port, stop), daemon=True).start()

    srv = _ThreadedHTTPServer(("0.0.0.0", args.http_port), _Handler)
    url = f"http://localhost:{args.http_port}/monitor.html"
    print(f"[monitor] HTTP serving on {url}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[monitor] stopping.")
    finally:
        stop.set()
        srv.server_close()


if __name__ == "__main__":
    main()
