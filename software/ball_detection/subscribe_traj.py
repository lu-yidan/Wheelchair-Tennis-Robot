#!/usr/bin/env python3
"""
subscribe_traj.py — 轨迹订阅最小示例

用法
----
  python subscribe_traj.py                        # 连本机
  python subscribe_traj.py --host 192.168.1.100   # 连局域网另一台机器

消息字段说明
-----------
  stamp    : float  发布时的 Unix 时间戳（秒）
  t_obs    : float  EKF 最后一次收到相机实测的时刻
  detected : bool   True  = 有新相机测量（200ms内）
                    False = EKF 惯性预测，无新测量
  pos      : [x,y,z]      当前球位置（米，AprilTag 坐标系，Z向上）
  vel      : [vx,vy,vz]   当前球速度（米/秒）
  traj     : [[x,y,z,t],...]  预测轨迹；t = 距 stamp 的秒数，约 200 点覆盖 2 秒
"""

import argparse
import json
import sys
import time

try:
    import zmq
except ImportError:
    print("请先安装: pip install pyzmq", file=sys.stderr)
    sys.exit(1)


def main():
    ap = argparse.ArgumentParser(description="轨迹订阅最小示例")
    ap.add_argument("--host", default="127.0.0.1", help="fusion 的 IP（默认 127.0.0.1）")
    ap.add_argument("--port", type=int, default=5580, help="ZMQ PUB 端口（默认 5580）")
    ap.add_argument("--idle-sec", type=float, default=5.0,
                    help="无消息时每隔多少秒打印一次等待提示（默认 5.0）")
    args = ap.parse_args()

    addr = f"tcp://{args.host}:{args.port}"
    print(f"连接 {addr} … （Ctrl+C 退出）\n")

    ctx = zmq.Context()
    sock = ctx.socket(zmq.SUB)
    sock.setsockopt(zmq.RCVHWM, 2)          # 只保留最新帧，不积压
    sock.setsockopt(zmq.RCVTIMEO, 200)       # recv 最多阻塞 200ms，然后检查 idle 定时器
    sock.connect(addr)
    sock.setsockopt_string(zmq.SUBSCRIBE, "")

    last_msg_t = time.time()   # 上次收到消息的时刻
    last_idle_print_t = 0.0    # 上次打印等待提示的时刻

    try:
        while True:
            # ── 接收消息（非阻塞超时，保证可以定期打印等待提示）────────────
            try:
                raw = sock.recv()
            except zmq.Again:
                # 200ms 内没有新消息，检查是否该打印等待提示
                now = time.time()
                if now - last_msg_t >= args.idle_sec and \
                   now - last_idle_print_t >= args.idle_sec:
                    last_idle_print_t = now
                    print(f"[{_ts()}] 等待球… （已 {now - last_msg_t:.0f}s 无消息）")
                continue

            now = time.time()
            last_msg_t = now

            # ── 解析 ─────────────────────────────────────────────────────────
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue

            # ── 使用数据（在这里写你的机器人逻辑）────────────────────────────
            handle(msg)

    except KeyboardInterrupt:
        print("\n退出。")
    finally:
        sock.close()
        ctx.term()


def handle(msg):
    """
    处理一条轨迹消息。
    把你的机器人逻辑写在这里，或者 import 后在外部调用。
    """
    detected = msg["detected"]
    pos      = msg["pos"]          # [x, y, z]  米
    vel      = msg["vel"]          # [vx, vy, vz]  米/秒
    traj     = msg["traj"]         # [[x,y,z,t], ...]  预测轨迹
    latency  = (time.time() - msg["stamp"]) * 1000

    status = "BALL" if detected else "COAST"   # COAST = 惯性预测，无实测

    # 只打印第一个和最后一个预测点，避免刷屏
    t_first = traj[0]  if traj else None
    t_last  = traj[-1] if traj else None

    print(
        f"[{_ts()}] {status:5s}  "
        f"pos=({pos[0]:+.2f},{pos[1]:+.2f},{pos[2]:+.2f})m  "
        f"vel=({vel[0]:+.1f},{vel[1]:+.1f},{vel[2]:+.1f})m/s  "
        f"traj={len(traj)}pts  "
        f"end=({t_last[0]:+.2f},{t_last[1]:+.2f},{t_last[2]:+.2f})@{t_last[3]:.2f}s  "
        f"lat={latency:.1f}ms"
    )


def _ts():
    """当前时间字符串，精确到毫秒。"""
    t = time.time()
    ms = int((t % 1) * 1000)
    return time.strftime("%H:%M:%S", time.localtime(t)) + f".{ms:03d}"


if __name__ == "__main__":
    main()
