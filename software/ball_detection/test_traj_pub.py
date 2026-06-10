#!/usr/bin/env python3
"""
test_traj_pub.py — 轨迹发布器本机自测脚本

用途
----
在 run_dual_d455.sh 启动后，用本脚本验证 fusion.py 的 ZMQ 轨迹发布是否正常工作，
无需同学的机器人参与。脚本会持续接收轨迹消息并打印统计信息。

前提条件
--------
1. conda activate catchball
2. ./run_dual_d455.sh  （另一个终端）
3. python test_traj_pub.py

参数
----
  --host    fusion 所在机器的 IP（默认 127.0.0.1，即本机）
  --port    ZMQ PUB 端口（默认 5580，与 run_dual_d455.sh 一致）
  --timeout 等待第一条消息的超时秒数（默认 10）
  --count   收到多少条后退出，0 = 一直运行（默认 0）

示例
----
  # 本机自测（fusion 和本脚本在同一台机器）
  python test_traj_pub.py

  # 测试局域网连通性（在同学的机器上运行，填本机 IP）
  python test_traj_pub.py --host 192.168.1.100

  # 只收 30 条后退出
  python test_traj_pub.py --count 30

消息格式（fusion.py 发布的 JSON）
---------------------------------
  {
    "stamp":    1749562345.123,   # Unix 时间戳（发布时刻）
    "t_obs":    1749562345.089,   # 最后一次相机测量被 EKF 接收的时刻
    "detected": true,             # false = EKF 在惯性预测，无新测量值
    "pos":  [x, y, z],           # 当前球位置（米，AprilTag 坐标系，Z 向上）
    "vel":  [vx, vy, vz],        # 当前球速度（米/秒）
    "traj": [[x,y,z,t], ...],    # 预测轨迹，t = 距 stamp 的秒数
  }

坐标系说明
----------
  原点 = AprilTag 中心 (0, 0, 0)
  Z+   = 竖直向上
  X+/Y+ = tag 平面内（由 tag 朝向决定）
  单位：米、秒
"""

import argparse
import json
import sys
import time

try:
    import zmq
except ImportError:
    print("[ERROR] pyzmq 未安装，运行: pip install pyzmq", file=sys.stderr)
    sys.exit(1)


# ── 终端颜色（不强依赖，不存在时退化为纯文本）──────────────────────────────
_GREEN  = "\033[92m"
_YELLOW = "\033[93m"
_RED    = "\033[91m"
_CYAN   = "\033[96m"
_RESET  = "\033[0m"
_BOLD   = "\033[1m"


def _color(text, code):
    return f"{code}{text}{_RESET}"


def _fmt_vec(v, unit=""):
    """把 [x, y, z] 格式化为带符号的对齐字符串。"""
    return f"({v[0]:+.3f}, {v[1]:+.3f}, {v[2]:+.3f}){unit}"


def _check_message(msg):
    """
    检查消息字段完整性，返回问题列表（空列表 = 合格）。
    此函数不抛出异常，供循环内调用。
    """
    issues = []
    for key in ("stamp", "t_obs", "detected", "pos", "vel", "traj"):
        if key not in msg:
            issues.append(f"缺少字段: {key}")
    if "pos" in msg and len(msg["pos"]) != 3:
        issues.append(f"pos 长度应为 3，实际 {len(msg['pos'])}")
    if "vel" in msg and len(msg["vel"]) != 3:
        issues.append(f"vel 长度应为 3，实际 {len(msg['vel'])}")
    if "traj" in msg:
        traj = msg["traj"]
        if not isinstance(traj, list) or len(traj) == 0:
            issues.append("traj 为空或非列表")
        elif len(traj[0]) != 4:
            issues.append(f"traj[0] 长度应为 4 (x,y,z,t)，实际 {len(traj[0])}")
    return issues


def main():
    ap = argparse.ArgumentParser(
        description="fusion.py ZMQ 轨迹发布器本机自测",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--host",    default="127.0.0.1",
                    help="fusion 的 IP 地址（默认 127.0.0.1）")
    ap.add_argument("--port",    type=int, default=5580,
                    help="ZMQ PUB 端口（默认 5580）")
    ap.add_argument("--timeout", type=int, default=10,
                    help="等待第一条消息的超时秒数（默认 10）")
    ap.add_argument("--count",   type=int, default=0,
                    help="收到 N 条后退出，0 = 一直运行（默认 0）")
    args = ap.parse_args()

    addr = f"tcp://{args.host}:{args.port}"
    print(f"\n{_color('[ ZMQ 轨迹订阅测试 ]', _BOLD)}")
    print(f"  连接地址 : {_color(addr, _CYAN)}")
    print(f"  等待超时 : {args.timeout}s")
    print(f"  接收数量 : {'无限' if args.count == 0 else args.count}")
    print()

    ctx = zmq.Context()
    sock = ctx.socket(zmq.SUB)

    # 只保留最新 2 条消息，避免测试端积压造成延迟假象
    sock.setsockopt(zmq.RCVHWM, 2)
    sock.setsockopt(zmq.RCVTIMEO, args.timeout * 1000)  # 超时单位：毫秒
    sock.connect(addr)
    sock.setsockopt_string(zmq.SUBSCRIBE, "")  # 订阅所有 topic

    print(f"等待消息（最多 {args.timeout}s）...")

    # ── 统计变量 ─────────────────────────────────────────────────────────────
    n_received    = 0       # 总收到条数
    n_detected    = 0       # detected=True 的条数
    n_coasting    = 0       # detected=False 的条数（EKF 惯性预测）
    n_issues      = 0       # 格式异常条数
    latencies     = []      # 端到端延迟（ms）列表，用于统计
    prev_stamp    = None    # 上一条消息的 stamp，用于计算实际发布频率
    freq_samples  = []      # 相邻消息间隔（秒）

    t_start = time.time()

    try:
        while True:
            # ── 接收消息 ─────────────────────────────────────────────────────
            try:
                raw = sock.recv()
            except zmq.Again:
                elapsed = time.time() - t_start
                print(f"\n{_color('[TIMEOUT]', _RED)} {elapsed:.1f}s 内未收到任何消息。")
                print("  请确认 run_dual_d455.sh 正在运行，且 --traj-pub-port 5580 已传给 fusion.py。")
                print(f"  查看日志: tail -f /tmp/fusion-logs/fusion.log")
                sys.exit(1)

            recv_t = time.time()

            # ── 解析 JSON ────────────────────────────────────────────────────
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError as e:
                print(f"{_color('[WARN]', _YELLOW)} JSON 解析失败: {e}")
                n_issues += 1
                continue

            # ── 字段校验 ─────────────────────────────────────────────────────
            issues = _check_message(msg)
            if issues:
                print(f"{_color('[WARN]', _YELLOW)} 消息格式问题: {'; '.join(issues)}")
                n_issues += 1
                continue

            n_received += 1
            if n_received == 1:
                print(f"{_color('[OK]', _GREEN)} 收到第一条消息！\n")

            # ── 计算延迟与频率 ────────────────────────────────────────────────
            latency_ms = (recv_t - msg["stamp"]) * 1000.0
            latencies.append(latency_ms)

            if prev_stamp is not None:
                interval = msg["stamp"] - prev_stamp
                if 0 < interval < 1.0:   # 过滤异常间隔
                    freq_samples.append(interval)
            prev_stamp = msg["stamp"]

            # ── 统计 detected 状态 ───────────────────────────────────────────
            detected = msg["detected"]
            if detected:
                n_detected += 1
                status_str = _color("DETECTED", _GREEN)
            else:
                n_coasting += 1
                status_str = _color("COASTING", _YELLOW)  # EKF 惯性滑行，无新测量

            # ── 提取轨迹信息 ─────────────────────────────────────────────────
            traj      = msg["traj"]
            traj_len  = len(traj)
            t_horizon = traj[-1][3] if traj else 0.0   # 预测最远时刻（秒）

            # ── 打印当前帧 ───────────────────────────────────────────────────
            obs_age_ms = (msg["stamp"] - msg["t_obs"]) * 1000.0
            print(
                f"#{n_received:4d}  {status_str}  "
                f"pos={_fmt_vec(msg['pos'], 'm')}  "
                f"vel={_fmt_vec(msg['vel'], 'm/s')}  "
                f"traj={traj_len}pts/{t_horizon:.2f}s  "
                f"latency={latency_ms:.1f}ms  "
                f"obs_age={obs_age_ms:.0f}ms"
            )

            # ── 每 30 条打印一次汇总统计 ─────────────────────────────────────
            if n_received % 30 == 0:
                avg_lat  = sum(latencies[-30:]) / 30
                max_lat  = max(latencies[-30:])
                avg_freq = 1.0 / (sum(freq_samples[-29:]) / max(len(freq_samples[-29:]), 1)) \
                           if freq_samples else 0.0
                det_pct  = 100.0 * n_detected / n_received
                print()
                print(_color(f"  ── 汇总（最近 30 条）──────────────────────────────", _CYAN))
                print(f"  平均延迟  : {avg_lat:.2f} ms   最大延迟: {max_lat:.2f} ms")
                print(f"  实际频率  : {avg_freq:.1f} Hz")
                print(f"  detected  : {n_detected}/{n_received} ({det_pct:.0f}%)")
                print(f"  coasting  : {n_coasting}/{n_received}")
                if n_issues:
                    print(f"  {_color(f'格式异常  : {n_issues} 条', _RED)}")
                print()

            # ── 达到指定条数后退出 ───────────────────────────────────────────
            if args.count > 0 and n_received >= args.count:
                break

    except KeyboardInterrupt:
        print("\n\n中断。")

    # ── 最终汇总 ─────────────────────────────────────────────────────────────
    if n_received == 0:
        print(_color("[FAIL] 未收到任何有效消息。", _RED))
        sys.exit(1)

    avg_lat  = sum(latencies) / len(latencies)
    max_lat  = max(latencies)
    min_lat  = min(latencies)
    avg_freq = 1.0 / (sum(freq_samples) / len(freq_samples)) if freq_samples else 0.0
    det_pct  = 100.0 * n_detected / n_received

    print(_color("\n══════════════ 最终统计 ══════════════", _BOLD))
    print(f"  总计接收  : {n_received} 条")
    print(f"  实际频率  : {avg_freq:.1f} Hz  （目标 30 Hz）")
    print(f"  端到端延迟: 均值 {avg_lat:.2f} ms  最小 {min_lat:.2f} ms  最大 {max_lat:.2f} ms")
    print(f"  detected  : {n_detected} 条 ({det_pct:.0f}%)  "
          f"coasting: {n_coasting} 条")
    if n_issues:
        print(f"  {_color(f'格式异常  : {n_issues} 条', _RED)}")
    else:
        print(f"  {_color('格式校验  : 全部通过', _GREEN)}")

    # 延迟判断（局域网 < 5ms 正常）
    if avg_lat < 5.0:
        print(f"  {_color('延迟评估  : 优秀 ✓', _GREEN)}")
    elif avg_lat < 20.0:
        print(f"  {_color('延迟评估  : 正常', _GREEN)}")
    else:
        print(f"  {_color(f'延迟评估  : 偏高，检查网络或 CPU 负载', _YELLOW)}")

    sock.close()
    ctx.term()


if __name__ == "__main__":
    main()
