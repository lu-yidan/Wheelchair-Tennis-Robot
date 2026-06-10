#!/usr/bin/env bash
# run_dual_d455.sh — 启动双 D455 + fusion.py + monitor 网页
#
# 用法:
#   ./run_dual_d455.sh
#   FUSION_PORT=5570 ./run_dual_d455.sh
#
# 打开:
#   http://localhost:8080/monitor.html  ← 融合后球的位置 + 预测弹道
#
# 日志:
#   tail -f /tmp/fusion-logs/fusion.log
#   tail -f /tmp/fusion-logs/d455_a.log
#   tail -f /tmp/fusion-logs/d455_b.log
#
# 关闭:
#   Ctrl+C

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── conda env ───────────────────────────────────────────────────────────────
CONDA_SH=""
for _root in "${CONDA_PREFIX_1:-}" "${CONDA_PREFIX:-}" \
             "$HOME/miniconda3" "$HOME/anaconda3" \
             "/opt/miniconda3" "/opt/anaconda3" "/opt/conda"; do
    [ -n "$_root" ] && [ -f "$_root/etc/profile.d/conda.sh" ] && \
        { CONDA_SH="$_root/etc/profile.d/conda.sh"; break; }
done
if [ -z "$CONDA_SH" ]; then
    echo "[ERROR] cannot find conda.sh" >&2
    exit 1
fi
source "$CONDA_SH"
conda activate catchball || { echo "[ERROR] conda activate catchball failed" >&2; exit 1; }
echo "[run_dual_d455] env: $CONDA_DEFAULT_ENV  ($(which python))"

FUSION_PORT="${FUSION_PORT:-5570}"
MONITOR_PORT="${MONITOR_PORT:-8080}"
WEBVIEW_PORT="${WEBVIEW_PORT:-5571}"
TRAJ_PUB_PORT="${TRAJ_PUB_PORT:-5580}"
LOGS=/tmp/fusion-logs
mkdir -p "$LOGS"

PIDS=()
_CLEANED=0

cleanup() {
    [ "$_CLEANED" = 1 ] && return
    _CLEANED=1
    echo
    echo "[run_dual_d455] shutting down..."
    for pid in "${PIDS[@]:-}"; do
        kill "$pid" 2>/dev/null
    done
    pkill -P $$ 2>/dev/null
    sleep 0.5
    echo "[run_dual_d455] done."
}
trap cleanup INT TERM EXIT

pkill -f "ball_detection.py" 2>/dev/null
pkill -f "fusion.py"         2>/dev/null
sleep 0.5

echo "[run_dual_d455] fusion port=$FUSION_PORT  monitor port=$MONITOR_PORT"

# ── 1. fusion ───────────────────────────────────────────────────────────────
python fusion.py --port "$FUSION_PORT" --webview-port "$WEBVIEW_PORT" \
    --traj-pub-port "$TRAJ_PUB_PORT" \
    > "$LOGS/fusion.log" 2>&1 &
PIDS+=($!)
sleep 1

# ── 2. D455 A (SN 260722302887, 端口基准: 5565-5568) ───────────────────────
python ball_detection.py --config config/d455_a.yaml \
    > "$LOGS/d455_a.log" 2>&1 &
PIDS+=($!)
sleep 2

# ── 3. D455 B (SN 152522251463, 端口偏移 +100: 5665-5668) ──────────────────
python ball_detection.py --config config/d455_b.yaml \
    > "$LOGS/d455_b.log" 2>&1 &
PIDS+=($!)
sleep 2

# ── 4. monitor (HTTP + SSE) ─────────────────────────────────────────────────
python monitor.py --http-port "$MONITOR_PORT" --udp-port "$WEBVIEW_PORT" \
    --dir "$SCRIPT_DIR" > "$LOGS/monitor.log" 2>&1 &
PIDS+=($!)
sleep 1

cat <<EOF

═══════════════════════════════════════════════════════════════════════
  网页:      http://localhost:${MONITOR_PORT}/monitor.html
  D455 A:   SN 260722302887  MJPEG → http://localhost:5568/main
  D455 B:   SN 152522251463  MJPEG → http://localhost:5668/main
  轨迹发布:  tcp://<本机IP>:${TRAJ_PUB_PORT}  (ZMQ SUB, 30 Hz, 2s预测)
  融合日志:  tail -f $LOGS/fusion.log
  按 Ctrl+C 停止全部
═══════════════════════════════════════════════════════════════════════

EOF

exec tail -f "$LOGS/fusion.log"
