#!/usr/bin/env bash
# run_fusion.sh — 启动 fusion.py + AONI + Razer + monitor 网页
#
# 用法:
#   ./run_fusion.sh                    # 默认 (AONI + Razer)
#   FUSION_PORT=5570 ./run_fusion.sh   # 覆盖 fusion 端口
#
# 打开:
#   http://localhost:8080/monitor.html  ← 两路相机画面
#
# 实时融合数字:
#   tail -f /tmp/fusion-logs/fusion.log
#
# 关闭:
#   Ctrl+C  (脚本会清理所有子进程)

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── conda env ───────────────────────────────────────────────────────────────
if ! command -v conda >/dev/null 2>&1; then
    if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
        source "$HOME/miniconda3/etc/profile.d/conda.sh"
    elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
        source "$HOME/anaconda3/etc/profile.d/conda.sh"
    fi
fi
conda activate catchball 2>/dev/null || {
    echo "[ERROR] cannot activate conda env 'catchball'" >&2
    exit 1
}

FUSION_PORT="${FUSION_PORT:-5570}"
MONITOR_PORT="${MONITOR_PORT:-8080}"
LOGS=/tmp/fusion-logs
mkdir -p "$LOGS"

PIDS=()

cleanup() {
    echo
    echo "[run_fusion] shutting down..."
    for pid in "${PIDS[@]:-}"; do
        kill "$pid" 2>/dev/null
    done
    # belt-and-suspenders: catch any orphan children from python multiprocessing
    pkill -P $$ 2>/dev/null
    sleep 0.5
    echo "[run_fusion] done."
    exit 0
}
trap cleanup INT TERM EXIT

# Pre-clean any stale instances from previous runs
pkill -f "ball_detection.py" 2>/dev/null
pkill -f "fusion.py" 2>/dev/null
sleep 0.5

echo "[run_fusion] fusion port = $FUSION_PORT, monitor port = $MONITOR_PORT"
echo "[run_fusion] logs → $LOGS/"

# ── 1. fusion ───────────────────────────────────────────────────────────────
echo "[run_fusion] starting fusion..."
python fusion.py --port "$FUSION_PORT" > "$LOGS/fusion.log" 2>&1 &
PIDS+=($!)
sleep 1

# ── 2. AONI ────────────────────────────────────────────────────────────────
echo "[run_fusion] starting AONI detector (config/webcam.yaml)..."
python ball_detection.py --config config/webcam.yaml --no-viz \
    > "$LOGS/aoni.log" 2>&1 &
PIDS+=($!)
sleep 2

# ── 3. Razer ───────────────────────────────────────────────────────────────
echo "[run_fusion] starting Razer detector (config/razer.yaml)..."
python ball_detection.py --config config/razer.yaml --no-viz \
    > "$LOGS/razer.log" 2>&1 &
PIDS+=($!)
sleep 2

# ── 4. monitor 页面 ─────────────────────────────────────────────────────────
echo "[run_fusion] serving monitor.html on http://localhost:${MONITOR_PORT}/monitor.html"
python -m http.server "$MONITOR_PORT" --directory "$SCRIPT_DIR" \
    > "$LOGS/http.log" 2>&1 &
PIDS+=($!)
sleep 1

# ── status ─────────────────────────────────────────────────────────────────
cat <<EOF

═══════════════════════════════════════════════════════════════════════
  网页:    http://localhost:${MONITOR_PORT}/monitor.html
  融合状态: tail -f $LOGS/fusion.log
  AONI 日志: tail -f $LOGS/aoni.log
  Razer 日志: tail -f $LOGS/razer.log
  按 Ctrl+C 停止全部
═══════════════════════════════════════════════════════════════════════

EOF

# 实时显示 fusion 输出 (终端始终能看到 FUSED 数字滚动)
exec tail -f "$LOGS/fusion.log"
