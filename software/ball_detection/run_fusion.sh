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
# Always source conda.sh — in non-interactive shells the user's .bashrc-defined
# `conda` shell function isn't inherited, so `conda activate` would fail with
# "shell not properly initialised".
CONDA_SH=""
for _root in "${CONDA_PREFIX_1:-}" "${CONDA_PREFIX:-}" \
             "$HOME/miniconda3" "$HOME/anaconda3" \
             "/opt/miniconda3" "/opt/anaconda3" "/opt/conda"; do
    [ -n "$_root" ] && [ -f "$_root/etc/profile.d/conda.sh" ] && \
        { CONDA_SH="$_root/etc/profile.d/conda.sh"; break; }
done
if [ -z "$CONDA_SH" ]; then
    echo "[ERROR] cannot find conda.sh — set CONDA_PREFIX or install miniconda" >&2
    exit 1
fi
# shellcheck disable=SC1090
source "$CONDA_SH"

if ! conda activate catchball; then
    echo "[ERROR] cannot activate conda env 'catchball'" >&2
    echo "[HINT]  available envs:" >&2
    conda env list 2>&1 | sed -n '/^#/!p' >&2 || true
    exit 1
fi
echo "[run_fusion] using env: $CONDA_DEFAULT_ENV  ($(which python))"

FUSION_PORT="${FUSION_PORT:-5570}"
MONITOR_PORT="${MONITOR_PORT:-8080}"
WEBVIEW_PORT="${WEBVIEW_PORT:-5571}"     # fusion → monitor SSE relay channel
LOGS=/tmp/fusion-logs
mkdir -p "$LOGS"

PIDS=()
_CLEANED=0

cleanup() {
    [ "$_CLEANED" = 1 ] && return
    _CLEANED=1
    echo
    echo "[run_fusion] shutting down..."
    for pid in "${PIDS[@]:-}"; do
        kill "$pid" 2>/dev/null
    done
    # belt-and-suspenders: catch any orphan children from python multiprocessing
    pkill -P $$ 2>/dev/null
    sleep 0.5
    echo "[run_fusion] done."
}
trap cleanup INT TERM EXIT

# Pre-clean any stale instances from previous runs
pkill -f "ball_detection.py" 2>/dev/null
pkill -f "fusion.py" 2>/dev/null
sleep 0.5

echo "[run_fusion] fusion port = $FUSION_PORT, monitor port = $MONITOR_PORT"
echo "[run_fusion] logs → $LOGS/"

# ── 1. fusion (broadcasts state to monitor.py via webview-port) ─────────────
echo "[run_fusion] starting fusion..."
python fusion.py --port "$FUSION_PORT" --webview-port "$WEBVIEW_PORT" \
    > "$LOGS/fusion.log" 2>&1 &
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

# ── 4. monitor.py — HTTP + SSE relay (replaces python -m http.server) ───────
echo "[run_fusion] serving monitor.py on http://localhost:${MONITOR_PORT}/monitor.html"
python monitor.py --http-port "$MONITOR_PORT" --udp-port "$WEBVIEW_PORT" \
    --dir "$SCRIPT_DIR" > "$LOGS/monitor.log" 2>&1 &
PIDS+=($!)
sleep 1

# ── status ─────────────────────────────────────────────────────────────────
cat <<EOF

═══════════════════════════════════════════════════════════════════════
  网页:    http://localhost:${MONITOR_PORT}/monitor.html
             ← 球的位置 + 预测弹道实时显示 (SSE)
  融合数字: tail -f $LOGS/fusion.log
  AONI 日志: tail -f $LOGS/aoni.log
  Razer 日志: tail -f $LOGS/razer.log
  monitor 日志: tail -f $LOGS/monitor.log
  按 Ctrl+C 停止全部
═══════════════════════════════════════════════════════════════════════

EOF

# 实时显示 fusion 输出 (终端始终能看到 FUSED 数字滚动)
exec tail -f "$LOGS/fusion.log"
