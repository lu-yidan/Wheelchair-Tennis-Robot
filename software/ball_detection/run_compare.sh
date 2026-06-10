#!/usr/bin/env bash
# run_compare.sh — 双 D455 对比测试：在两台相机上分别跑不同的检测方法组合
#
# 用法:
#   ./run_compare.sh <方法A> <方法B>
#
# 方法名 (预设):
#   hsv-vis     HSV+MOG2,  纯视觉深度  (vis_weight=1.0)
#   hsv-sensor  HSV+MOG2,  纯传感器深度 (vis_weight=0.0)
#   hsv-fused   HSV+MOG2,  深度融合    (vis_weight=0.5, 当前默认)
#   yolo-vis    YOLO,      纯视觉深度
#   yolo-sensor YOLO,      纯传感器深度
#   yolo-fused  YOLO,      深度融合
#   both-vis    HSV+YOLO,  纯视觉深度  (分屏对比两种检测器)
#   both-fused  HSV+YOLO,  深度融合
#
# 示例:
#   ./run_compare.sh hsv-fused yolo-fused       # 对比 HSV vs YOLO（都用深度融合）
#   ./run_compare.sh hsv-vis   hsv-sensor       # 对比纯视觉 vs 纯传感器深度
#   ./run_compare.sh both-vis  both-fused       # 对比有无深度融合（两侧都分屏显示 HSV/YOLO）
#
# 查看结果:
#   D455 A  MJPEG → http://localhost:5568/main
#   D455 B  MJPEG → http://localhost:5668/main
#   融合视图 → http://localhost:8080/monitor.html
#
# 日志:
#   tail -f /tmp/fusion-logs/cam_a.log
#   tail -f /tmp/fusion-logs/cam_b.log

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

METHOD_A="${1:-hsv-fused}"
METHOD_B="${2:-yolo-fused}"

# ── conda ───────────────────────────────────────────────────────────────────
CONDA_SH=""
for _root in "${CONDA_PREFIX_1:-}" "${CONDA_PREFIX:-}" \
             "$HOME/miniconda3" "$HOME/anaconda3" \
             "/opt/miniconda3" "/opt/anaconda3" "/opt/conda"; do
    [ -n "$_root" ] && [ -f "$_root/etc/profile.d/conda.sh" ] && \
        { CONDA_SH="$_root/etc/profile.d/conda.sh"; break; }
done
[ -z "$CONDA_SH" ] && { echo "[ERROR] conda.sh not found" >&2; exit 1; }
source "$CONDA_SH"
conda activate catchball || { echo "[ERROR] conda activate catchball failed" >&2; exit 1; }

# ── 方法 → CLI 参数 转换 ────────────────────────────────────────────────────
method_to_args() {
    case "$1" in
        hsv-vis)     echo "--detector hsv  --vis-weight 1.0" ;;
        hsv-sensor)  echo "--detector hsv  --vis-weight 0.0" ;;
        hsv-fused)   echo "--detector hsv  --vis-weight 0.5" ;;
        yolo-vis)    echo "--detector yolo --vis-weight 1.0" ;;
        yolo-sensor) echo "--detector yolo --vis-weight 0.0" ;;
        yolo-fused)  echo "--detector yolo --vis-weight 0.5" ;;
        both-vis)    echo "--detector both --vis-weight 1.0" ;;
        both-fused)  echo "--detector both --vis-weight 0.5" ;;
        *)
            echo "[ERROR] 未知方法: $1" >&2
            echo "可选: hsv-vis hsv-sensor hsv-fused yolo-vis yolo-sensor yolo-fused both-vis both-fused" >&2
            exit 1
            ;;
    esac
}

ARGS_A=$(method_to_args "$METHOD_A")
ARGS_B=$(method_to_args "$METHOD_B")

FUSION_PORT="${FUSION_PORT:-5570}"
MONITOR_PORT="${MONITOR_PORT:-8080}"
WEBVIEW_PORT="${WEBVIEW_PORT:-5571}"
LOGS=/tmp/fusion-logs
mkdir -p "$LOGS"

PIDS=()
_CLEANED=0

cleanup() {
    [ "$_CLEANED" = 1 ] && return
    _CLEANED=1
    echo
    echo "[run_compare] shutting down..."
    for pid in "${PIDS[@]:-}"; do kill "$pid" 2>/dev/null; done
    pkill -P $$ 2>/dev/null
    sleep 0.4
    echo "[run_compare] done."
}
trap cleanup INT TERM EXIT

pkill -f "ball_detection.py" 2>/dev/null
pkill -f "fusion.py"         2>/dev/null
sleep 0.4

echo "
┌─────────────────────────────────────────────────────┐
│  D455 A  (SN 260722302887)  →  方法: $METHOD_A
│  D455 B  (SN 152522251463)  →  方法: $METHOD_B
│  CLI A: $ARGS_A
│  CLI B: $ARGS_B
└─────────────────────────────────────────────────────┘
"

# ── 1. fusion ───────────────────────────────────────────────────────────────
python fusion.py --port "$FUSION_PORT" --webview-port "$WEBVIEW_PORT" \
    > "$LOGS/fusion.log" 2>&1 &
PIDS+=($!)
sleep 0.8

# ── 2. D455 A ───────────────────────────────────────────────────────────────
# shellcheck disable=SC2086
python ball_detection.py --config config/d455_a.yaml $ARGS_A \
    > "$LOGS/cam_a.log" 2>&1 &
PIDS+=($!)
sleep 1.5

# ── 3. D455 B ───────────────────────────────────────────────────────────────
# shellcheck disable=SC2086
python ball_detection.py --config config/d455_b.yaml $ARGS_B \
    > "$LOGS/cam_b.log" 2>&1 &
PIDS+=($!)
sleep 1.5

# ── 4. monitor ──────────────────────────────────────────────────────────────
python monitor.py --http-port "$MONITOR_PORT" --udp-port "$WEBVIEW_PORT" \
    --dir "$SCRIPT_DIR" > "$LOGS/monitor.log" 2>&1 &
PIDS+=($!)
sleep 0.8

cat <<EOF

═══════════════════════════════════════════════════════════════
  D455 A  [$METHOD_A]  → http://localhost:5568/main
  D455 B  [$METHOD_B]  → http://localhost:5668/main
  融合视图              → http://localhost:${MONITOR_PORT}/monitor.html
  Ctrl+C 停止全部
═══════════════════════════════════════════════════════════════

EOF

exec tail -f "$LOGS/fusion.log"
