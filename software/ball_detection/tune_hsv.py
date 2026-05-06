"""
tune_hsv.py

Interactive HSV + MOG2 mask tuner for D455.

四个窗口同时显示：
  1. 原始彩色帧 + 检测圆（左上）
  2. HSV 颜色掩码（右上）
  3. MOG2 运动掩码（左下）
  4. HSV ∩ MOG2 合并掩码 + 轮廓（右下）

滑动条（在窗口顶部）：
  H_low / H_high / S_min / V_min — HSV 范围
  MOG2_thresh — 背景减除灵敏度（越小越敏感）
  Min_radius  — 最小检测半径(px)
  Circularity — 最小圆度 × 100

按键：
  s — 打印当前参数到终端（可直接粘贴到启动命令）
  r — 重置 MOG2 背景模型
  q / ESC — 退出

Usage:
    conda activate catchball
    python tune_hsv.py
    python tune_hsv.py --width 848 --height 480
"""

import argparse
import time
import numpy as np
import cv2
import pyrealsense2 as rs

# ── defaults ──────────────────────────────────────────────────────────────────
DEF_H_LOW   = 10
DEF_H_HIGH  = 35
DEF_S_MIN   = 170
DEF_V_MIN   = 170
DEF_MOG2    = 50    # varThreshold
DEF_MIN_R   = 3
DEF_CIRC    = 55    # circularity × 100

BALL_RADIUS = 0.0335   # m


def _hw_reset_and_start(pipeline, width, height):
    print("[INFO] Hardware reset...")
    devs = rs.context().query_devices()
    if not devs:
        raise RuntimeError("No RealSense device found.")
    devs[0].hardware_reset()
    time.sleep(6)
    pipeline_new = rs.pipeline()
    for fps in (60, 30, 15):
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16,  fps)
        try:
            profile = pipeline_new.start(cfg)
            pipeline_new.wait_for_frames(timeout_ms=6000)
            print(f"[INFO] RealSense OK ({fps} Hz)")
            return pipeline_new, profile
        except RuntimeError as e:
            try:
                pipeline_new.stop()
            except Exception:
                pass
            pipeline_new = rs.pipeline()
    raise RuntimeError("Could not start RealSense pipeline.")


def main():
    parser = argparse.ArgumentParser(description="Interactive HSV/MOG2 mask tuner for D455")
    parser.add_argument("--width",  type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()

    pipeline, profile = _hw_reset_and_start(rs.pipeline(), args.width, args.height)
    color_intrin = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    depth_scale  = profile.get_device().first_depth_sensor().get_depth_scale()
    fx = color_intrin.fx
    print(f"[INFO] fx={fx:.1f}  depth_scale={depth_scale:.5f}")
    print(f"[INFO] Max range ≈ {fx * BALL_RADIUS / DEF_MIN_R:.1f} m  "
          f"(fx={fx:.0f}, R={BALL_RADIUS}m, min_r={DEF_MIN_R}px)")

    # ── Create windows — trackbars live on the colour window ─────────────────
    # WINDOW_NORMAL is created AFTER trackbars; Qt requires the window to exist
    # first via namedWindow before createTrackbar, and AUTOSIZE works more
    # reliably with Qt when fonts are missing.
    WIN_CTRL_WIN = "1 Color+detect"
    cv2.namedWindow(WIN_CTRL_WIN, cv2.WINDOW_NORMAL)
    cv2.createTrackbar("H_low",    WIN_CTRL_WIN, DEF_H_LOW,   179, lambda _: None)
    cv2.createTrackbar("H_high",   WIN_CTRL_WIN, DEF_H_HIGH,  179, lambda _: None)
    cv2.createTrackbar("S_min",    WIN_CTRL_WIN, DEF_S_MIN,   255, lambda _: None)
    cv2.createTrackbar("V_min",    WIN_CTRL_WIN, DEF_V_MIN,   255, lambda _: None)
    cv2.createTrackbar("MOG2_thr", WIN_CTRL_WIN, DEF_MOG2,    200, lambda _: None)
    cv2.createTrackbar("Min_r_px", WIN_CTRL_WIN, DEF_MIN_R,    50, lambda _: None)
    cv2.createTrackbar("Circ100",  WIN_CTRL_WIN, DEF_CIRC,    100, lambda _: None)

    # Remaining output windows
    cv2.namedWindow("2 HSV mask",     cv2.WINDOW_NORMAL)
    cv2.namedWindow("3 MOG2 motion",  cv2.WINDOW_NORMAL)
    cv2.namedWindow("4 Combined",     cv2.WINDOW_NORMAL)
    for wn in ("1 Color+detect", "2 HSV mask", "3 MOG2 motion", "4 Combined"):
        cv2.resizeWindow(wn, args.width // 2, args.height // 2)
    # Tile windows (approximate positions)
    cv2.moveWindow("1 Color+detect", 0,               0)
    cv2.moveWindow("2 HSV mask",     args.width // 2, 0)
    cv2.moveWindow("3 MOG2 motion",  0,               args.height // 2 + 60)
    cv2.moveWindow("4 Combined",     args.width // 2, args.height // 2 + 60)

    # ── State ──────────────────────────────────────────────────────────────────
    back_sub   = cv2.createBackgroundSubtractorMOG2(
        history=100, varThreshold=DEF_MOG2, detectShadows=False)
    prev_mog2_thr = DEF_MOG2

    print("\n[INFO] Controls: s=save params  r=reset MOG2  q/ESC=quit")
    print("[INFO] Drag trackbars to tune HSV and MOG2 parameters live.\n")

    while True:
        frames = pipeline.wait_for_frames(timeout_ms=3000)
        cf = frames.get_color_frame()
        df = frames.get_depth_frame()
        if not cf or not df:
            continue

        color     = np.asanyarray(cf.get_data()).copy()
        depth_arr = np.asanyarray(df.get_data())

        # Read trackbar values
        TB = "1 Color+detect"
        h_low  = cv2.getTrackbarPos("H_low",    TB)
        h_high = cv2.getTrackbarPos("H_high",   TB)
        s_min  = cv2.getTrackbarPos("S_min",    TB)
        v_min  = cv2.getTrackbarPos("V_min",    TB)
        mog2_t = cv2.getTrackbarPos("MOG2_thr", TB)
        min_r  = max(cv2.getTrackbarPos("Min_r_px", TB), 1)
        circ   = cv2.getTrackbarPos("Circ100",  TB) / 100.0

        # Recreate MOG2 if threshold changed
        if mog2_t != prev_mog2_thr:
            back_sub = cv2.createBackgroundSubtractorMOG2(
                history=100, varThreshold=mog2_t, detectShadows=False)
            prev_mog2_thr = mog2_t

        hsv_low  = np.array([h_low, s_min, v_min], dtype=np.uint8)
        hsv_high = np.array([h_high, 255, 255],    dtype=np.uint8)

        # ── 1. HSV mask ───────────────────────────────────────────────────────
        hsv      = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
        hsv_mask = cv2.inRange(hsv, hsv_low, hsv_high)
        kern     = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        hsv_mask = cv2.morphologyEx(hsv_mask, cv2.MORPH_CLOSE, kern, iterations=2)
        hsv_mask = cv2.morphologyEx(hsv_mask, cv2.MORPH_OPEN,  kern, iterations=1)

        # ── 2. MOG2 motion mask ────────────────────────────────────────────────
        BG_RESIZE = 0.4
        small = cv2.resize(color, (0, 0), fx=BG_RESIZE, fy=BG_RESIZE,
                           interpolation=cv2.INTER_AREA)
        fgmask = back_sub.apply(small)
        h, w   = color.shape[:2]
        motion_mask = cv2.resize(fgmask, (w, h), interpolation=cv2.INTER_NEAREST)
        kern5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_DILATE, kern5, iterations=2)

        # ── 3. Combined mask ───────────────────────────────────────────────────
        combined = cv2.bitwise_and(hsv_mask, motion_mask)

        # ── 4. Detect candidates from combined mask ────────────────────────────
        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        vis = color.copy()
        best = None

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < np.pi * min_r**2:
                continue
            peri = cv2.arcLength(cnt, True)
            if peri == 0:
                continue
            c = 4 * np.pi * area / (peri**2)
            (cx_f, cy_f), r = cv2.minEnclosingCircle(cnt)
            if r < min_r or r > 200:
                continue

            # Draw all candidates in grey
            cv2.circle(vis, (int(cx_f), int(cy_f)), int(r), (100, 100, 100), 1)
            cv2.putText(vis, f"c={c:.2f}", (int(cx_f) - int(r), int(cy_f) - int(r) - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)

            if c >= circ:
                score = area * c
                if best is None or score > best[0]:
                    best = (score, int(cx_f), int(cy_f), r, c)

        depth_m = 0.0
        if best is not None:
            _, bx, by, br, bc = best
            # Visual depth
            depth_vis = fx * BALL_RADIUS / br if br > 0 else 0.0
            # Quick sensor depth
            d_raw = depth_arr[np.clip(by, 0, depth_arr.shape[0]-1),
                              np.clip(bx, 0, depth_arr.shape[1]-1)]
            depth_sens = d_raw * depth_scale + BALL_RADIUS if d_raw > 0 else 0.0
            depth_m = depth_vis if depth_vis > 0 else depth_sens

            cv2.circle(vis, (bx, by), int(br), (0, 255, 0), 2)
            cv2.circle(vis, (bx, by), 3, (0, 0, 255), -1)
            cv2.putText(vis, f"BALL  r={br:.0f}px  d={depth_m:.2f}m  c={bc:.2f}",
                        (bx - int(br), max(by - int(br) - 6, 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        else:
            cv2.putText(vis, "No ball", (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)

        # ── HUD on color frame ─────────────────────────────────────────────────
        cv2.putText(vis,
                    f"H=[{h_low},{h_high}] S>={s_min} V>={v_min} "
                    f"MOG2={mog2_t} r>={min_r}px circ>={circ:.2f}",
                    (8, vis.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)

        # ── Colourize masks for clarity ────────────────────────────────────────
        # HSV mask: yellow tint
        hsv_disp = cv2.cvtColor(hsv_mask, cv2.COLOR_GRAY2BGR)
        hsv_disp[hsv_mask > 0] = [0, 220, 220]

        # Motion mask: blue tint
        mot_disp = cv2.cvtColor(motion_mask, cv2.COLOR_GRAY2BGR)
        mot_disp[motion_mask > 0] = [200, 80, 0]

        # Combined: green tint + detected circle
        comb_disp = cv2.cvtColor(combined, cv2.COLOR_GRAY2BGR)
        comb_disp[combined > 0] = [0, 200, 80]
        if best is not None:
            _, bx, by, br, _ = best
            cv2.circle(comb_disp, (bx, by), int(br), (0, 255, 0), 2)

        # Candidate count overlay
        n_cand = sum(1 for cnt in contours
                     if cv2.contourArea(cnt) >= np.pi * min_r**2
                     and cv2.arcLength(cnt, True) > 0
                     and 4*np.pi*cv2.contourArea(cnt)/cv2.arcLength(cnt,True)**2 >= circ)
        for disp, label in [(hsv_disp,  f"HSV mask  H=[{h_low},{h_high}] S>={s_min} V>={v_min}"),
                            (mot_disp,  f"MOG2 motion  thr={mog2_t}"),
                            (comb_disp, f"Combined  candidates={len(contours)}  passing={n_cand}")]:
            cv2.putText(disp, label, (8, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # ── Display ────────────────────────────────────────────────────────────
        cv2.imshow("1 Color+detect", vis)
        cv2.imshow("2 HSV mask",     hsv_disp)
        cv2.imshow("3 MOG2 motion",  mot_disp)
        cv2.imshow("4 Combined",     comb_disp)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord('s'):
            print(f"\n[PARAMS] Copy-paste to ball_detection_d455.py or command line:")
            print(f"  --h-low {h_low} --h-high {h_high} --s-min {s_min} --v-min {v_min}")
            print(f"  (MOG2 varThreshold={mog2_t}, min_r={min_r}px, circularity={circ:.2f})")
            print(f"\n  settings.yaml equivalent:")
            print(f"    low_H: {h_low}")
            print(f"    high_H: {h_high}")
            print(f"    low_S: {s_min}")
            print(f"    low_V: {v_min}")
        elif key == ord('r'):
            back_sub = cv2.createBackgroundSubtractorMOG2(
                history=100, varThreshold=mog2_t, detectShadows=False)
            print("[INFO] MOG2 background model reset.")

    pipeline.stop()
    cv2.destroyAllWindows()
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
