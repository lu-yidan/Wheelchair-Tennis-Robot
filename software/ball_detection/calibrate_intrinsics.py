"""
calibrate_intrinsics.py — 交互式棋盘格内参标定 (一次性)

适用任何 V4L2 / UVC 摄像头.  专门给 webcam.yaml 准备 fx/fy/ppx/ppy/dist.

用法:
    python calibrate_intrinsics.py --device 2 --width 1920 --height 1080
    # 默认 9×6 内角点 (10×7 方格)，每格 25 mm — A4 打印的标准盘

按键 (要在 OpenCV 窗口里按):
    SPACE  — 当前画面里检测到棋盘 → 捕获
    c      — 至少捕获 15 张后跑标定
    q      — 不保存退出

打印棋盘:
    OpenCV 自带：opencv-4.x/doc/pattern.png
    或下载: https://github.com/opencv/opencv/blob/4.x/doc/pattern.png
    A4 打印不缩放, 量一下方格实际边长 → 用 --square 传 (米)

标定流程:
    1. 棋盘平摆桌面，相机距离 ~50 cm
    2. 慢慢倾斜/平移/旋转，每个角度按 SPACE 一次
    3. 至少 15 张，覆盖：左上、右上、左下、右下、中央、近、远、各种倾角
    4. 按 c 出结果，复制 YAML 到 config/webcam.yaml 的 camera.intrinsics 下

输出指标:
    reprojection_error < 0.5 px = 优秀
    reprojection_error < 1.0 px = 可用
    > 1.0 px → 重新拍, 棋盘没拍到边角或反光严重
"""

import argparse
import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", type=int, default=2, help="V4L2 device index (默认 2 = /dev/video2)")
    ap.add_argument("--width",  type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--fps",    type=int, default=30)
    ap.add_argument("--fourcc", default="MJPG")
    ap.add_argument("--rows",   type=int, default=9, help="棋盘内角点列数 (默认 9)")
    ap.add_argument("--cols",   type=int, default=6, help="棋盘内角点行数 (默认 6)")
    ap.add_argument("--square", type=float, default=0.025,
                    help="单个方格物理边长 (米, 默认 25mm)")
    ap.add_argument("--min-shots", type=int, default=15,
                    help="最少需要的有效图数 (默认 15)")
    args = ap.parse_args()

    pattern = (args.rows, args.cols)
    # 物体世界坐标 (棋盘在 Z=0 平面上)
    objp = np.zeros((args.rows * args.cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:args.rows, 0:args.cols].T.reshape(-1, 2) * args.square

    cap = cv2.VideoCapture(args.device, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap = cv2.VideoCapture(args.device)
    if not cap.isOpened():
        raise SystemExit(f"open /dev/video{args.device} failed")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*args.fourcc))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)

    aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] /dev/video{args.device} opened at {aw}×{ah}  pattern={args.rows}×{args.cols} "
          f"square={args.square*1000:.1f}mm")

    obj_points, img_points = [], []
    win = "calibrate (SPACE=capture | c=calibrate | q=quit)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    sub_crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    last_h, last_w = ah, aw
    last_capture_ts = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[WARN] grab failed")
            continue
        last_h, last_w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # findChessboardCornersSB 比 findChessboardCorners 快 + 准 (OpenCV 4.x)
        try:
            found, corners = cv2.findChessboardCornersSB(
                gray, pattern, cv2.CALIB_CB_NORMALIZE_IMAGE)
        except AttributeError:
            found, corners = cv2.findChessboardCorners(
                gray, pattern,
                flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK
                      + cv2.CALIB_CB_NORMALIZE_IMAGE)
            if found:
                corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), sub_crit)

        viz = frame.copy()
        col_ok = (60, 220, 60)
        col_no = (60, 60, 220)
        if found:
            cv2.drawChessboardCorners(viz, pattern, corners, found)
        cv2.rectangle(viz, (0, 0), (viz.shape[1], 70), (0, 0, 0), -1)
        cv2.putText(viz, f"captured: {len(obj_points)} / {args.min_shots}+    "
                    f"{'BOARD FOUND - press SPACE' if found else 'no board'}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    col_ok if found else col_no, 2)
        cv2.putText(viz, "SPACE=capture   c=calibrate   q=quit",
                    (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        cv2.imshow(win, viz)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            cap.release(); cv2.destroyAllWindows()
            print("[INFO] quit without saving")
            return
        if key == ord(" ") and found:
            import time
            now = time.perf_counter()
            if now - last_capture_ts < 0.3:        # debounce
                continue
            last_capture_ts = now
            obj_points.append(objp)
            img_points.append(corners)
            print(f"[+] capture #{len(obj_points)}")
        if key == ord("c"):
            if len(obj_points) < args.min_shots:
                print(f"[WARN] need ≥{args.min_shots} shots, have {len(obj_points)}")
                continue
            break

    cap.release()
    cv2.destroyAllWindows()

    print(f"\n[INFO] calibrating from {len(obj_points)} images at {last_w}×{last_h} ...")
    rms, K, dist, _, _ = cv2.calibrateCamera(
        obj_points, img_points, (last_w, last_h), None, None)

    fx, fy = K[0, 0], K[1, 1]
    ppx, ppy = K[0, 2], K[1, 2]
    d = dist.flatten()[:5]

    quality = "excellent" if rms < 0.5 else "good" if rms < 1.0 else "rough"
    print(f"\n[RESULT]  reprojection error = {rms:.4f} px  ({quality})")
    print(f"          fx={fx:.2f}  fy={fy:.2f}")
    print(f"          ppx={ppx:.2f}  ppy={ppy:.2f}")
    print(f"          dist={[round(float(x), 6) for x in d]}")

    print("\n" + "─"*70)
    print("# Paste under camera.intrinsics in config/webcam.yaml:")
    print("─"*70)
    print(f"  intrinsics:")
    print(f"    fx:  {fx:.2f}")
    print(f"    fy:  {fy:.2f}")
    print(f"    ppx: {ppx:.2f}")
    print(f"    ppy: {ppy:.2f}")
    print(f"    dist: [{', '.join(f'{x:.6f}' for x in d)}]")
    print("─"*70)
    print(f"# Also update top-level width/height to {last_w}/{last_h} if changed.")


if __name__ == "__main__":
    main()
