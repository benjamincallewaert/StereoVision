from __future__ import annotations
import argparse
import os
import threading
import queue
import time
from dataclasses import dataclass, field

import numpy as np
import cv2

from basler_grabber import BaslerStereoPair

PREVIEW_DIR = "calibrations"

PATTERN_COLS   = 7
PATTERN_ROWS   = 5
CHECKER_SIZE   = 34
MARKER_SIZE    = 24
ARUCO_DICT     = cv2.aruco.DICT_4X4_1000

MIN_CORNERS = 15 

dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
board = cv2.aruco.CharucoBoard(
    (PATTERN_COLS, PATTERN_ROWS),
    CHECKER_SIZE,
    MARKER_SIZE,
    dictionary
)
board.setLegacyPattern(True)
detector = cv2.aruco.CharucoDetector(board)


# =========================
# Countdown + Burst Capture
# =========================
def delayed_capture(capture_event, delay=1, burst=50, interval=0.6):
    for i in range(delay, 0, -1):
        print(f"📸 Capturing in {i}...", end="\r", flush=True)
        time.sleep(1)

    print("📸 Capturing burst!        ")

    for _ in range(burst):
        capture_event.set()
        time.sleep(2)
        print("CAPTURING!!!!!!!!!!!")
        time.sleep(interval)


@dataclass
class CalibData:
    obj_l: list = field(default_factory=list)
    pts_l: list = field(default_factory=list)
    obj_r: list = field(default_factory=list)
    pts_r: list = field(default_factory=list)

    obj_s: list = field(default_factory=list)
    pts_sl: list = field(default_factory=list)
    pts_sr: list = field(default_factory=list)

    img_shape: tuple | None = None
    captured: int = 0
    used: int = 0

    @property
    def status(self) -> str:
        return f"captured={self.captured} used={self.used}"


def calibrate(
    cols: int,
    rows: int,
    square_mm: float,
    serial_left: str | None = None,
    serial_right: str | None = None,
    session_name: str = "test1",
    output_path: str = "stereo_calibration.npz",
):
    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_mm

    data = CalibData()
    fq = queue.Queue(maxsize=30)
    stop = threading.Event()
    capture = threading.Event()

    os.makedirs(f"{PREVIEW_DIR}/{session_name}/left/", exist_ok=True)
    os.makedirs(f"{PREVIEW_DIR}/{session_name}/right/", exist_ok=True)

    # =========================
    # PRODUCER (capture frames)
    # =========================
    def producer(pair: BaslerStereoPair):
        while not stop.is_set():
            if not capture.wait(timeout=0.05):
                continue
            capture.clear()

            fl, fr = pair.grab()
            if fl is None or fr is None:
                continue

            try:
                data.captured += 1

                fq.put((fl.copy(), fr.copy()), timeout=0.05)

                cv2.imwrite(f"{PREVIEW_DIR}/{session_name}/left/{data.captured:03d}.jpg", fl)
                cv2.imwrite(f"{PREVIEW_DIR}/{session_name}/right/{data.captured:03d}.jpg", fr)

                print(f"✓ Captured {data.captured}")
            except queue.Full:
                print("⚠ Queue full — dropping frame")

    # =========================
    # CONSUMER (detect + filter)
    # =========================
    def consumer():
        while not stop.is_set() or not fq.empty():
            try:
                fl, fr = fq.get(timeout=0.1)
            except queue.Empty:
                continue

            if data.img_shape is None:
                data.img_shape = fl.shape[:2][::-1]

            gl = cv2.cvtColor(fl, cv2.COLOR_BGR2GRAY)
            gr = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)

            cl, il, _, ml = detector.detectBoard(gl)
            cr, ir, _, mr = detector.detectBoard(gr)

            nL = 0 if il is None else len(il)
            nR = 0 if ir is None else len(ir)
            if il is None or ir is None or nL < MIN_CORNERS or nR < MIN_CORNERS:
                print(f"[SKIP] L={nL} R={nR}")
                continue

            print(f"[OK]   L={nL} R={nR}")

            data.obj_l.append(objp)
            data.pts_l.append(cl)

            data.obj_r.append(objp)
            data.pts_r.append(cr)

            data.obj_s.append(objp)
            data.pts_sl.append(cl)
            data.pts_sr.append(cr)

            data.used += 1

    print("\nENTER → capture | d → calibrate | q → quit\n")

    with BaslerStereoPair(serial_left=serial_left, serial_right=serial_right) as pair:
        tp = threading.Thread(target=producer, args=(pair,), daemon=True)
        tc = threading.Thread(target=consumer, daemon=True)

        tp.start()
        tc.start()

        while True:
            cmd = input(f"[{data.status}] > ").strip().lower()

            if cmd == "q":
                stop.set()
                return

            if cmd == "d":
                if data.used < 5:
                    print("⚠ Need at least 5 GOOD frames")
                    continue
                break

            # 🔥 Countdown + burst trigger
            threading.Thread(
                target=delayed_capture,
                args=(capture,),
                daemon=True
            ).start()

        stop.set()
        tp.join()
        tc.join()

    shape = data.img_shape

    print("\n📐 Mono LEFT...")
    rms_l, K_l, d_l, *_ = cv2.calibrateCamera(data.obj_l, data.pts_l, shape, None, None)
    print(f"RMS {rms_l:.4f}")

    print("📐 Mono RIGHT...")
    rms_r, K_r, d_r, *_ = cv2.calibrateCamera(data.obj_r, data.pts_r, shape, None, None)
    print(f"RMS {rms_r:.4f}")

    print(f"📐 Stereo ({len(data.obj_s)} pairs)...")
    rms, K_l, d_l, K_r, d_r, R, T, E, F = cv2.stereoCalibrate(
        data.obj_s,
        data.pts_sl,
        data.pts_sr,
        K_l,
        d_l,
        K_r,
        d_r,
        shape,
        flags=cv2.CALIB_FIX_INTRINSIC,
    )
    print(f"RMS {rms:.4f}")

    np.savez(
        output_path,
        K_l=K_l, d_l=d_l,
        K_r=K_r, d_r=d_r,
        R=R, T=T,
    )

    print(f"\n✅ Saved → {output_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cols", type=int, default=7)
    ap.add_argument("--rows", type=int, default=5)
    ap.add_argument("--square_mm", type=float, default=25.0)
    ap.add_argument("--session_name", type=str, default="test1")
    ap.add_argument("--serial_l", type=str, default="24856866")
    ap.add_argument("--serial_r", type=str, default="24856867")
    ap.add_argument("--out", type=str, default="stereo_calibration.npz")

    args = ap.parse_args()
    output_path = rf'{args.session_name}/{args.out}'
    calibrate(
        cols=args.cols,
        rows=args.rows,
        square_mm=args.square_mm,
        serial_left=args.serial_l,
        serial_right=args.serial_r,
        output_path= output_path,
    )