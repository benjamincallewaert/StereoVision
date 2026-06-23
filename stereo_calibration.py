import cv2
import numpy as np
import os
import argparse

# ── CONFIGURATION ────────────────────────────────────────────
PATTERN_COLS   = 7
PATTERN_ROWS   = 5
CHECKER_SIZE   = 35          # mm
MARKER_SIZE    = 24         # mm
ARUCO_DICT     = cv2.aruco.DICT_4X4_100
IMAGE_EXT      = ('*.jpg','*.jpeg','*.png','*.bmp','*.tiff')

MIN_CORNERS = 24
MAX_ARUCO = 17
# ─────────────────────────────────────────────────────────────


def preprocess_images(input_dir: str, output_dir: str) -> None:
    """
    Load images from input_dir, apply a preprocessing pipeline to improve
    Charuco board detection, and save results to output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)

    image_files = sorted([
        f for f in os.listdir(input_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))
    ])

    if not image_files:
        print("No images found in input directory.")
        return

    print(f"Preprocessing {len(image_files)} images...\n")

    for filename in image_files:
        in_path  = os.path.join(input_dir, filename)
        out_path = os.path.join(output_dir, filename)

        image = cv2.imread(in_path)
        if image is None:
            print(f"  [SKIP] Could not read {filename}")
            continue
        print(f"Sharpeness score {sharpness_score(image, gray = False)}")
        processed = _pipeline(image)
        print(f"Sharpeness score processed {sharpness_score(processed)}")
        cv2.imwrite(out_path, processed)
        print(f"  [OK]   {filename} → {out_path}")

    print(f"\nDone. {len(image_files)} images saved to: {output_dir}")

def sharpness_score(img, gray = True):
    if not gray:
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = img
    return cv2.Laplacian(gray_image, cv2.CV_64F).var()

def _pipeline(image: np.ndarray) -> np.ndarray:

    # ── 1. Convert to grayscale ───────────────────────────────────────
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # ── 2. Gentle denoise ─────────────────────────────────────────────
    denoised = cv2.fastNlMeansDenoising(gray, h=6, templateWindowSize=7, searchWindowSize=21)

    # ── 3. CLAHE — much gentler clip to avoid blowing out highlights ──
    clahe     = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
    equalized = clahe.apply(denoised)

    gamma = 1.2
    inv_gamma = 1.0 / gamma
    lut = np.array([
        ((i / 255.0) ** inv_gamma) * 255
        for i in range(256)
    ], dtype=np.uint8)
    corrected = cv2.LUT(equalized, lut)

    # ── 5. Mild unsharp mask — sharpens edges without binarising ─────
    blurred   = cv2.GaussianBlur(corrected, (0, 0), sigmaX=2)
    sharpened = cv2.addWeighted(corrected, 1.5, blurred, -0.5, 0)

    return sharpened

def calibrate_and_save_parameters(INPUT_DIR, visualize = False):

    # ── Step 1: Build board and detector ─────────────────────
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    board      = cv2.aruco.CharucoBoard((PATTERN_COLS, PATTERN_ROWS), CHECKER_SIZE, MARKER_SIZE, dictionary)
    detector   = cv2.aruco.CharucoDetector(board)
    img = board.generateImage(outSize=(1400, 1000), marginSize=20, borderBits=1)
    board.setLegacyPattern(True)
    if visualize:
        cv2.imshow("CharucoBoard", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imwrite(f"charuco_boards/charuco_board_{PATTERN_COLS}x{PATTERN_ROWS}_{CHECKER_SIZE}mm_{MARKER_SIZE}mm_{str(ARUCO_DICT)}.jpg", img)
        
    # ── Step 2: Load images ───────────────────────────────────
    image_files = sorted([
        os.path.join(INPUT_DIR, f)
        for f in os.listdir(INPUT_DIR)
        if f.lower().endswith(".jpg")
    ])
    print(f"Found {len(image_files)} images.")
    print(cv2.__version__)
    all_charuco_corners = []
    all_charuco_ids     = []
    image_size          = None
    used_images         = 0

    # ── Step 3: Detect corners in every image ─────────────────
    for image_file in image_files:
        image = cv2.imread(image_file)
        if image is None:
            print(f"  [SKIP] Could not read {image_file}")
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        print(f"Sharpness {sharpness_score(gray)}")

        charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(gray)

        # ── Visualization ─────────────────────────────────────────────


        n_markers = len(marker_ids)  if marker_ids  is not None else 0
        n_charuco = len(charuco_ids) if charuco_ids is not None else 0
        is_ok     = charuco_ids is not None and n_charuco >= MIN_CORNERS and n_markers <= MAX_ARUCO
        color     = (0, 200, 0) if is_ok else (0, 0, 220)

        lines = [
            os.path.basename(image_file),
            f"ArUco markers:   {n_markers}",
            f"ChArUco corners: {n_charuco} / {MIN_CORNERS} required",
            f"Status: {'OK' if is_ok else 'SKIP'}",
            f"Sharpness: {sharpness_score(gray):.1f}",
        ]
        if visualize:
            vis = image.copy()
            if marker_ids is not None and len(marker_ids) > 0:
                cv2.aruco.drawDetectedMarkers(vis, marker_corners, marker_ids)
            if charuco_ids is not None and len(charuco_ids) > 0:
                cv2.aruco.drawDetectedCornersCharuco(vis, charuco_corners, charuco_ids, (0, 255, 0))
            for i, line in enumerate(lines):
                y = 30 + i * 28
                cv2.putText(vis, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(vis, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color,     1, cv2.LINE_AA)

            h, w = vis.shape[:2]
            scale = min(1.0, 1400 / w, 900 / h)
            if scale < 1.0:
                vis = cv2.resize(vis, (int(w * scale), int(h * scale)))

            cv2.imshow("Marker Detection — any key to advance, ESC to quit", vis)
            if cv2.waitKey(0) == 27:
                cv2.destroyAllWindows()
                break

        if not is_ok:
            print(f"[SKIP] {os.path.basename(image_file)} — "
                  f"only {n_charuco} Charuco corners found "
                  f"(markers detected: {n_markers})")
            continue

        print(f"  [OK]   {os.path.basename(image_file)} — {n_charuco} Charuco corners "
              f"(markers detected: {n_markers})")

        all_charuco_corners.append(charuco_corners.astype(np.float32))
        all_charuco_ids.append(charuco_ids.astype(np.int32))

        image_size = gray.shape[::-1]
        used_images += 1

    cv2.destroyAllWindows()

    print(f"\nUsing {used_images}/{len(image_files)} images for calibration.")

    if used_images < 4:
        print("ERROR: Not enough valid images. Need at least 4.")
        return

    # ── Step 4: Calibrate ─────────────────────────────────────
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
        all_charuco_corners,   # positional only — keyword args break overload resolution
        all_charuco_ids,
        board,
        image_size,
        None,
        None
    )

    print(f"\nCalibration RMS re-projection error: {ret:.4f} px")
    print(f"Camera matrix:\n{camera_matrix}")
    print(f"Distortion coefficients:\n{dist_coeffs.ravel()}")

    # ── Step 5: Save parameters ───────────────────────────────
    out_path = os.path.join(INPUT_DIR, "calibration.npz")
    np.savez(
        out_path,
        camera_matrix = camera_matrix,
        dist_coeffs   = dist_coeffs,
        rvecs         = np.array(rvecs, dtype=object),
        tvecs         = np.array(tvecs, dtype=object),
        rms_error     = ret
    )
    print(f"\nCalibration saved to: {out_path}")

"""
stereo_calibrate_cameras — improved pair selection

Replaces the original stereo_calibrate_cameras() and adds the helpers:
  _score_pair()       — per-pair quality metric
  _select_best_pairs() — greedy diversity + score filter

Drop-in: the function signature is identical to the original.
"""

# ── Pair-selection knobs (tune freely) ───────────────────────────────────────
MAX_PAIRS           = 30 
W_REPROJ            = 0.6   # weight for (1 / mean_reproj_error) in composite score
W_CORNERS           = 0.4   # weight for normalised corner count
MAX_REPROJ_THRESH   = 2.0   # discard any pair whose per-pair reproj error > this


# ─────────────────────────────────────────────────────────────────────────────
# Helper 1 — score a single pair
# ─────────────────────────────────────────────────────────────────────────────

def _score_pair(
    obj_pts:      np.ndarray,   # (N,3) float32
    img_pts_1:    np.ndarray,   # (N,1,2) float32
    img_pts_2:    np.ndarray,   # (N,1,2) float32
    K1: np.ndarray, D1: np.ndarray,
    K2: np.ndarray, D2: np.ndarray,
    rvec1: np.ndarray, tvec1: np.ndarray,
    rvec2: np.ndarray, tvec2: np.ndarray,
) -> float:
    """
    Mean per-corner reprojection error for this pair using the known intrinsics
    and a single-camera PnP solve.  Lower = better.
    Returns np.inf if PnP fails.
    """
    try:
        proj1, _ = cv2.projectPoints(obj_pts, rvec1, tvec1, K1, D1)
        proj2, _ = cv2.projectPoints(obj_pts, rvec2, tvec2, K2, D2)

        err1 = np.linalg.norm(img_pts_1.reshape(-1, 2) - proj1.reshape(-1, 2), axis=1)
        err2 = np.linalg.norm(img_pts_2.reshape(-1, 2) - proj2.reshape(-1, 2), axis=1)

        return float((err1.mean() + err2.mean()) / 2.0)
    except Exception:
        return np.inf


# ─────────────────────────────────────────────────────────────────────────────
# Helper 2 — coverage bitmask for a set of 2-D corner points
# ─────────────────────────────────────────────────────────────────────────────

def _coverage_mask(
    corners:     np.ndarray,   # (N,1,2) float32
    image_size:  tuple,        # (width, height)
    grid_cols:   int,
    grid_rows:   int,
) -> set:
    """
    Return the set of (col, row) grid cells that contain at least one corner.
    Used by the greedy selection to measure how much new area a pair adds.
    """
    W, H = image_size
    cell_w = W / grid_cols
    cell_h = H / grid_rows
    cells = set()
    for pt in corners.reshape(-1, 2):
        c = min(int(pt[0] / cell_w), grid_cols - 1)
        r = min(int(pt[1] / cell_h), grid_rows - 1)
        cells.add((c, r))
    return cells


# ─────────────────────────────────────────────────────────────────────────────
# Helper 3 — greedy pair selection
# ─────────────────────────────────────────────────────────────────────────────

def _select_best_pairs(candidates: list, max_pairs: int) -> list:
    """
    Given a list of candidate dicts (each must have 'score', 'reproj_err',
    'coverage_mask', and the calibration arrays), return a subset of at most
    max_pairs entries chosen to maximise spatial coverage while still
    preferring high-quality (low reprojection error) candidates.

    Strategy
    --------
    1. Hard-reject pairs above MAX_REPROJ_THRESH.
    2. Normalise score across remaining candidates.
    3. Greedy: at each step pick the candidate that maximises
           alpha * normalised_score  +  (1-alpha) * new_cell_fraction
       where new_cell_fraction is the proportion of grid cells this candidate
       adds that are not yet covered by the already-selected set.
       alpha = 0.35  (coverage is the dominant criterion once we have a few
                       good-quality pairs; score is a tie-breaker)
    """
    ALPHA = 0.35
    TOTAL_CELLS = PATTERN_COLS * PATTERN_ROWS

    # Step 1: hard-reject noisy pairs
    valid = [c for c in candidates if c["reproj_err"] < MAX_REPROJ_THRESH]
    if not valid:
        print("[Select] WARNING — no pairs below MAX_REPROJ_THRESH; using all.")
        valid = candidates[:]

    # Step 2: normalise score to [0,1]
    scores = np.array([c["score"] for c in valid], dtype=float)
    s_min, s_max = scores.min(), scores.max()
    denom = (s_max - s_min) if s_max > s_min else 1.0
    for i, c in enumerate(valid):
        c["norm_score"] = (c["score"] - s_min) / denom

    # Step 3: greedy selection
    selected = []
    covered_cells: set = set()
    remaining = valid[:]

    while remaining and len(selected) < max_pairs:
        best_val = -1.0
        best_idx = 0
        for i, cand in enumerate(remaining):
            new_cells = cand["coverage_mask"] - covered_cells
            new_frac  = len(new_cells) / TOTAL_CELLS
            value     = ALPHA * cand["norm_score"] + (1 - ALPHA) * new_frac
            if value > best_val:
                best_val = value
                best_idx = i

        chosen = remaining.pop(best_idx)
        selected.append(chosen)
        covered_cells |= chosen["coverage_mask"]

        cells_pct = 100 * len(covered_cells) / TOTAL_CELLS
        print(
            f"  [Select #{len(selected):02d}]  {chosen['name']:<28s}  "
            f"reproj={chosen['reproj_err']:.3f}px  "
            f"score={chosen['norm_score']:.2f}  "
            f"coverage={cells_pct:.0f}%"
        )

    return selected


# ─────────────────────────────────────────────────────────────────────────────
# Main function — drop-in replacement
# ─────────────────────────────────────────────────────────────────────────────

def stereo_calibrate_cameras(
    cam1_dir:   str,
    cam2_dir:   str,
    output_dir: str,
) -> None:
    """
    Stereo calibration with intelligent pair selection.

    Improvements over the original
    --------------------------------
    * Each detected pair is scored on reprojection error (using fixed single-
      camera intrinsics + solvePnP) and corner count.
    * A greedy set-cover pass then selects up to MAX_PAIRS pairs that together
      cover as much of the image plane as possible, prioritising low-error pairs.
    * Only the selected pairs are fed into cv2.stereoCalibrate, reducing
      sensitivity to outlier frames and redundant near-duplicate poses.

    Parameters / outputs are identical to the original function.
    """

    # ── 1. Load single-camera intrinsics ────────────────────────────────────
    cal1_path = os.path.join(cam1_dir, "calibration.npz")
    cal2_path = os.path.join(cam2_dir, "calibration.npz")

    for p in (cal1_path, cal2_path):
        if not os.path.isfile(p):
            raise FileNotFoundError(
                f"Single-camera calibration not found: {p}\n"
                "Run calibrate_and_save_parameters() first."
            )

    cal1 = np.load(cal1_path, allow_pickle=True)
    cal2 = np.load(cal2_path, allow_pickle=True)

    K1 = cal1["camera_matrix"].astype(np.float64)
    D1 = cal1["dist_coeffs"].astype(np.float64)
    K2 = cal2["camera_matrix"].astype(np.float64)
    D2 = cal2["dist_coeffs"].astype(np.float64)

    print(f"\n[Stereo] Loaded intrinsics from:\n  {cal1_path}\n  {cal2_path}")

    # ── 2. Build board and detector ─────────────────────────────────────────
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    board      = cv2.aruco.CharucoBoard(
        (PATTERN_COLS, PATTERN_ROWS), CHECKER_SIZE, MARKER_SIZE, dictionary
    )
    board.setLegacyPattern(True)
    detector = cv2.aruco.CharucoDetector(board)

    # ── 3. Find paired image filenames ──────────────────────────────────────
    exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    names1 = {f for f in os.listdir(cam1_dir) if f.lower().endswith(exts)}
    names2 = {f for f in os.listdir(cam2_dir) if f.lower().endswith(exts)}
    paired_names = sorted(names1 & names2)

    if not paired_names:
        raise RuntimeError(
            "No matching filenames found between the two camera directories.\n"
            "Images must share identical filenames (e.g. frame_001.jpg in both)."
        )

    print(f"[Stereo] Found {len(paired_names)} paired image files.")

    # ── 4. Detect and score every candidate pair ─────────────────────────────
    candidates  = []
    image_size  = None

    for name in paired_names:
        path1 = os.path.join(cam1_dir, name)
        path2 = os.path.join(cam2_dir, name)

        img1 = cv2.imread(path1)
        img2 = cv2.imread(path2)
        if img1 is None or img2 is None:
            print(f"  [SKIP] Could not read pair: {name}")
            continue

        try:
            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        except Exception:
            gray1, gray2 = img1, img2

        c_corners1, c_ids1, _, _ = detector.detectBoard(gray1)
        c_corners2, c_ids2, _, _ = detector.detectBoard(gray2)

        n1 = len(c_ids1) if c_ids1 is not None else 0
        n2 = len(c_ids2) if c_ids2 is not None else 0

        if n1 < MIN_CORNERS or n2 < MIN_CORNERS:
            print(f"  [SKIP] {name}  cam1={n1}  cam2={n2}  (below MIN_CORNERS={MIN_CORNERS})")
            continue

        # ── Find common corner IDs ───────────────────────────────────────────
        ids1_flat  = c_ids1.flatten()
        ids2_flat  = c_ids2.flatten()
        common_ids = np.intersect1d(ids1_flat, ids2_flat)

        if len(common_ids) < MIN_CORNERS:
            print(f"  [SKIP] {name}  only {len(common_ids)} common corners")
            continue

        idx1   = np.where(np.isin(ids1_flat, common_ids))[0]
        idx2   = np.where(np.isin(ids2_flat, common_ids))[0]
        order1 = np.argsort(ids1_flat[idx1])
        order2 = np.argsort(ids2_flat[idx2])

        matched_corners1 = c_corners1[idx1[order1]].astype(np.float32)
        matched_corners2 = c_corners2[idx2[order2]].astype(np.float32)
        matched_ids      = common_ids[order1]

        obj_pts = board.getChessboardCorners()[matched_ids].astype(np.float32)

        image_size = gray1.shape[::-1]   # (width, height)

        # ── Per-pair reprojection error via solvePnP ─────────────────────────
        # We get pose for each camera independently, then measure how well
        # the known intrinsics + this pose reproduce the detected corners.
        reproj_err = np.inf
        ok1, rvec1, tvec1 = cv2.solvePnP(obj_pts, matched_corners1, K1, D1)
        ok2, rvec2, tvec2 = cv2.solvePnP(obj_pts, matched_corners2, K2, D2)

        if ok1 and ok2:
            reproj_err = _score_pair(
                obj_pts,
                matched_corners1, matched_corners2,
                K1, D1, K2, D2,
                rvec1, tvec1, rvec2, tvec2,
            )

        # ── Coverage: union of cells visible in both cameras ─────────────────
        cov1 = _coverage_mask(matched_corners1, image_size, PATTERN_COLS, PATTERN_ROWS)
        cov2 = _coverage_mask(matched_corners2, image_size, PATTERN_COLS, PATTERN_ROWS)
        coverage = cov1 | cov2   # union: if either camera sees a region it counts

        # ── Composite score (higher = better) ────────────────────────────────
        # Protect against zero reproj error (perfect synthetic data, or bad PnP)
        safe_reproj = max(reproj_err, 1e-6)
        n_corners   = len(matched_ids)
        raw_score   = W_REPROJ * (1.0 / safe_reproj) + W_CORNERS * n_corners

        candidates.append({
            "name":          name,
            "obj_pts":       obj_pts,
            "img_pts_1":     matched_corners1,
            "img_pts_2":     matched_corners2,
            "reproj_err":    reproj_err,
            "n_corners":     n_corners,
            "score":         raw_score,
            "coverage_mask": coverage,
        })

        print(
            f"  [CAND] {name:<28s}  common={n_corners:2d}  reproj={reproj_err:.3f}px"
        )

    print(f"\n[Stereo] {len(candidates)} candidate pairs collected.")

    if len(candidates) < 6:
        print("ERROR: Need at least 6 valid pairs for reliable stereo calibration.")
        return

    # ── 5. Select the best MAX_PAIRS pairs ──────────────────────────────────
    print(f"\n[Stereo] Selecting up to {MAX_PAIRS} pairs "
          f"(grid {PATTERN_COLS}×{PATTERN_ROWS}, α={0.35})…")
    selected = _select_best_pairs(candidates, max_pairs=MAX_PAIRS)

    print(f"\n[Stereo] Selected {len(selected)}/{len(candidates)} pairs.")
    print(f"  Mean reproj error of selection : "
          f"{np.mean([c['reproj_err'] for c in selected]):.3f} px")
    print(f"  Mean reproj error of all cands : "
          f"{np.mean([c['reproj_err'] for c in candidates]):.3f} px")

    # ── 6. Build the calibration arrays from selected pairs ─────────────────
    obj_points_all = [c["obj_pts"]    for c in selected]
    img_points_1   = [c["img_pts_1"]  for c in selected]
    img_points_2   = [c["img_pts_2"]  for c in selected]

    # ── 7. Stereo calibration ────────────────────────────────────────────────
    stereo_flags = (
        cv2.CALIB_USE_INTRINSIC_GUESS     # trust single-camera results
        # Remove CALIB_RATIONAL_MODEL if you want the simpler 5-param model;
        # it's only beneficial with enough data and genuine distortion.
        # | cv2.CALIB_RATIONAL_MODEL
    )

    rms, K1_out, D1_out, K2_out, D2_out, R, T, E, F = cv2.stereoCalibrate(
        obj_points_all,
        img_points_1,
        img_points_2,
        K1, D1,
        K2, D2,
        image_size,
        flags=stereo_flags,
        criteria=(cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 200, 1e-7),
    )

    baseline_mm = float(np.linalg.norm(T))
    print(f"\n[Stereo] RMS re-projection error : {rms:.4f} px")
    print(f"[Stereo] Baseline                : {baseline_mm:.2f} mm")
    print(f"[Stereo] Rotation (Rodrigues)    : {cv2.Rodrigues(R)[0].ravel()}")
    print(f"[Stereo] Translation (mm)        : {T.ravel()}")

    # ── 8. Stereo rectification ──────────────────────────────────────────────
    alpha = 1   # 0 = crop to valid pixels, 1 = full sensor area

    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        K1_out, D1_out,
        K2_out, D2_out,
        image_size,
        R, T,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=alpha,
        newImageSize=image_size,
    )

    map1x, map1y = cv2.initUndistortRectifyMap(
        K1_out, D1_out, R1, P1, image_size, cv2.CV_32FC1
    )
    map2x, map2y = cv2.initUndistortRectifyMap(
        K2_out, D2_out, R2, P2, image_size, cv2.CV_32FC1
    )

    # ── 9. Save ──────────────────────────────────────────────────────────────
    np.savez(
        output_dir,
        camera_matrix_1 = K1_out,
        dist_coeffs_1   = D1_out,
        camera_matrix_2 = K2_out,
        dist_coeffs_2   = D2_out,
        R               = R,
        T               = T,
        E               = E,
        F               = F,
        R1              = R1,
        R2              = R2,
        P1              = P1,
        P2              = P2,
        Q               = Q,
        map1x           = map1x,
        map1y           = map1y,
        map2x           = map2x,
        map2y           = map2y,
        roi1            = np.array(roi1),
        roi2            = np.array(roi2),
        rms_error       = rms,
        image_size      = np.array(image_size),
        baseline_mm     = baseline_mm,
        selected_pairs  = np.array([c["name"] for c in selected]),   # audit trail
    )
    print(f"\n[Stereo] Calibration saved to: {output_dir}")
    print("[Stereo] Done.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--serial_l", type=str, default="24856866")
    ap.add_argument("--serial_r", type=str, default="24856867")
    ap.add_argument("--session_name", type=str, default="test2")
    ap.add_argument("--preprocess", type=bool, default=False)

    args = ap.parse_args()
    SESSION_NAME = args.session_name
    CAM1_SRC = fr"calibrations/{SESSION_NAME}/left"
    CAM2_SRC = fr"calibrations/{SESSION_NAME}/right"

    if args.preprocess:
        CAM1_OUT = fr"{CAM1_SRC}/preprocessed"
        CAM2_OUT = fr"{CAM2_SRC}/preprocessed"
        preprocess_images(CAM1_SRC, CAM1_OUT)
        preprocess_images(CAM2_SRC, CAM2_OUT)
    else:
        CAM1_OUT = CAM1_SRC
        CAM2_OUT = CAM2_SRC

    calibrate_and_save_parameters(CAM1_OUT)
    calibrate_and_save_parameters(CAM2_OUT)
    
    STEREO_OUT = fr'calibrations/{SESSION_NAME}/stereo_calibration.npz'

    stereo_calibrate_cameras(
        cam1_dir   = CAM1_SRC,
        cam2_dir   = CAM2_SRC,
        output_dir = STEREO_OUT,
    )


