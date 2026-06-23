from __future__ import annotations
import cv2
import cv2.aruco as aruco
import numpy as np

from typing import Optional
import json
from pathlib import Path

from wheel_matching_utils import (
    WheelPose, MARKER_IDS, MatchResult, save_visualisations,
    _annotate_single, NUMBER_OF_MARKERS, detect_markers,
)
# Detector, dictionary, params and preprocessing now live in wheel_matching_utils
# (single shared, tuned detector — see detect_markers()).
MATCH_MAX_ANGLE_DEG = 10.0

# ── Marker detection ──────────────────────────────────────────────────────────

def _img_angle(corners: np.ndarray) -> float:
    """
    Robust marker orientation using the vector from marker centre
    to the midpoint of the top edge (corners 0→1).

    Using the midpoint of an edge rather than a single corner:
      - Averages out per-corner localisation noise
      - Invariant to which corner is labelled '0' within that edge
      - The diagonal-intersection centre is stable under mild perspective

    corners: shape (4, 2), order [TL, TR, BR, BL] as returned by OpenCV
    """
    c0, c1, c2, c3 = corners[0], corners[1], corners[2], corners[3]

    # Stable centre: intersection of the two diagonals
    centre = (c0 + c2) / 2   # == (c1 + c3) / 2 for a perfect square;
                               # averaging both diagonals is even more robust:
    centre = ((c0 + c2) + (c1 + c3)) / 4

    # Top-edge midpoint (the edge between corner-0 and corner-1)
    top_mid = (c0 + c1) / 2

    dx, dy = top_mid - centre
    return float(np.degrees(np.arctan2(dy, dx)) % 360)


def process_image(src: np.ndarray | str | Path,
                  label: Optional[str] = None,
                  verbose: bool = False) -> Optional[WheelPose]:
    if isinstance(src, (str, Path)):
        label = label or str(src)
        img   = cv2.imread(str(src))
        if img is None:
            raise FileNotFoundError(src)
    else:
        img = src

    # Shared, tuned detector with CLAHE + upscale preprocessing.  Corners come
    # back in original-image coordinates.
    corners_list, ids = detect_markers(img)

    marker_angles: dict[int, float] = {}
    kept_corners: list = []
    if ids is not None:
        for corners, mid in zip(corners_list, ids.flatten()):
            mid = int(mid)
            if mid in MARKER_IDS:
                marker_angles[mid] = _img_angle(corners[0])
                kept_corners.append(corners)   # aligned with marker_angles order

    name = Path(label).name if label else "image"
    if not marker_angles:
        if verbose:
            print(f"  {name}: no markers detected")
        return None

    if verbose:
        ids_str = ", ".join(f"ID{m}={a:.1f}°" for m, a in sorted(marker_angles.items()))
        print(f"  {name}: [{ids_str}]")

    return WheelPose(
        marker_angles   = marker_angles,
        n_visible       = len(marker_angles),
        confidence      = len(marker_angles) / NUMBER_OF_MARKERS,
        image_path      = label,
        corners_list    = kept_corners,
    )

def process_folder(folder: str | Path, visualize: bool = False,
                   verbose: bool = True) -> list[Optional[WheelPose]]:
    folder = Path(folder)
    exts   = {".jpg", ".jpeg", ".png", ".bmp"}
    paths  = sorted(p for p in folder.iterdir() if p.suffix.lower() in exts)
    poses: list[Optional[WheelPose]] = []

    if visualize:
        cv2.namedWindow("Pose Preview", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Pose Preview", 800, 600)
        cv2.moveWindow("Pose Preview", 100, 100)

    n_detected = 0
    for p in paths:
        pose = process_image(p, verbose=verbose)
        poses.append(pose)
        if pose is not None:
            n_detected += 1
        if visualize:
            img = cv2.imread(str(p))
            annotated = _annotate_single(img, pose) if pose is not None else img
            cv2.imshow("Pose Preview", annotated)
            if cv2.waitKey(1) == 27:
                break

    if visualize:
        cv2.destroyAllWindows()

    print(f"  {folder}: detected markers in {n_detected}/{len(paths)} frames")
    return poses


# ── Pose distance: angle-vector diff ─────────────────────────────────────────

def _angle_diff_scalar(a: float, b: float) -> float:
    """Shortest angular distance [0, 180]."""
    d = abs(a - b) % 360
    return min(d, 360 - d)


def pose_distance(p: WheelPose, q: WheelPose) -> tuple[float, int]:
    """
    Mean angular difference over markers visible in BOTH poses.
    Returns (mean_diff_deg, n_shared).  If no shared markers, returns (180, 0).
    """
    shared_ids = set(p.marker_angles) & set(q.marker_angles)
    if not shared_ids:
        return 180.0, 0

    diffs = [_angle_diff_scalar(p.marker_angles[mid], q.marker_angles[mid])
             for mid in shared_ids]
    return float(np.mean(diffs)), len(shared_ids)


# ── Matching (Hungarian, same structure as before) ────────────────────────────

def match_runs(left_poses:  list[Optional[WheelPose]],
               right_poses: list[Optional[WheelPose]],
               max_angle:   float = MATCH_MAX_ANGLE_DEG,
               min_conf:    float = 1 / len(MARKER_IDS)) -> list[MatchResult]:

    valid_l = [(i, p) for i, p in enumerate(left_poses)
               if p is not None and p.confidence >= min_conf]
    valid_r = [(j, p) for j, p in enumerate(right_poses)
               if p is not None and p.confidence >= min_conf]

    if not valid_l or not valid_r:
        return []

    # Build cost matrix using the per-marker angle vector distance
    D        = np.zeros((len(valid_l), len(valid_r)))
    N_shared = np.zeros_like(D, dtype=int)

    for li, (_, lp) in enumerate(valid_l):
        for ri, (_, rp) in enumerate(valid_r):
            D[li, ri], N_shared[li, ri] = pose_distance(lp, rp)

    from scipy.optimize import linear_sum_assignment
    row_ind, col_ind = linear_sum_assignment(D)

    results = []
    for li, ri in zip(row_ind, col_ind):
        diff     = D[li, ri]
        n_shared = N_shared[li, ri]
        if diff > max_angle or n_shared == 0:
            continue
        orig_li, lpose = valid_l[li]
        orig_ri, rpose = valid_r[ri]
        results.append(MatchResult(
            left_idx   = orig_li,
            right_idx  = orig_ri,
            left_path  = lpose.image_path,
            right_path = rpose.image_path,
            angle_diff = round(diff, 3),
            n_shared   = n_shared,
            score      = round(
                (1 - diff / max_angle)
                * (n_shared / len(MARKER_IDS))   # reward more shared markers
                * min(lpose.confidence, rpose.confidence),
                4
            ),
        ))

    return sorted(results, key=lambda r: r.angle_diff)

# ── JSON export ───────────────────────────────────────────────────────────────
def export_json(matches: list[MatchResult], output: str | Path) -> None:
    def _default(o):
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        raise TypeError(f"Not serializable: {type(o)}")

    data = [vars(m) for m in matches]
    with open(output, "w") as f:
        json.dump(data, f, indent=2, default=_default)
    print(f"Saved {len(data)} matches → {output}")

# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Match wheel runs via ArUco poses")
    ap.add_argument("--left_folder",  default=r"sync_capture\depot_3kmph_001_left\wheel")
    ap.add_argument("--right_folder", default=r"sync_capture\depot_3kmph_002_left\wheel")
    ap.add_argument("--max-angle",    type=float, default=MATCH_MAX_ANGLE_DEG)
    ap.add_argument("--min-conf",     type=float, default=1/len(MARKER_IDS))
    ap.add_argument("--output",       default="matches.json")
    ap.add_argument("--no-vis",       action="store_true", help="Skip visualisation")
    args = ap.parse_args()

    print(f"\nLeft folder:  {args.left_folder}")
    left_poses  = process_folder(args.left_folder)

    print(f"\nRight folder: {args.right_folder}")
    right_poses = process_folder(args.right_folder)

    n_l = sum(p is not None for p in left_poses)
    n_r = sum(p is not None for p in right_poses)
    print(f"\nValid poses → left: {n_l}/{len(left_poses)}  right: {n_r}/{len(right_poses)}")

    matches = match_runs(left_poses, right_poses,
                         max_angle=args.max_angle, min_conf=args.min_conf)
    print(f"\nMatches found: {len(matches)}")
    for m in matches[:10]:
        print(f"  L[{m.left_idx:4d}] ↔ R[{m.right_idx:4d}]  "
              f"Δθ={m.angle_diff:6.2f}°  score={m.score:.3f}")

    export_json(matches, args.output)

    if not args.no_vis:
        exts       = {".jpg", ".jpeg", ".png", ".bmp"}
        left_paths  = sorted(p for p in Path(args.left_folder).iterdir()
                             if p.suffix.lower() in exts)
        right_paths = sorted(p for p in Path(args.right_folder).iterdir()
                             if p.suffix.lower() in exts)
        save_visualisations(matches, left_paths, right_paths, left_poses, right_poses)