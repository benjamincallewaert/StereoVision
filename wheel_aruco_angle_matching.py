from __future__ import annotations
import cv2
import cv2.aruco as aruco
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import json, os
from wheel_matching_utils import MARKER_IDS, MARKERS_OFFSET
# ── Configuration ────────────────────────────────────────────────────────────
                       # IDs on each wheel
  # degrees CW from 12-o'clock
MARKER_SIZE_MM  = 38.0                                 # physical side length

ARUCO_DICT   = aruco.getPredefinedDictionary(aruco.DICT_4X4_100)
ARUCO_PARAMS = aruco.DetectorParameters()
ARUCO_PARAMS.adaptiveThreshWinSizeMin   = 3
ARUCO_PARAMS.adaptiveThreshWinSizeMax   = 53
ARUCO_PARAMS.adaptiveThreshWinSizeStep  = 4
ARUCO_PARAMS.minMarkerPerimeterRate     = 0.01
ARUCO_PARAMS.maxMarkerPerimeterRate     = 4.0
ARUCO_PARAMS.polygonalApproxAccuracyRate= 0.08
ARUCO_PARAMS.cornerRefinementMethod     = aruco.CORNER_REFINE_SUBPIX
ARUCO_PARAMS.minCornerDistanceRate      = 0.01
ARUCO_PARAMS.minDistanceToBorder        = 2

MATCH_MAX_ANGLE_DEG = 10.0
VIS_FOLDER          = Path("aruco_wheel")

# ── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class WheelPose:
    wheel_angle_deg: float          # canonical wheel angle [0, 360)
    confidence:      float          # n_visible / len(MARKER_IDS)
    n_visible:       int
    marker_ids:      list[int]      # which IDs were seen
    image_path:      Optional[str] = None

@dataclass
class MatchResult:
    left_idx:   int
    right_idx:  int
    left_path:  Optional[str]
    right_path: Optional[str]
    angle_diff: float               # degrees, lower = better
    score:      float               # 0..1, higher = better

# ── Marker detection & WheelPose ─────────────────────────────────────────────
def _img_angle(corners: np.ndarray) -> float:
    """In-plane yaw of a marker: angle of corner-0 → corner-1 vector (degrees)."""
    dx, dy = corners[1] - corners[0]
    return float(np.degrees(np.arctan2(dy, dx)))


def process_image(src: np.ndarray | str | Path,
                  label: Optional[str] = None) -> Optional[WheelPose]:
    """
    Detect ArUco markers and derive the canonical wheel angle.

    Each visible marker k contributes:
        wheel_angle = image_plane_angle(k) − MARKER_OFFSETS[k]

    A circular mean over all contributions gives the final angle.
    Returns None when no relevant markers are detected.
    """
    if isinstance(src, (str, Path)):
        label = label or str(src)
        img   = cv2.imread(str(src))
        if img is None:
            raise FileNotFoundError(src)
    else:
        img = src

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    corners_list, ids, _ = aruco.ArucoDetector(ARUCO_DICT, ARUCO_PARAMS).detectMarkers(gray)

    seen = []
    if ids is not None:
        for corners, mid in zip(corners_list, ids.flatten()):
            if mid in MARKER_IDS:
                seen.append((int(mid), corners[0]))   # corners[0] → shape (4,2)

    # ── print detections ──────────────────────────────────────────────────
    name = Path(label).name if label else "image"
    if seen:
        ids_str = ", ".join(f"ID{m}" for m, _ in seen)
        print(f"  {name}: found [{ids_str}]")
    else:
        print(f"  {name}: no markers detected")

    if not seen:
        return None

    # Circular mean of wheel-angle estimates
    estimates = [(_img_angle(c) - MARKERS_OFFSET[mid]) % 360 for mid, c in seen]
    rad       = np.radians(estimates)
    mean_deg  = float(np.degrees(np.arctan2(np.mean(np.sin(rad)),
                                            np.mean(np.cos(rad)))) % 360)

    return WheelPose(
        wheel_angle_deg = mean_deg,
        confidence      = len(seen) / len(MARKER_IDS),
        n_visible       = len(seen),
        marker_ids      = [m for m, _ in seen],
        image_path      = label,
    )


def process_folder(folder: str | Path) -> list[Optional[WheelPose]]:
    folder = Path(folder)
    exts   = {".jpg", ".jpeg", ".png", ".bmp"}
    paths  = sorted(p for p in folder.iterdir() if p.suffix.lower() in exts)
    poses  = []
    for p in paths:
        try:   poses.append(process_image(p))
        except Exception as e:
            print(f"  [WARN] {p.name}: {e}")
            poses.append(None)
    return poses

# ── Matching ─────────────────────────────────────────────────────────────────

def _angle_diff(a: float, b: float) -> float:
    """Shortest angular distance [0, 180] between two angles in [0, 360)."""
    d = abs(a - b) % 360
    return min(d, 360 - d)


def match_runs(left_poses:  list[Optional[WheelPose]],
               right_poses: list[Optional[WheelPose]],
               max_angle:   float = MATCH_MAX_ANGLE_DEG,
               min_conf:    float = 1 / len(MARKER_IDS)) -> list[MatchResult]:
    """
    Optimal one-to-one assignment via the Hungarian algorithm on the angular
    distance matrix, then filter by max_angle threshold.

    Compared to greedy assignment, this minimises total angular error across
    all matched pairs rather than just picking the locally best pair first.
    """
    valid_l = [(i, p) for i, p in enumerate(left_poses)
               if p is not None and p.confidence >= min_conf]
    valid_r = [(j, p) for j, p in enumerate(right_poses)
               if p is not None and p.confidence >= min_conf]

    if not valid_l or not valid_r:
        return []

    # Build distance matrix
    D = np.array([[_angle_diff(lp.wheel_angle_deg, rp.wheel_angle_deg)
                   for _, rp in valid_r]
                  for _, lp in valid_l])

    # Hungarian assignment (scipy wraps LAPJV)
    from scipy.optimize import linear_sum_assignment
    row_ind, col_ind = linear_sum_assignment(D)

    results = []
    for li, ri in zip(row_ind, col_ind):
        diff = D[li, ri]
        if diff > max_angle:
            continue
        orig_li, lpose = valid_l[li]
        orig_ri, rpose = valid_r[ri]
        results.append(MatchResult(
            left_idx   = orig_li,
            right_idx  = orig_ri,
            left_path  = lpose.image_path,
            right_path = rpose.image_path,
            angle_diff = round(diff, 3),
            score      = round((1 - diff / max_angle) * min(lpose.confidence,
                                                             rpose.confidence), 4),
        ))

    return sorted(results, key=lambda r: r.angle_diff)

# ── Visualisation ─────────────────────────────────────────────────────────────

_COLOURS = {1: (0, 220, 80), 2: (255, 165, 0), 3: (80, 120, 255), 4: (200, 60, 200)}

def draw_detections(img: np.ndarray, pose: Optional[WheelPose]) -> np.ndarray:
    out = img.copy()
    if pose is None:
        cv2.putText(out, "No markers", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        return out

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    corners_list, ids, _ = aruco.ArucoDetector(ARUCO_DICT, ARUCO_PARAMS).detectMarkers(gray)

    if ids is not None:
        for corners, mid in zip(corners_list, ids.flatten()):
            if mid not in MARKER_IDS:
                continue
            c   = corners[0].astype(int)
            col = _COLOURS.get(mid, (200, 200, 200))
            cv2.polylines(out, [c], True, col, 2)
            cx, cy = c.mean(axis=0).astype(int)
            cv2.circle(out, tuple(cx), 4, col, -1) if False else None  # skip (handled below)
            cv2.circle(out, (int(c[:,0].mean()), int(c[:,1].mean())), 5, col, -1)
            cv2.putText(out, f"ID{mid}", (c[0,0]+6, c[0,1]-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)

    txt = (f"wheel={pose.wheel_angle_deg:.1f}deg  "
           f"vis={pose.n_visible}/{len(MARKER_IDS)}  conf={pose.confidence:.2f}")
    # shadow for readability
    for colour, thickness in [((30, 30, 30), 3), ((255, 255, 255), 1)]:
        cv2.putText(out, txt, (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, thickness)
    return out


def save_visualisations(matches:     list[MatchResult],
                         left_paths:  list[Path],
                         right_paths: list[Path],
                         left_poses:  list[Optional[WheelPose]],
                         right_poses: list[Optional[WheelPose]]) -> None:
    vis_l = VIS_FOLDER / "left"
    vis_r = VIS_FOLDER / "right"
    vis_l.mkdir(parents=True, exist_ok=True)
    vis_r.mkdir(parents=True, exist_ok=True)

    for m in matches:
        for idx, paths, poses, subdir in [
            (m.left_idx,  left_paths,  left_poses,  vis_l),
            (m.right_idx, right_paths, right_poses, vis_r),
        ]:
            if idx >= len(paths):
                continue
            img = cv2.imread(str(paths[idx]))
            if img is not None:
                ann = draw_detections(img, poses[idx])
                cv2.imwrite(str(subdir / paths[idx].name), ann)

    print(f"Visualisations saved → {VIS_FOLDER}/")

# ── JSON export ───────────────────────────────────────────────────────────────

def export_json(matches: list[MatchResult], output: str | Path) -> None:
    data = [vars(m) for m in matches]
    with open(output, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved {len(data)} matches → {output}")

# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Match wheel runs via ArUco poses")
    ap.add_argument("--left_folder",  default=r"sync_capture\depot_3kmph_002_left\wheel")
    ap.add_argument("--right_folder", default=r"sync_capture\depot_3kmph_002_right\wheel")
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