import math
import cv2
import numpy as np
from typing import Optional
from dataclasses import dataclass
import cv2.aruco as aruco
from pathlib import Path

VIS_FOLDER          = Path("aruco_wheel")
_VIS_SCALE           = 1.5
NUMBER_OF_WHEELS    = 24
NUMBER_OF_MARKERS   = 4
MARKER_IDS          = [a + i for a in range(10, (NUMBER_OF_WHEELS*10+1), 10) for i in range(1, NUMBER_OF_MARKERS + 1)]
MARKERS_OFFSET      = {m: (m % 10 - 1) * 90 for m in MARKER_IDS}
MARKER_SIZE_MM      = 38.0

# ── ArUco dictionary ───────────────────────────────────────────────────────────
# MARKER_IDS run up to 244 (24 wheels x 4).  DICT_4X4_100 only encodes IDs 0-99,
# so 60 of the 96 markers (wheels 10-24) were silently *undetectable*.  The 4x4
# dictionaries are nested (the first 100 codes of DICT_4X4_250 are identical to
# DICT_4X4_100), so this is safe for existing markers and matches the capture
# default in grab_sync_footage.py.
ARUCO_DICT          = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)

# ── Detector parameters ─────────────────────────────────────────────────────
# Tuned for the intended rig: 4 m FOV on the 3840 px sensor (~1.04 mm/px), with
# 38 mm markers on the wheel flange -> ~37 px/side, ~148 px perimeter.
#   * minMarkerPerimeterRate = 0.025 -> min perimeter ~96 px (admits markers
#     >=~24 px/side) while rejecting the small noise quads that caused false IDs.
#     NOTE: the old depot footage was a much wider FOV with ~14 px markers; to
#     re-process THAT footage you must drop this back to ~0.008.
#   * adaptiveThreshWinSize: a short sweep (5..45 step 8) is enough and far
#     faster than the old 3..53 step 4 (13 windows) on a 10 MP frame.
#   * errorCorrectionRate / detectInvertedMarker kept sane: pushing them higher
#     made the detector hallucinate markers out of sensor noise, poisoning the
#     matching.  If you still miss real markers, raise errorCorrectionRate toward
#     0.8 — but watch the spurious-ID rate.
ARUCO_PARAMS        = aruco.DetectorParameters()
ARUCO_PARAMS.adaptiveThreshWinSizeMin    = 5
ARUCO_PARAMS.adaptiveThreshWinSizeMax    = 45
ARUCO_PARAMS.adaptiveThreshWinSizeStep   = 8
ARUCO_PARAMS.minMarkerPerimeterRate      = 0.025
ARUCO_PARAMS.maxMarkerPerimeterRate      = 4.0
ARUCO_PARAMS.polygonalApproxAccuracyRate = 0.06
ARUCO_PARAMS.cornerRefinementMethod      = aruco.CORNER_REFINE_SUBPIX
ARUCO_PARAMS.cornerRefinementWinSize     = 5
ARUCO_PARAMS.cornerRefinementMaxIterations = 40
ARUCO_PARAMS.minCornerDistanceRate       = 0.03
ARUCO_PARAMS.minDistanceToBorder         = 1
ARUCO_PARAMS.errorCorrectionRate         = 0.6   # OpenCV default; raise with care
ARUCO_PARAMS.detectInvertedMarker        = False # standard black-on-white markers

# Optional pre-detection upscale.  Counter-intuitively this *reduced* the true
# detection count and roughly doubled runtime on these frames (cubic upscaling
# smears the already-tiny markers), so it is OFF by default.  CLAHE contrast
# stretching is what actually helps with the underexposed frames.
ARUCO_UPSCALE       = 1

# Build the detector ONCE and reuse it.  (The old code constructed a fresh
# ArucoDetector on every single image — and sometimes twice per image.)
_DETECTOR           = aruco.ArucoDetector(ARUCO_DICT, ARUCO_PARAMS)
_CLAHE              = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

_COLOURS            = {1: (0, 220, 80), 2: (255, 165, 0), 3: (80, 120, 255), 4: (200, 60, 200)}
_ARROW_LEN          = 40   # pixels for the orientation arrow


def preprocess_for_aruco(img: np.ndarray, upscale: int = ARUCO_UPSCALE) -> np.ndarray:
    """Grey -> CLAHE (fix underexposure) -> optional upscale (help tiny markers)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    gray = _CLAHE.apply(gray)
    if upscale and upscale != 1:
        gray = cv2.resize(gray, None, fx=upscale, fy=upscale,
                          interpolation=cv2.INTER_CUBIC)
    return gray


def detect_markers(img: np.ndarray, upscale: int = ARUCO_UPSCALE):
    """
    Detect ArUco markers using the shared, tuned detector.

    Returns ``(corners_list, ids)`` where corner coordinates are expressed in
    the *original* image frame (the upscale factor is divided back out).
    """
    gray = preprocess_for_aruco(img, upscale)
    corners, ids, _ = _DETECTOR.detectMarkers(gray)
    if upscale and upscale != 1 and corners:
        corners = [c / float(upscale) for c in corners]
    return corners, ids

WHEEL_MAP = {
    11: 1, 12: 1, 13: 1, 14: 1,
    21: 2, 22: 2, 23: 2, 24: 2,
}

WHEEL_IDS = {
    1: {11, 12, 13, 14},
    2: {21, 22, 23, 24},
}

WINDOW_SIZE = 10  # frames to keep wheel active

# ── Data classes ──────────────────────────────────────────────────────────────
@dataclass
class MatchResult:
    left_idx:   int
    right_idx:  int
    left_path:  Optional[str]
    right_path: Optional[str]
    angle_diff: float          # mean angular difference over shared markers
    n_shared:   int            # how many marker IDs were visible in both
    score:      float          # 0..1

@dataclass
class WheelPose:
    """
    Stores the raw image-plane angle for every visible marker.
    No MARKER_OFFSETS subtraction — we keep the absolute angles.
    """
    marker_angles: dict[int, float]   # {marker_id: img_plane_angle_deg}
    n_visible: int
    confidence: float                 # n_visible / len(MARKER_IDS)
    image_path: Optional[str] = None
    corners_list: Optional[list] = None

    @property
    def marker_ids(self) -> list[int]:
        return list(self.marker_angles.keys())
    
def _draw_arrow(img: np.ndarray, origin: tuple[int, int],
                angle_deg: float, colour: tuple, length: int = _ARROW_LEN) -> None:
    """Draw an arrow from origin in the direction of angle_deg (image-plane CW from right)."""
    rad = math.radians(angle_deg)
    tip = (
        int(origin[0] + length * math.cos(rad)),
        int(origin[1] + length * math.sin(rad)),
    )
    cv2.arrowedLine(img, origin, tip, (20, 20, 20), 5, tipLength=0.35)  # shadow
    cv2.arrowedLine(img, origin, tip, colour,       2, tipLength=0.35)


def _annotate_single(img: np.ndarray, pose: Optional[WheelPose]) -> np.ndarray:
    out = img.copy()

    def T(x): return int(x * _VIS_SCALE)

    if pose is None:
        cv2.putText(out, "No markers", (T(20), T(40)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2 * _VIS_SCALE, (0, 0, 255), 3)
        return out

    for corners, mid in zip(pose.corners_list, pose.marker_ids):
        mid = int(mid)
        if mid not in MARKER_IDS:
            continue

        c = corners[0].astype(int)
        col = _COLOURS.get(mid, (200, 200, 200))

        cv2.polylines(out, [c], True, col, T(2))
        c0 = tuple(c[0])
        cv2.circle(out, c0, T(5), col, -1)

        angle = pose.marker_angles.get(mid)
        if angle is not None:
            _draw_arrow(out, c0, angle, col)

            rad = math.radians(angle)
            lx = int(c0[0] + (_ARROW_LEN + 14) * math.cos(rad))
            ly = int(c0[1] + (_ARROW_LEN + 14) * math.sin(rad))

            label = f"{angle:.1f}°"
            cv2.putText(out, label, (lx, ly),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9 * _VIS_SCALE, (0, 0, 0), T(3))
            cv2.putText(out, label, (lx, ly),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9 * _VIS_SCALE, col, T(1))

        cv2.putText(out, f"ID{mid}", (c[0,0] + T(6), c[0,1] - T(8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9 * _VIS_SCALE, (0, 0, 0), T(3))
        cv2.putText(out, f"ID{mid}", (c[0,0] + T(6), c[0,1] - T(8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9 * _VIS_SCALE, col, T(1))

    summary = (f"vis={pose.n_visible}/{len(MARKER_IDS)}  conf={pose.confidence:.2f}")
    cv2.putText(out, summary, (T(16), T(36)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8 * _VIS_SCALE, (0,0,0), T(3))
    cv2.putText(out, summary, (T(16), T(36)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8 * _VIS_SCALE, (255,255,255), T(1))

    return out


def _make_side_by_side(left_img:  np.ndarray,
                       right_img: np.ndarray,
                       match:     MatchResult,
                       left_pose: Optional[WheelPose],
                       right_pose: Optional[WheelPose]) -> np.ndarray:
    """
    Stack two annotated images horizontally with a divider and a header bar
    showing the match statistics.
    """
    ann_l = _annotate_single(left_img,  left_pose)
    ann_r = _annotate_single(right_img, right_pose)

    # ── normalise heights ─────────────────────────────────────────────────────
    h     = max(ann_l.shape[0], ann_r.shape[0])
    def _pad_h(im, target_h):
        if im.shape[0] == target_h:
            return im
        pad = np.zeros((target_h - im.shape[0], im.shape[1], 3), dtype=np.uint8)
        return np.vstack([im, pad])

    ann_l = _pad_h(ann_l, h)
    ann_r = _pad_h(ann_r, h)

    # ── divider ───────────────────────────────────────────────────────────────
    divider = np.full((h, 6, 3), 80, dtype=np.uint8)

    # ── header bar ───────────────────────────────────────────────────────────
    total_w = ann_l.shape[1] + 6 + ann_r.shape[1]
    header  = np.zeros((52, total_w, 3), dtype=np.uint8)

    left_name  = Path(match.left_path).name  if match.left_path  else f"L[{match.left_idx}]"
    right_name = Path(match.right_path).name if match.right_path else f"R[{match.right_idx}]"
    header_txt = (f"{left_name}  ↔  {right_name}"
                  f"   |   Δθ={match.angle_diff:.2f}°"
                  f"   shared={match.n_shared}/{len(MARKER_IDS)}"
                  f"   score={match.score:.3f}")

    cv2.putText(header, header_txt, (12, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 180, 180), 1)

    panel = np.hstack([ann_l, divider, ann_r])
    return np.vstack([header, panel])


def save_visualisations(matches:     list[MatchResult],
                        left_paths:  list[Path],
                        right_paths: list[Path],
                        left_poses:  list[Optional[WheelPose]],
                        right_poses: list[Optional[WheelPose]],
                        out_folder:  Path = VIS_FOLDER / "pairs") -> None:
    """
    Save one side-by-side PNG per match.
    Filename encodes rank, both source names, and the angle difference.
    """
    out_folder.mkdir(parents=True, exist_ok=True)

    for rank, m in enumerate(matches):
        # ── load images ───────────────────────────────────────────────────────
        if m.left_idx >= len(left_paths) or m.right_idx >= len(right_paths):
            continue
        img_l = cv2.imread(str(left_paths[m.left_idx]))
        img_r = cv2.imread(str(right_paths[m.right_idx]))
        if img_l is None or img_r is None:
            continue

        panel = _make_side_by_side(img_l, img_r, m,
                                   left_poses[m.left_idx],
                                   right_poses[m.right_idx])

        # ── filename ──────────────────────────────────────────────────────────
        l_stem = Path(left_paths[m.left_idx]).stem
        r_stem = Path(right_paths[m.right_idx]).stem
        fname  = f"{rank:03d}_{l_stem}_vs_{r_stem}_dtheta{m.angle_diff:.2f}.png"
        cv2.imwrite(str(out_folder / fname), panel)

    print(f"Saved {len(matches)} pair visualisations → {out_folder}/")