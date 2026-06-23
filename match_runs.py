"""
match_runs.py
=============
Match left/right stereo frame pairs from **run 1** to those from **run 2**
using the ArUco-marker positions on the inspection wheel as a rotational
position key.

Wheel geometry
--------------
Each wheel carries three ArUco markers (IDs 0, 1, 2) at fixed angular
positions relative to each other.  As the rig travels along the rail the
wheel rotates; two frames from different runs are at the *same rail position*
when the wheel angle is the same.

Algorithm
---------
1. Detect all visible markers in every wheel image.
2. Estimate the wheel centre as the centroid of all detected marker centroids.
3. Compute each marker's bearing from the wheel centre (degrees, [0 360)).
4. **Inter-marker offset calibration** – from every frame where ≥ 2 markers
   are visible, record the angle difference between pairs.  The median across
   all such frames gives stable offsets (e.g. offset[1] = median(a1 – a0)).
   These offsets let us recover a "marker-0-equivalent" canonical angle even
   when marker 0 is occluded.
5. Build an N × M cost matrix of circular angular distances between every
   run-1 and run-2 canonical angle.
6. Solve the **assignment problem** (scipy linear_sum_assignment / Hungarian
   method) for globally optimal 1-to-1 matching.
7. Write *matches.csv* and side-by-side preview montages.

Directory layout expected
-------------------------
run1/
    wheel/  wheel_0000.png …
    left/   left_0000.png  …
    right/  right_0000.png …
run2/
    wheel/  wheel_0000.png …
    left/   left_0000.png  …
    right/  right_0000.png …

Usage
-----
python match_runs.py --run1 path/to/run1 --run2 path/to/run2 --out matches/
python match_runs.py --run1 run1 --run2 run2 --out out --max-angle-diff 15 --preview-n 20
"""

from __future__ import annotations

import argparse
import csv
import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from scipy.optimize import linear_sum_assignment



# ──────────────────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class WheelState:
    """
    All wheel-position information extracted from one wheel-camera frame.

    Attributes
    ----------
    frame_idx:
        Index within its run (0-based).
    image_path:
        Absolute path to the wheel image on disk.
    detected_ids:
        List of ArUco marker IDs that were successfully detected.
    marker_centers:
        ``{id: (cx, cy)}`` pixel centroid of each detected marker.
    wheel_center:
        ``(wx, wy)`` estimated wheel rotation centre (mean of marker centroids).
    raw_angles_deg:
        ``{id: angle}`` bearing of each marker from *wheel_center* in [0, 360).
    canonical_angle_deg:
        Single representative wheel angle (degrees), normalised so it is
        equivalent to "where marker 0 would be" even when marker 0 is occluded.
        ``None`` if no markers were detected.
    confidence:
        0.0 – 1.0.  1.0 = all three markers visible; lower when fewer markers
        or when the canonical angle had to be inferred from an offset estimate.
    """
    frame_idx:           int
    image_path:          Path
    detected_ids:        list[int]               = field(default_factory=list)
    marker_centers:      dict[int, tuple]        = field(default_factory=dict)
    wheel_center:        tuple[float, float]     = (0.0, 0.0)
    raw_angles_deg:      dict[int, float]        = field(default_factory=dict)
    canonical_angle_deg: float | None            = None
    confidence:          float                   = 0.0


@dataclass
class FrameMatch:
    """One matched pair (run-1 frame ↔ run-2 frame)."""
    run1_idx:              int
    run2_idx:              int
    run1_angle_deg:        float
    run2_angle_deg:        float
    angle_diff_deg:        float
    run1_markers:          list[int]
    run2_markers:          list[int]
    run1_confidence:       float
    run2_confidence:       float
    match_confidence:      float


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _wrap180(diff: float) -> float:
    """Wrap an angle difference into (−180, +180]."""
    return ((diff + 180.0) % 360.0) - 180.0


def _angular_dist(a: float, b: float) -> float:
    """Smallest absolute angular distance between two angles [0 360)."""
    return abs(_wrap180(a - b))


def _marker_centroid(corners: np.ndarray) -> tuple[float, float]:
    """Mean of the four corners of one ArUco detection result."""
    pts = corners[0]          # shape (4, 2)
    return float(pts[:, 0].mean()), float(pts[:, 1].mean())


def _sorted_image_paths(directory: Path) -> list[Path]:
    """Return all .png / .jpg images in *directory*, sorted by name."""
    paths = sorted(
        p for p in directory.iterdir()
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tiff", ".bmp"}
    )
    if not paths:
        raise FileNotFoundError(f"No images found in {directory}")
    return paths


# ──────────────────────────────────────────────────────────────────────────────
# Offset calibrator
# ──────────────────────────────────────────────────────────────────────────────

class OffsetCalibrator:
    """
    Estimate stable inter-marker angular offsets from a list of WheelStates.

    For every frame that has ≥ 2 detected markers we record
    ``angle[id_k] – angle[id_0]`` (where id_0 is the lowest available ID).
    The median across all such observations is used as the offset so that any
    single visible marker can be normalised to a "marker-0-equivalent" angle.

    Parameters
    ----------
    expected_ids:
        The three marker IDs on the wheel (default ``(0, 1, 2)``).
    """

    def __init__(self, expected_ids: tuple[int, ...] = (0, 1, 2)) -> None:
        self._expected = set(expected_ids)
        # offset[k] = angle[k] - angle[0]  (canonical reference = marker 0)
        self._samples: dict[int, list[float]] = {k: [] for k in expected_ids}
        self._offsets: dict[int, float] = {k: 0.0 for k in expected_ids}
        self._calibrated = False

    def update(self, state: WheelState) -> None:
        """Feed one WheelState into the offset accumulator."""
        if len(state.detected_ids) < 2:
            return
        angles = state.raw_angles_deg
        id0 = min(state.detected_ids)          # lowest available ID as reference
        a0  = angles[id0]
        # Record offset for every other marker relative to id0, then relate
        # everything back to global marker-0.
        for kid in state.detected_ids:
            if kid == id0:
                continue
            raw_offset = _wrap180(angles[kid] - a0)
            # We want offset relative to global marker-0; if id0 ≠ 0 we adjust.
            global_offset = raw_offset + self._offsets.get(id0, 0.0)
            self._samples[kid].append(global_offset)

    def finalise(self) -> dict[int, float]:
        """
        Compute and return ``{marker_id: offset_deg}``.

        Must be called after all WheelStates have been fed via ``update()``.
        """
        self._offsets = {}
        for kid, samples in self._samples.items():
            if kid == 0 or not samples:
                self._offsets[kid] = 0.0
            else:
                self._offsets[kid] = float(np.median(samples))
        self._calibrated = True
        return dict(self._offsets)

    @property
    def offsets(self) -> dict[int, float]:
        return dict(self._offsets)

    def canonical(self, state: WheelState) -> float | None:
        """
        Return the canonical (marker-0-equivalent) wheel angle for *state*.

        Returns ``None`` if no markers were detected.
        """
        if not state.detected_ids:
            return None
        # Prefer marker 0; fall back to lowest detected ID.
        best_id = 0 if 0 in state.detected_ids else min(state.detected_ids)
        raw = state.raw_angles_deg[best_id]
        canonical = (raw - self._offsets.get(best_id, 0.0)) % 360.0
        return float(canonical)

    def confidence(self, state: WheelState) -> float:
        """
        Return a confidence in [0, 1] for the canonical angle estimate.

        Full confidence (1.0) requires all expected markers detected AND
        the calibration has been finalised from ≥ 2 markers per frame.
        """
        if not state.detected_ids:
            return 0.0
        frac_visible = len(state.detected_ids) / max(len(self._expected), 1)
        calib_bonus  = 0.2 if (self._calibrated and 0 in state.detected_ids) else 0.0
        return min(1.0, frac_visible + calib_bonus)


# ──────────────────────────────────────────────────────────────────────────────
# Wheel-directory analyser
# ──────────────────────────────────────────────────────────────────────────────

def analyse_wheel_dir(
    wheel_dir:     Path,
    aruco_dict_id: int           = cv2.aruco.DICT_4X4_50,
    expected_ids:  tuple[int, ...] = (0, 1, 2),
    verbose:       bool          = True,
) -> tuple[list[WheelState], OffsetCalibrator]:
    """
    Detect ArUco markers in every image under *wheel_dir* and return
    a list of :class:`WheelState` objects plus a fitted
    :class:`OffsetCalibrator`.

    The calibrator is fitted on the detected states so it is ready to
    produce canonical angles immediately.

    Parameters
    ----------
    wheel_dir:
        Path to the ``wheel/`` sub-directory of a run.
    aruco_dict_id:
        Any ``cv2.aruco.DICT_*`` constant (must match what was used during
        capture).
    expected_ids:
        Marker IDs present on the wheel.
    verbose:
        Print per-frame detection summary.
    """
    wheel_dir = Path(wheel_dir)
    paths     = _sorted_image_paths(wheel_dir)

    aruco_dict    = cv2.aruco.getPredefinedDictionary(aruco_dict_id)
    aruco_params  = cv2.aruco.DetectorParameters()
    detector      = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

    calibrator = OffsetCalibrator(expected_ids)
    states: list[WheelState] = []

    n_no_detect = 0

    for idx, path in enumerate(paths):
        img  = cv2.imread(str(path))
        if img is None:
            warnings.warn(f"Could not read image: {path}")
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        corners_list, ids, _ = detector.detectMarkers(gray)

        state = WheelState(frame_idx=idx, image_path=path)

        if ids is not None and len(ids) > 0:
            flat_ids = ids.flatten().tolist()
            # Keep only expected IDs (ignore spurious detections)
            for corner, mid in zip(corners_list, flat_ids):
                if mid in expected_ids:
                    cx, cy = _marker_centroid(corner)
                    state.detected_ids.append(mid)
                    state.marker_centers[mid] = (cx, cy)

            if state.detected_ids:
                # Wheel centre = centroid of all detected marker centres
                xs = [v[0] for v in state.marker_centers.values()]
                ys = [v[1] for v in state.marker_centers.values()]
                wx, wy = float(np.mean(xs)), float(np.mean(ys))
                state.wheel_center = (wx, wy)

                # Bearing of each marker from wheel centre
                for mid, (cx, cy) in state.marker_centers.items():
                    angle = np.degrees(np.arctan2(cy - wy, cx - wx)) % 360.0
                    state.raw_angles_deg[mid] = float(angle)

                calibrator.update(state)
        else:
            n_no_detect += 1

        states.append(state)

    # Finalise calibration, then assign canonical angles and confidence
    calibrator.finalise()
    n_valid = 0
    for s in states:
        s.canonical_angle_deg = calibrator.canonical(s)
        s.confidence          = calibrator.confidence(s)
        if s.canonical_angle_deg is not None:
            n_valid += 1

    offsets = calibrator.offsets
    if verbose:
        print(
            f"  {wheel_dir}: {len(states)} frames | "
            f"{n_valid} with angle | "
            f"{n_no_detect} no detection | "
            f"offsets → {offsets}"
        )

    return states, calibrator


# ──────────────────────────────────────────────────────────────────────────────
# Cost matrix and matching
# ──────────────────────────────────────────────────────────────────────────────

def build_cost_matrix(
    states1: list[WheelState],
    states2: list[WheelState],
) -> np.ndarray:
    """
    Build an N × M matrix of circular angular distances.

    Rows correspond to run-1 frames, columns to run-2 frames.
    Frames with no detected markers are assigned ``np.inf`` in all cells.
    """
    n = len(states1)
    m = len(states2)
    cost = np.full((n, m), np.inf, dtype=np.float64)

    for i, s1 in enumerate(states1):
        if s1.canonical_angle_deg is None:
            continue
        for j, s2 in enumerate(states2):
            if s2.canonical_angle_deg is None:
                continue
            cost[i, j] = _angular_dist(s1.canonical_angle_deg,
                                        s2.canonical_angle_deg)
    return cost


def match_hungarian(
    states1:        list[WheelState],
    states2:        list[WheelState],
    cost:           np.ndarray,
    max_angle_diff: float = 10.0,
) -> list[FrameMatch]:
    """
    Globally optimal 1-to-1 matching via the Hungarian algorithm.

    Pairs whose angular distance exceeds *max_angle_diff* are excluded.
    Requires scipy.
    """

    # Replace inf with a large finite number so the solver can run
    finite_cost = np.where(np.isinf(cost), 1e9, cost)
    row_ind, col_ind = linear_sum_assignment(finite_cost)

    matches: list[FrameMatch] = []
    for r, c in zip(row_ind, col_ind):
        diff = cost[r, c]
        if diff > max_angle_diff or np.isinf(diff):
            continue
        s1, s2 = states1[r], states2[c]
        matches.append(FrameMatch(
            run1_idx=r,
            run2_idx=c,
            run1_angle_deg=s1.canonical_angle_deg,
            run2_angle_deg=s2.canonical_angle_deg,
            angle_diff_deg=diff,
            run1_markers=sorted(s1.detected_ids),
            run2_markers=sorted(s2.detected_ids),
            run1_confidence=s1.confidence,
            run2_confidence=s2.confidence,
            match_confidence=min(s1.confidence, s2.confidence) * max(0.0, 1.0 - diff / 180.0),
        ))
    return sorted(matches, key=lambda m: m.run1_idx)


def match_nearest_neighbour(
    states1:        list[WheelState],
    states2:        list[WheelState],
    cost:           np.ndarray,
    max_angle_diff: float = 10.0,
) -> list[FrameMatch]:
    """
    For every run-1 frame find the closest run-2 frame (may be many-to-one).

    Used as a fallback when scipy is unavailable, or to supplement the
    Hungarian results for inspection purposes.
    """
    matches: list[FrameMatch] = []
    for r, s1 in enumerate(states1):
        row = cost[r]
        if np.all(np.isinf(row)):
            continue
        c    = int(np.argmin(row))
        diff = cost[r, c]
        if diff > max_angle_diff or np.isinf(diff):
            continue
        s2 = states2[c]
        matches.append(FrameMatch(
            run1_idx=r,
            run2_idx=c,
            run1_angle_deg=s1.canonical_angle_deg,
            run2_angle_deg=s2.canonical_angle_deg,
            angle_diff_deg=diff,
            run1_markers=sorted(s1.detected_ids),
            run2_markers=sorted(s2.detected_ids),
            run1_confidence=s1.confidence,
            run2_confidence=s2.confidence,
            match_confidence=min(s1.confidence, s2.confidence) * max(0.0, 1.0 - diff / 180.0),
        ))
    return sorted(matches, key=lambda m: m.run1_idx)


# ──────────────────────────────────────────────────────────────────────────────
# Output helpers
# ──────────────────────────────────────────────────────────────────────────────
def save_csv(
    matches:  list[FrameMatch],
    out_path: Path,
) -> None:
    """Write matches.csv with one row per matched pair."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run1_frame", "run2_frame",
        "run1_angle_deg", "run2_angle_deg", "angle_diff_deg",
        "run1_markers", "run2_markers",
        "run1_confidence", "run2_confidence", "match_confidence",
    ]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for m in matches:
            writer.writerow({
                "run1_frame":       m.run1_idx,
                "run2_frame":       m.run2_idx,
                "run1_angle_deg":   f"{m.run1_angle_deg:.2f}",
                "run2_angle_deg":   f"{m.run2_angle_deg:.2f}",
                "angle_diff_deg":   f"{m.angle_diff_deg:.2f}",
                "run1_markers":     str(m.run1_markers),
                "run2_markers":     str(m.run2_markers),
                "run1_confidence":  f"{m.run1_confidence:.3f}",
                "run2_confidence":  f"{m.run2_confidence:.3f}",
                "match_confidence": f"{m.match_confidence:.3f}",
            })
    print(f"  Saved {len(matches)} matches → {out_path}")


def save_json(
    matches:    list[FrameMatch],
    states1:    list[WheelState],
    states2:    list[WheelState],
    offsets1:   dict,
    offsets2:   dict,
    out_path:   Path,
) -> None:
    """Write a machine-readable JSON summary (matches + metadata)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "run1_frames_total":  len(states1),
        "run2_frames_total":  len(states2),
        "matches_count":      len(matches),
        "run1_marker_offsets_deg": offsets1,
        "run2_marker_offsets_deg": offsets2,
        "matches": [
            {
                "run1_frame":       m.run1_idx,
                "run2_frame":       m.run2_idx,
                "run1_angle_deg":   round(m.run1_angle_deg, 3),
                "run2_angle_deg":   round(m.run2_angle_deg, 3),
                "angle_diff_deg":   round(m.angle_diff_deg, 3),
                "run1_markers":     m.run1_markers,
                "run2_markers":     m.run2_markers,
                "match_confidence": round(m.match_confidence, 4),
            }
            for m in matches
        ],
    }
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved JSON  → {out_path}")


def _annotate_wheel(img: np.ndarray, state: WheelState, label: str) -> np.ndarray:
    """
    Draw marker centroids, wheel centre, and canonical angle on a copy
    of *img*.
    """
    out = img.copy()

    COLOURS = {0: (0, 255, 0), 1: (0, 165, 255), 2: (255, 0, 128)}

    for mid, (cx, cy) in state.marker_centers.items():
        colour = COLOURS.get(mid, (200, 200, 200))
        cv2.circle(out, (int(cx), int(cy)), 8, colour, -1)
        cv2.putText(out, f"ID{mid}", (int(cx) + 10, int(cy) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)

    if state.detected_ids:
        wx, wy = int(state.wheel_center[0]), int(state.wheel_center[1])
        cv2.drawMarker(out, (wx, wy), (255, 255, 255),
                       cv2.MARKER_CROSS, 20, 2)

    angle_txt = (
        f"{state.canonical_angle_deg:.1f} deg"
        if state.canonical_angle_deg is not None else "no detection"
    )
    conf_txt  = f"conf={state.confidence:.2f}"
    cv2.putText(out, label, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(out, angle_txt, (10, 56),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2)
    cv2.putText(out, conf_txt,  (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)
    return out


def _load_side_image(run_dir: Path, sub: str, idx: int) -> np.ndarray | None:
    """
    Load image number *idx* from ``<run_dir>/<sub>/``.

    Returns ``None`` (instead of raising) if the file does not exist.
    """
    try:
        paths = _sorted_image_paths(run_dir / sub)
        if idx >= len(paths):
            return None
        img = cv2.imread(str(paths[idx]))
        return img
    except FileNotFoundError:
        return None


def _stack_pair(img1: np.ndarray | None, img2: np.ndarray | None,
                label1: str, label2: str,
                target_w: int = 1280) -> np.ndarray:
    """
    Place *img1* and *img2* side by side, resized to fit *target_w* pixels
    wide, with small text labels.
    """
    PLACEHOLDER_H = 200

    def _prep(img: np.ndarray | None, label: str) -> np.ndarray:
        if img is None:
            ph = np.zeros((PLACEHOLDER_H, target_w // 2, 3), np.uint8)
            cv2.putText(ph, f"[{label} – missing]", (10, PLACEHOLDER_H // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)
            return ph
        # Convert mono/uint16 to colour
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.dtype == np.uint16:
            norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
            img  = cv2.cvtColor(norm.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        h, w = img.shape[:2]
        new_w = target_w // 2
        new_h = int(h * new_w / w)
        return cv2.resize(img, (new_w, new_h))

    left_img  = _prep(img1, label1)
    right_img = _prep(img2, label2)

    # Pad heights to match
    h1, h2 = left_img.shape[0], right_img.shape[0]
    max_h   = max(h1, h2)
    if h1 < max_h:
        left_img  = np.pad(left_img,  ((0, max_h - h1), (0, 0), (0, 0)))
    if h2 < max_h:
        right_img = np.pad(right_img, ((0, max_h - h2), (0, 0), (0, 0)))

    return np.hstack([left_img, right_img])


def save_previews(
    matches:   list[FrameMatch],
    states1:   list[WheelState],
    states2:   list[WheelState],
    run1_dir:  Path,
    run2_dir:  Path,
    out_dir:   Path,
    n:         int   = 20,
    target_w:  int   = 1280,
) -> None:
    """
    Save side-by-side preview montages for the *n* highest-confidence matches.

    Each preview is a three-row image:
        Row 0 – wheel images (annotated with marker positions & angle)
        Row 1 – left stereo images
        Row 2 – right stereo images
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Sort by confidence descending, take top-n
    ranked = sorted(matches, key=lambda m: m.match_confidence, reverse=True)[:n]

    for rank, m in enumerate(ranked):
        s1 = states1[m.run1_idx]
        s2 = states2[m.run2_idx]

        # ── Wheel row ─────────────────────────────────────────────────────────
        w1_img = cv2.imread(str(s1.image_path))
        w2_img = cv2.imread(str(s2.image_path))
        if w1_img is not None:
            w1_img = _annotate_wheel(w1_img, s1, f"Run1 fr{m.run1_idx}")
        if w2_img is not None:
            w2_img = _annotate_wheel(w2_img, s2, f"Run2 fr{m.run2_idx}")

        wheel_row = _stack_pair(w1_img, w2_img,
                                f"run1 wheel {m.run1_idx}",
                                f"run2 wheel {m.run2_idx}",
                                target_w)

        # ── Left stereo row ───────────────────────────────────────────────────
        l1 = _load_side_image(run1_dir, "left",  m.run1_idx)
        l2 = _load_side_image(run2_dir, "left",  m.run2_idx)
        left_row = _stack_pair(l1, l2,
                               f"run1 left {m.run1_idx}",
                               f"run2 left {m.run2_idx}",
                               target_w)

        # ── Right stereo row ──────────────────────────────────────────────────
        r1 = _load_side_image(run1_dir, "right", m.run1_idx)
        r2 = _load_side_image(run2_dir, "right", m.run2_idx)
        right_row = _stack_pair(r1, r2,
                                f"run1 right {m.run1_idx}",
                                f"run2 right {m.run2_idx}",
                                target_w)

        # ── Combine rows + header banner ──────────────────────────────────────
        # Pad rows to same width
        max_w = max(wheel_row.shape[1], left_row.shape[1], right_row.shape[1])
        def _pad_w(row, mw):
            if row.shape[1] < mw:
                return np.pad(row, ((0, 0), (0, mw - row.shape[1]), (0, 0)))
            return row
        wheel_row = _pad_w(wheel_row, max_w)
        left_row  = _pad_w(left_row,  max_w)
        right_row = _pad_w(right_row, max_w)

        # Divider lines
        div = np.full((4, max_w, 3), 80, np.uint8)

        banner = np.zeros((36, max_w, 3), np.uint8)
        banner_txt = (
            f"Match #{rank+1}  |  "
            f"Run1 fr{m.run1_idx} ({m.run1_angle_deg:.1f}°)  ↔  "
            f"Run2 fr{m.run2_idx} ({m.run2_angle_deg:.1f}°)  |  "
            f"Δ={m.angle_diff_deg:.2f}°  conf={m.match_confidence:.2f}"
        )
        cv2.putText(banner, banner_txt, (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 1)

        montage = np.vstack([banner, wheel_row, div, left_row, div, right_row])

        fname = out_dir / f"preview_{rank+1:03d}_r1f{m.run1_idx:04d}_r2f{m.run2_idx:04d}.jpg"
        cv2.imwrite(str(fname), montage, [cv2.IMWRITE_JPEG_QUALITY, 90])

    print(f"  Saved {min(n, len(matches))} preview montages → {out_dir}")


def save_angle_plot(
    states1:  list[WheelState],
    states2:  list[WheelState],
    matches:  list[FrameMatch],
    out_path: Path,
) -> None:
    """
    Save a simple angle-vs-frame plot as an image (no matplotlib required).
    Run-1 canonical angles are plotted in blue, run-2 in orange, and matched
    pairs are connected by thin green lines.
    """
    valid1 = [(s.frame_idx, s.canonical_angle_deg)
              for s in states1 if s.canonical_angle_deg is not None]
    valid2 = [(s.frame_idx, s.canonical_angle_deg)
              for s in states2 if s.canonical_angle_deg is not None]

    if not valid1 or not valid2:
        return

    W, H = 1600, 600
    margin_l, margin_r, margin_t, margin_b = 60, 20, 20, 40
    plot_w = W - margin_l - margin_r
    plot_h = H - margin_t - margin_b

    canvas = np.ones((H, W, 3), np.uint8) * 30

    # Map frame index → x,  angle → y
    max_idx = max(max(v[0] for v in valid1), max(v[0] for v in valid2))
    def fx(idx):   return margin_l + int(idx / max(max_idx, 1) * plot_w)
    def fy(angle): return margin_t + plot_h - int(angle / 360.0 * plot_h)

    # Grid lines at 90-degree intervals
    for a in (0, 90, 180, 270, 360):
        y = fy(a)
        cv2.line(canvas, (margin_l, y), (W - margin_r, y), (60, 60, 60), 1)
        cv2.putText(canvas, f"{a}°", (4, y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)

    # Match connector lines
    for m in matches:
        s1a = states1[m.run1_idx].canonical_angle_deg
        s2a = states2[m.run2_idx].canonical_angle_deg
        if s1a is None or s2a is None:
            continue
        p1 = (fx(m.run1_idx), fy(s1a))
        p2 = (fx(m.run2_idx), fy(s2a))
        alpha = max(0.2, m.match_confidence)
        colour = (int(0 * alpha), int(200 * alpha), int(80 * alpha))
        cv2.line(canvas, p1, p2, colour, 1, cv2.LINE_AA)

    # Run-2 trace (orange)
    for i in range(len(valid2) - 1):
        cv2.line(canvas,
                 (fx(valid2[i][0]),   fy(valid2[i][1])),
                 (fx(valid2[i+1][0]), fy(valid2[i+1][1])),
                 (0, 140, 255), 2, cv2.LINE_AA)

    # Run-1 trace (blue)
    for i in range(len(valid1) - 1):
        cv2.line(canvas,
                 (fx(valid1[i][0]),   fy(valid1[i][1])),
                 (fx(valid1[i+1][0]), fy(valid1[i+1][1])),
                 (255, 140, 0), 2, cv2.LINE_AA)

    # Legend
    cv2.putText(canvas, "Run 1", (margin_l, H - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 140, 0), 2)
    cv2.putText(canvas, "Run 2", (margin_l + 80, H - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 140, 255), 2)
    cv2.putText(canvas, "Matches", (margin_l + 160, H - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 80), 2)
    cv2.putText(canvas, "Wheel angle (deg) vs. frame index",
                (W // 2 - 180, H - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas)
    print(f"  Saved angle plot  → {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Top-level pipeline
# ──────────────────────────────────────────────────────────────────────────────

def run_matching(
    run1_dir:       Path,
    run2_dir:       Path,
    out_dir:        Path,
    aruco_dict_id:  int   = cv2.aruco.DICT_4X4_50,
    expected_ids:   tuple = (0, 1, 2),
    max_angle_diff: float = 10.0,
    preview_n:      int   = 20,
    use_hungarian:  bool  = True,
) -> list[FrameMatch]:
    """
    Full pipeline: analyse both wheel directories, match frames, save results.

    Returns the list of :class:`FrameMatch` objects.
    """
    run1_dir = Path(run1_dir)
    run2_dir = Path(run2_dir)
    out_dir  = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'─'*60}")
    print(f"Analysing Run 1:  {run1_dir}")
    states1, cal1 = analyse_wheel_dir(
        run1_dir / "wheel", aruco_dict_id, expected_ids
    )

    print(f"Analysing Run 2:  {run2_dir}")
    states2, cal2 = analyse_wheel_dir(
        run2_dir / "wheel", aruco_dict_id, expected_ids
    )

    # Cross-validate calibrations
    print(f"\n  Run 1 inter-marker offsets: {cal1.offsets}")
    print(f"  Run 2 inter-marker offsets: {cal2.offsets}")
    for kid in expected_ids:
        o1 = cal1.offsets.get(kid, 0.0)
        o2 = cal2.offsets.get(kid, 0.0)
        if kid != 0 and abs(o1 - o2) > 5.0:
            warnings.warn(
                f"  [WARN] Marker {kid} offset differs between runs by "
                f"{abs(o1-o2):.1f}° — check wheel geometry or detection quality.",
                RuntimeWarning,
            )

    print(f"\nBuilding {len(states1)} × {len(states2)} cost matrix…")
    cost = build_cost_matrix(states1, states2)

    finite_pairs = int(np.sum(np.isfinite(cost)))
    print(f"  {finite_pairs} finite entries  "
          f"({100*finite_pairs / max(cost.size,1):.1f}% of cells)")

    # ── Matching ──────────────────────────────────────────────────────────────
    if use_hungarian:
        print("Running Hungarian (optimal 1-to-1) matching…")
        matches = match_hungarian(states1, states2, cost, max_angle_diff)
        method  = "hungarian"
    else:
        matches = match_nearest_neighbour(states1, states2, cost, max_angle_diff)
        method  = "nearest_neighbour"

    if not matches:
        print(
            f"\n  [WARN] No matches found within {max_angle_diff}° — "
            f"try increasing --max-angle-diff."
        )
        return []

    diffs = [m.angle_diff_deg for m in matches]
    print(
        f"\n  {len(matches)} matches found  ({method})\n"
        f"  Angle diff — mean: {np.mean(diffs):.2f}°  "
        f"median: {np.median(diffs):.2f}°  "
        f"max: {np.max(diffs):.2f}°"
    )

    # ── Save results ──────────────────────────────────────────────────────────
    print(f"\nSaving results to {out_dir} …")
    save_csv(matches,  out_dir / "matches.csv")
    save_json(
        matches, states1, states2,
        cal1.offsets, cal2.offsets,
        out_dir / "matches.json",
    )
    save_angle_plot(states1, states2, matches, out_dir / "angle_plot.png")
    save_previews(
        matches, states1, states2,
        run1_dir, run2_dir,
        out_dir / "previews",
        n=preview_n,
    )

    print(f"\n{'─'*60}")
    print(f"Done.  {len(matches)} matched pairs written to {out_dir.resolve()}")
    return matches


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Match stereo frame pairs across two rail-scanning runs "
                    "using ArUco wheel-marker positions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--run1",  required=True,
                   help="Root directory of run 1 (contains left/ right/ wheel/).")
    p.add_argument("--run2",  required=True,
                   help="Root directory of run 2.")
    p.add_argument("--out",   default="matches",
                   help="Output directory for CSV, JSON, and previews.")
    p.add_argument("--max-angle-diff", type=float, default=10.0,
                   help="Maximum allowed canonical-angle difference (degrees) "
                        "for a pair to be considered a match.")
    p.add_argument("--preview-n",      type=int,   default=20,
                   help="Number of preview montages to save.")
    p.add_argument("--no-hungarian",   action="store_true",
                   help="Use nearest-neighbour instead of Hungarian matching.")
    p.add_argument("--aruco-dict",     type=int,
                   default=cv2.aruco.DICT_4X4_50,
                   help="cv2.aruco.DICT_* constant (integer).")
    p.add_argument("--expected-ids",   type=int, nargs="+", default=[0, 1, 2],
                   help="ArUco marker IDs present on the wheel.")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_matching(
        run1_dir       = Path(args.run1),
        run2_dir       = Path(args.run2),
        out_dir        = Path(args.out),
        aruco_dict_id  = args.aruco_dict,
        expected_ids   = tuple(args.expected_ids),
        max_angle_diff = args.max_angle_diff,
        preview_n      = args.preview_n,
        use_hungarian  = not args.no_hungarian,
    )