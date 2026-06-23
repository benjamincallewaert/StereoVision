from __future__ import annotations

import csv
import json
import cv2
import numpy as np
from pathlib import Path
from typing import Optional

def load_stereo_calibration(npz_path: str) -> dict:
    data = np.load(npz_path)

    print("\n[INFO] Calibration keys:", data.files)

    return {
        "K1": data["camera_matrix_1"],
        "D1": data["dist_coeffs_1"],
        "K2": data["camera_matrix_2"],
        "D2": data["dist_coeffs_2"],
        "R": data["R"],
        "T": data["T"],
        "E": data["E"],
        "F": data["F"],
        "R1": data["R1"],
        "R2": data["R2"],
        "P1": data["P1"],
        "P2": data["P2"],
        "Q": data["Q"],
        "map1x": data["map1x"],
        "map1y": data["map1y"],
        "map2x": data["map2x"],
        "map2y": data["map2y"],
        "baseline_mm": float(data["baseline_mm"]),
    }

def discover_image_pairs(images_dir: str) -> list[dict]:

    base = Path(images_dir)

    l_dir = base / "left"
    r_dir = base / "right"

    if not l_dir.is_dir() or not r_dir.is_dir():
        raise FileNotFoundError(
            f"[ERROR] Expected:\n{l_dir}\n{r_dir}"
        )

    EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

    left_map = {
        p.stem: p
        for p in l_dir.iterdir()
        if p.suffix.lower() in EXTS
    }

    right_map = {
        p.stem: p
        for p in r_dir.iterdir()
        if p.suffix.lower() in EXTS
    }

    common = sorted(set(left_map) & set(right_map))

    if not common:
        raise ValueError("[ERROR] No matching stereo image pairs found")

    pairs = [
        {
            "stem": stem,
            "left": left_map[stem],
            "right": right_map[stem],
        }
        for stem in common
    ]

    print(f"[INFO] Found {len(pairs)} stereo pair(s)")

    return pairs

def rectify_images(left_img, right_img, calib):

    rect_left = cv2.remap(
        left_img,
        calib["map1x"],
        calib["map1y"],
        cv2.INTER_LINEAR,
    )

    rect_right = cv2.remap(
        right_img,
        calib["map2x"],
        calib["map2y"],
        cv2.INTER_LINEAR,
    )

    return rect_left, rect_right


ANNOTATION_VERSION = 2


def annotation_paths(images_dir: str, stem: str):

    base = Path(images_dir)

    left_ann_dir  = base / "left"  / "annotations"
    right_ann_dir = base / "right" / "annotations"

    left_ann_dir.mkdir(parents=True, exist_ok=True)
    right_ann_dir.mkdir(parents=True, exist_ok=True)

    return (
        left_ann_dir  / f"{stem}.json",
        right_ann_dir / f"{stem}.json",
    )


def load_annotation(images_dir: str, stem: str) -> Optional[dict]:

    left_path, right_path = annotation_paths(images_dir, stem)

    if not left_path.exists() or not right_path.exists():
        return None

    try:
        with open(left_path) as f:
            left_ann = json.load(f)

        with open(right_path) as f:
            right_ann = json.load(f)

        # Support both old bbox format and new polygon format
        if "polygon" in left_ann and "polygon" in right_ann:
            ann = {
                "polygon_left":  left_ann["polygon"],
                "polygon_right": right_ann["polygon"],
            }
        elif "bbox" in left_ann and "bbox" in right_ann:
            # Backwards-compat: convert bbox → 4-point polygon
            ann = {
                "polygon_left":  _bbox_to_polygon(left_ann["bbox"]),
                "polygon_right": _bbox_to_polygon(right_ann["bbox"]),
            }
        else:
            return None

        print(f"[INFO] Loaded annotations for '{stem}'")
        return ann

    except Exception as e:
        print(f"[WARN] Failed loading annotations: {e}")
        return None


def save_annotation(images_dir: str, stem: str, polygon_left, polygon_right):
    """Save polygon annotations (lists of [x, y] integer pairs)."""

    left_path, right_path = annotation_paths(images_dir, stem)

    left_ann = {
        "version": ANNOTATION_VERSION,
        "stem":    stem,
        "camera":  "left",
        "polygon": [[int(x), int(y)] for x, y in polygon_left],
    }

    right_ann = {
        "version": ANNOTATION_VERSION,
        "stem":    stem,
        "camera":  "right",
        "polygon": [[int(x), int(y)] for x, y in polygon_right],
    }

    with open(left_path, "w") as f:
        json.dump(left_ann, f, indent=2)

    with open(right_path, "w") as f:
        json.dump(right_ann, f, indent=2)

    print(f"[INFO] Left  annotation saved -> {left_path}")
    print(f"[INFO] Right annotation saved -> {right_path}")


def _bbox_to_polygon(bbox):
    """Convert [x, y, w, h] into a 4-corner polygon."""
    x, y, w, h = bbox
    return [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]


def select_polygon_on_image(
    window_name: str,
    image,
    display_scale: float = 0.25,
) -> list[list[int]]:
    """
    Interactive polygon drawing tool.

    Controls
    --------
    Left-click      : add a vertex
    Backspace       : remove the last vertex
    Enter / Space   : confirm polygon (need ≥ 3 points)
    Escape          : cancel and return empty list

    Returns
    -------
    List of [x, y] points in **full-resolution** coordinates.
    """

    h, w = image.shape[:2]
    small = cv2.resize(image, (int(w * display_scale), int(h * display_scale)))

    points: list[list[int]] = []   # points in display-scale coords
    done   = False
    cancel = False

    instructions = [
        "Left-click: add vertex",
        "Backspace:  undo last",
        "Enter/Space: confirm (>=3 pts)",
        "Escape: cancel",
    ]

    def _render():
        vis = small.copy()

        # Draw edges
        if len(points) > 1:
            cv2.polylines(
                vis,
                [np.array(points, dtype=np.int32)],
                isClosed=False,
                color=(0, 255, 0),
                thickness=2,
            )
        # Draw closing edge preview
        if len(points) > 2:
            cv2.line(vis, tuple(points[-1]), tuple(points[0]), (0, 255, 0), 1)

        # Draw vertices
        for pt in points:
            cv2.circle(vis, tuple(pt), 5, (0, 80, 255), -1)

        # Instruction overlay
        for i, txt in enumerate(instructions):
            y = 20 + i * 22
            cv2.putText(vis, txt, (8, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(vis, txt, (8, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (255, 255, 255), 1, cv2.LINE_AA)

        point_count_txt = f"Points: {len(points)}"
        cv2.putText(vis, point_count_txt, (8, small.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow(window_name, vis)

    def _on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append([x, y])
            _render()

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 900, 600)
    cv2.setMouseCallback(window_name, _on_mouse)
    _render()

    while not done and not cancel:
        key = cv2.waitKey(20) & 0xFF

        if key in (13, 32):           # Enter or Space → confirm
            if len(points) >= 3:
                done = True
            else:
                print("[WARN] Need at least 3 points to confirm polygon.")

        elif key == 8:                # Backspace → undo
            if points:
                points.pop()
                _render()

        elif key == 27:               # Escape → cancel
            cancel = True

    cv2.destroyWindow(window_name)

    if cancel or len(points) < 3:
        print("[WARN] Polygon selection cancelled or too few points.")
        return []

    # Scale back to full-resolution coordinates
    full_res_points = [
        [int(x / display_scale), int(y / display_scale)]
        for x, y in points
    ]

    return full_res_points


def get_or_draw_annotation(
    stem: str,
    rect_left,
    rect_right,
    images_dir: str,
    display_scale: float = 0.25,
):
    ann = load_annotation(images_dir, stem)

    if ann is not None:
        return ann

    print(f"\n[ANNOTATE] No annotation found for '{stem}'")
    print("Draw polygon on LEFT image, then RIGHT image.")
    print("Left-click to add points | Backspace to undo | Enter to confirm")

    polygon_left = select_polygon_on_image(
        f"LEFT [{stem}]",
        rect_left,
        display_scale,
    )

    polygon_right = select_polygon_on_image(
        f"RIGHT [{stem}]",
        rect_right,
        display_scale,
    )

    if not polygon_left or not polygon_right:
        print(f"[ERROR] Polygon selection aborted for '{stem}'")
        return None

    save_annotation(images_dir, stem, polygon_left, polygon_right)

    return {
        "polygon_left":  polygon_left,
        "polygon_right": polygon_right,
    }

def triangulate_point(pt_left, pt_right, P1, P2):

    pt_left  = np.array(pt_left,  dtype=np.float32).reshape(2, 1)
    pt_right = np.array(pt_right, dtype=np.float32).reshape(2, 1)

    pts4d = cv2.triangulatePoints(P1, P2, pt_left, pt_right)

    return (pts4d[:3] / pts4d[3]).flatten()


def _polygon_to_mask(polygon: list, shape: tuple) -> np.ndarray:
    """Return a binary mask (uint8) with the polygon interior filled."""
    mask = np.zeros(shape[:2], dtype=np.uint8)
    pts  = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(mask, [pts], 255)
    return mask


def _polygon_bounding_rect(polygon: list):
    """Return (x, y, w, h) bounding rect of a polygon."""
    pts = np.array(polygon, dtype=np.int32)
    return cv2.boundingRect(pts)


def robust_disparity(polygon_left: list, polygon_right: list,
                     img_shape: tuple, step: int = 5) -> float:
    """
    Estimate disparity by sampling corresponding pixels that fall
    *inside* both polygon masks and computing the median x-shift.

    Parameters
    ----------
    polygon_left / polygon_right : list of [x, y] full-res points
    img_shape : (height, width) or (height, width, channels) of the image
    step      : pixel stride for the sampling grid
    """

    mask1 = _polygon_to_mask(polygon_left,  img_shape)
    mask2 = _polygon_to_mask(polygon_right, img_shape)

    x1, y1, w1, h1 = _polygon_bounding_rect(polygon_left)
    x2, y2, w2, h2 = _polygon_bounding_rect(polygon_right)

    # Sample a grid over the union bounding box
    y_min = min(y1, y2)
    y_max = max(y1 + h1, y2 + h2)
    x_min = min(x1, x2)
    x_max = max(x1 + w1, x2 + w2)

    disparities = []

    for y in range(y_min, y_max, step):
        for x in range(x_min, x_max, step):
            if (0 <= y < img_shape[0]) and (0 <= x < img_shape[1]):
                if mask1[y, x] and mask2[y, x]:
                    # Disparity = horizontal shift between the two rectified images
                    # For the same row, the matching pixel in right is at x - disparity
                    # We estimate it as (x offset of left centroid) - (x offset of right centroid)
                    disparities.append(float(x1 - x2))   # consistent with bbox approach

    # Fall back to centroid difference if not enough overlapping samples
    if len(disparities) < 5:
        # Use bounding-rect centre difference as fallback
        cx1 = x1 + w1 / 2
        cx2 = x2 + w2 / 2
        return float(cx1 - cx2)

    arr = np.array(disparities)
    med = np.median(arr)
    mad = np.median(np.abs(arr - med))

    if mad < 1e-6:
        return float(med)

    filtered = arr[np.abs(arr - med) < 2.5 * mad]
    return float(np.median(filtered)) if len(filtered) > 0 else float(med)


def estimate_object_size_and_distance(
    polygon_left:  list,
    polygon_right: list,
    calib:         dict,
    img_shape:     tuple,
):
    """
    Estimate distance, width and height using polygon annotations.

    Width and height are derived from the **bounding rect** of each polygon
    so that the measurement spans the full extent of the drawn region.
    """

    P1       = calib["P1"]
    P2       = calib["P2"]
    baseline = calib["baseline_mm"]
    fx       = P1[0, 0]

    disparity = robust_disparity(polygon_left, polygon_right, img_shape)

    if disparity <= 0:
        raise ValueError(f"[ERROR] Invalid disparity: {disparity:.3f}")

    Z = (fx * baseline) / disparity

    # Bounding rects for physical size estimation
    x1, y1, w1, h1 = _polygon_bounding_rect(polygon_left)
    x2, y2, w2, h2 = _polygon_bounding_rect(polygon_right)

    w_px = min(w1, w2)
    h_px = min(h1, h2)

    width_mm  = (w_px * Z) / fx
    height_mm = (h_px * Z) / fx

    # 3-D centre from polygon centroids
    cx1 = x1 + w1 / 2
    cy1 = y1 + h1 / 2
    cx2 = x2 + w2 / 2
    cy2 = y2 + h2 / 2

    center_3d = triangulate_point((cx1, cy1), (cx2, cy2), P1, P2)

    return {
        "distance_mm":  float(Z),
        "width_mm":     float(width_mm),
        "height_mm":    float(height_mm),
        "disparity_px": float(disparity),
        "center_3d_mm": center_3d,
    }

def draw_polygon(img, polygon: list, color=(0, 255, 0), thickness=3):
    """Draw a closed polygon on *img* in-place."""
    pts = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(img, [pts], isClosed=True, color=color, thickness=thickness)


def make_side_by_side(
    rect_left,
    rect_right,
    polygon_left:  list,
    polygon_right: list,
    scale: float = 0.25,
):
    vis_l = rect_left.copy()
    vis_r = rect_right.copy()

    draw_polygon(vis_l, polygon_left,  color=(0, 255, 0))
    draw_polygon(vis_r, polygon_right, color=(0, 180, 255))

    vis_l = cv2.resize(vis_l, None, fx=scale, fy=scale)
    vis_r = cv2.resize(vis_r, None, fx=scale, fy=scale)

    return np.hstack([vis_l, vis_r])


def run_directory_pipeline(
    calib_path:    str,
    images_dir:    str,
    output_dir:    str   = "results",
    display_scale: float = 0.25,
    save_vis:      bool  = True,
    show_windows:  bool  = True,
    rectify:       bool = False
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    calib = load_stereo_calibration(calib_path)
    pairs = discover_image_pairs(images_dir)

    all_results = []

    for pair in pairs:

        stem = pair["stem"]

        print(f"\n{'=' * 50}")
        print(f"Pair: {stem}")
        print(f"{'=' * 50}")

        left_img  = cv2.imread(str(pair["left"]))
        right_img = cv2.imread(str(pair["right"]))

        if left_img is None or right_img is None:
            print(f"[WARN] Failed reading '{stem}'")
            continue
        if rectify:
            rect_left, rect_right = rectify_images(left_img, right_img, calib)
        else:
            rect_left, rect_right = left_img, right_img

        ann = get_or_draw_annotation(
            stem,
            rect_left,
            rect_right,
            images_dir,
            display_scale,
        )
        if ann is None:
            continue
        1
        polygon_left  = ann["polygon_left"]
        polygon_right = ann["polygon_right"]

        try:
            res = estimate_object_size_and_distance(
                polygon_left,
                polygon_right,
                calib,
                img_shape=rect_left.shape,
            )

        except ValueError as exc:
            print(f"[ERROR] {exc}")
            continue

        row = {
            "stem":         stem,
            "distance_mm":  round(res["distance_mm"],  2),
            "width_mm":     round(res["width_mm"],2),
            "height_mm":    round(res["height_mm"], 2),
            "disparity_px": round(res["disparity_px"], 3),
            "cx_mm":        round(float(res["center_3d_mm"][0]), 2),
            "cy_mm":        round(float(res["center_3d_mm"][1]), 2),
            "cz_mm":        round(float(res["center_3d_mm"][2]), 2),
        }

        all_results.append(row)

        print(f"Distance  : {row['distance_mm']:.2f} mm")
        print(f"Width     : {row['width_mm']:.2f} mm")
        print(f"Height    : {row['height_mm']:.2f} mm")
        print(f"Disparity : {row['disparity_px']:.3f} px")

        canvas = make_side_by_side(
            rect_left,
            rect_right,
            polygon_left,
            polygon_right,
            scale=display_scale,
        )

        label = (
            f"{stem} | "
            f"D:{row['distance_mm']} mm | "
            f"W:{row['width_mm']} mm | "
            f"H:{row['height_mm']} mm"
        )

        cv2.putText(
            canvas, label, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8,
            (0, 255, 255), 2, cv2.LINE_AA,
        )

        if save_vis:
            vis_path = output_path / f"{stem}_result.png"
            cv2.imwrite(str(vis_path), canvas)
            print(f"[INFO] Saved visualization -> {vis_path}")

        if show_windows:
            cv2.imshow(f"Result [{stem}]", canvas)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    if all_results:
        csv_path = output_path / "results.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\n[INFO] CSV written -> {csv_path}")
    else:
        print("\n[WARN] No valid results")

    return all_results


if __name__ == "__main__":

    SESSION_NAME = "test3"

    CALIB_PATH = rf"calibrations\{SESSION_NAME}\stereo_calibration.npz"
    IMAGES_DIR = rf"output\{SESSION_NAME}"
    OUTPUT_DIR = rf"results\{SESSION_NAME}"

    results = run_directory_pipeline(
        calib_path=CALIB_PATH,
        images_dir=IMAGES_DIR,
        output_dir=OUTPUT_DIR,
        display_scale=0.25,
        save_vis=True,
        show_windows=True,
    )

    print("\n===== FINAL SUMMARY =====")
    for r in results:
        print(
            f"{r['stem']:20s} "
            f"dist={r['distance_mm']:8.2f} mm "
            f"w={r['width_mm']:8.2f} mm "
            f"h={r['height_mm']:8.2f} mm"
        )