"""
demosaic_raw.py
===============
Offline debayer for raw-Bayer captures produced by
``grab_sync_footage.py --raw-bayer``.

The capture saves the single-channel Bayer mosaic (1 byte/px) as ``.npy`` (or a
lossless image), skipping the host demosaic during acquisition.  This script
turns those into normal BGR images.

Bayer pattern: the Basler acA3800-14uc sensor is **BayerBG8**, which maps to
OpenCV's ``COLOR_BayerRG2BGR`` (verified empirically against the pylon converter;
the GenICam vs OpenCV naming is intentionally "opposite-corner").  Override with
``--pattern`` if you use a different sensor.

Usage
-----
    # whole capture (left/right/wheel sub-folders) -> jpg
    python demosaic_raw.py --src sync_capture/run1 --out sync_capture/run1_bgr

    # a single folder, keep lossless
    python demosaic_raw.py --src sync_capture/run1/wheel \
                           --out sync_capture/run1/wheel_bgr --format png
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

# Basler GenICam name -> OpenCV demosaic code (opposite-corner convention).
_PATTERNS = {
    "bg": cv2.COLOR_BayerRG2BGR,   # BayerBG8 (this rig)
    "gb": cv2.COLOR_BayerGR2BGR,
    "rg": cv2.COLOR_BayerBG2BGR,
    "gr": cv2.COLOR_BayerGB2BGR,
}

_RAW_EXTS = {".npy", ".png", ".tif", ".tiff", ".bmp"}


def _load_bayer(path: Path) -> np.ndarray | None:
    if path.suffix.lower() == ".npy":
        return np.load(str(path))
    return cv2.imread(str(path), cv2.IMREAD_UNCHANGED)


def _demosaic_one(path: Path, out_dir: Path, code: int, ext: str,
                  params: list[int]) -> bool:
    bayer = _load_bayer(path)
    if bayer is None:
        print(f"[WARN] could not read {path}")
        return False
    if bayer.ndim == 3:          # already colour — just convert container
        bgr = bayer
    else:
        bgr = cv2.cvtColor(bayer, code)
    out_dir.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(out_dir / (path.stem + ext)), bgr, params))


def _iter_dirs(src: Path):
    """Yield (input_dir, relative_name) for src and any left/right/wheel subdirs."""
    subs = [d for d in ("left", "right", "wheel") if (src / d).is_dir()]
    if subs:
        for d in subs:
            yield src / d, d
    else:
        yield src, ""


def main() -> None:
    ap = argparse.ArgumentParser(description="Offline debayer for raw-Bayer captures")
    ap.add_argument("--src", required=True, help="capture dir (with left/right/wheel) or a single folder")
    ap.add_argument("--out", required=True, help="output root")
    ap.add_argument("--format", default="jpg", choices=["jpg", "png"], help="output format")
    ap.add_argument("--quality", type=int, default=95, help="JPEG quality")
    ap.add_argument("--pattern", default="bg", choices=list(_PATTERNS),
                    help="Basler Bayer pattern (default bg = BayerBG8)")
    ap.add_argument("--workers", type=int, default=None, help="thread count (default: CPU count)")
    args = ap.parse_args()

    code = _PATTERNS[args.pattern]
    ext  = ".jpg" if args.format == "jpg" else ".png"
    params = [cv2.IMWRITE_JPEG_QUALITY, args.quality] if ext == ".jpg" else [cv2.IMWRITE_PNG_COMPRESSION, 1]

    src, out_root = Path(args.src), Path(args.out)
    total = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = []
        for in_dir, rel in _iter_dirs(src):
            out_dir = out_root / rel if rel else out_root
            files = sorted(p for p in in_dir.iterdir() if p.suffix.lower() in _RAW_EXTS)
            print(f"{in_dir}: {len(files)} files -> {out_dir}")
            for p in files:
                futs.append(pool.submit(_demosaic_one, p, out_dir, code, ext, params))
        ok = sum(1 for f in futs if f.result())
        total = len(futs)
    print(f"Done: {ok}/{total} demosaiced -> {out_root}")


if __name__ == "__main__":
    main()
