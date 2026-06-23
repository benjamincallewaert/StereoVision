# StereoVision — wheelflat capture & wheel-phase matching

A three-camera rig for inspecting **wheelflats on a moving train**:

- a **stereo pair** (left + right), mounted roughly parallel to the track, used to
  measure the wheel/flat geometry, and
- a **wheel (side) camera** ~4 m from the track that reads **ArUco markers placed
  at 90° around the wheel** to estimate the wheel's **angular phase**.

The angular phase lets you **match a run recorded on the left side of the track to
a run recorded on the right side** (frames at the same wheel rotation), so the two
viewpoints can be compared/fused.

> All three cameras are hardware-triggered together (e.g. by a Raspberry Pi pulse
> on `Line3`), so a given trigger index corresponds to the same instant on every
> camera.

---

## Hardware

| Role  | Default serial | Camera | Notes |
|-------|----------------|--------|-------|
| left  | `24856866` | Basler **acA3800-14uc** (10 MP, 3840×2748, color, USB3) | stereo |
| right | `24856867` | Basler acA3800-14uc | stereo |
| wheel | `24856864` | Basler acA3800-14uc | ArUco phase |

Camera abstractions also exist for **Allied Vision** (VmbPy) and **FLIR/Spinnaker**
(PySpin), selectable via `CameraType`, but the rig defaults to Basler.

**Two hard ceilings to remember** (see *Performance & tuning*):
- acA3800-14uc does **~14 fps at full resolution** — a faster trigger cannot
  produce more frames unless you crop (ROI).
- Three 10 MP USB3 cameras ≈ **~440 MB/s** aggregate; put them on **separate USB3
  host controllers**, not one hub.

---

## Repository layout

| File | Purpose |
|------|---------|
| `grab_sync_footage.py` | **Main capture.** 3-camera synchronised capture (`SyncRig`); hardware-trigger or free-run; lossless queue pipeline; per-camera ROI; format/throughput controls. |
| `stereo_pair.py` | Camera wrappers (`BaslerCamera`, `AlliedVisionCamera`, `FlirCamera`), `CameraWorker`, `StereoPair`, `WheelCamera`, `_make_camera` factory. |
| `wheel_aruco_matching.py` | **Primary** wheel-phase detection + left/right run matching (per-marker angle, Hungarian). Writes `matches.json` and side-by-side previews. |
| `wheel_matching_utils.py` | Shared, tuned ArUco detector (`detect_markers`, CLAHE preprocessing), `WheelPose`, `MatchResult`, visualisation helpers, marker-ID scheme. |
| `wheel_aruco_angle_matching.py` | Older matching variant (single canonical wheel angle via circular mean). Superseded. |
| `match_runs.py` | Alternative matcher (wheel-centre bearings + inter-marker offset calibration). Uses a different marker scheme (`DICT_4X4_50`, IDs 0/1/2). |
| `grab_calibration_footage.py` | Live Charuco capture **and** immediate stereo calibration (Basler). |
| `stereo_calibration.py` | Stereo calibration from saved Charuco image pairs → `stereo_calibration.npz`. |
| `estimate_size.py` | Wheelflat size estimation from rectified stereo pairs using the calibration. |
| `estimate_cylindrical_size.py` | **Experimental/WIP** wheel-tread segmentation via SAM3 (GPU). |
| `grab_stereo_footage.py` | Quick stereo trigger-sync drift measurement. |
| `basler_grabber.py` | Minimal `BaslerStereoPair` (used by the live calibration tool). |
| `stereo_utils.py` | `thermal_to_bgr` helper (FLIR). |

Capture/output data (`sync_capture/`, `aruco_wheel/`, `output/`, `calibrations/`,
the `.venv/`, etc.) is **git-ignored** — see `.gitignore`.

---

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
pip install numpy scipy opencv-python   # cv2.aruco is included
pip install pypylon                      # Basler (required for the default rig)
# optional, only for those camera types:
pip install vmbpy                         # Allied Vision
# PySpin / Spinnaker SDK                  # FLIR (vendor installer)
# estimate_cylindrical_size.py (experimental): torch + transformers + a CUDA GPU
```

Python 3.10/3.11 recommended (matches the vendor SDK wheels).

---

## Workflow

### 1. Calibrate the stereo pair

Live (capture Charuco views and calibrate in one go):
```bash
python grab_calibration_footage.py --session_name test1 --square_mm 35
```
Or calibrate from already-captured pairs:
```bash
python stereo_calibration.py            # see flags in the file
```
Produces `stereo_calibration.npz` (intrinsics, `R`, `T`, rectification maps, `Q`,
baseline) used by `estimate_size.py`.

### 2. Capture a run

```bash
python grab_sync_footage.py \
    --test-name depot_3kmph_001_left \
    --save-format jpg \
    --stereo-roi-height 1400 \
    --wheel-roi-height 1000 \
    --idle-timeout 20
```

Frames are written to `sync_capture/<test-name>/{left,right,wheel}/`.

**In hardware-trigger mode** capture uses the lossless queue pipeline: every frame
each camera delivers is saved, named by its **frame_id (BlockID)** so the three
streams stay aligned by trigger index even if a camera drops a frame. The run ends
after `--num-frames` per camera, on a keypress (preview mode), or after
`--idle-timeout` seconds with no new frames *and* the write queue drained.

The end-of-run line reports both **Delivered** (camera→us) and **Saved**, the
achieved fps/camera, and **dropped triggers** (BlockID gaps) per camera.

#### `grab_sync_footage.py` options

| Flag | Default | Meaning |
|------|---------|---------|
| `--test-name` | `depot_3kmph_002_left` | output folder under `sync_capture/` |
| `--exposure` | `2000` | exposure (µs) — keep short to freeze motion |
| `--gain` | `10` | stereo gain (dB) |
| `--wheel-gain` | `4` | wheel-camera gain (dB) |
| `--save-format` | `png` | `png`/`jpg`/`bmp`/`tiff` — **`jpg` is ~5–10× faster to write** |
| `--num-frames` | `2000` | frames per camera |
| `--wheel-roi-height` | full | sensor-side vertical crop (px) for the wheel cam |
| `--stereo-roi-height` | full | sensor-side vertical crop (px) for both stereo cams |
| `--idle-timeout` | `10` | seconds of no new frames before stopping |

Trigger mode, serials, and ROI vertical offsets are set near the top of `__main__`
/ in the `SyncRig(...)` call (`USE_HW_TRIGGER`, `stereo_roi_offset_y`,
`wheel_roi_offset_y`; ROIs are vertically centred by default).

### 3. Detect wheel phase & match left/right runs

```bash
python wheel_aruco_matching.py \
    --left_folder  sync_capture/depot_3kmph_001_left/wheel \
    --right_folder sync_capture/depot_3kmph_001_right/wheel \
    --max-angle 10 \
    --output matches.json
```
Writes `matches.json` (matched frame pairs with angle difference + score) and
side-by-side preview PNGs under `aruco_wheel/pairs/`.

**Marker scheme** (`wheel_matching_utils.py`): IDs are `wheel*10 + position`, e.g.
wheel 1 → `11,12,13,14`, with `position` (1–4) at 0/90/180/270°. Detection uses
`DICT_4X4_250` (must cover all IDs in use) with a shared, tuned detector + CLAHE.

### 4. Estimate wheelflat size

```bash
python estimate_size.py     # consumes stereo_calibration.npz + a stereo pair folder
```

---

## Output structure

```
sync_capture/<run>/
  left/   left_000123.jpg     # trigger mode: filename = frame_id (BlockID)
  right/  right_000123.jpg
  wheel/  wheel_000123.jpg
```

> **Pair stereo left/right by the frame_id in the filename, not by sorted
> position.** If a camera drops a frame, positions shift but matching frame_ids
> still correspond to the same trigger. (Free-run mode uses a sequential
> `left_0000.png` index instead.)

---

## Performance & tuning

The data rate is the binding constraint: **10.5 MP × 3 cams × fps**. At full res
that is far beyond real-time disk/encode, so:

- **Format is the biggest write lever.** PNG of a 10 MP frame is ~seconds; JPEG is
  ~150–220 ms; BMP ~50 ms but huge. Use `--save-format jpg` (fine for the wheel
  cam; consider fidelity for the stereo pair).
- **ROI cuts data *and* raises the fps cap.** Frame rate scales ~inversely with
  rows: full 2748 rows → ~14 fps; ~1000 rows → ~38 fps. Crop the **stereo pair**
  too (`--stereo-roi-height`) — it dominates the data and is usually what stalls a
  fast trigger.
- `drops = 0` does **not** mean every pulse was caught: a camera that is too busy
  to service a trigger never increments BlockID, so that pulse is invisible to the
  gap counter. Watch **Delivered fps** and set the Pi pulse rate to the sustainable
  value you measure after choosing format + ROI.
- Spread the cameras across **separate USB3 controllers**.

### Detection robustness (lighting, not code)

The markers can only go on the wheel **flange (~40 mm tall)** → ~38 mm square →
~37 px (≈6 px/cell) at a 4 m field of view. That decodes reliably **only with good
contrast and low gain**; the depot failures were dark + noisy frames, not a code
problem. Practical fixes: **more light → lower gain → shorter exposure**, ideally
**retroreflective markers + a coaxial ring/strobe** (wide-entrance-angle prismatic
sheeting so return holds across the full FOV).

---

## Known limitations / gotchas

- Lossless (PNG) full-res at high trigger rates is not achievable — drop the rate
  or reduce resolution/ROI.
- `match_runs.py` uses a *different* marker convention (`DICT_4X4_50`, IDs 0/1/2)
  than `wheel_aruco_matching.py`; pick one pipeline.
- `estimate_cylindrical_size.py` is experimental (depends on SAM3 + GPU) and not
  wired into the main flow.
