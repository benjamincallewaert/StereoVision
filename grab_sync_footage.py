from __future__ import annotations

import os
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np
import vmbpy
import argparse

# OpenCV's own ImageCodecs thread pool fights the pipeline workers for cores.
# We do our own parallelism, so let each imwrite run single-threaded.
try:
    cv2.setNumThreads(1)
except Exception:
    pass


# ──────────────────────────────────────────────────────────────────────────────
# Output format presets
# ──────────────────────────────────────────────────────────────────────────────

def _format_to_ext_params(fmt: str) -> tuple[str, list[int]]:
    """
    Map a short format name to (file_extension, cv2.imwrite params).

    Speed / size trade-off for a 10 MP frame (measure on your own machine):
      - "png"  : lossless, but PNG encoding is CPU-heavy.  Compression level 1
                 is the fastest *lossless* option OpenCV offers.
      - "jpg"  : ~5-10x faster to encode and ~10x smaller on disk, but lossy.
                 Fine for the ArUco/wheel camera; think twice for the stereo
                 measurement pair if you need sub-pixel accuracy.
      - "bmp"  : essentially zero encoding cost, but huge files -> disk-bandwidth
                 bound.  Good when the SSD is fast and CPU is the bottleneck.
    """
    fmt = fmt.lower().lstrip(".")
    if fmt == "png":
        return ".png", [cv2.IMWRITE_PNG_COMPRESSION, 1]
    if fmt in ("jpg", "jpeg"):
        return ".jpg", [cv2.IMWRITE_JPEG_QUALITY, 95]
    if fmt == "bmp":
        return ".bmp", []
    if fmt in ("tif", "tiff"):
        return ".tiff", []
    raise ValueError(f"Unknown save_format {fmt!r} (use png/jpg/bmp/tiff)")

from stereo_pair import (
    CameraType,
    CameraFrame,
    ArucoDetection,
    CameraWorker,
    _make_camera,
)
from stereo_utils import thermal_to_bgr


# ──────────────────────────────────────────────────────────────────────────────
# Trio frame dataclass
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class TrioFrame:
    """One synchronised capture from all three cameras."""
    left:      CameraFrame | None
    right:     CameraFrame | None
    wheel:     CameraFrame | None
    detection: ArucoDetection | None = field(default=None)  # populated after ArUco pass

    @property
    def valid(self) -> bool:
        return (
            self.left  is not None and
            self.right is not None and
            self.wheel is not None
        )

    def ts_delta_ms(self) -> dict[str, float]:
        """Return pairwise hardware-timestamp deltas in milliseconds."""
        if not self.valid:
            return {}
        tl = self.left.hw_ts_ns
        tr = self.right.hw_ts_ns
        tw = self.wheel.hw_ts_ns
        return {
            "left-right_ms":  abs(tl - tr) / 1e6,
            "left-wheel_ms":  abs(tl - tw) / 1e6,
            "right-wheel_ms": abs(tr - tw) / 1e6,
        }

    def __len__(self) -> int:
        return sum(1 for x in (self.left, self.right, self.wheel) if x is not None)


# ──────────────────────────────────────────────────────────────────────────────
# SyncRig
# ──────────────────────────────────────────────────────────────────────────────

class SyncRig:
    """
    Three-camera synchronised rig: left + right (stereo pair) + wheel.

    All three cameras are opened with a single shared VmbSystem context when
    any of them is Allied Vision.  Three CameraWorker threads continuously
    drain frames in the background; grab_sync() snapshots all three workers
    as close together as possible.

    grab_sequence() is fully pipelined:
      - The main loop runs the pace-maker and grab calls only.
      - A single shared ThreadPoolExecutor handles saving and preview
        rendering for each trio in parallel.
      - A backpressure semaphore caps the number of in-flight (grabbed but
        not yet written) frames so RAM cannot overflow.
    """

    # Pipeline worker threads.  Image encoding (cv2.imwrite) releases the GIL,
    # so encoding scales across cores.  One worker encodes one trio's three
    # images, so up to _PIPELINE_WORKERS encodes run concurrently.  Sizing this
    # to the core count is what lets disk write-out keep up with capture.
    _PIPELINE_WORKERS = max(4, (os.cpu_count() or 4))

    # Maximum number of frames that may be grabbed but not yet written to disk.
    # Each in-flight trio holds three full frames in RAM, so this is the main
    # RAM-vs-throughput knob.  (48 trios * 3 * ~32 MB ≈ 4.6 GB for 10 MP cams.)
    _MAX_IN_FLIGHT = 48

    def __init__(
        self,
        # ── Camera identities ──────────────────────────────────────────────
        serial_left:   str | None = None,
        type_left:     CameraType = CameraType.BASLER,
        serial_right:  str | None = None,
        type_right:    CameraType = CameraType.BASLER,
        serial_wheel:  str | None = None,
        type_wheel:    CameraType = CameraType.BASLER,
        # ── Shared sensor settings ─────────────────────────────────────────
        exposure_us:   float = None,
        stereo_gain_db:       float = None,
        wheel_gain_db:        float = None,
        # ── Trigger settings ───────────────────────────────────────────────
        use_external_trigger:  bool = False,
        trigger_source_left:   str  = "Line3",
        trigger_source_right:  str  = "Line3",
        trigger_source_wheel:  str  = "Line3",
        # ── ArUco (wheel camera) ───────────────────────────────────────────
        aruco_dict_id:   int               = cv2.aruco.DICT_4X4_250,
        camera_matrix:   np.ndarray | None = None,
        dist_coeffs:     np.ndarray | None = None,
        marker_length_m: float | None      = None,
        # ── Wheel-camera vertical ROI (sensor-side crop) ────────────────────
        # Crop the wheel camera to a horizontal band to cut its data rate.
        # None = full frame.  offset None = centred vertically.
        wheel_roi_height:   int | None = None,
        wheel_roi_offset_y: int | None = None,
    ) -> None:

        # ── Shared VmbSystem context (required if any camera is Allied Vision) ──
        self._vmb = None
        needs_vmb = any(
            t == CameraType.ALLIED_VISION
            for t in (type_left, type_right, type_wheel)
        )
        if needs_vmb:
            self._vmb = vmbpy.VmbSystem.get_instance().__enter__()

        # ── Open three cameras ────────────────────────────────────────────────
        self._cam_l = _make_camera(
            serial_left,  type_left,  exposure_us, stereo_gain_db,
            use_external_trigger, trigger_source_left,  vmb=self._vmb,
        )
        self._cam_r = _make_camera(
            serial_right, type_right, exposure_us, stereo_gain_db,
            use_external_trigger, trigger_source_right, vmb=self._vmb,
        )
        self._cam_w = _make_camera(
            serial_wheel, type_wheel, exposure_us, wheel_gain_db,
            use_external_trigger, trigger_source_wheel, vmb=self._vmb,
            roi_height=wheel_roi_height, roi_offset_y=wheel_roi_offset_y,
        )

        # ── Background grab workers (one per camera) ──────────────────────────
        self._worker_l = CameraWorker(self._cam_l)
        self._worker_r = CameraWorker(self._cam_r)
        self._worker_w = CameraWorker(self._cam_w)

        for w in (self._worker_l, self._worker_r, self._worker_w):
            w.start()

        # Tracks the last hw_ts_ns consumed from each worker so that
        # grab_sync_blocking can detect truly *new* frames.
        self._last_ts: Dict[str, Optional[int]] = {"l": None, "r": None, "w": None}

        # ── ArUco detector ────────────────────────────────────────────────────
        aruco_dict   = cv2.aruco.getPredefinedDictionary(aruco_dict_id)
        aruco_params = cv2.aruco.DetectorParameters()
        self._detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

        self._camera_matrix   = camera_matrix
        self._dist_coeffs     = dist_coeffs
        self._marker_length_m = marker_length_m
        self._pose_ready = (
            camera_matrix   is not None and
            dist_coeffs     is not None and
            marker_length_m is not None
        )

        print(
            f"[SyncRig] Three cameras ready — "
            f"trigger={'external' if use_external_trigger else 'free-run'} | "
            f"ArUco pose={'on' if self._pose_ready else 'off'}"
        )

    # ── Warmup ───────────────────────────────────────────────────────────────
    def _warmup(self, timeout_s: float = 10.0) -> None:
        """
        Block until every worker has produced at least one frame.

        Called automatically at the start of grab_sequence() so the capture
        loop never sees None frames simply because the cameras hadn't finished
        initialising yet.  Raises TimeoutError if a camera doesn't respond.
        """
        deadline = time.monotonic() + timeout_s
        workers  = {
            "left":  self._worker_l,
            "right": self._worker_r,
            "wheel": self._worker_w,
        }
        waiting = set(workers.keys())

        print("[SyncRig] Waiting for cameras to deliver first frames...", end="", flush=True)
        while waiting and time.monotonic() < deadline:
            for label in list(waiting):
                with workers[label].lock:
                    ready = workers[label].latest_frame is not None
                if ready:
                    waiting.discard(label)
            if waiting:
                time.sleep(0.0005)

        if waiting:
            raise TimeoutError(
                f"[SyncRig] Camera(s) {waiting} did not produce a frame "
                f"within {timeout_s:.1f}s — check connections."
            )
        print(" ready.")

    # ── Synchronised grab ─────────────────────────────────────────────────────
    def grab_sync(self) -> TrioFrame:
        """
        Snapshot the latest frame from all three background workers as close
        together as possible.

        In hardware-trigger mode the three workers will all have been woken by
        the same electrical pulse, so the frames are already hardware-latched
        at the same instant; this call just reads them out.

        In free-run mode this method returns the most recently grabbed frame
        from each camera.  Use TrioFrame.ts_delta_ms() to inspect skew.
        """
        # Acquire locks in a consistent order (by id) to prevent deadlock.
        workers = [self._worker_l, self._worker_r, self._worker_w]
        for w in sorted(workers, key=id):
            w.lock.acquire()
        try:
            fl = self._worker_l.latest_frame
            fr = self._worker_r.latest_frame
            fw = self._worker_w.latest_frame
        finally:
            for w in workers:
                w.lock.release()

        return TrioFrame(left=fl, right=fr, wheel=fw)

    def grab_sync_blocking(self, timeout_s: float = 2.0) -> TrioFrame:
        """
        Wait until all three workers have each produced at least one new frame
        since the last call, then return them as a TrioFrame.

        Checking ``latest_frame`` is a cheap lock-protected read, so all three
        cameras are polled inline in a single loop.  (The previous version spun
        up three throw-away threads on every call — at 100 fps that is 300
        thread creations per second of pure overhead.)

        Raises TimeoutError if any camera does not deliver within timeout_s.
        """
        deadline = time.monotonic() + timeout_s
        prev_ts  = self._last_ts

        workers = {"l": self._worker_l, "r": self._worker_r, "w": self._worker_w}
        results: Dict[str, CameraFrame] = {}

        while len(results) < 3 and time.monotonic() < deadline:
            for label, worker in workers.items():
                if label in results:
                    continue
                with worker.lock:
                    f = worker.latest_frame
                if f is not None and f.hw_ts_ns != prev_ts[label]:
                    results[label] = f
            if len(results) < 3:
                time.sleep(0.0005)

        missing = [lbl for lbl in workers if lbl not in results]
        if missing:
            raise TimeoutError(
                f"[SyncRig] grab_sync_blocking: camera(s) {missing} timed out "
                f"after {timeout_s:.1f}s"
            )

        fl, fr, fw = results["l"], results["r"], results["w"]
        self._last_ts = {
            "l": fl.hw_ts_ns,
            "r": fr.hw_ts_ns,
            "w": fw.hw_ts_ns,
        }

        return TrioFrame(left=fl, right=fr, wheel=fw)

    # ── Per-frame post-processing ─────────────────────────────────────────────

    def _process_trio(
        self,
        trio:        TrioFrame,
        idx:         int,
        save_dir:    Path | None,
        ext:         str,
        imwrite_params: list[int],
        show_preview: bool,
        preview_images: dict[str, np.ndarray | None],
        preview_lock:   threading.Lock,
        backpressure:   threading.Semaphore,
    ) -> TrioFrame:
        """
        Encode + save all three images and optionally prepare preview frames.

        Runs inside the shared pipeline ThreadPoolExecutor.  Concurrency comes
        from the pool itself (one worker per trio, _PIPELINE_WORKERS in
        parallel); the three writes within a trio are done sequentially.  This
        avoids the previous design's per-frame nested thread pool, whose
        creation/teardown ran tens of thousands of times over a long capture.

        The backpressure semaphore is always released in the finally block,
        even if an exception occurs, so the capture loop can never deadlock.
        """
        try:
            if save_dir is not None:
                for sub, img in (
                    ("left",  trio.left.image),
                    ("right", trio.right.image),
                    ("wheel", trio.wheel.image),
                ):
                    ok = cv2.imwrite(
                        str(save_dir / sub / f"{sub}_{idx:04d}{ext}"),
                        img,
                        imwrite_params,
                    )
                    if not ok:
                        print(f"[WARN] Frame {idx:04d}: cv2.imwrite failed for {sub}")

            if show_preview:
                def _to_bgr(img: np.ndarray) -> np.ndarray:
                    return thermal_to_bgr(img) if img.dtype == np.uint16 else img

                pl, pr, pw = (
                    _to_bgr(trio.left.image),
                    _to_bgr(trio.right.image),
                    _to_bgr(trio.wheel.image),
                )
                with preview_lock:
                    preview_images["left"]  = pl
                    preview_images["right"] = pr
                    preview_images["wheel"] = pw

        except Exception as exc:
            print(f"[SyncRig] Pipeline error on frame {idx:04d}: {exc}")
        finally:
            # Always release the slot so the capture loop can never deadlock.
            backpressure.release()

        # Return nothing: the capture loop must not retain a reference to the
        # trio (or its three full-resolution images) once it has been written,
        # otherwise RAM grows without bound over a long sequence.
        return None

    # ── Triggered capture (lossless queue pipeline, matched by BlockID) ────────

    def _grab_sequence_triggered(
        self,
        num_frames:     int,
        save_dir:       Path | None,
        ext:            str,
        imwrite_params: list[int],
        show_preview:   bool,
        idle_timeout_s: float = 10.0,
    ) -> int:
        """
        Hardware-trigger capture that keeps EVERY triggered frame.

        Each camera worker pushes every frame it grabs into a shared bounded
        queue; a pool of writer threads drains it and saves each image named by
        its frame_id (BlockID).  Because all three cameras share the Raspberry
        Pi trigger pulse, files with the same frame_id form a synchronised trio,
        so the streams stay aligned by BlockID even when a camera drops a frame.

        No software pace-maker and no latest-only overwrite: the trigger sets the
        rate and nothing is silently discarded.  If the writers fall behind, the
        bounded queue back-pressures the grab threads; any frame the camera then
        has to drop is detected via the BlockID gap (worker.dropped).

        Stops once every camera has saved >= num_frames frames, on a keypress
        (preview mode), or after idle_timeout_s with no new frames.
        """
        work_q: "queue.Queue" = queue.Queue(maxsize=self._MAX_IN_FLIGHT * 3)
        STOP = object()

        saved      = {"left": 0, "right": 0, "wheel": 0}
        fallback   = {"left": 0, "right": 0, "wheel": 0}  # for cams without BlockID
        saved_lock = threading.Lock()

        def _writer() -> None:
            while True:
                item = work_q.get()
                try:
                    if item is STOP:
                        return
                    label, frame = item
                    if save_dir is not None:
                        if frame.frame_id is not None:
                            name = f"{label}_{frame.frame_id:06d}{ext}"
                        else:
                            with saved_lock:
                                idx = fallback[label]; fallback[label] += 1
                            name = f"{label}_{idx:06d}{ext}"
                        ok = cv2.imwrite(str(save_dir / label / name),
                                         frame.image, imwrite_params)
                        if not ok:
                            print(f"[WARN] imwrite failed: {label} id={frame.frame_id}")
                    with saved_lock:
                        saved[label] += 1
                finally:
                    work_q.task_done()

        writers = [
            threading.Thread(target=_writer, name=f"sync_rig_writer_{i}", daemon=True)
            for i in range(self._PIPELINE_WORKERS)
        ]
        for t in writers:
            t.start()

        # Route every grabbed frame into the queue from now on.
        self._worker_l.attach_sink(work_q, "left")
        self._worker_r.attach_sink(work_q, "right")
        self._worker_w.attach_sink(work_q, "wheel")

        print(f"[SyncRig] Triggered capture — waiting for {num_frames} frames/camera "
              f"(format={ext}).")
        start_time    = time.time()
        last_progress = start_time
        last_min      = 0
        drain_start   = start_time
        try:
            while True:
                with saved_lock:
                    cur_min = min(saved.values())
                if cur_min >= num_frames:
                    break

                now = time.time()
                if cur_min > last_min:
                    last_min, last_progress = cur_min, now
                elif now - last_progress > idle_timeout_s:
                    print(f"[SyncRig] No new frames for {idle_timeout_s:.0f}s — "
                          f"stopping (min saved={cur_min}).")
                    break

                if show_preview:
                    for title, w in (("Left",  self._worker_l),
                                     ("Right", self._worker_r),
                                     ("Wheel", self._worker_w)):
                        f = w.get_latest()
                        if f is not None:
                            img = thermal_to_bgr(f.image) if f.image.dtype == np.uint16 else f.image
                            cv2.imshow(title, img)
                    if cv2.waitKey(1) != -1:
                        print("[SyncRig] Keypress — stopping early.")
                        break
                else:
                    time.sleep(0.02)
        finally:
            # Stop feeding the queue, drain it, then stop the writers.
            self._worker_l.detach_sink()
            self._worker_r.detach_sink()
            self._worker_w.detach_sink()

            drain_start = time.time()
            work_q.join()                      # all queued frames written to disk
            for _ in writers:
                work_q.put(STOP)
            for t in writers:
                t.join()
            if show_preview:
                cv2.destroyAllWindows()

        elapsed  = max(time.time() - start_time, 1e-9)
        with saved_lock:
            total = dict(saved)
        captured = min(total.values())
        print(
            f"[SyncRig] Triggered done. Saved L={total['left']} R={total['right']} "
            f"W={total['wheel']} | achieved {captured / elapsed:.1f} fps/camera | "
            f"dropped at camera (BlockID gaps): L={self._worker_l.dropped} "
            f"R={self._worker_r.dropped} W={self._worker_w.dropped} | "
            f"final write drain {time.time() - drain_start:.1f}s"
        )
        return captured

    # ── Sequence capture + save ───────────────────────────────────────────────

    def grab_sequence(
        self,
        num_frames:   int,
        fps:          float,
        save_dir:     str | Path | None = None,
        show_preview: bool = True,
        blocking:     bool = False,
        save_format:  str  = "png",
        log_every:    int  = 0,
    ) -> int:
        """
        Capture num_frames synchronised trios at the requested fps.

        The capture loop runs on the calling thread at the requested pace.
        Saving and preview rendering are dispatched to a shared background
        ThreadPoolExecutor.  A backpressure semaphore (_MAX_IN_FLIGHT) ensures
        the capture loop blocks before RAM can overflow, regardless of how
        fast the camera produces frames relative to disk write speed.

        Frames are streamed to disk and *not* retained in RAM, so a long
        high-fps run uses only as much memory as the in-flight backpressure
        window allows.

        Parameters
        ----------
        num_frames:
            Total frames to capture.
        fps:
            Target capture rate (soft pace-maker).
        save_dir:
            Root directory.  Sub-directories left/, right/, and wheel/ are
            created automatically.  Pass None to skip saving.
        show_preview:
            Display live OpenCV windows for all three streams.  Window updates
            are marshalled back to the main thread to satisfy OpenCV's GUI
            requirements.
        blocking:
            Use grab_sync_blocking() instead of grab_sync().
            Recommended when use_external_trigger=True.
        save_format:
            "png" (lossless, slow), "jpg" (fast, lossy), "bmp" (no encode,
            large) or "tiff".  See _format_to_ext_params for the trade-offs.
            The dominant lever for "writing out takes forever".
        log_every:
            Print per-frame timestamp-skew diagnostics every N frames.
            0 (default) disables per-frame printing entirely — at high fps the
            console writes themselves become a bottleneck on Windows.

        Returns
        -------
        int
            Number of valid trios captured and dispatched for saving.
        """
        period  = 1.0 / fps
        grab_fn = self.grab_sync_blocking if blocking else self.grab_sync
        ext, imwrite_params = _format_to_ext_params(save_format)

        # ── Create output directories ─────────────────────────────────────────
        if save_dir is not None:
            save_dir = Path(save_dir)
            for sub in ("left", "right", "wheel"):
                (save_dir / sub).mkdir(parents=True, exist_ok=True)
            print(f"[SyncRig] Saving to: {save_dir.resolve()}  (format={ext})")

        # ── Warmup: wait for all cameras to deliver their first frame ─────────
        self._warmup(timeout_s=5.0)
        time.sleep(1.0)

        # ── Hardware-trigger path: lossless queue pipeline (Tier 3) ───────────
        # Keeps every triggered frame, names files by BlockID so the three
        # streams stay aligned.  The free-run path below keeps the older
        # latest-frame + pace-maker behaviour.
        if blocking:
            return self._grab_sequence_triggered(
                num_frames, save_dir, ext, imwrite_params, show_preview,
            )

        # ── Backpressure: cap RAM usage regardless of disk speed ──────────────
        # One permit = one trio held in RAM.  The pipeline releases a permit
        # only after all three images are written to disk.
        backpressure = threading.Semaphore(self._MAX_IN_FLIGHT)

        # ── Shared preview state (OpenCV windows must update on main thread) ──
        preview_lock   = threading.Lock()
        preview_images: dict[str, np.ndarray | None] = {
            "left": None, "right": None, "wheel": None,
        }

        # ── Pipeline executor (shared across all frames) ───────────────────────
        pipeline = ThreadPoolExecutor(
            max_workers=self._PIPELINE_WORKERS,
            thread_name_prefix="sync_rig_pipeline",
        )

        captured = 0  # valid trios dispatched for saving (not retained in RAM)
        # Dropped-trigger detection: the camera's frame_id (BlockID) increments
        # once per trigger, so a jump > 1 between consecutive grabs means the
        # camera produced frames we never consumed (we couldn't keep up).
        prev_fid: Dict[str, Optional[int]] = {"l": None, "r": None, "w": None}
        dropped:  Dict[str, int]           = {"l": 0, "r": 0, "w": 0}
        start_time = time.time()
        try:
            for idx in range(num_frames):
                t0 = time.perf_counter()

                # Block here if _MAX_IN_FLIGHT frames are already in-flight.
                # This is the key guard against RAM overflow.
                backpressure.acquire()

                trio = grab_fn()

                if not trio.valid:
                    # Release the permit immediately — nothing will be written.
                    backpressure.release()
                    print(
                        f"[WARN] Frame {idx:04d} — one or more cameras returned "
                        f"None ({len(trio)}/3); skipping."
                    )
                    continue

                # Count triggers the cameras produced but we never consumed.
                for key, frame in (("l", trio.left), ("r", trio.right), ("w", trio.wheel)):
                    fid = frame.frame_id
                    if fid is not None and prev_fid[key] is not None:
                        gap = fid - prev_fid[key] - 1
                        if gap > 0:
                            dropped[key] += gap
                    prev_fid[key] = fid

                # Timestamp-skew diagnostics (throttled — printing every frame
                # at high fps is itself a bottleneck).
                if log_every and idx % log_every == 0:
                    deltas = trio.ts_delta_ms()
                    print(
                        f"  [{idx:04d}]  "
                        f"L-R: {deltas.get('left-right_ms',  float('nan')):.3f} ms | "
                        f"L-W: {deltas.get('left-wheel_ms',  float('nan')):.3f} ms | "
                        f"R-W: {deltas.get('right-wheel_ms', float('nan')):.3f} ms"
                    )

                # Dispatch post-processing to the pipeline.
                # _process_trio owns the permit and releases it when done.
                pipeline.submit(
                    self._process_trio,
                    trio, idx, save_dir, ext, imwrite_params, show_preview,
                    preview_images, preview_lock, backpressure,
                )
                captured += 1
                # Drop our own reference so the only live reference is inside the
                # pipeline task; once it finishes writing, the images are freed.
                trio = None

                # ── Update OpenCV windows on the main thread ──────────────────
                if show_preview:
                    with preview_lock:
                        imgs = dict(preview_images)
                    for label, title in (("left", "Left"), ("right", "Right"), ("wheel", "Wheel")):
                        if imgs[label] is not None:
                            cv2.imshow(title, imgs[label])
                    if cv2.waitKey(1) != -1:
                        print("[SyncRig] Keypress — stopping early.")
                        break

                # ── Pace-maker ────────────────────────────────────────────────
                # ONLY in free-run mode.  In hardware-trigger mode the Raspberry
                # Pi pulse train sets the rate; sleeping here would throttle the
                # loop below the trigger rate and silently drop every trigger we
                # slept through (the workers keep only the latest frame).
                if not blocking:
                    elapsed = time.perf_counter() - t0
                    if elapsed < period:
                        time.sleep(period - elapsed)

        finally:
            # Drain: reacquire every backpressure permit.  A permit is released
            # only when a pipeline task has finished writing its trio, so once
            # we hold them all, every image is on disk.
            finish_time = time.time()
            FPS = captured /(finish_time - start_time)
            total_dropped = max(dropped.values()) if any(dropped.values()) else 0
            print(
                f"[SyncRig] Captured {captured} / {num_frames} frames. "
                f"Achieved FPS: {FPS:.1f}. "
                f"Dropped triggers (camera produced, not consumed): "
                f"L={dropped['l']} R={dropped['r']} W={dropped['w']}. "
                f"Flushing remaining writes to disk..."
            )
            for _ in range(self._MAX_IN_FLIGHT):
                backpressure.acquire()

            pipeline.shutdown(wait=True)

            if show_preview:
                cv2.destroyAllWindows()
        print(f"[SyncRig] Writing out took {str(time.time()-finish_time)} s")
        print(f"[SyncRig] Done — {captured} frames captured and written.")

        return captured

    # ── Resource management ───────────────────────────────────────────────────

    def close(self) -> None:
        print("[SyncRig] Shutting down...")
        for w in (self._worker_l, self._worker_r, self._worker_w):
            try:
                w.stop()
            except Exception:
                pass
        for cam in (self._cam_l, self._cam_r, self._cam_w):
            try:
                cam.close()
            except Exception:
                pass
        if self._vmb is not None:
            try:
                self._vmb.__exit__(None, None, None)
            except Exception:
                pass
            self._vmb = None
        print("[SyncRig] All cameras closed.")

    def __enter__(self) -> "SyncRig":
        return self

    def __exit__(self, *_) -> None:
        self.close()


# ──────────────────────────────────────────────────────────────────────────────
# CONFIG & ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--test-name",
        default="depot_3kmph_002_left",
        help="Name of capture session"
    )

    parser.add_argument(
        "--exposure",
        type=float,
        default=2000,
        help="Exposure time in microseconds"
    )

    parser.add_argument(
        "--gain",
        type=float,
        default=10,
        help="Gain in dB"
    )
    parser.add_argument(
        "--wheel-gain",
        type=float,
        default=4,
        help="Gain in dB for wheel camera"
    )
    parser.add_argument(
        "--wheel-roi-height",
        type=int,
        default=None,
        help="Vertical ROI (sensor-side crop) height in px for the wheel camera; "
             "full width, centred. Omit for full frame. Cuts the wheel camera's "
             "data rate so capture keeps up with a fast trigger."
    )
    parser.add_argument(
        "--save-format",
        default="png",
        choices=["png", "jpg", "jpeg", "bmp", "tiff"],
        help="Image format. 'jpg' is ~5-10x faster to write than 'png'."
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=2000,
        help="Frames per camera to capture."
    )
    args = parser.parse_args()

    LEFT_SERIAL  = "24856866";  LEFT_TYPE  = CameraType.BASLER
    RIGHT_SERIAL = "24856867";  RIGHT_TYPE = CameraType.BASLER
    WHEEL_SERIAL = "24856864";  WHEEL_TYPE = CameraType.BASLER

    EXPOSURE_US = args.exposure
    STEREO_GAIN_DB = args.gain
    WHEEL_GAIN_DB = args.wheel_gain

    TEST = args.test_name

    USE_HW_TRIGGER     = True
    TRIGGER_LINE_LEFT  = "Line3"
    TRIGGER_LINE_RIGHT = "Line3"
    TRIGGER_LINE_WHEEL = "Line3"

    NUM_FRAMES   = args.num_frames
    FPS_TARGET   = 10.0
    SAVE_DIR     = Path(f"sync_capture/{TEST}")
    SHOW_PREVIEW = False
    SAVE_FORMAT  = args.save_format
    LOG_EVERY    = 0
    WHEEL_ROI_HEIGHT = args.wheel_roi_height   # None = full frame

    BLOCKING_GRAB = USE_HW_TRIGGER

    with SyncRig(
        serial_left=LEFT_SERIAL,   type_left=LEFT_TYPE,
        serial_right=RIGHT_SERIAL, type_right=RIGHT_TYPE,
        serial_wheel=WHEEL_SERIAL, type_wheel=WHEEL_TYPE,

        exposure_us=EXPOSURE_US,
        stereo_gain_db=STEREO_GAIN_DB,
        wheel_gain_db=WHEEL_GAIN_DB,

        use_external_trigger=USE_HW_TRIGGER,
        trigger_source_left=TRIGGER_LINE_LEFT,
        trigger_source_right=TRIGGER_LINE_RIGHT,
        trigger_source_wheel=TRIGGER_LINE_WHEEL,

        wheel_roi_height=WHEEL_ROI_HEIGHT,
    ) as rig:

        rig.grab_sequence(
            num_frames=NUM_FRAMES,
            fps=FPS_TARGET,
            save_dir=SAVE_DIR,
            show_preview=SHOW_PREVIEW,
            blocking=BLOCKING_GRAB,
            save_format=SAVE_FORMAT,
            log_every=LOG_EVERY,
        )

    os._exit(0)