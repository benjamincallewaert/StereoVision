from __future__ import annotations

from dataclasses import dataclass
import time
from pathlib import Path

import threading

import os
import vmbpy
import pyspin as PySpin
from pypylon import pylon

import enum
from stereo_utils import thermal_to_bgr
import cv2
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Public enum
# ──────────────────────────────────────────────────────────────────────────────

class CameraType(enum.Enum):
    BASLER        = "basler"
    ALLIED_VISION = "allied_vision"
    FLIR          = "flir"


# ──────────────────────────────────────────────────────────────────────────────
# Abstract base
# ──────────────────────────────────────────────────────────────────────────────

class CameraBase:
    """Minimal interface every concrete camera wrapper must implement."""

    def grab(self) -> tuple[np.ndarray, int] | None:
        """
        Capture a single frame.

        Returns
        -------
        (np.ndarray, int)
            - Array: BGR uint8 for colour cameras, Mono uint16 for FLIR thermal.
            - int: hardware timestamp in **nanoseconds**, latched at exposure.
        None
            On grab failure.
        """
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError

    def __enter__(self) -> "CameraBase":
        return self

    def __exit__(self, *_) -> None:
        self.close()


# ──────────────────────────────────────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ArucoDetection:
    """Result of one ArUco detection pass on a single frame."""
    ids:      np.ndarray | None   # shape (N, 1) int32; None when no markers found
    corners:  list                # list of (1, 4, 2) float32 corner arrays
    annotated: np.ndarray         # BGR copy of frame with markers drawn
    # Populated only when camera_matrix + dist_coeffs + marker_length_m are set
    rvecs: np.ndarray | None = None  # (N, 1, 3) rotation vectors
    tvecs: np.ndarray | None = None  # (N, 1, 3) translation vectors


@dataclass
class StereoFrame:
    left:           np.ndarray | None
    right:          np.ndarray | None
    left_hw_ts_ns:  int | None = None
    right_hw_ts_ns: int | None = None


@dataclass
class WheelFrame:
    image:              np.ndarray | None
    hw_ts_ns:           int | None = None
    detection:          ArucoDetection | None = None
    angular_position:   int | None = None

@dataclass(slots=True)
class CameraFrame:
    image: np.ndarray
    hw_ts_ns: int
    frame_id: int | None = None

# ──────────────────────────────────────────────────────────────────────────────
# Basler (pypylon)
# ──────────────────────────────────────────────────────────────────────────────

class BaslerCamera(CameraBase):
    """
    Wraps a single Basler camera via pypylon.

    Timestamp calibration
    ---------------------
    ``result.GetTimeStamp()`` tick frequency varies by model and transport
    layer.  The constructor auto-calibrates by timing two free-run grabs
    200 ms apart against the wall clock, giving the true ``ns_per_tick``
    without any hard-coded assumptions.
    """

    def __init__(
        self,
        serial: str | None,
        exposure_us: float = 35_000.0,
        gain_db: float = 15.0,
        use_external_trigger: bool = False,
        trigger_source: str = "Line1",
        roi_height: int | None = None,
        roi_offset_y: int | None = None,
    ) -> None:

        tlf     = pylon.TlFactory.GetInstance()
        devices = tlf.EnumerateDevices()
        if not devices:
            print("No Basler cameras detected.")
            os._exit(1)

        device = None
        if serial is not None:
            for d in devices:
                if str(d.GetSerialNumber()) == str(serial):
                    device = d
                    break
            if device is None:
                print(
                    f"Basler camera with serial {serial!r} not found. "
                    f"Available: {[str(d.GetSerialNumber()) for d in devices]}"
                )
                os._exit(1)
        else:
            device = devices[0]

        self._cam   = pylon.InstantCamera(tlf.CreateDevice(device))
        self._pylon = pylon
        self._cam.Open()
        if exposure_us is not None:
            self._cam.ExposureAuto.SetValue("Off")
            self._cam.ExposureTime.SetValue(exposure_us)
        else:
            self._cam.ExposureAuto.SetValue("Continuous")
        
        if gain_db is not None:
            self._cam.GainAuto.SetValue("Off")
            self._cam.Gain.SetValue(gain_db)
        else:
            self._cam.GainAuto.SetValue("Continuous")

        # Optional: Enable auto white balance (for color cameras)
        self._cam.BalanceWhiteAuto.SetValue("Continuous")
        self._converter = pylon.ImageFormatConverter()
        self._converter.OutputPixelFormat  = pylon.PixelType_BGR8packed
        self._converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

        self._use_external_trigger = use_external_trigger

        if use_external_trigger:
            self._cam.TriggerSelector.SetValue("FrameStart")
            self._cam.TriggerSource.SetValue(trigger_source)
            self._cam.TriggerActivation.SetValue("RisingEdge")
            self._cam.TriggerMode.SetValue("On")
            grab_strategy = pylon.GrabStrategy_OneByOne
        else:
            self._cam.TriggerMode.SetValue("Off")
            grab_strategy = pylon.GrabStrategy_OneByOne

        # ── Vertical ROI (sensor-side AOI) ────────────────────────────────────
        # Cropping the height ON THE SENSOR (not in software) cuts the data that
        # is transferred and encoded, which is the real lever for keeping up with
        # a fast trigger.  Width is left full; the band is centred unless an
        # explicit offset is given.  Must be set while not grabbing.
        if roi_height is not None:
            try:
                self._cam.OffsetY.SetValue(0)
                h_max = self._cam.Height.GetMax()
                h_inc = self._cam.Height.GetInc() or 1
                h     = max(self._cam.Height.GetMin(), min(int(roi_height), h_max))
                h    -= h % h_inc
                self._cam.Height.SetValue(h)

                off   = (h_max - h) // 2 if roi_offset_y is None else int(roi_offset_y)
                o_inc = self._cam.OffsetY.GetInc() or 1
                off  -= off % o_inc
                off   = max(0, min(off, h_max - h))
                self._cam.OffsetY.SetValue(off)
                print(f"[Basler] ROI applied: Height={h} OffsetY={off} (full width)")
            except Exception as exc:
                print(f"[Basler] ROI set failed ({exc}); using full frame.")

        # Enlarge the driver-side ring buffer so a brief disk/encoder stall does
        # not immediately overrun the camera and drop triggers.  Each buffer
        # holds one full frame (~32 MB at 10 MP), so this trades RAM for burst
        # tolerance — raise/lower to taste.
        try:
            self._cam.MaxNumBuffer.SetValue(64)
        except Exception:
            try:
                self._cam.MaxNumBuffer = 64
            except Exception:
                pass

        self._cam.StartGrabbing(grab_strategy)

        trigger_info = (
            f"trigger=external({trigger_source}, rising-edge)"
            if use_external_trigger else "trigger=free-run"
        )
        print(
            f"[Basler] serial={device.GetSerialNumber()} "
            f"exposure={exposure_us}µs gain={gain_db}dB {trigger_info} "
        )


    def grab(self) -> CameraFrame | None:

        timeout_ms = 5_000 if self._use_external_trigger else 500

        result = self._cam.RetrieveResult(
            timeout_ms,
            self._pylon.TimeoutHandling_ThrowException
        )

        if not result.GrabSucceeded():
            result.Release()
            return None

        hw_ts_ns = int(result.GetTimeStamp())

        frame_id = None
        try:
            frame_id = int(result.GetBlockID())
        except Exception:
            pass

        # ── Convert to proper BGR image ─────────────────────────
        converted = self._converter.Convert(result)

        arr = converted.GetArray().copy()

        result.Release()

        return CameraFrame(
            image=arr,
            hw_ts_ns=hw_ts_ns,
            frame_id=frame_id,
        )

    def _set_fps(self, fps: float) -> None:
        try:
            self._cam.AcquisitionFrameRateEnable.SetValue(True)
            self._cam.AcquisitionFrameRate.SetValue(fps)
        except Exception:
            pass

    def close(self) -> None:
        if self._cam.IsGrabbing():
            self._cam.StopGrabbing()
        self._cam.Close()


# ──────────────────────────────────────────────────────────────────────────────
# Allied Vision Alvium 
# ──────────────────────────────────────────────────────────────────────────────

class AlliedVisionCamera(CameraBase):
    """
    Wraps a single Allied Vision Alvium camera via VmbPy.

    ``frame.get_timestamp()`` is documented as nanoseconds for Alvium models;
    ``GevTimestampTickFrequency`` is read as a cross-check if available.
    """

    def __init__(
        self,
        serial: str | None,
        vmb,
        exposure_us: float = 35_000.0,
        gain_db: float = 15.0,
        use_external_trigger: bool = False,
        trigger_source: str = "Line1",
    ) -> None:

        cameras = vmb.get_all_cameras()
        if not cameras:
            print("No Allied Vision cameras detected.")
            os._exit(1)

        cam_obj = None
        if serial is not None:
            for c in cameras:
                if c.get_id() == str(serial) or c.get_serial() == str(serial):
                    cam_obj = c; break
            if cam_obj is None:
                print(
                    f"Allied Vision camera {serial!r} not found. "
                    f"Available: {[c.get_id() for c in cameras]}"
                )
                os._exit(1)
        else:
            cam_obj = cameras[0]

        self._cam   = cam_obj
        self._vmbpy = vmbpy
        self._use_external_trigger = use_external_trigger
        self._cam.__enter__()

        with self._cam:
            if exposure_us is not None:
                try:
                    self._cam.ExposureMode.set("Off")
                    self._cam.ExposureTime.set(exposure_us)
                except:
                    pass
            if gain_db is not None:
                try:
                    self._cam.GainAuto.set("Off")
                    self._cam.Gain.set(gain_db)
                except Exception:
                    pass
            if use_external_trigger:
                try:
                    self._cam.TriggerSelector.set("FrameStart")
                    self._cam.TriggerSource.set(trigger_source)
                    self._cam.TriggerActivation.set("RisingEdge")
                    self._cam.TriggerMode.set("On")
                except Exception as exc:
                    print(
                        f"[AlliedVision] Trigger config failed: {exc}"
                    )
                    os._exit(1)
            else:
                try:
                    self._cam.TriggerMode.set("Off")
                except Exception:
                    pass

        trigger_info = (
            f"trigger=external({trigger_source}, rising-edge)"
            if use_external_trigger else "trigger=free-run"
        )
        print(
            f"[AlliedVision] serial={cam_obj.get_serial()} "
            f"exposure={exposure_us}µs gain={gain_db}dB {trigger_info} "
        )

    def _set_fps(self, fps: float) -> None:
        try:
            self._cam.get_feature_by_name("AcquisitionFrameRateEnable").set(True)
            self._cam.get_feature_by_name("AcquisitionFrameRate").set(fps)
        except Exception:
            print(f"[Allied Vision] Could not set FPS to {fps}")

    def grab(self) -> tuple[np.ndarray, int] | None:
        timeout_ms = 500_000 if self._use_external_trigger else 500
        try:
            frame     = self._cam.get_frame(timeout_ms=timeout_ms)
            hw_ts_ns  = int(frame.get_timestamp())
            frame_id = None
            try:
                frame_id = int(frame.get_id())
            except Exception:
                pass
            return CameraFrame(
                image=frame.as_numpy_ndarray().copy(),
                hw_ts_ns=hw_ts_ns,
                frame_id=frame_id,
            )
        except Exception as exc:
            print(f"[AlliedVision] grab failed: {exc}")
        return None

    def close(self) -> None:
        try:
            self._cam.__exit__(None, None, None)
        except Exception:
            pass


# ──────────────────────────────────────────────────────────────────────────────
# FLIR thermal (PySpin)
# ──────────────────────────────────────────────────────────────────────────────

class FlirCamera(CameraBase):
    """
    Wraps a single FLIR camera via PySpin (Spinnaker SDK).
    ``image.GetTimeStamp()`` returns ns natively.
    """

    def __init__(
        self,
        serial: str | None,
        exposure_us: float = 35_000.0,
        gain_db: float = 15.0,
        use_external_trigger: bool = False,
        trigger_source: str = "Line3",
    ) -> None:

        self._PySpin = PySpin
        self._system = PySpin.System.GetInstance()
        cam_list     = self._system.GetCameras()

        if cam_list.GetSize() == 0:
            cam_list.Clear(); self._system.ReleaseInstance()
            print("No FLIR cameras detected.")
            os._exit(1)

        cam_obj = (
            cam_list.GetBySerial(str(serial)) if serial is not None
            else cam_list.GetByIndex(0)
        )
        cam_list.Clear()

        if cam_obj is None or not cam_obj.IsValid():
            self._system.ReleaseInstance()
            print(f"FLIR camera with serial {serial!r} not found.")
            os._exit(1)

        self._cam = cam_obj
        self._use_external_trigger = use_external_trigger
        self._cam.Init()
        nodemap = self._cam.GetNodeMap()

        try:
            node_exp_auto = PySpin.CEnumerationPtr(nodemap.GetNode("ExposureAuto"))
            if PySpin.IsAvailable(node_exp_auto) and PySpin.IsWritable(node_exp_auto):
                entry = node_exp_auto.GetEntryByName("Off")
                if PySpin.IsAvailable(entry) and PySpin.IsReadable(entry):
                    node_exp_auto.SetIntValue(entry.GetValue())
            node_exp = PySpin.CFloatPtr(nodemap.GetNode("ExposureTime"))
            if PySpin.IsAvailable(node_exp) and PySpin.IsWritable(node_exp):
                node_exp.SetValue(float(np.clip(
                    exposure_us, node_exp.GetMin(), node_exp.GetMax()
                )))
        except PySpin.SpinnakerException:
            pass

        self._configure_trigger(nodemap, trigger_source, enable=use_external_trigger)
        self._cam.BeginAcquisition()

        sn = self._cam.TLDevice.DeviceSerialNumber.GetValue()
        trigger_info = (
            f"trigger=external({trigger_source}, rising-edge)"
            if use_external_trigger else "trigger=free-run"
        )
        print(f"[FLIR] serial={sn} exposure={exposure_us}µs {trigger_info}")

    def _configure_trigger(self, nodemap, trigger_source: str, *, enable: bool) -> None:
        PySpin = self._PySpin

        def _get_enum(name):
            ptr = PySpin.CEnumerationPtr(nodemap.GetNode(name))
            if not (PySpin.IsAvailable(ptr) and PySpin.IsWritable(ptr)):
                print(f"Node {name!r} not available/writable.")
                os._exit(1)
            return ptr

        def _set_enum(ptr, entry_name):
            entry = ptr.GetEntryByName(entry_name)
            if not (PySpin.IsAvailable(entry) and PySpin.IsReadable(entry)):
                print(f"Entry {entry_name!r} unavailable.")
                os._exit(1)
            ptr.SetIntValue(entry.GetValue())

        try:
            tm = _get_enum("TriggerMode")
            _set_enum(tm, "Off")
            if enable:
                _set_enum(_get_enum("TriggerSelector"), "FrameStart")
                _set_enum(_get_enum("TriggerSource"),   trigger_source)
                try: _set_enum(_get_enum("TriggerActivation"), "RisingEdge")
                except Exception: pass
                _set_enum(tm, "On")
        except PySpin.SpinnakerException as exc:
            print(f"[FLIR] trigger config failed: {exc}")
            os._exit()

    def _set_fps(self, fps: float) -> None:
        PySpin = self._PySpin
        try:
            if self._cam.AcquisitionFrameRateAuto.GetAccessMode() == PySpin.RW:
                self._cam.AcquisitionFrameRateAuto.SetValue(
                    PySpin.AcquisitionFrameRateAuto_Off
                )
                self._cam.AcquisitionFrameRateEnable.SetValue(True)
                self._cam.AcquisitionFrameRate.SetValue(fps)
        except Exception:
            print(f"[FLIR] Could not set FPS to {fps}")

    def grab(self) -> tuple[np.ndarray, int] | None:
        PySpin     = self._PySpin
        timeout_ms = 5_000 if self._use_external_trigger else 1_000
        try:
            image = self._cam.GetNextImage(timeout_ms)
            if image.IsIncomplete():
                image.Release(); return None
            hw_ts_ns = int(image.GetTimeStamp())
            arr = (
                image.GetNDArray().copy()
                if image.GetPixelFormatName().startswith("Mono")
                else image.Convert(PySpin.PixelFormat_BGR8).GetNDArray().copy()
            )
            frame_id = None
            try:
                frame_id = int(image.GetFrameID())
            except Exception:
                pass
            image.Release()
            return CameraFrame(
                image=arr,
                hw_ts_ns=hw_ts_ns,
                frame_id=frame_id,
            )
        except PySpin.SpinnakerException as exc:
            print(f"[FLIR] grab failed: {exc}")
            return None

    def close(self) -> None:
        for fn in (self._cam.EndAcquisition, self._cam.DeInit):
            try: fn()
            except Exception: pass
        try:
            del self._cam
            self._system.ReleaseInstance()
        except Exception: pass


# ──────────────────────────────────────────────────────────────────────────────
# Internal factory
# ──────────────────────────────────────────────────────────────────────────────

def _make_camera(
    serial: str | None,
    cam_type: CameraType,
    exposure_us: float,
    gain_db: float,
    use_external_trigger: bool,
    trigger_source: str,
    vmb=None,
    roi_height: int | None = None,
    roi_offset_y: int | None = None,
) -> CameraBase:
    """Create the appropriate CameraBase subclass.

    roi_height / roi_offset_y apply a sensor-side vertical crop (Basler only for
    now); pass None for the full frame.
    """
    if cam_type == CameraType.BASLER:
        return BaslerCamera(
            serial,
            exposure_us=exposure_us, gain_db=gain_db,
            use_external_trigger=use_external_trigger,
            trigger_source=trigger_source,
            roi_height=roi_height, roi_offset_y=roi_offset_y,
        )
    elif cam_type == CameraType.ALLIED_VISION:
        if vmb is None:
            print("vmb context required for Allied Vision cameras.")
            os._exit(1)
        return AlliedVisionCamera(
            serial, vmb,
            exposure_us=exposure_us, gain_db=gain_db,
            use_external_trigger=use_external_trigger,
            trigger_source=trigger_source,
        )
    elif cam_type == CameraType.FLIR:
        return FlirCamera(
            serial,
            exposure_us=exposure_us, gain_db=gain_db,
            use_external_trigger=use_external_trigger,
            trigger_source=trigger_source,
        )
    else:
        print(f"Unknown CameraType: {cam_type!r}")
        os._exit(1)

# ──────────────────────────────────────────────────────────────────────────────
# CAMERA WORKER
# ──────────────────────────────────────────────────────────────────────────────
class CameraWorker:
    """
    Background grab thread.

    Two modes:
      * latest-frame (default): keeps only the most recent frame, for free-run
        snapshotting (SyncRig.grab_sync).
      * sink mode: when a sink queue is attached via attach_sink(), EVERY grabbed
        frame is pushed to the queue as ``(label, CameraFrame)`` — this is what
        the triggered capture uses so no triggered frame is dropped in software.
        The queue is bounded, so if the writers fall behind the put() blocks,
        which back-pressures the camera (and any resulting real drop is detected
        via the frame_id gap below).
    """

    def __init__(self, cam: CameraBase):
        self.cam = cam

        self.latest_frame: CameraFrame | None = None

        self.lock = threading.Lock()
        self.running = False

        # ── sink mode state ───────────────────────────────────────────────────
        self._sink = None                 # queue.Queue | None
        self._label: str | None = None
        self._prev_fid: int | None = None
        self.dropped = 0                  # triggers the camera produced but we missed

        self.thread = threading.Thread(
            target=self._run,
            daemon=True,
        )

    # ── sink control ───────────────────────────────────────────────────────────
    def attach_sink(self, sink, label: str) -> None:
        """Route every subsequent frame to ``sink`` as ``(label, frame)``."""
        self._prev_fid = None
        self.dropped   = 0
        self._label    = label
        self._sink     = sink            # set last: the read in _run is then valid

    def detach_sink(self) -> None:
        self._sink = None

    def start(self):
        self.running = True
        self.thread.start()

    def stop(self):
        self.running = False
        self.thread.join()

    def _run(self):
        while self.running:
            try:
                frame = self.cam.grab()
            except Exception:
                # A transient grab failure (e.g. timeout) must not kill the
                # worker thread or spin the CPU at 100%.
                time.sleep(0.001)
                continue

            if frame is None:
                time.sleep(0.001)
                continue

            with self.lock:
                self.latest_frame = frame

            sink = self._sink
            if sink is not None:
                # Detect triggers the camera produced but we never retrieved
                # (frame_id / BlockID increments once per trigger).
                if frame.frame_id is not None and self._prev_fid is not None:
                    gap = frame.frame_id - self._prev_fid - 1
                    if gap > 0:
                        self.dropped += gap
                self._prev_fid = frame.frame_id
                # Blocks when writers are behind -> back-pressure, not RAM blow-up.
                sink.put((self._label, frame))

    def get_latest(self):
        with self.lock:
            return self.latest_frame

# ──────────────────────────────────────────────────────────────────────────────
# WheelCamera
# ──────────────────────────────────────────────────────────────────────────────

class WheelCamera:
    """
    Single triggered camera pointed at the wheel side for ArUco marker detection.

    Uses the same :func:`_make_camera` factory as :class:`StereoPair`, so any
    supported camera type can be used.

    Parameters
    ----------
    serial:
        Camera serial number. Pass *None* for the first detected camera of
        ``cam_type``.
    cam_type:
        One of the :class:`CameraType` enum values.
    exposure_us / gain_db:
        Sensor settings.
    use_external_trigger:
        Put the camera into hardware-trigger mode. Should match the stereo pair.
    trigger_source:
        Physical input line, e.g. ``"Line1"``.
    aruco_dict_id:
        Any ``cv2.aruco.DICT_*`` constant.
        Default: ``DICT_4X4_50`` (large, robust markers).
    camera_matrix:
        3 × 3 intrinsic matrix (``np.ndarray float64``).
        Required for pose estimation; pass *None* to skip pose.
    dist_coeffs:
        Distortion coefficients (1 × 5 or similar).
        Required for pose estimation; pass *None* to skip pose.
    marker_length_m:
        Physical side length of the marker in metres.
        Required for pose estimation; ignored otherwise.

    Notes on VmbPy
    --------------
    If ``cam_type`` is ``ALLIED_VISION`` and a :class:`StereoPair` that also
    uses Allied Vision cameras already exists, pass its ``_vmb`` attribute as
    ``vmb`` so both share the singleton ``VmbSystem`` context::

        pair = StereoPair(...)
        wheel = WheelCamera(..., cam_type=CameraType.ALLIED_VISION,
                            vmb=pair._vmb)
    """

    def __init__(
        self,
        serial: str | None,
        cam_type: CameraType = CameraType.BASLER,
        exposure_us: float = 35_000.0,
        gain_db: float = 15.0,
        use_external_trigger: bool = False,
        trigger_source: str = "Line1",
        aruco_dict_id: int = cv2.aruco.DICT_4X4_50,
        camera_matrix: np.ndarray | None = None,
        dist_coeffs:   np.ndarray | None = None,
        marker_length_m: float | None = None,
        vmb=None,  # pass an existing VmbSystem context for Allied Vision
    ) -> None:

        self._vmb_owned = False  # True if *we* opened the VmbSystem context

        # Open VmbSystem only if this camera needs it and no context was passed.
        if cam_type == CameraType.ALLIED_VISION and vmb is None:
            vmb = vmbpy.VmbSystem.get_instance().__enter__()
            self._vmb_owned = True
        self._vmb = vmb

        self._cam: CameraBase = _make_camera(
            serial, cam_type, exposure_us, gain_db,
            use_external_trigger, trigger_source,
            vmb=self._vmb,
        )
        self._worker = CameraWorker(self._cam)
        # ── ArUco detector ─────────────────────────────────────────────────
        aruco_dict   = cv2.aruco.getPredefinedDictionary(aruco_dict_id)
        aruco_params = cv2.aruco.DetectorParameters()
        self._detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

        # ── Pose estimation (optional) ─────────────────────────────────────
        self._camera_matrix    = camera_matrix
        self._dist_coeffs      = dist_coeffs
        self._marker_length_m  = marker_length_m
        self._pose_ready = (
            camera_matrix   is not None and
            dist_coeffs     is not None and
            marker_length_m is not None
        )

        print(
            f"[WheelCamera] ArUco dict={aruco_dict_id} "
            f"pose_estimation={'on' if self._pose_ready else 'off (no calibration)'}"
        )

    # ── Core grab ──────────────────────────────────────────────────────────
    def grab(self) -> tuple[np.ndarray, int] | None:
        return self._cam.grab()

    # ── Grab + detect ──────────────────────────────────────────────────────
    def grab_and_detect(self) -> WheelFrame | None:
        """
        Capture one frame and run ArUco marker detection on it.

        Returns
        -------
        :class:`WheelFrame`
            ``image``     – raw BGR frame (uint8).
            ``hw_ts_ns``  – hardware timestamp in nanoseconds.
            ``detection`` – :class:`ArucoDetection` with corner positions,
                            IDs, annotated preview image, and (optionally)
                            pose vectors.
        None
            If the grab itself failed.
        """
        result = self._cam.grab()
        if result is None:
            return None

        img, hw_ts_ns = result
        detection     = self._detect(img)
        return WheelFrame(image=img, hw_ts_ns=hw_ts_ns, detection=detection)

    # ── ArUco internals ────────────────────────────────────────────────────
    def _detect(self, bgr: np.ndarray) -> ArucoDetection:
        """Run marker detection (and optionally pose estimation) on *bgr*."""
        gray              = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        corners, ids, _   = self._detector.detectMarkers(gray)

        annotated = bgr.copy()
        rvecs = tvecs = None

        if ids is not None and len(ids) > 0:
            cv2.aruco.drawDetectedMarkers(annotated, corners, ids)

            if self._pose_ready:
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                    corners,
                    self._marker_length_m,
                    self._camera_matrix,
                    self._dist_coeffs,
                )
                for rvec, tvec in zip(rvecs, tvecs):
                    cv2.drawFrameAxes(
                        annotated,
                        self._camera_matrix, self._dist_coeffs,
                        rvec, tvec,
                        self._marker_length_m * 0.5,
                    )

        return ArucoDetection(
            ids=ids, corners=corners, annotated=annotated,
            rvecs=rvecs, tvecs=tvecs,
        )

    # ── Resource management ────────────────────────────────────────────────
    def close(self) -> None:
        try:
            self._worker.stop()
            self._cam.close()
        except Exception:
            pass
        if self._vmb_owned and self._vmb is not None:
            try:
                self._vmb.__exit__(None, None, None)
            except Exception:
                pass
            self._vmb = None

    def __enter__(self) -> "WheelCamera":
        return self

    def __exit__(self, *_) -> None:
        self.close()


# ──────────────────────────────────────────────────────────────────────────────
# StereoPair
# ──────────────────────────────────────────────────────────────────────────────

class StereoPair:
    """
    Open two cameras (of any supported type) and provide synchronised frame
    capture.
    """
    def __init__(
        self,
        serial_left:           str | None  = "24856866",
        type_left:             CameraType  = CameraType.BASLER,
        serial_right:          str | None  = "24856864",
        type_right:            CameraType  = CameraType.BASLER,
        exposure_us:           float       = 35_000.0,
        gain_db:               float       = 15.0,
        use_external_trigger:  bool        = False,
        trigger_source_left:   str         = "Line1",
        trigger_source_right:  str         = "Line1",
    ) -> None:
        self._vmb   = None
        self._cam_l: CameraBase
        self._cam_r: CameraBase

        needs_vmb = (
            type_left  == CameraType.ALLIED_VISION or
            type_right == CameraType.ALLIED_VISION
        )
        if needs_vmb:
            self._vmb = vmbpy.VmbSystem.get_instance().__enter__()

        self._cam_l = _make_camera(
            serial_left,  type_left,  exposure_us, gain_db,
            use_external_trigger, trigger_source_left,  vmb=self._vmb,
        )
        self._cam_r = _make_camera(
            serial_right, type_right, exposure_us, gain_db,
            use_external_trigger, trigger_source_right, vmb=self._vmb,
        )
        self._worker_l = CameraWorker(self._cam_l)
        self._worker_r = CameraWorker(self._cam_r)

        self._worker_l.start()
        self._worker_r.start()

    def grab(self) -> StereoFrame:
        left_result  = self._cam_l.grab()
        right_result = self._cam_r.grab()
        left_img,  left_ts  = left_result  if left_result  is not None else (None, None)
        right_img, right_ts = right_result if right_result is not None else (None, None)
        return StereoFrame(left_img, right_img, left_ts, right_ts)

    def grab_sequence(
        self,
        num_frames: int,
        fps: float,
        save_dir: str | Path | None = None,
        show_preview: bool = True,
    ) -> list[StereoFrame]:
        period = 1.0 / fps
        frames: list[StereoFrame] = []

        if save_dir is not None:
            save_dir  = Path(save_dir)
            (save_dir / "left").mkdir(parents=True, exist_ok=True)
            (save_dir / "right").mkdir(parents=True, exist_ok=True)

        for idx in range(num_frames):
            t0     = time.perf_counter()
            stereo = self.grab()

            if stereo.left is None or stereo.right is None:
                print(f"[WARN] Failed frame {idx}")
                continue
            frames.append(stereo)

            if save_dir is not None:
                cv2.imwrite(str(save_dir / "left"  / f"left_{idx:04d}.png"),  stereo.left)
                cv2.imwrite(str(save_dir / "right" / f"right_{idx:04d}.png"), stereo.right)

            if show_preview:
                l_show = thermal_to_bgr(stereo.left)  if stereo.left.dtype  == np.uint16 else stereo.left
                r_show = thermal_to_bgr(stereo.right) if stereo.right.dtype == np.uint16 else stereo.right
                cv2.imshow("Left",  l_show); cv2.imshow("Right", r_show)
                cv2.waitKey(1)

            remaining = period - (time.perf_counter() - t0)
            if remaining > 0:
                time.sleep(remaining)

        return frames

    def close(self) -> None:
        self._worker_l.stop()
        self._worker_r.stop()
        for cam in (self._cam_l, self._cam_r):
            try: cam.close()
            except Exception: pass
        if self._vmb is not None:
            try: self._vmb.__exit__(None, None, None)
            except Exception: pass
            self._vmb = None

    def __enter__(self) -> "StereoPair":
        return self

    def __exit__(self, *_) -> None:
        self.close()


# ──────────────────────────────────────────────────────────────────────────────
# Demo
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    LEFT_SERIAL  = "DEV_Cam1"; LEFT_TYPE  = CameraType.BASLER
    RIGHT_SERIAL = "24856864"; RIGHT_TYPE = CameraType.BASLER
    WHEEL_SERIAL = "24856866"; WHEEL_TYPE = CameraType.BASLER

    USE_HW_TRIGGER = False

    with StereoPair(
        serial_left=LEFT_SERIAL,   type_left=LEFT_TYPE,
        serial_right=RIGHT_SERIAL, type_right=RIGHT_TYPE,
        exposure_us=350.0, gain_db=10.0,
        use_external_trigger=USE_HW_TRIGGER,
        trigger_source_left="Line0", trigger_source_right="Line3",
    ) as pair:
        with WheelCamera(
            serial=WHEEL_SERIAL, cam_type=WHEEL_TYPE,
            exposure_us=350.0, gain_db=10.0,
            use_external_trigger=USE_HW_TRIGGER,
            trigger_source="Line3",
        ) as wheel:

            print("Grabbing — press any key to exit.")
            while True:
                stereo      = pair.grab()
                wheel_frame = wheel.grab_and_detect()

                if stereo.left is None or stereo.right is None or wheel_frame is None:
                    print("Grab returned None; retrying…"); continue

                l_show = thermal_to_bgr(stereo.left)  if stereo.left.dtype  == np.uint16 else stereo.left
                r_show = thermal_to_bgr(stereo.right) if stereo.right.dtype == np.uint16 else stereo.right

                cv2.imshow("Left",  l_show)
                cv2.imshow("Right", r_show)
                cv2.imshow("Wheel", wheel_frame.detection.annotated)

                det = wheel_frame.detection
                if det.ids is not None:
                    print(f"  Detected markers: {det.ids.flatten().tolist()}")

                if cv2.waitKey(1) != -1:
                    break

    cv2.destroyAllWindows()
    os._exit(0)