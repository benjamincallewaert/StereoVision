from pypylon import pylon
import numpy as np
import time

class BaslerStereoPair:
    def __init__(
        self,
        serial_left: str | None = 24856866,
        serial_right: str | None = 24856867,
        gain_db: float | None = None,
        exposure_us: float |None  = None,
        array: bool = False
    ):
        tlf = pylon.TlFactory.GetInstance()
        devices = tlf.EnumerateDevices()

        if len(devices) < 2:
            raise RuntimeError(
                f"Expected ≥ 2 Basler cameras, found {len(devices)}."
            )

        def _find_device(serial: str | None, exclude_idx: int | None = None):
            if serial is not None:
                for i, d in enumerate(devices):
                    print(i,d.GetSerialNumber())
                    if str(d.GetSerialNumber()) == str(serial) and i != exclude_idx:
                        return i, d
                raise RuntimeError(f"Camera with serial {serial!r} not found.")
            # Pick the first device that is not excluded
            for i, d in enumerate(devices):
                if i != exclude_idx:
                    return i, d
            raise RuntimeError("Not enough cameras.")

        if array:
            self.cameras = pylon.InstantCameraArray(2)
            for i in range(2):
                self.cameras[i].Attach(tlf.CreateDevice(devices[i]))
            self._cam_l = self.cameras[0]
            self._cam_r = self.cameras[1]
        else:
            idx_l, dev_l = _find_device(serial_left)
            idx_r, dev_r = _find_device(serial_right, exclude_idx=idx_l)
            self._cam_l = pylon.InstantCamera(tlf.CreateDevice(dev_l))
            self._cam_r = pylon.InstantCamera(tlf.CreateDevice(dev_r))
            self.cameras = (self._cam_l, self._cam_r)

        for cam in self.cameras:
            cam.Open()
            if not array:
                cam.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        
        if exposure_us is not None:
            self._set_exposure(exposure_us)
        
        if gain_db is not None:
            self._set_gain(gain_db)
  
        self._converter = pylon.ImageFormatConverter()
        self._converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        self._converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned


    def grab(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        left = self._grab_one(self._cam_l)
        right = self._grab_one(self._cam_r)
        return left, right
    
    def grab_with_count(self, delay = 5) -> tuple[np.ndarray | None, np.ndarray | None]:
        for i in range(delay, 0, -1):
            print(f"📸 Capturing in {i}...", end="\r", flush=True)
            time.sleep(1)
        return self.grab()
        
    def _grab_one(self, cam) -> np.ndarray | None:
        result = cam.RetrieveResult(500, pylon.TimeoutHandling_ThrowException)
        if result.GrabSucceeded():
            img = self._converter.Convert(result)
            arr = img.GetArray()
            result.Release()
            return arr
        result.Release()
        return None
    
    def _get_cameras(self):
        tlf = pylon.TlFactory.GetInstance()
        devices = tlf.EnumerateDevices()
        cameras = pylon.InstantCameraArray(2)
        for i in range(2):
            cameras[i].Attach(tlf.CreateDevice(devices[i]))
        return cameras
    
    def _set_exposure(self, exposure: int):
        for cam in (self._cam_l, self._cam_r):
            cam.ExposureAuto.SetValue("Off")
            cam.ExposureTime.SetValue(exposure)
    
    def _set_gain(self, gain:int):
        for cam in (self._cam_l, self._cam_r):
            cam.ExposureAuto.SetValue("Off")
            cam.ExposureTime.SetValue(gain)
    
    def close(self):
        for cam in (self._cam_l, self._cam_r):
            cam.StopGrabbing()
            cam.Close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()