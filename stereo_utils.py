import numpy as np
import cv2

def thermal_to_bgr(
    mono16: np.ndarray, colormap: int = cv2.COLORMAP_INFERNO
    ) -> np.ndarray:
    norm = cv2.normalize(mono16, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    return cv2.applyColorMap(norm, colormap)