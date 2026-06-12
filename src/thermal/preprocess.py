import cv2
import numpy as np

# CLAHE settings: modest clip so we enhance local contrast without amplifying noise.
_CLAHE_CLIP = 2.0
_CLAHE_GRID = (8, 8)


def preprocess(img_rgb: np.ndarray) -> np.ndarray:
    """Enhance a colorized thermal image for the detector.

    Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) to the L
    (lightness) channel in LAB space, leaving the color/palette intact. This
    sharpens local structure (transformer edges, thin conductors) so YOLO
    localizes better on low-contrast thermal frames, without distorting the
    global palette the defect layer relies on.

    Input/output: HxWx3 uint8 RGB. Must be applied identically at training and
    inference time to avoid train/serve skew.
    """
    if img_rgb.ndim != 3 or img_rgb.shape[2] != 3:
        raise ValueError("preprocess expects an HxWx3 RGB image")
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    lightness, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=_CLAHE_CLIP, tileGridSize=_CLAHE_GRID)
    lightness = clahe.apply(lightness)
    return cv2.cvtColor(cv2.merge((lightness, a, b)), cv2.COLOR_LAB2RGB)
