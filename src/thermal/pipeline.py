import cv2
import numpy as np

from thermal.colormap import ColorToHeat
from thermal.defects import analyze_wires, analyze_transformer
from thermal.preprocess import preprocess
from thermal.schema import Detection

# Median Lab-distance above which we consider the palette mismatched.
CALIBRATION_DIST_THRESHOLD = 20.0
# Pad fraction around a detected transformer before running the wire detector on
# the crop. Matches the build-time crop pad (data_prep) so train == serve scale.
WIRE_CROP_PAD = 0.15


def _pad_clip(bbox, pad, w, h):
    x0, y0, x1, y1 = bbox
    px, py = int(round((x1 - x0) * pad)), int(round((y1 - y0) * pad))
    return max(0, x0 - px), max(0, y0 - py), min(w, x1 + px), min(h, y1 + py)


def analyze_image(img_rgb: np.ndarray, transformer_detector, wire_detector,
                  c2h: ColorToHeat, pad: float = WIRE_CROP_PAD):
    """Two-stage cascade. Returns (findings, intensity_map, calibration_ok).

    1. CLAHE-preprocess the frame (matching training) and detect transformers.
    2. For each transformer: crop it (+pad), detect wires INSIDE the crop, and map
       those wire boxes back to full-frame coords. This restricts wire detection to
       the transformer region (kills background false positives) and matches the
       crop scale the wire model trained on.
    3. Read RELATIVE heat on the raw palette: a localized transformer hotspot, and
       each transformer's wires compared against their own siblings.

    `*_detector` are anything exposing `detect(img) -> list[Detection]`, fed a BGR
    array (Ultralytics/cv2 convention — same channel order the models trained on).
    """
    # Adaptive CLAHE matches the `clahe` training variant; convert to BGR for the
    # detectors. Feeding RGB here silently swaps R/B and wrecks detection on a
    # palette image (the colors ARE the signal) — the cause of "boxes all wrong".
    proc_bgr = cv2.cvtColor(preprocess(img_rgb), cv2.COLOR_RGB2BGR)
    h, w = img_rgb.shape[:2]
    intensity, dist = c2h.to_intensity(img_rgb)   # heat read on the RAW RGB palette
    calibration_ok = bool(np.median(dist) < CALIBRATION_DIST_THRESHOLD)

    findings = []
    for t in transformer_detector.detect(proc_bgr):
        findings.append(analyze_transformer(intensity, t))
        x0, y0, x1, y1 = _pad_clip(t.bbox, pad, w, h)
        crop = proc_bgr[y0:y1, x0:x1]
        if crop.size == 0:
            continue
        wires_full = [
            Detection("wire", (x0 + wx0, y0 + wy0, x0 + wx1, y0 + wy1), wd.conf)
            for wd in wire_detector.detect(crop)
            for (wx0, wy0, wx1, wy1) in (wd.bbox,)
        ]
        findings += analyze_wires(intensity, wires_full)
    return findings, intensity, calibration_ok
