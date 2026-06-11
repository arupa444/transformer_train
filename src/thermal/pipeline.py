import numpy as np
from thermal.colormap import ColorToHeat
from thermal.defects import analyze_wires, analyze_tank

# Median Lab-distance above which we consider the palette mismatched.
CALIBRATION_DIST_THRESHOLD = 20.0


def analyze_image(img_rgb: np.ndarray, detector,
                  c2h: ColorToHeat) -> tuple[list, np.ndarray, bool]:
    """Run the full pipeline. Returns (findings, intensity_map, calibration_ok).

    `detector` is any object exposing `detect(img_rgb) -> list[Detection]`.
    """
    dets = detector.detect(img_rgb)
    intensity, dist = c2h.to_intensity(img_rgb)
    calibration_ok = bool(np.median(dist) < CALIBRATION_DIST_THRESHOLD)

    wires = [d for d in dets if d.cls == "wire"]
    tanks = [d for d in dets if d.cls == "tank"]

    findings = analyze_wires(intensity, wires)
    findings += [analyze_tank(intensity, t) for t in tanks]
    return findings, intensity, calibration_ok
