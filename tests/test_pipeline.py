import numpy as np
from thermal.schema import Detection
from thermal.colormap import build_lut, ColorToHeat
from thermal.pipeline import analyze_image


class FakeDetector:
    """Returns fixed detections regardless of input."""
    def __init__(self, dets):
        self._dets = dets
    def detect(self, img_rgb, conf=0.25):
        return self._dets


def test_analyze_image_returns_findings_and_calibration():
    img = np.zeros((100, 100, 3), dtype=np.uint8)  # all black -> low intensity
    detector = FakeDetector([
        Detection("transformer", (0, 0, 100, 100)),
        Detection("wire", (10, 10, 20, 90)),
        Detection("wire", (40, 10, 50, 90)),
    ])
    c2h = ColorToHeat(build_lut("inferno"))
    findings, intensity, calib_ok = analyze_image(img, detector, c2h)
    assert intensity.shape == (100, 100)
    components = sorted({f.component for f in findings})
    assert components == ["transformer", "wire"]
    assert isinstance(calib_ok, bool)
