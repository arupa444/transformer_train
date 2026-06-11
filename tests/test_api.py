import base64
import numpy as np
import cv2
from fastapi.testclient import TestClient
from thermal.schema import Detection
from thermal.colormap import build_lut, ColorToHeat
from api import create_app


class FakeDetector:
    def detect(self, img_rgb, conf=0.25):
        return [
            Detection("transformer", (0, 0, 100, 100)),
            Detection("wire", (10, 10, 20, 90)),
            Detection("wire", (40, 10, 50, 90)),
        ]


def _png_bytes():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".png", img)
    return buf.tobytes()


def _make_client():
    app = create_app(FakeDetector(), ColorToHeat(build_lut("inferno")))
    return TestClient(app)


def test_analyze_endpoint_returns_report():
    client = _make_client()
    resp = client.post("/analyze", files={"file": ("t.png", _png_bytes(), "image/png")})
    assert resp.status_code == 200
    body = resp.json()
    assert "defects" in body
    assert "calibration_ok" in body
    assert "annotated_image_png_b64" in body
    # annotated image must be valid base64-decodable PNG bytes
    raw = base64.b64decode(body["annotated_image_png_b64"])
    assert raw[:8] == b"\x89PNG\r\n\x1a\n"


def test_analyze_endpoint_defects_content():
    """FakeDetector returns 1 transformer + 2 wires, so findings must be 3
    entries with the expected component values and required keys."""
    client = _make_client()
    resp = client.post("/analyze", files={"file": ("t.png", _png_bytes(), "image/png")})
    assert resp.status_code == 200
    defects = resp.json()["defects"]
    # 2 wire findings + 1 transformer finding = 3 total
    assert len(defects) == 3
    components = [d["component"] for d in defects]
    assert components.count("wire") == 2
    assert components.count("transformer") == 1
    # every entry must have the required schema keys
    for d in defects:
        assert "component" in d
        assert "bbox" in d
        assert "severity" in d
        assert "relative_delta" in d
        assert isinstance(d["bbox"], list) and len(d["bbox"]) == 4


def test_analyze_invalid_image_returns_422():
    """Uploading non-image bytes must return HTTP 422, not 200."""
    client = _make_client()
    resp = client.post(
        "/analyze",
        files={"file": ("bad.png", b"not an image", "image/png")},
    )
    assert resp.status_code == 422
