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
            Detection("tank", (0, 0, 100, 100)),
            Detection("wire", (10, 10, 20, 90)),
            Detection("wire", (40, 10, 50, 90)),
        ]


def _png_bytes():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".png", img)
    return buf.tobytes()


def test_analyze_endpoint_returns_report():
    app = create_app(FakeDetector(), ColorToHeat(build_lut("inferno")))
    client = TestClient(app)
    resp = client.post("/analyze", files={"file": ("t.png", _png_bytes(), "image/png")})
    assert resp.status_code == 200
    body = resp.json()
    assert "defects" in body
    assert "calibration_ok" in body
    assert "annotated_image_png_b64" in body
    # annotated image must be valid base64-decodable PNG bytes
    raw = base64.b64decode(body["annotated_image_png_b64"])
    assert raw[:8] == b"\x89PNG\r\n\x1a\n"
