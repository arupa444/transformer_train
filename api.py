import os
import base64
import numpy as np
import cv2
from fastapi import FastAPI, UploadFile, File

from thermal.colormap import build_lut, ColorToHeat
from thermal.pipeline import analyze_image
from thermal.report import annotate, to_json


def create_app(detector, c2h: ColorToHeat) -> FastAPI:
    app = FastAPI(title="Transformer Thermal Defect Classifier")

    @app.post("/analyze")
    async def analyze(file: UploadFile = File(...)):
        data = await file.read()
        img_bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if img_bgr is None:
            return {"error": "could not decode image"}
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        findings, _intensity, calib_ok = analyze_image(img_rgb, detector, c2h)
        annotated = annotate(img_bgr, findings)
        ok, buf = cv2.imencode(".png", annotated)
        b64 = base64.b64encode(buf.tobytes()).decode()

        return {
            "calibration_ok": calib_ok,
            "defects": to_json(findings),
            "annotated_image_png_b64": b64,
        }

    return app


# Default app for `uvicorn api:app`. Guarded so tests can import without weights.
_weights = os.environ.get("THERMAL_WEIGHTS", "models/best.pt")
if os.path.exists(_weights):
    from thermal.detector import TransformerDetector
    app = create_app(TransformerDetector(_weights), ColorToHeat(build_lut("inferno")))
