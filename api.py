import os
import base64
import numpy as np
import cv2
from fastapi import FastAPI, UploadFile, File, HTTPException

from thermal.colormap import build_lut, ColorToHeat
from thermal.pipeline import analyze_image
from thermal.report import annotate, to_json


def create_app(transformer_detector, wire_detector, c2h: ColorToHeat) -> FastAPI:
    app = FastAPI(title="Transformer Thermal Defect Classifier (cascade)")

    @app.post("/analyze")
    async def analyze(file: UploadFile = File(...)):
        data = await file.read()
        img_bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise HTTPException(status_code=422, detail="could not decode image")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        findings, _intensity, calib_ok = analyze_image(
            img_rgb, transformer_detector, wire_detector, c2h)
        annotated = annotate(img_bgr, findings)
        encode_ok, buf = cv2.imencode(".png", annotated)
        if not encode_ok:
            raise HTTPException(status_code=500, detail="failed to encode annotated image")
        return {
            "calibration_ok": calib_ok,
            "defects": to_json(findings),
            "annotated_image_png_b64": base64.b64encode(buf.tobytes()).decode(),
        }

    return app


# Default app for `uvicorn api:app`. Guarded so tests can import without weights.
_transformer_w = os.environ.get("THERMAL_TRANSFORMER_WEIGHTS", "models/transformer.pt")
_wire_w = os.environ.get("THERMAL_WIRE_WEIGHTS", "models/wire.pt")
if os.path.exists(_transformer_w) and os.path.exists(_wire_w):
    from thermal.detector import YoloDetector
    app = create_app(YoloDetector(_transformer_w), YoloDetector(_wire_w),
                     ColorToHeat(build_lut("inferno")))
