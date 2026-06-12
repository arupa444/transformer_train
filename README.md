# Transformer Thermal Defect Classifier (2-model cascade)

Detects overheating wires and transformer hotspots in colorized thermal images.
A **cascade of two YOLO26x detectors** localizes the parts; a classical-CV layer
reads *relative* heat and flags defects.

> Output is **relative severity** (Normal/Watch/Investigate/Critical), not absolute °C —
> the input is colorized screenshots with no embedded temperature.

## Architecture

```
image → CLAHE → [YOLO26x transformer] → for each transformer:
                     crop + 0.15 pad → [YOLO26x wire] → map boxes back
                                            ↓
        relative-heat CV (raw palette): transformer hotspot + wire-vs-siblings → defects
```

Two single-class detectors (not one multi-class) because: it restricts wire
detection to the transformer region (kills foliage/background false positives),
matches the wire model's train scale to the crop it runs on, and lets each stage
be tuned independently. The detector only localizes; the CV layer decides defects.

## Setup
```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
pytest -q          # CV core + cascade pipeline + API tests pass without any model
```

## Get the two models
1. **Label** with labelImg — classes `transformer`, `wire` (see `docs/annotation-guide.md`).
2. **Build datasets** (dedup + leakage-safe split + adaptive CLAHE + transformer-anchored
   wire crops):
   ```bash
   python -m thermal.data_prep.build --subset both
   # -> /Volumes/dronisight/yolo_thermal_transformer  and  /Volumes/dronisight/Yolo_thermal_wire
   ```
3. **Train** on Colab — `notebooks/train_cascade_yolo26.ipynb` trains both YOLO26x
   detectors. Download `transformer.pt` + `wire.pt` into `models/`.

## Run the API
```bash
uv pip install -e ".[inference]"   # ultralytics/torch for the real detectors
THERMAL_TRANSFORMER_WEIGHTS=models/transformer.pt \
THERMAL_WIRE_WEIGHTS=models/wire.pt \
uvicorn api:app --reload
```
`POST /analyze` an image → JSON: `calibration_ok`, `defects[]` (component, bbox,
severity, relative_delta), and `annotated_image_png_b64`.

## Code map
- `src/thermal/data_prep/` — VOC parse + canonical names, **content-hash dedup/merge**
  (collapses byte-identical duplicates, unions partial labels), capture-time grouping,
  leakage-safe split, adaptive CLAHE, transformer-anchored wire crops, build orchestrator.
- `src/thermal/preprocess.py` — adaptive CLAHE (same transform at train + inference).
- `src/thermal/detector.py` — `YoloDetector`: generic single-model wrapper (one per stage).
- `src/thermal/colormap.py` — iron palette → 0–1 heat map (on the raw image).
- `src/thermal/defects.py` — relative hotspot + wire-vs-siblings + severity.
- `src/thermal/pipeline.py` — the cascade (`analyze_image`).
- `src/thermal/report.py` — annotated image + JSON.
- `api.py` — FastAPI `/analyze`.

## Tuning
Severity thresholds: `src/thermal/defects.py` (`_WATCH`/`_INVESTIGATE`/`_CRITICAL`).
Wire crop pad: `WIRE_CROP_PAD` in `pipeline.py` (build) and `data_prep/build.py` (must match).
If a different camera palette is used, change the colormap in `build_lut(...)` and re-check
`calibration_ok`.
