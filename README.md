# Transformer Thermal Defect Classifier

Detects overheating wires and transformer hotspots in colorized thermal images.
A YOLOv8 detector crops the parts; a CV layer reads relative heat and flags defects.

> Output is **relative severity** (Normal/Watch/Investigate/Critical), not absolute °C.
> The input is colorized screenshots with no embedded temperature.

## Setup
```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
pytest -q          # all CV + API tests pass without a model
```

## Get a model
1. Label images with **labelImg** (classes: `transformer`, `wire`) — see
   `docs/annotation-guide.md`.
2. Train on Colab — open `notebooks/train_yolo.ipynb`, run all cells.
3. Download `best.pt` into `models/`.

## Run the API
```bash
uv pip install -e ".[inference]"   # installs ultralytics/torch for the real detector
THERMAL_WEIGHTS=models/best.pt uvicorn api:app --reload
```
Then POST an image:
```bash
curl -F "file=@your_image.jpg" http://127.0.0.1:8000/analyze
```
Returns JSON: `calibration_ok`, `defects[]` (component, bbox, severity,
relative_delta), and `annotated_image_png_b64` (base64 PNG with boxes).

## How it works
1. **Detector** (`src/thermal/detector.py`) finds `transformer` and `wire` boxes.
2. **Colormap** (`src/thermal/colormap.py`) inverts the iron palette to a 0–1 heat map.
3. **Defects** (`src/thermal/defects.py`): a wire much hotter than its siblings,
   or a localized hotspot on the transformer body (background pixels inside the
   box are excluded via an Otsu split), is flagged; severity scales with the gap.
4. **Report** (`src/thermal/report.py`) draws boxes + emits JSON.

The **pipeline** (`src/thermal/pipeline.py`) chains steps 1→4, and the shared
`Detection` / `DefectFinding` types live in `src/thermal/schema.py`.

## Tuning
Severity thresholds live in `src/thermal/defects.py` (`_WATCH`, `_INVESTIGATE`,
`_CRITICAL`). If a different camera palette is used, change the colormap name in
`api.py` / `build_lut(...)` and re-check `calibration_ok`.
