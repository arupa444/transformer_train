# Transformer Thermal Defect Classifier

Detects overheating conductors/connections and hot spots on transformers from
colorized thermal images. One **YOLO26x detector** localizes the transformer; a
classical-CV layer reads *relative* heat inside it and flags hotspots.

> Output is **relative severity** (Normal/Watch/Investigate/Critical), not absolute °C —
> the input is colorized screenshots with no embedded temperature.

## Architecture

```
image → CLAHE → [YOLO26x transformer] → for each transformer:
                     pad 0.15 + scan crop for hot regions (relative heat, raw palette)
                                            ↓
                        connected-component hotspots → severity by heat-above-body
```

**One learned model, not two.** The transformer is a big, consistent object → YOLO
nails it. Wires are thin, densely-clustered conductors that a detector could not learn
on this dataset — so hot conductors/connections are found by **CV** (a region hotter
than the transformer body), which is exactly the defect signal and needs no training.
A *uniformly* hot tank can't be flagged from relative heat alone (needs absolute °C).

## Setup
```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
pytest -q          # CV core + pipeline + API tests pass without any model
```

## Get the model
1. **Label** with labelImg — classes `transformer`, `wire` (see `docs/annotation-guide.md`).
   (The `wire` boxes still feed the dataset build; only the *transformer* detector is trained.)
2. **Build datasets** (dedup + leakage-safe split + adaptive CLAHE):
   ```bash
   python -m thermal.data_prep.build --subset both
   # -> /Volumes/dronisight/yolo_thermal_transformer  (+ Yolo_thermal_wire, currently unused)
   ```
3. **Train + test** on Colab — zip `yolo_thermal_transformer`, upload to Google Drive, and
   run `notebooks/train_cascade_yolo26.ipynb` (mounts Drive → unzips → clones this repo →
   trains the YOLO26x transformer detector → evaluates → runs the full pipeline on test
   frames). The weights are saved to Drive; drop `transformer.pt` into `models/`.

## Run the API
```bash
uv pip install -e ".[inference]"   # ultralytics/torch for the real detector
THERMAL_TRANSFORMER_WEIGHTS=models/transformer.pt uvicorn api:app --reload
```
`POST /analyze` an image → JSON: `calibration_ok`, `defects[]` (component=`hotspot`,
bbox, severity, relative_delta), and `annotated_image_png_b64`.

## Code map
- `src/thermal/data_prep/` — VOC parse + canonical names, **content-hash dedup/merge**
  (collapses byte-identical duplicates, unions partial labels), capture-time grouping,
  leakage-safe split, adaptive CLAHE, build orchestrator.
- `src/thermal/preprocess.py` — adaptive CLAHE (same transform at train + inference).
- `src/thermal/detector.py` — `YoloDetector`: generic single-model wrapper.
- `src/thermal/colormap.py` — iron palette → 0–1 heat map (on the raw image).
- `src/thermal/defects.py` — `find_hotspots`: CV hot-region detection + severity.
- `src/thermal/pipeline.py` — `analyze_image`: transformer detect → CV hotspots.
- `src/thermal/report.py` — annotated image + JSON.
- `api.py` — FastAPI `/analyze`.

## Tuning
- Hotspot sensitivity / severity: `src/thermal/defects.py` — `_HOTSPOT_MARGIN`,
  `_WATCH`/`_INVESTIGATE`/`_CRITICAL` (default `0.45`/`0.55`/`0.62`). These are
  **calibrated on 755 real crops**: a normal warm connection sits ~+0.22..+0.44 above
  the transformer body; hot-wire defects are the tail at ~+0.55..+0.65. Lower the floor
  for more sensitivity (and more warm-connection noise).
- Crop pad around the transformer: `HOTSPOT_PAD` in `pipeline.py`.
- Different camera palette → change the colormap in `build_lut(...)` and re-check
  `calibration_ok`.
