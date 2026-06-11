# Transformer Thermal Defect Classifier — Design

**Date:** 2026-06-11
**Status:** Approved (direction)

## Goal

Automatically detect overheating defects in transformers from **colorized thermal
images** (iron palette screenshots — no embedded temperature, no scale bar). The
signal of interest: a wire/bushing or part of the tank that is **abnormally hot
relative to its surroundings** (e.g. a white-hot conductor while the rest of the
unit glows orange).

## Core constraints (locked)

1. **Input is colorized screenshots only.** No radiometric data → output is
   **relative severity**, never absolute °C. Reports must state this.
2. **Background contains heat-like distractors** (sun band, sky, hot grating).
   Localization must isolate the transformer so these don't cause false alarms.
3. **Fully automatic** — no human draws the ROI at inference time.
4. Defect granularity for v1: **`hot wire` and `hot tank`** (component-level,
   not full root-cause diagnosis).

## Architectural decision: detector crops, CV judges

Two stages with strictly separated jobs:

- **Object detector (YOLOv8-nano):** finds and crops physical parts only.
  Learns shapes — *not* defects.
- **CV layer:** reads heat from the colors inside each crop and decides what is a
  defect via **relative** heat comparison.

Rationale: labeling "defect vs normal" is subjective and defect examples are rare,
so it makes a poor training target. Labeling parts (`tank`, `wire`) is objective
and present in every image. Train the model on the easy objective task; let CV +
physics handle the heat judgment. This also removes the fragile heuristic auto-ROI
from the original plan — the trained detector does localization robustly.

## Classes (annotation)

Exactly two, both physical parts:

| Class  | What is boxed |
|--------|---------------|
| `tank` | The main rectangular transformer body |
| `wire` | The cables / bushing conductors entering the unit |

No defect/normal classes. (`bushing` may be added later for finer detail.)

## Pipeline

```
image → [detector: YOLO] → crops (tank, wire boxes)
                              ↓
        [colormap inversion] → 0–1 heat-intensity map
                              ↓
        [defect CV] → relative hotspot + wire-to-wire comparison + severity
                              ↓
        [report] → annotated image + JSON
```

### Stage 1 — Detector (`detector.py` + Colab training)
- YOLOv8-nano, transfer learning, 2 classes.
- Trained in a Colab `.ipynb` on the annotated dataset (exported from Roboflow).
- Output: `best.pt` weights, loaded by the inference service.

### Stage 2 — Colormap inversion (`colormap.py`)
- Reference 256-step iron-palette LUT (black→navy→purple→magenta→orange→yellow→white
  = cold→hot).
- For each pixel: nearest palette color in **Lab** space (robust to JPEG noise) →
  position along palette → **0–1 heat intensity**.
- Calibration check: if pixels match the LUT poorly (wrong palette), warn rather
  than emit garbage.

### Stage 3 — Defect CV (`defects.py`)
- **Relative hotspot:** within each crop, flag regions above a robust threshold
  derived from that region itself (e.g. `median + k·MAD`). Catches a white-hot
  wire even when the whole tank glows.
- **Wire-to-wire comparison:** group detected wires; one significantly hotter than
  its siblings = classic loose-connection / overload signature.
- **Severity:** bucket by how far above reference the hotspot sits →
  `Normal / Watch / Investigate / Critical`. Relative units only.

### Stage 4 — Report (`report.py`)
- Annotated image (boxes + heatmap overlay).
- JSON: per-defect `{ component, bbox, severity, relative_delta }`.

### Stage 5 — Inference service (`api.py`, FastAPI)
- Endpoint: `POST /analyze` — accepts an image, returns annotated image + JSON.
- Loads `best.pt` once at startup; runs colormap → defect CV per request.

## Tech stack

- Python, managed with **`uv`** (`uv pip install` inside an activated `.venv`).
- Detection/training: **Ultralytics YOLOv8**, trained on **Google Colab**.
- CV: OpenCV, NumPy, scikit-image.
- Serving: **FastAPI** (+ uvicorn).
- Annotation: **Roboflow** (web, free) — exports YOLO format directly.

## Annotation workflow

1. Upload images to Roboflow.
2. Draw boxes: `tank`, `wire`. Objective and quick.
3. (Optional bootstrap once detector exists) export detector predictions as
   pre-labels and correct them — faster than from scratch.
4. Target ~150–300 images, train/val split, export YOLO format → Colab.

## Validation (no labels exist yet)

- Manually mark true defects on ~20–30 held-out images as a small eval set.
- Track precision/recall of the defect CV as thresholds are tuned.
- Detector accuracy measured by standard YOLO mAP on the val split.

## Risks

- **Palette assumption** — different camera palettes require re-calibrating the LUT
  per source.
- **No absolute temperature** — permanent; relative severity only.
- **Small dataset** — detector quality scales with labeled count; start small,
  grow via pre-label bootstrapping.

## Out of scope (v1)

- Absolute temperature readings.
- Root-cause diagnosis (loose connection vs overload vs oil level).
- Live video; batch/single-image only.
- `bushing` as a separate class (deferred).
