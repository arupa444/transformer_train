# INSTRUCTION.md — Transformer Thermal Defect Classifier (end-to-end)

A practical, plain-language guide from raw thermal images to a running defect detector.
The system is **one YOLO26x detector + a CV layer**: YOLO localizes the `transformer`,
then classical CV reads **relative heat** inside it to flag hot conductors/connections
and hot spots (Normal / Watch / Investigate / Critical).

> ⚠️ Input is **colorized thermal screenshots** (no embedded °C). Output is therefore
> **relative severity**, never an absolute temperature. A *uniformly* hot tank can't be
> flagged from relative heat alone — only localized hot regions.

> **Why no wire model?** Thin, densely-clustered conductors were not learnable as a YOLO
> class on this dataset (mAP stalled ~0.13). CV finds *hot* conductors directly (a region
> hotter than the transformer body), which is the actual defect signal — and needs no training.

---

## 0. The whole flow at a glance

```
1 Annotate (labelImg)        →  transformer + wire boxes (VOC .xml)
2 Build dataset              →  python -m thermal.data_prep.build --subset both
3 Train ONE YOLO26x model    →  notebooks/train_cascade_yolo26.ipynb (Colab GPU)
4 Run inference / API        →  uvicorn api:app   (POST /analyze)
```

| Step | Where | Status if you cloned this repo |
|------|-------|--------------------------------|
| 1 Annotate | labelImg on your Mac | done by the team |
| 2 Build dataset | this repo | code ready (`thermal.data_prep`) |
| 3 Train transformer detector | Google Colab | **you run this** to get `transformer.pt` |
| 4 Inference (detect + CV) | this repo | code ready + tested |

---

## 1. Setup (one time)

Always use `uv` (never bare `pip`).

```bash
cd /Users/arupanandaswain/PycharmProjects/transformer
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"        # CV + data-prep + tests (NO torch — light & fast)
pytest -q                          # sanity: 33 tests pass without any model
```

Install the heavy detector dep only when you run the real model:
```bash
uv pip install -e ".[inference]"   # adds ultralytics / torch
```

---

## 2. Annotate (labelImg)

Full conventions: `docs/annotation-guide.md`. Essentials:

- **Two classes, in this exact order in `classes.txt`:** `transformer` (id 0), `wire` (id 1).
  Same file for every annotator, or ids get swapped.
- **`transformer`** = one tight-ish box around the unit (some background is fine).
- **`wire`** = each conductor / bushing, boxed tightly. *(These currently feed only the
  dataset build; the detector trained is the transformer. Keeping them lets you revisit a
  learned wire/bushing model later.)*
- Save format **YOLO** in labelImg (it also writes the `.xml` we actually use).

Audit label health / class frequency:
```bash
python scripts/analyze_annotations.py
```

---

## 3. Build the dataset

```bash
python -m thermal.data_prep.build --subset both
```

Reusing the proven `trainDronisight` design, this:
- reads the **VOC `.xml`** labels (source of truth; index-based `.txt` is ignored);
- **content-hash dedup/merge** — collapses byte-identical duplicates (the `_T.JPG` /
  `_T_1.JPG` pairs + cross-annotator copies) into one image and **unions partial labels**.
  *Critical:* prevents the same photo leaking into both train and val;
- **leakage-safe split** by capture-sequence group (train/val/test ≈ 80/15/5);
- **adaptive CLAHE** — writes both an `orig` and a `clahe` image variant.

Outputs (pole-style layout): `/Volumes/dronisight/yolo_thermal_transformer/` is the one you
train on (`Yolo_thermal_wire/` is also built but currently unused). Each has
`images|labels/{train,val,test}/{orig,clahe}`, `data_{orig,clahe}.yaml`, `manifest.csv`,
`dataset_meta.json`. Quick check:
```bash
cat /Volumes/dronisight/yolo_thermal_transformer/dataset_meta.json
```

---

## 4. Train the transformer detector + test the pipeline (Google Colab)

The notebook does everything (mount Drive → unzip → clone repo → train → evaluate → run the
full pipeline). You only zip one folder and upload it.

### 4a. Zip the transformer dataset and upload to Google Drive
```bash
cd /Volumes/dronisight
zip -r transformer.zip yolo_thermal_transformer
```
Upload `transformer.zip` to your Google Drive (e.g. `MyDrive/`).

### 4b. Run the notebook
Open `notebooks/train_cascade_yolo26.ipynb` in Colab → **Runtime → Change runtime type →
GPU**. In the **paths** cell set `TRANSFORMER_ZIP`, then **Runtime → Run all**. The cells:

1. **mount** — mounts Google Drive.
2. **setup** — pins `ultralytics>=8.4.60` (older silently downgrades YOLO26 → nano) +
   `pillow==11.2.1` + `scikit-image`, and clones this repo (for the inference code).
3. **unzip** — unzips the dataset and repoints `data.yaml` to `/content`.
4. **train** — trains the YOLO26x transformer detector.
5. **save-weights** — copies `transformer.pt` to Drive (`thermal_weights/`) and the repo's
   `models/` (survives a runtime reset).
6. **eval-map** — transformer mAP on the held-out **test** split.
7. **cascade-test** — runs the full pipeline (`analyze_image`: detect + CV hotspots) on test
   frames, displays annotated defects inline; outputs in `/content/cascade_out/`.

### 4c. Use the weights locally
Download `transformer.pt` from Drive `thermal_weights/` into this repo's `models/`.

**OOM on a free T4:** in the `train-args` cell set `MODEL = 'yolo26m.pt'` (or lower
`batch=4` → `2`). For ~600 images, `m`/`l` often generalize **better** than `x`.
**Overfitting:** open `results.png` in the run folder — widening train-vs-val gap → smaller
model / fewer epochs.

---

## 5. Run inference / the API

```bash
uv pip install -e ".[inference]"
THERMAL_TRANSFORMER_WEIGHTS=models/transformer.pt uvicorn api:app --reload   # :8000
```

Analyze an image:
```bash
curl -s -F "file=@/path/to/thermal.jpg" http://127.0.0.1:8000/analyze | jq
```

Response:
```json
{
  "calibration_ok": true,
  "defects": [
    {"component": "hotspot", "bbox": [x1,y1,x2,y2], "severity": "Critical", "relative_delta": 0.41}
  ],
  "annotated_image_png_b64": "iVBORw0K..."
}
```
- `defects` is empty when nothing is abnormally hot (a healthy unit).
- `calibration_ok: false` ⇒ the image's palette doesn't match the assumed iron palette
  (heat reading unreliable — see Tuning).
- Decode `annotated_image_png_b64` (base64 → PNG) to see boxes colored by severity.

---

## 6. How it works

1. **Preprocess** (`src/thermal/preprocess.py`) — adaptive CLAHE, the *same* transform used
   to build the `clahe` dataset (train/serve parity). Fed to the detector as **BGR**.
2. **Transformer detect** (`YoloDetector` + `models/transformer.pt`) on the full frame.
3. **Colormap inversion** (`src/thermal/colormap.py`) on the **raw** image → 0–1 heat map
   (relative temps stay true; CLAHE is only for the detector).
4. **CV hotspots** (`src/thermal/defects.find_hotspots`): inside each transformer (+0.15 pad),
   estimate the warm-body reference (Otsu, background excluded), then every connected region
   hotter than body by `_HOTSPOT_MARGIN` is a `hotspot`, scored by how far above body it sits.
5. **Report** (`src/thermal/report.py`) — annotated image + JSON.

Orchestrated in `src/thermal/pipeline.py` (`analyze_image(img_rgb, transformer_detector, c2h)`).

---

## 7. Tuning knobs

| What | Where |
|------|-------|
| Hotspot floor / severity | `src/thermal/defects.py` → `_HOTSPOT_MARGIN`, `_WATCH`/`_INVESTIGATE`/`_CRITICAL` (default `0.45`/`0.55`/`0.62`, **calibrated on 755 crops**: normal warm connection ≈ +0.22..+0.44 above body, defects ≈ +0.55..+0.65). Lower for more sensitivity. |
| Crop pad around transformer | `src/thermal/pipeline.py` → `HOTSPOT_PAD` |
| Palette / calibration | `build_lut("inferno")` in `api.py`; re-check `calibration_ok` |
| Detector confidence | `YoloDetector.detect(..., conf=0.25)` |
| Train aug / model size | `notebooks/train_cascade_yolo26.ipynb` (`MODEL`, `TRAIN_ARGS`) |
| Train/val/test ratio, seed, group gap | `src/thermal/data_prep/build.py` |

---

## 8. Troubleshooting

- **`0 images, N backgrounds` during training** — stale Ultralytics label cache. Delete
  `*.cache` under the dataset's `labels/` and retry.
- **macOS `._*` / `.DS_Store` sidecars** on the exFAT SSD crash YOLO. The build self-cleans
  them; when copying prefer `rsync --exclude '._*' --exclude '.DS_Store'`.
- **YOLO26 trained but acts like nano** — Colab had old `ultralytics`; ensure `>=8.4.60`
  (the notebook pins it).
- **`calibration_ok: false`** — different camera palette than `inferno`; swap the colormap in
  `build_lut(...)` to match your camera.
- **No hotspots flagged** — either the unit is genuinely uniform (healthy), or the transformer
  was missed (check the detector first), or sensitivity is too low (`_HOTSPOT_MARGIN`).
- **Too many hotspots / noise** — raise `_HOTSPOT_MARGIN` or `_MIN_HOTSPOT_AREA_FRAC`.

---

## 9. Tests & layout

```bash
pytest -q          # 33 tests; run with no model / no torch
```

```
src/thermal/
  data_prep/   labels, imaging (CLAHE), assemble (merge/group/split/crops), build
  preprocess.py  detector.py  colormap.py  defects.py  pipeline.py  report.py  schema.py
api.py                      FastAPI /analyze (detect + CV)
notebooks/train_cascade_yolo26.ipynb   scripts/make_notebook.py  scripts/analyze_annotations.py
docs/annotation-guide.md    models/ (transformer.pt, gitignored)
```
