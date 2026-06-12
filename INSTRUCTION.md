# INSTRUCTION.md — Transformer Thermal Defect Classifier (end-to-end)

A practical, plain-language guide to go from raw thermal images to a running defect
detector. The system is a **2-model cascade**: a YOLO26x `transformer` detector, then
a YOLO26x `wire` detector that runs on each transformer crop, then a classical-CV layer
that reads **relative heat** and flags defects (Normal / Watch / Investigate / Critical).

> ⚠️ Input is **colorized thermal screenshots** (no embedded °C). Output is therefore
> **relative severity**, never an absolute temperature.

---

## 0. The whole flow at a glance

```
1 Annotate (labelImg)        →  transformer + wire boxes (VOC .xml)
2 Build datasets             →  python -m thermal.data_prep.build --subset both
3 Train two YOLO26x models   →  notebooks/train_cascade_yolo26.ipynb (Colab GPU)
4 Run inference / API        →  uvicorn api:app   (POST /analyze)
```

| Step | Where | Status if you cloned this repo |
|------|-------|--------------------------------|
| 1 Annotate | labelImg on your Mac | done by the team |
| 2 Build datasets | this repo | code ready (`thermal.data_prep`) |
| 3 Train | Google Colab | **you run this** to get the weights |
| 4 Inference | this repo | code ready + tested |

---

## 1. Setup (one time)

Always use `uv` (never bare `pip`).

```bash
cd /Users/arupanandaswain/PycharmProjects/transformer
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"        # CV + data-prep + tests (NO torch — light & fast)
pytest -q                          # sanity: 27 tests pass without any model
```

Install the heavy detector deps only when you actually run the real models:
```bash
uv pip install -e ".[inference]"   # adds ultralytics / torch
```

---

## 2. Annotate (labelImg)

Full conventions: `docs/annotation-guide.md`. The essentials:

- **Two classes, in this exact order in `classes.txt`:** `transformer` (id 0), `wire` (id 1).
  Same file for every annotator, or ids get swapped.
- **`transformer`** = one tight-ish box around the unit (some background is fine).
- **`wire`** = each conductor / bushing, boxed **individually and tightly**.
- Save format **YOLO** in labelImg (it also writes the `.xml` we actually use).
- Don't box pure background / cold (blue) regions.

The 4-annotator folders live under `/Volumes/dronisight/thermal/` (e.g.
`JUNE 11 MEM 1/_T_collected`, `Jun 11 Mem 2 `, `JUNE 11 MEM 3/_T_collected`,
`JUNE 11 MEM 4`). To audit label health / class frequency first:
```bash
python scripts/analyze_annotations.py
```

---

## 3. Build the datasets

```bash
python -m thermal.data_prep.build --subset both
```

This (reusing the proven `trainDronisight` design) does the hard parts for you:
- reads the **VOC `.xml`** labels (the source of truth; index-based `.txt` is ignored);
- **content-hash dedup/merge** — collapses byte-identical duplicates (the `_T.JPG` /
  `_T_1.JPG` pairs + cross-annotator copies) into one image and **unions partial labels**.
  *This is critical:* it prevents the same photo leaking into both train and val;
- **leakage-safe split** by capture-sequence group (train/val/test ≈ 80/15/5);
- **adaptive CLAHE** — writes both an `orig` and a `clahe` image variant;
- **transformer-anchored wire crops** — the wire dataset is each transformer box + 15% pad
  (so the wire model trains at the scale it runs on in the cascade).

Outputs (pole-style layout each):
```
/Volumes/dronisight/yolo_thermal_transformer/   (full-frame transformer)
/Volumes/dronisight/Yolo_thermal_wire/          (transformer crops)
  ├── data_orig.yaml   data_clahe.yaml
  ├── images/{train,val,test}/{orig,clahe}/
  ├── labels/{train,val,test}/{orig,clahe}/
  ├── manifest.csv     dataset_meta.json
```

Quick check after building:
```bash
cat /Volumes/dronisight/yolo_thermal_transformer/dataset_meta.json
```

---

## 4. Train + test both YOLO26x models (Google Colab)

The notebook does everything (mount Drive → unzip → clone repo → train → evaluate →
run the cascade). You only zip the two folders and upload them to Drive.

### 4a. Zip the two dataset folders and upload to Google Drive
```bash
cd /Volumes/dronisight
zip -r transformer.zip yolo_thermal_transformer
zip -r wire.zip Yolo_thermal_wire
```
Upload `transformer.zip` and `wire.zip` to your Google Drive (e.g. `MyDrive/`).

### 4b. Run the notebook
Open `notebooks/train_cascade_yolo26.ipynb` in Colab →
**Runtime → Change runtime type → GPU**. In the **paths** cell set `TRANSFORMER_ZIP` /
`WIRE_ZIP` to where you uploaded the zips, then **Runtime → Run all**. The cells:

1. **mount** — mounts Google Drive.
2. **setup** — pins `ultralytics>=8.4.60` (older silently downgrades YOLO26 → nano) +
   `pillow==11.2.1`, and clones this repo (for the cascade inference code).
3. **paths / unzip** — unzips both datasets and repoints each `data.yaml` to `/content`.
4. **train-transformer / train-wire** — one YOLO26x each.
5. **save-weights** — copies `transformer.pt` + `wire.pt` to Drive (`thermal_weights/`)
   and into the cloned repo's `models/` (so they survive a runtime reset).
6. **eval-map** — per-model mAP on the held-out **test** split.
7. **cascade-test** — runs the full cascade (this repo's `analyze_image`) on test frames
   and displays the annotated defects inline; outputs in `/content/cascade_out/`.

### 4c. Use the weights locally
They're saved in your Drive `thermal_weights/` folder — download `transformer.pt` +
`wire.pt` into this repo's `models/` to run the local API (Step 5).

**If a free T4 runs out of memory (OOM):** in the `train-args` cell set
`MODEL = 'yolo26m.pt'` (or lower `batch=4` → `2`). For ~600–700 training images, `m`/`l`
often generalize **better** than `x` anyway.
**Watch overfitting:** open `results.png` in each `runs/.../` folder — a widening
train-vs-val gap means use a smaller model or fewer epochs.

---

## 5. Run inference / the API

```bash
uv pip install -e ".[inference]"
THERMAL_TRANSFORMER_WEIGHTS=models/transformer.pt \
THERMAL_WIRE_WEIGHTS=models/wire.pt \
uvicorn api:app --reload        # http://127.0.0.1:8000
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
    {"component": "transformer", "bbox": [x1,y1,x2,y2], "severity": "Watch", "relative_delta": 0.14},
    {"component": "wire",        "bbox": [x1,y1,x2,y2], "severity": "Critical", "relative_delta": 0.41}
  ],
  "annotated_image_png_b64": "iVBORw0K..."
}
```
- `calibration_ok: false` ⇒ the image's palette doesn't match the assumed iron palette
  (the heat reading is unreliable — see Tuning).
- Decode `annotated_image_png_b64` (base64 → PNG) to see boxes colored by severity.

---

## 6. How it works (the cascade)

1. **Preprocess** (`src/thermal/preprocess.py`) — adaptive CLAHE, the *same* transform
   used to build the `clahe` dataset (train/serve parity).
2. **Transformer detect** (`YoloDetector` + `models/transformer.pt`) on the full frame.
3. For each transformer: **crop + 0.15 pad → wire detect** (`models/wire.pt`) on the crop,
   then map wire boxes back to full-frame coords. (Restricts wires to the transformer →
   no background false positives; matches the wire model's training scale.)
4. **Colormap inversion** (`src/thermal/colormap.py`) on the **raw** image → 0–1 heat map
   (relative temps stay true; CLAHE is only for the detector).
5. **Defect CV** (`src/thermal/defects.py`): a localized **transformer hotspot** (vs the
   warm-body reference, background excluded via an Otsu split) and each transformer's
   **wires compared to their siblings**; severity scales with the gap.
6. **Report** (`src/thermal/report.py`) — annotated image + JSON.

Orchestrated in `src/thermal/pipeline.py` (`analyze_image`).

---

## 7. Tuning knobs

| What | Where |
|------|-------|
| Severity thresholds | `src/thermal/defects.py` → `_WATCH` / `_INVESTIGATE` / `_CRITICAL` |
| Wire crop pad (must match build) | `pipeline.py` `WIRE_CROP_PAD` **and** `data_prep/build.py` `WIRE_CROP_PAD` |
| Palette / calibration | `build_lut("inferno")` in `api.py`; re-check `calibration_ok` |
| Detector confidence | `YoloDetector.detect(..., conf=0.25)` |
| Train aug / model size | `notebooks/train_cascade_yolo26.ipynb` (`MODEL`, `train_args`) |
| Train/val/test ratio, seed, group gap | `src/thermal/data_prep/build.py` |

---

## 8. Troubleshooting

- **`0 images, N backgrounds` during training** — a stale Ultralytics label cache.
  Delete `*.cache` under the dataset's `labels/` and retry.
- **macOS `._*` / `.DS_Store` sidecars** on the exFAT SSD crash YOLO. The build self-cleans
  AppleDouble files; when copying datasets prefer
  `rsync --exclude '._*' --exclude '.DS_Store'`.
- **YOLO26 trained but acts like nano** — Colab had old `ultralytics`; ensure `>=8.4.60`
  (the notebook pins it).
- **`calibration_ok: false`** — the camera used a different palette than `inferno`; swap the
  colormap name in `build_lut(...)` (and rebuild the LUT) to match your camera.
- **Few/no wires detected** — the wire model only runs inside detected transformers; if the
  transformer was missed, its wires are lost. Check the transformer detector's recall first.

---

## 9. Tests & layout

```bash
pytest -q          # 27 tests; run with no model / no torch
```

```
src/thermal/
  data_prep/   labels, imaging (CLAHE), assemble (merge/group/split/crops), build
  preprocess.py  detector.py  colormap.py  defects.py  pipeline.py  report.py  schema.py
api.py                      FastAPI /analyze (cascade)
notebooks/train_cascade_yolo26.ipynb   scripts/make_notebook.py  scripts/analyze_annotations.py
docs/annotation-guide.md    models/ (transformer.pt + wire.pt, gitignored)
```
