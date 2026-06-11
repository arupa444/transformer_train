# Annotation Guide — labelImg

You are labeling **parts, not defects**. Every image has a transformer and wires;
you just trace their boxes. The model learns shapes; the CV layer decides heat.

## Classes (exactly two, in this order)

| id | class         | what you box |
|----|---------------|--------------|
| 0  | `transformer` | the transformer unit (body + radiator). One box per unit. |
| 1  | `wire`        | each conductor / bushing connection. Box each one separately. |

> ⚠️ **Order matters.** labelImg assigns the class **id** from the order in your
> `classes.txt` (a.k.a. `predefined_classes.txt`). It MUST read:
> ```
> transformer
> wire
> ```
> Every annotator must use the **same** file, or ids 0/1 get swapped and half the
> labels train inverted. The `data.yaml` `names:` list must match this order too.

## Box conventions (this is what "good" looks like)

- **`transformer`** — hug the equipment. Some background inside the box is fine
  (the CV layer ignores cold background via an Otsu split), but don't draw it much
  larger than the unit, or the detector learns sloppy localization.
- **`wire`** — box **each** conductor **individually and tightly**, including the
  bushing where it connects (bushings go in the `wire` class). The hottest
  conductor stubs are your most valuable labels — make sure each is captured.
- **Do not** box pure background, foliage, or cold (blue/purple) regions.
- **Be consistent** — same rules on every image, every annotator. Aim for 2+ wire
  boxes per image so the wire-to-wire comparison has something to compare against.

## Steps in labelImg

1. Install: `uv pip install labelImg` (inside an activated venv) or `pipx install labelImg`.
2. Run `labelImg`. Set the save format to **YOLO** (the left-toolbar button toggles
   PascalVOC / YOLO — it must say **YOLO**).
3. `Open Dir` → your images folder. `Change Save Dir` → where the `.txt` labels go.
4. For each image: press `W`, draw a box, pick `transformer` or `wire`, repeat.
   `Ctrl+S` saves a `<image>.txt` (one row per box: `class_id cx cy w h`, normalized).
5. Aim for **150–300 images**. Keep variety: angles, distances, lighting, units.

labelImg writes a `classes.txt` next to your labels — confirm it is exactly the two
lines above, in that order.

## Organize for training

Split into train/validation (~80/20) in this layout (YOLO standard — Ultralytics
finds labels by swapping `/images/` → `/labels/`):

```
dataset/
├── data.yaml
├── train/
│   ├── images/   *.jpg
│   └── labels/   *.txt
└── valid/
    ├── images/
    └── labels/
```

`data.yaml`:
```yaml
path: /content/dataset    # absolute path on Colab (the notebook sets this)
train: train/images
val: valid/images
nc: 2
names: ['transformer', 'wire']
```

Zip the `dataset/` folder and upload it in the Colab notebook
(`notebooks/train_yolo.ipynb`).

## Tip: bootstrap later
Once you have a trained `best.pt`, run new images through it, export the
predictions as YOLO `.txt` pre-labels, and just **correct** them in labelImg —
far faster than labeling from scratch.
