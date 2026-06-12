"""Generate notebooks/train_cascade_yolo26.ipynb — trains the TWO cascade detectors
(transformer + wire) with YOLO26x on Google Colab.

Regenerate after editing:  python scripts/make_notebook.py
"""
import nbformat as nbf


def build_notebook() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    cells = []

    def md(id_, text):
        c = nbf.v4.new_markdown_cell(text); c["id"] = id_; cells.append(c)

    def code(id_, src):
        c = nbf.v4.new_code_cell(src); c["id"] = id_; cells.append(c)

    md("intro",
       "# Train the thermal cascade — YOLO26x (transformer + wire)\n"
       "Runtime → Change runtime type → **GPU** first.\n\n"
       "Trains **two single-class detectors** on the datasets built by "
       "`thermal.data_prep.build` (already deduped + CLAHE-preprocessed):\n"
       "- `transformer` — full frame\n"
       "- `wire` — on transformer crops (the cascade)\n\n"
       "Both train on the `clahe` image variant, with thermal-tuned augmentation "
       "(**no hue jitter** — it would scramble the heat palette).")

    code("install",
         "# YOLO26 gotchas (learned on Colab): 8.3.x silently degrades yolo26 -> nano,\n"
         "# and pillow 11.3+ is broken on Colab. Pin both.\n"
         '!pip install -q -U "ultralytics>=8.4.60" "pillow==11.2.1"')

    md("upload-md",
       "### Upload the datasets\n"
       "On your machine, zip each dataset's **clahe** variant (smaller upload):\n"
       "```bash\n"
       "cd /Volumes/dronisight\n"
       "zip -r transformer.zip yolo_thermal_transformer/data_clahe.yaml \\\n"
       "    yolo_thermal_transformer/images/*/clahe yolo_thermal_transformer/labels/*/clahe\n"
       "zip -r wire.zip Yolo_thermal_wire/data_clahe.yaml \\\n"
       "    Yolo_thermal_wire/images/*/clahe Yolo_thermal_wire/labels/*/clahe\n"
       "```\n"
       "Then run the next cell and pick **both** zips.")

    code("upload",
         "import zipfile, os\n"
         "from google.colab import files\n"
         "for name, up in files.upload().items():\n"
         "    with zipfile.ZipFile(name) as z:\n"
         "        z.extractall('/content')\n"
         "    print('extracted', name)\n"
         "print(os.listdir('/content'))")

    code("fix-yaml",
         "# The build-time data.yaml `path:` points at the build machine; repoint to /content.\n"
         "import yaml, glob\n"
         "DATASETS = {\n"
         "    'transformer': '/content/yolo_thermal_transformer',\n"
         "    'wire': '/content/Yolo_thermal_wire',\n"
         "}\n"
         "DATA_YAML = {}\n"
         "for key, root in DATASETS.items():\n"
         "    p = f'{root}/data_clahe.yaml'\n"
         "    d = yaml.safe_load(open(p)); d['path'] = root\n"
         "    yaml.safe_dump(d, open(p, 'w'), sort_keys=False)\n"
         "    DATA_YAML[key] = p\n"
         "    print(key, '->', d)")

    code("train-args",
         "# Thermal-tuned training. NO hue jitter (palette = heat). x is heavy for ~600-700\n"
         "# train imgs -> early stopping + regularization. Drop to yolo26m.pt / lower batch on OOM.\n"
         "MODEL = 'yolo26x.pt'\n"
         "def train_args(data_yaml, scale):\n"
         "    return dict(\n"
         "        data=data_yaml, epochs=150, imgsz=1280, batch=4, seed=1337,\n"
         "        hsv_h=0.0, hsv_s=0.2, hsv_v=0.3,\n"
         "        fliplr=0.5, flipud=0.0, degrees=10.0, translate=0.1, scale=scale,\n"
         "        mosaic=1.0, close_mosaic=10,\n"
         "        weight_decay=0.0005, dropout=0.1, cos_lr=True, patience=30, amp=True,\n"
         "    )")

    code("train-transformer",
         "from ultralytics import YOLO\n"
         "# full frame, large object -> moderate scale jitter\n"
         "m_t = YOLO(MODEL)\n"
         "m_t.train(project='runs/transformer', name='yolo26x', **train_args(DATA_YAML['transformer'], scale=0.5))")

    code("train-wire",
         "# transformer crops, thin objects -> wider scale jitter to mimic crop zoom\n"
         "m_w = YOLO(MODEL)\n"
         "m_w.train(project='runs/wire', name='yolo26x', **train_args(DATA_YAML['wire'], scale=0.9))")

    code("validate",
         "for tag, m in (('transformer', m_t), ('wire', m_w)):\n"
         "    r = m.val()\n"
         "    print(f'{tag}: mAP50-95={r.box.map:.3f}  mAP50={r.box.map50:.3f}')")

    code("download",
         "# Download both best.pt, renamed for the cascade.\n"
         "import shutil\n"
         "from google.colab import files\n"
         "shutil.copy('runs/transformer/yolo26x/weights/best.pt', 'transformer.pt')\n"
         "shutil.copy('runs/wire/yolo26x/weights/best.pt', 'wire.pt')\n"
         "files.download('transformer.pt'); files.download('wire.pt')")

    md("next",
       "Put `transformer.pt` and `wire.pt` into the repo's `models/` folder, then run the "
       "cascade inference / API. Watch `results.png` per run: a widening train-vs-val gap = "
       "overfitting → try `MODEL='yolo26m.pt'` or fewer epochs.")

    nb["cells"] = cells
    return nb


def main() -> None:
    nb = build_notebook()
    with open("notebooks/train_cascade_yolo26.ipynb", "w") as f:
        nbf.write(nb, f)
    print("wrote notebooks/train_cascade_yolo26.ipynb")


if __name__ == "__main__":
    main()
