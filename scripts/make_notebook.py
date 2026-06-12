"""Generate notebooks/train_cascade_yolo26.ipynb — mounts Google Drive, unzips the
two dataset folders, trains BOTH YOLO26x detectors (transformer + wire), then TESTS
the full cascade using this repo's inference code. All in one Colab notebook.

You just: zip the two dataset folders, upload the zips to Drive, set their paths in
the notebook. The notebook does the rest (unzip -> train -> eval -> cascade demo).

Regenerate:  python scripts/make_notebook.py
"""
import nbformat as nbf

REPO_URL = "https://github.com/arupa444/transformer_train.git"


def build_notebook() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    cells = []

    def md(id_, text):
        c = nbf.v4.new_markdown_cell(text); c["id"] = id_; cells.append(c)

    def code(id_, src):
        c = nbf.v4.new_code_cell(src); c["id"] = id_; cells.append(c)

    md("intro",
       "# Thermal cascade — train + test (YOLO26x)\n"
       "Runtime → Change runtime type → **GPU** first.\n\n"
       "**You do:** zip the two dataset folders and upload the zips to Google Drive:\n"
       "```bash\n"
       "cd /Volumes/dronisight\n"
       "zip -r transformer.zip yolo_thermal_transformer\n"
       "zip -r wire.zip Yolo_thermal_wire\n"
       "```\n"
       "**This notebook does:** mount Drive → unzip → train `transformer` + `wire` "
       "YOLO26x → evaluate → run the full cascade inference (this repo's code) on test images.")

    code("mount", "from google.colab import drive\n"
                  "drive.mount('/content/drive')")

    code("setup",
         "# Pins that matter on Colab: 8.3.x silently degrades yolo26 -> nano;\n"
         "# pillow 11.3+ is broken. Then clone this repo for the cascade inference code.\n"
         '!pip install -q -U "ultralytics>=8.4.60" "pillow==11.2.1"\n'
         f"!git clone -q {REPO_URL} /content/transformer_train\n"
         "import sys; sys.path.insert(0, '/content/transformer_train/src')\n"
         "print('ok')")

    md("paths-md",
       "### Point these at YOUR zips on Drive\n"
       "Edit the two paths to wherever you uploaded `transformer.zip` / `wire.zip`.")

    code("paths",
         "TRANSFORMER_ZIP = '/content/drive/MyDrive/transformer.zip'\n"
         "WIRE_ZIP        = '/content/drive/MyDrive/wire.zip'\n"
         "# where trained weights are copied so they survive the session\n"
         "DRIVE_WEIGHTS_DIR = '/content/drive/MyDrive/thermal_weights'")

    code("unzip",
         "import zipfile, os, yaml\n"
         "for z in (TRANSFORMER_ZIP, WIRE_ZIP):\n"
         "    with zipfile.ZipFile(z) as zf:\n"
         "        zf.extractall('/content')\n"
         "    print('unzipped', z)\n"
         "DATASETS = {'transformer': '/content/yolo_thermal_transformer',\n"
         "            'wire': '/content/Yolo_thermal_wire'}\n"
         "# repoint each data.yaml `path:` (built on the Mac) to the Colab location\n"
         "DATA_YAML = {}\n"
         "for key, root in DATASETS.items():\n"
         "    p = f'{root}/data_clahe.yaml'\n"
         "    d = yaml.safe_load(open(p)); d['path'] = root\n"
         "    yaml.safe_dump(d, open(p, 'w'), sort_keys=False)\n"
         "    DATA_YAML[key] = p\n"
         "    print(key, '->', d)")

    code("train-args",
         "# Thermal-tuned. NO hue jitter (palette = heat). x is heavy for ~600-700 imgs ->\n"
         "# early stopping + regularization. On OOM: MODEL='yolo26m.pt' or lower batch.\n"
         "MODEL = 'yolo26x.pt'\n"
         "def train_args(data_yaml, scale):\n"
         "    return dict(data=data_yaml, epochs=150, imgsz=1280, batch=4, seed=1337,\n"
         "                hsv_h=0.0, hsv_s=0.2, hsv_v=0.3,\n"
         "                fliplr=0.5, flipud=0.0, degrees=10.0, translate=0.1, scale=scale,\n"
         "                mosaic=1.0, close_mosaic=10,\n"
         "                weight_decay=0.0005, dropout=0.1, cos_lr=True, patience=30, amp=True)")

    code("train-transformer",
         "from ultralytics import YOLO\n"
         "m_t = YOLO(MODEL)  # full frame, big object\n"
         "m_t.train(project='runs/transformer', name='yolo26x',\n"
         "          **train_args(DATA_YAML['transformer'], scale=0.5))")

    code("train-wire",
         "m_w = YOLO(MODEL)  # transformer crops, thin objects -> wider scale jitter\n"
         "m_w.train(project='runs/wire', name='yolo26x',\n"
         "          **train_args(DATA_YAML['wire'], scale=0.9))")

    code("save-weights",
         "# Persist weights to Drive (survive runtime resets) and into the repo's models/.\n"
         "import os, shutil\n"
         "os.makedirs(DRIVE_WEIGHTS_DIR, exist_ok=True)\n"
         "os.makedirs('/content/transformer_train/models', exist_ok=True)\n"
         "T = 'runs/transformer/yolo26x/weights/best.pt'\n"
         "W = 'runs/wire/yolo26x/weights/best.pt'\n"
         "for src, name in ((T, 'transformer.pt'), (W, 'wire.pt')):\n"
         "    shutil.copy(src, f'{DRIVE_WEIGHTS_DIR}/{name}')\n"
         "    shutil.copy(src, f'/content/transformer_train/models/{name}')\n"
         "print('weights saved to', DRIVE_WEIGHTS_DIR, 'and repo models/')")

    md("eval-md", "## Test\n"
                  "Per-model mAP on the held-out **test** split, then the full **cascade** "
                  "(transformer → crop → wire → relative-heat CV) on test images.")

    code("eval-map",
         "for tag, m, key in (('transformer', m_t, 'transformer'), ('wire', m_w, 'wire')):\n"
         "    r = m.val(data=DATA_YAML[key], split='test')\n"
         "    print(f'{tag} TEST: mAP50-95={r.box.map:.3f}  mAP50={r.box.map50:.3f}')")

    code("cascade-test",
         "# Full cascade using THIS repo's inference code on a few test frames.\n"
         "import glob, os, cv2\n"
         "import matplotlib.pyplot as plt\n"
         "from thermal.detector import YoloDetector\n"
         "from thermal.colormap import build_lut, ColorToHeat\n"
         "from thermal.pipeline import analyze_image\n"
         "from thermal.report import annotate, to_json\n"
         "\n"
         "td = YoloDetector('runs/transformer/yolo26x/weights/best.pt')\n"
         "wd = YoloDetector('runs/wire/yolo26x/weights/best.pt')\n"
         "c2h = ColorToHeat(build_lut('inferno'))\n"
         "\n"
         "# run on the ORIG test frames (raw palette -> true heat)\n"
         "imgs = sorted(glob.glob('/content/yolo_thermal_transformer/images/test/orig/*.jpg'))[:6]\n"
         "os.makedirs('/content/cascade_out', exist_ok=True)\n"
         "fig, axes = plt.subplots(1, len(imgs), figsize=(4*len(imgs), 4))\n"
         "for ax, p in zip(axes if len(imgs) > 1 else [axes], imgs):\n"
         "    bgr = cv2.imread(p); rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)\n"
         "    findings, _, calib = analyze_image(rgb, td, wd, c2h)\n"
         "    out = annotate(bgr, findings)\n"
         "    cv2.imwrite(f\"/content/cascade_out/{os.path.basename(p)}\", out)\n"
         "    print(os.path.basename(p), 'calib_ok=', calib, to_json(findings))\n"
         "    ax.imshow(cv2.cvtColor(out, cv2.COLOR_BGR2RGB)); ax.axis('off')\n"
         "plt.tight_layout(); plt.show()")

    md("next",
       "Annotated cascade outputs are in `/content/cascade_out/` and the weights in your "
       "Drive `thermal_weights/` folder. Download `transformer.pt` + `wire.pt` into the repo's "
       "`models/` to run the local API. If train-vs-val (`results.png`) diverges, try "
       "`MODEL='yolo26m.pt'`.")

    nb["cells"] = cells
    return nb


def main() -> None:
    nb = build_notebook()
    with open("notebooks/train_cascade_yolo26.ipynb", "w") as f:
        nbf.write(nb, f)
    print("wrote notebooks/train_cascade_yolo26.ipynb")


if __name__ == "__main__":
    main()
