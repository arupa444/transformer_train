"""Generate notebooks/train_cascade_yolo26.ipynb — trains the transformer detector
(YOLO26x) and tests the full pipeline (transformer detect + CV hotspot detection).

There is NO wire model: thin densely-clustered conductors were not learnable on this
dataset, so hot conductors/connections are found by CV (relative heat) inside the
transformer crop — no training needed.

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
       "# Thermal defect classifier — train + test (YOLO26x + CV)\n"
       "**Just: Runtime → Change runtime type → GPU, then Runtime → Run all.**\n\n"
       "Only requirement: your dataset must be on Google Drive. On your Mac run\n"
       "```bash\ncd /Volumes/Atrisol_D2\nzip -r transformer.zip yolo_thermal_transformer\n```\n"
       "and upload `transformer.zip` anywhere in **My Drive** (or one folder deep). This "
       "notebook auto-finds it — no paths to edit.\n\n"
       "**Architecture:** one YOLO26x detector localizes the **transformer**; hot "
       "conductors/connections and hotspots are then found by **CV (relative heat)** inside "
       "the transformer crop — there is **no wire model** (thin clustered conductors weren't "
       "learnable; CV directly targets 'where is it hot').")

    code("mount", "from google.colab import drive\n"
                  "drive.mount('/content/drive')")

    code("setup",
         "# Colab pins: 8.3.x silently degrades yolo26 -> nano; pillow 11.3+ is broken.\n"
         '!pip install -q -U "ultralytics>=8.4.60" "pillow==11.2.1" "scikit-image"\n'
         f"!git clone -q {REPO_URL} /content/transformer_train\n"
         "import sys; sys.path.insert(0, '/content/transformer_train/src')\n"
         "print('ok')")

    code("locate-data",
         "# Auto-find the dataset on Drive (a 'transformer.zip' or a 'yolo_thermal_transformer'\n"
         "# folder), in My Drive or one folder deep. No path editing needed.\n"
         "import glob, os, zipfile, shutil, yaml\n"
         "ROOT = '/content/yolo_thermal_transformer'\n"
         "DRIVE = '/content/drive/MyDrive'\n"
         "DRIVE_WEIGHTS_DIR = f'{DRIVE}/thermal_weights'\n"
         "if not os.path.isdir(ROOT):\n"
         "    zips = glob.glob(f'{DRIVE}/transformer.zip') + glob.glob(f'{DRIVE}/*/transformer.zip')\n"
         "    folders = (glob.glob(f'{DRIVE}/yolo_thermal_transformer')\n"
         "               + glob.glob(f'{DRIVE}/*/yolo_thermal_transformer'))\n"
         "    if zips:\n"
         "        print('unzipping', zips[0])\n"
         "        with zipfile.ZipFile(zips[0]) as z:\n"
         "            z.extractall('/content')\n"
         "    elif folders:\n"
         "        print('copying', folders[0])\n"
         "        shutil.copytree(folders[0], ROOT)\n"
         "    else:\n"
         "        raise FileNotFoundError(\n"
         "            \"Dataset not found. Upload 'transformer.zip' (zip of the \"\n"
         "            \"yolo_thermal_transformer folder) anywhere in your Google Drive.\")\n"
         "assert os.path.isdir(f'{ROOT}/images'), f'unexpected dataset layout under {ROOT}'\n"
         "p = f'{ROOT}/data_clahe.yaml'\n"
         "d = yaml.safe_load(open(p)); d['path'] = ROOT\n"
         "yaml.safe_dump(d, open(p, 'w'), sort_keys=False)\n"
         "DATA_YAML = p\n"
         "print('dataset ready ->', d)")

    code("train-args",
         "# Thermal-tuned. NO hue jitter (palette = heat). x is heavy for ~600 imgs ->\n"
         "# early stopping + regularization. On OOM: MODEL='yolo26m.pt' or lower batch.\n"
         "MODEL = 'yolo26x.pt'\n"
         "TRAIN_ARGS = dict(data=DATA_YAML, epochs=150, imgsz=1280, batch=4, seed=1337,\n"
         "                  hsv_h=0.0, hsv_s=0.2, hsv_v=0.3,\n"
         "                  fliplr=0.5, flipud=0.0, degrees=10.0, translate=0.1, scale=0.5,\n"
         "                  mosaic=1.0, close_mosaic=10,\n"
         "                  weight_decay=0.0005, dropout=0.1, cos_lr=True, patience=30, amp=True)")

    code("train",
         "from ultralytics import YOLO\n"
         "m = YOLO(MODEL)\n"
         "m.train(project='runs/transformer', name='yolo26x', **TRAIN_ARGS)")

    code("save-weights",
         "import os, shutil\n"
         "os.makedirs(DRIVE_WEIGHTS_DIR, exist_ok=True)\n"
         "os.makedirs('/content/transformer_train/models', exist_ok=True)\n"
         "best = str(m.trainer.best)\n"
         "shutil.copy(best, f'{DRIVE_WEIGHTS_DIR}/transformer.pt')\n"
         "shutil.copy(best, '/content/transformer_train/models/transformer.pt')\n"
         "print('saved transformer.pt from', best)")

    code("eval-map",
         "r = m.val(data=DATA_YAML, split='test')\n"
         "print(f'transformer TEST: mAP50-95={r.box.map:.3f}  mAP50={r.box.map50:.3f}')")

    md("test-md", "## Test the full pipeline (transformer detect + CV hotspots)")
    code("cascade-test",
         "import glob, os, cv2\n"
         "import matplotlib.pyplot as plt\n"
         "from thermal.detector import YoloDetector\n"
         "from thermal.colormap import build_lut, ColorToHeat\n"
         "from thermal.pipeline import analyze_image\n"
         "from thermal.report import annotate, to_json\n"
         "\n"
         "td = YoloDetector('/content/transformer_train/models/transformer.pt')\n"
         "c2h = ColorToHeat(build_lut('inferno'))\n"
         "\n"
         "imgs = sorted(glob.glob(f'{ROOT}/images/test/orig/*.jpg'))[:6]\n"
         "os.makedirs('/content/cascade_out', exist_ok=True)\n"
         "fig, axes = plt.subplots(1, len(imgs), figsize=(4*len(imgs), 4))\n"
         "for ax, p in zip(axes if len(imgs) > 1 else [axes], imgs):\n"
         "    bgr = cv2.imread(p); rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)\n"
         "    findings, _, calib = analyze_image(rgb, td, c2h)\n"
         "    out = annotate(bgr, findings)\n"
         "    cv2.imwrite(f\"/content/cascade_out/{os.path.basename(p)}\", out)\n"
         "    print(os.path.basename(p), 'calib_ok=', calib, to_json(findings))\n"
         "    ax.imshow(cv2.cvtColor(out, cv2.COLOR_BGR2RGB)); ax.axis('off')\n"
         "plt.tight_layout(); plt.show()")

    md("next",
       "Annotated outputs are in `/content/cascade_out/`; the transformer weights are in "
       "Drive `thermal_weights/`. Download `transformer.pt` into the repo's `models/` to run "
       "the local API. Tune hotspot sensitivity via `_HOTSPOT_MARGIN` / severity thresholds in "
       "`src/thermal/defects.py`.")

    nb["cells"] = cells
    return nb


def main() -> None:
    nb = build_notebook()
    with open("notebooks/train_cascade_yolo26.ipynb", "w") as f:
        nbf.write(nb, f)
    print("wrote notebooks/train_cascade_yolo26.ipynb")


if __name__ == "__main__":
    main()
