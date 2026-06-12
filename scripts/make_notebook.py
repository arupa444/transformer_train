"""Generate notebooks/train_yolo.ipynb for Google Colab YOLOv8 training."""
import nbformat as nbf


def build_notebook() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    cells = []

    cell = nbf.v4.new_markdown_cell(
        "# Train YOLO26x — Transformer Parts (transformer, wire)\n"
        "Runtime → Change runtime type → **GPU** before running.\n\n"
        "Trains from the consolidated dataset built by `scripts/build_dataset.py` "
        "(4 annotators merged, labels remapped to `transformer`/`wire`, images "
        "already CLAHE-preprocessed — **do not** preprocess again here)."
    )
    cell["id"] = "intro-header"
    cells.append(cell)

    cell = nbf.v4.new_code_cell(
        "# YOLO26 needs a recent ultralytics; latest pip has it.\n"
        "!pip install -q -U ultralytics"
    )
    cell["id"] = "install-deps"
    cells.append(cell)

    cell = nbf.v4.new_markdown_cell(
        "Upload `YOLO_thermal.zip` — zip the **contents** of the `YOLO_thermal/` "
        "folder so `train/` and `valid/` (each with `images/` + `labels/`) sit at "
        "the zip root."
    )
    cell["id"] = "upload-header"
    cells.append(cell)

    cell = nbf.v4.new_code_cell(
        "import zipfile, os\n"
        "from google.colab import files\n"
        "uploaded = files.upload()  # pick your dataset.zip\n"
        "zip_name = next(iter(uploaded))\n"
        'with zipfile.ZipFile(zip_name) as z:\n'
        '    z.extractall("/content/dataset")\n'
        'print("extracted:", os.listdir("/content/dataset"))'
    )
    cell["id"] = "upload-dataset"
    cells.append(cell)

    cell = nbf.v4.new_code_cell(
        "# Write the dataset config. names order MUST match labelImg classes.txt.\n"
        'data_yaml = "/content/dataset/data.yaml"\n'
        'with open(data_yaml, "w") as f:\n'
        '    f.write(\n'
        '        "path: /content/dataset\\n"\n'
        '        "train: train/images\\n"\n'
        '        "val: valid/images\\n"\n'
        '        "nc: 2\\n"\n'
        "        \"names: ['transformer', 'wire']\\n\"\n"
        "    )\n"
        "print(open(data_yaml).read())"
    )
    cell["id"] = "write-data-yaml"
    cells.append(cell)

    cell = nbf.v4.new_code_cell(
        "from ultralytics import YOLO\n"
        '# YOLO26x: largest, NMS-free end-to-end detector. Heavy for ~1.5k images,\n'
        "# so we lean on augmentation + early stopping to avoid overfitting.\n"
        'model = YOLO("yolo26x.pt")\n'
        "model.train(\n"
        "    data=data_yaml,\n"
        "    epochs=150, imgsz=640, patience=30,\n"
        "    batch=8,            # x is heavy: drop to 4 if a T4 OOMs, or batch=-1 to auto-fit\n"
        "    # Thermal-tuned aug: NO hue shift (it would scramble the heat palette);\n"
        "    # geometric aug + mosaic help the minority 'transformer' class generalize.\n"
        "    hsv_h=0.0, hsv_s=0.2, hsv_v=0.2,\n"
        "    fliplr=0.5, flipud=0.0, degrees=5.0,\n"
        "    mosaic=1.0, close_mosaic=10,\n"
        ")"
    )
    cell["id"] = "train-model"
    cells.append(cell)

    cell = nbf.v4.new_code_cell(
        "metrics = model.val()\n"
        'print("mAP50-95:", metrics.box.map)\n'
        'print("mAP50:   ", metrics.box.map50)'
    )
    cell["id"] = "validate-model"
    cells.append(cell)

    cell = nbf.v4.new_code_cell(
        "# Optional: eyeball predictions on the validation images (saved under runs/detect/predict).\n"
        'model.predict("/content/dataset/valid/images", conf=0.25, save=True)'
    )
    cell["id"] = "visual-check"
    cells.append(cell)

    cell = nbf.v4.new_code_cell(
        "# Download the trained weights to your machine.\n"
        "from google.colab import files\n"
        'files.download("runs/detect/train/weights/best.pt")'
    )
    cell["id"] = "download-weights"
    cells.append(cell)

    cell = nbf.v4.new_markdown_cell(
        "Put the downloaded `best.pt` into your project's `models/` folder, then run "
        "the API: `THERMAL_WEIGHTS=models/best.pt uvicorn api:app --reload`."
    )
    cell["id"] = "next-steps"
    cells.append(cell)

    nb["cells"] = cells
    return nb


def main() -> None:
    nb = build_notebook()
    with open("notebooks/train_yolo.ipynb", "w") as f:
        nbf.write(nb, f)
    print("wrote notebooks/train_yolo.ipynb")


if __name__ == "__main__":
    main()
