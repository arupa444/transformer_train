"""Generate notebooks/train_yolo.ipynb for Google Colab YOLOv8 training."""
import nbformat as nbf


def build_notebook() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    cells = []

    cell = nbf.v4.new_markdown_cell(
        "# Train YOLOv8 — Transformer Parts (transformer, wire)\n"
        "Runtime → Change runtime type → **GPU** before running.\n\n"
        "This trains from a **labelImg** dataset (YOLO `.txt` labels)."
    )
    cell["id"] = "intro-header"
    cells.append(cell)

    cell = nbf.v4.new_code_cell(
        "!pip install -q ultralytics"
    )
    cell["id"] = "install-deps"
    cells.append(cell)

    cell = nbf.v4.new_markdown_cell(
        "Upload your `dataset.zip`. It should contain `train/` and `valid/` at its "
        "root, each with `images/` and `labels/` subfolders (see "
        "`docs/annotation-guide.md`)."
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
        'model = YOLO("yolov8n.pt")  # nano: small, fast, enough for 2 classes\n'
        "model.train(\n"
        "    data=data_yaml,\n"
        "    # epochs/imgsz: standard starting point; batch=16 fits a Colab T4's VRAM.\n"
        "    epochs=100, imgsz=640, patience=20, batch=16,\n"
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
