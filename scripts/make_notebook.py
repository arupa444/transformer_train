"""Generate notebooks/train_yolo.ipynb for Google Colab YOLOv8 training."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

cells.append(nbf.v4.new_markdown_cell(
    "# Train YOLOv8 — Transformer Parts (tank, wire)\n"
    "Runtime → Change runtime type → **GPU** before running."
))

cells.append(nbf.v4.new_code_cell(
    "!pip install -q ultralytics roboflow"
))

cells.append(nbf.v4.new_code_cell(
    "# Paste the export snippet from Roboflow (Task 9, step 7).\n"
    "from roboflow import Roboflow\n"
    'rf = Roboflow(api_key="YOUR_API_KEY")\n'
    'project = rf.workspace("YOUR_WORKSPACE").project("transformer-thermal")\n'
    'dataset = project.version(1).download("yolov8")\n'
    "print(dataset.location)"
))

cells.append(nbf.v4.new_code_cell(
    "from ultralytics import YOLO\n"
    'model = YOLO("yolov8n.pt")  # nano: small, fast, enough for 2 classes\n'
    "model.train(\n"
    '    data=f"{dataset.location}/data.yaml",\n'
    "    epochs=100, imgsz=640, patience=20, batch=16,\n"
    ")"
))

cells.append(nbf.v4.new_code_cell(
    "metrics = model.val()\n"
    'print("mAP50-95:", metrics.box.map)\n'
    'print("mAP50:   ", metrics.box.map50)'
))

cells.append(nbf.v4.new_code_cell(
    "# Quick visual check on a validation image, then download weights.\n"
    "from google.colab import files\n"
    'files.download("runs/detect/train/weights/best.pt")'
))

cells.append(nbf.v4.new_markdown_cell(
    "Put the downloaded `best.pt` into your project's `models/` folder, then run "
    "the API: `THERMAL_WEIGHTS=models/best.pt uvicorn api:app --reload`."
))

nb["cells"] = cells
with open("notebooks/train_yolo.ipynb", "w") as f:
    nbf.write(nb, f)
print("wrote notebooks/train_yolo.ipynb")
