# Annotation Guide — Roboflow

You are labeling **parts, not defects**. Every image has a tank and wires; you
just trace their boxes. The model learns shapes; the CV layer decides heat.

## Classes (exactly two)
- `tank` — the large rectangular transformer body.
- `wire` — each cable / bushing conductor entering the unit. Box each one
  separately (you need 2+ per image for the wire-to-wire comparison to work).

## Steps
1. Go to https://roboflow.com, create a free account.
2. **Create New Project** → Project Type: **Object Detection**. Name it
   `transformer-thermal`. Annotation group: `parts`.
3. **Upload** all your thermal images.
4. Open the annotation tool. For each image:
   - Draw a tight box around the tank → label `tank`.
   - Draw a tight box around **each** wire/bushing → label `wire`.
   - Save and go to the next image.
5. Aim for **150–300 images**. Keep variety: different angles, lighting, models.
6. **Generate a Version**:
   - Train/Valid/Test split: 70/20/10.
   - Preprocessing: Auto-Orient + Resize to 640×640.
   - Augmentations (optional, helps small datasets): horizontal flip,
     ±15% brightness, ±10° rotation. Do **not** use hue/saturation shifts —
     that would break the color→heat assumption.
7. **Export** → format **YOLOv8** → choose "show download code". Copy the
   `roboflow.workspace(...).project(...).version(n).download("yolov8")` snippet —
   you paste it into the Colab notebook (Task 10).

## Tip: bootstrap later
Once you have a trained `best.pt`, run new images through it, export the
predictions as pre-labels, and just correct them in Roboflow — far faster than
labeling from scratch.
