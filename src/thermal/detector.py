from thermal.schema import Detection


class TransformerDetector:
    def __init__(self, weights_path: str):
        from ultralytics import YOLO  # lazy import keeps torch out of the test path
        self.model = YOLO(weights_path)
        self.names = self.model.names

    def detect(self, img_rgb, conf: float = 0.25):
        result = self.model(img_rgb, conf=conf)[0]
        dets = []
        for box in result.boxes:
            cls_name = self.names[int(box.cls[0])]
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
            dets.append(Detection(cls_name, (x1, y1, x2, y2), float(box.conf[0])))
        return dets
