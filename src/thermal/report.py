import cv2
import numpy as np
from typing import List
from thermal.schema import DefectFinding

# BGR colors (OpenCV order) per severity.
SEVERITY_COLORS = {
    "Normal": (0, 200, 0),
    "Watch": (0, 200, 200),
    "Investigate": (0, 140, 255),
    "Critical": (0, 0, 255),
}


def to_json(findings: List[DefectFinding]) -> list:
    return [{
        "component": f.component,
        "bbox": list(f.bbox),
        "severity": f.severity,
        "relative_delta": round(f.relative_delta, 4),
    } for f in findings]


def annotate(img_bgr: np.ndarray, findings: List[DefectFinding]) -> np.ndarray:
    out = img_bgr.copy()
    for f in findings:
        x1, y1, x2, y2 = f.bbox
        color = SEVERITY_COLORS[f.severity]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"{f.component}:{f.severity}"
        cv2.putText(out, label, (x1, max(12, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return out


def heatmap_overlay(img_bgr: np.ndarray, intensity: np.ndarray,
                    alpha: float = 0.4) -> np.ndarray:
    hm = (np.clip(intensity, 0, 1) * 255).astype(np.uint8)
    hm = cv2.applyColorMap(hm, cv2.COLORMAP_INFERNO)
    return cv2.addWeighted(img_bgr, 1 - alpha, hm, alpha, 0)
