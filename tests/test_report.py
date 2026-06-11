import numpy as np
from thermal.schema import DefectFinding
from thermal.report import to_json, annotate, heatmap_overlay


def test_to_json_structure():
    findings = [DefectFinding("wire", (1, 2, 3, 4), "Critical", 0.4123)]
    out = to_json(findings)
    assert out == [{
        "component": "wire",
        "bbox": [1, 2, 3, 4],
        "severity": "Critical",
        "relative_delta": 0.4123,
    }]


def test_annotate_keeps_image_shape():
    img = np.zeros((50, 50, 3), dtype=np.uint8)
    findings = [DefectFinding("tank", (5, 5, 40, 40), "Watch", 0.15)]
    out = annotate(img, findings)
    assert out.shape == img.shape
    assert out is not img            # must not mutate input


def test_heatmap_overlay_keeps_shape():
    img = np.zeros((30, 30, 3), dtype=np.uint8)
    intensity = np.full((30, 30), 0.5, dtype=np.float32)
    out = heatmap_overlay(img, intensity)
    assert out.shape == img.shape
