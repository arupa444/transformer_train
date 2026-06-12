import numpy as np
from thermal.schema import Detection
from thermal.defects import (
    severity_from_delta, analyze_wires, analyze_transformer,
)


def test_severity_buckets():
    assert severity_from_delta(0.05) == "Normal"
    assert severity_from_delta(0.15) == "Watch"
    assert severity_from_delta(0.30) == "Investigate"
    assert severity_from_delta(0.50) == "Critical"


def test_severity_nonfinite_is_normal_not_critical():
    # a non-finite delta (NaN/inf) must never silently become Critical
    assert severity_from_delta(float("nan")) == "Normal"
    assert severity_from_delta(float("inf")) == "Normal"


def test_crop_normalizes_reversed_bbox():
    from thermal.defects import _crop
    intensity = np.zeros((100, 100), dtype=np.float32)
    intensity[10:90, 40:60] = 1.0
    # reversed x (60>40) must still crop the 40..60 region, not a 1px sliver
    crop = _crop(intensity, (60, 10, 40, 90))
    assert crop.shape[1] >= 15 and float(crop.mean()) > 0.5


def test_analyze_wires_flags_the_hot_one():
    intensity = np.zeros((100, 100), dtype=np.float32)
    # Three wire regions: two cool (~0.2), one hot (~0.9).
    intensity[10:90, 10:20] = 0.2
    intensity[10:90, 40:50] = 0.2
    intensity[10:90, 70:80] = 0.9
    wires = [
        Detection("wire", (10, 10, 20, 90)),
        Detection("wire", (40, 10, 50, 90)),
        Detection("wire", (70, 10, 80, 90)),
    ]
    findings = analyze_wires(intensity, wires)
    by_box = {f.bbox: f for f in findings}
    assert by_box[(70, 10, 80, 90)].severity == "Critical"
    assert by_box[(10, 10, 20, 90)].severity == "Normal"


def test_analyze_wires_skips_when_fewer_than_two():
    intensity = np.full((50, 50), 0.5, dtype=np.float32)
    assert analyze_wires(intensity, [Detection("wire", (0, 0, 10, 10))]) == []


def test_analyze_transformer_detects_local_hotspot():
    intensity = np.full((100, 100), 0.3, dtype=np.float32)  # warm body
    intensity[45:55, 45:55] = 0.95                          # tiny hotspot
    finding = analyze_transformer(intensity, Detection("transformer", (0, 0, 100, 100)))
    assert finding.component == "transformer"
    assert finding.severity in ("Investigate", "Critical")


def test_analyze_transformer_ignores_cold_background():
    # Box dominated by cold foliage (60%) around a warm body (40%) with only a
    # mild warm spot — no real defect. The Otsu-based body reference must use the
    # warm pixels (~0.40), so the delta stays small instead of a false Critical.
    intensity = np.full((100, 100), 0.05, dtype=np.float32)  # cold background
    intensity[:, 60:] = 0.40                                 # warm body region
    intensity[10:20, 80:90] = 0.55                           # mild warm spot
    finding = analyze_transformer(intensity, Detection("transformer", (0, 0, 100, 100)))
    assert finding.severity != "Critical"
