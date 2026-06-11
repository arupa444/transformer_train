import numpy as np
from thermal.schema import Detection
from thermal.defects import (
    severity_from_delta, analyze_wires, analyze_tank,
)


def test_severity_buckets():
    assert severity_from_delta(0.05) == "Normal"
    assert severity_from_delta(0.15) == "Watch"
    assert severity_from_delta(0.30) == "Investigate"
    assert severity_from_delta(0.50) == "Critical"


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


def test_analyze_tank_detects_local_hotspot():
    intensity = np.full((100, 100), 0.3, dtype=np.float32)  # warm body
    intensity[44:56, 44:56] = 0.95                          # tiny hotspot (12x12 = 1.44%, > 1% needed for p99)
    finding = analyze_tank(intensity, Detection("tank", (0, 0, 100, 100)))
    assert finding.component == "tank"
    assert finding.severity in ("Investigate", "Critical")
