# Transformer Thermal Defect Classifier — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Automatically flag overheating wires and tank hotspots in colorized thermal images of transformers, served over a FastAPI endpoint.

**Architecture:** A YOLOv8 object detector finds the physical parts (`tank`, `wire`) and crops them. A classical CV layer inverts the color palette back to a 0–1 heat-intensity map, then decides defects by *relative* heat (a wire hotter than its siblings, a localized tank hotspot) and assigns severity. The detector only localizes; the CV layer makes every defect call.

**Tech Stack:** Python (managed with `uv`), OpenCV, NumPy, scikit-image, SciPy, Ultralytics YOLOv8 (trained on Google Colab), FastAPI + uvicorn, Roboflow for annotation, pytest.

---

## File Structure

```
transformer/
├── pyproject.toml                 # deps + pytest config
├── src/thermal/
│   ├── __init__.py
│   ├── schema.py                  # Detection, DefectFinding dataclasses (shared types)
│   ├── colormap.py                # color -> 0..1 heat intensity (palette inversion)
│   ├── defects.py                 # relative hotspot + wire-to-wire + severity
│   ├── report.py                  # annotated image + JSON output
│   ├── pipeline.py                # orchestrates detector -> colormap -> defects
│   └── detector.py                # YOLO wrapper (loads best.pt)
├── api.py                         # FastAPI service (create_app factory)
├── notebooks/train_yolo.ipynb     # Colab training notebook (generated)
├── scripts/make_notebook.py       # generates the notebook deterministically
├── docs/annotation-guide.md       # how to label in Roboflow
├── models/                        # best.pt lives here (gitignored)
└── tests/
    ├── test_colormap.py
    ├── test_defects.py
    ├── test_report.py
    ├── test_pipeline.py
    └── test_api.py
```

**Build order:** Part A (Tasks 1–8) is the CV core + API, fully testable with fake detections and synthetic images — **no model needed**. Part B (Tasks 9–10) is annotation + training. Part C (Task 11) is the final hookup + run guide.

---

## Task 1: Project setup

**Files:**
- Create: `pyproject.toml`
- Create: `src/thermal/__init__.py` (empty)
- Create: `tests/__init__.py` (empty)

- [ ] **Step 1: Create the virtual environment**

Run:
```bash
uv venv && source .venv/bin/activate
```
Expected: `.venv` created and activated.

- [ ] **Step 2: Write `pyproject.toml`**

```toml
[project]
name = "thermal-defect"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "numpy",
    "opencv-python-headless",
    "scikit-image",
    "scipy",
    "matplotlib",
    "fastapi",
    "uvicorn",
    "python-multipart",
]

[project.optional-dependencies]
# Heavy: pulls in torch. Only needed to run the REAL detector (after training).
# The CV core + API + full test suite run without it.
inference = ["ultralytics"]
dev = ["pytest", "nbformat", "httpx"]

[tool.pytest.ini_options]
pythonpath = ["src", "."]
testpaths = ["tests"]
```

- [ ] **Step 3: Install dependencies**

Run:
```bash
uv pip install -e ".[dev]"
```
Expected: all packages install without error.

- [ ] **Step 4: Create empty package files**

```bash
mkdir -p src/thermal tests notebooks scripts docs models
touch src/thermal/__init__.py tests/__init__.py
```

- [ ] **Step 5: Verify pytest runs (no tests yet)**

Run:
```bash
pytest -q
```
Expected: `no tests ran` (exit 0 or 5), no import errors.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml src/thermal/__init__.py tests/__init__.py
git commit -m "chore: project setup with uv and pytest"
```

---

## Task 2: Shared types (`schema.py`)

**Files:**
- Create: `src/thermal/schema.py`
- Test: `tests/test_defects.py` (will import these; no standalone test needed yet)

- [ ] **Step 1: Write `src/thermal/schema.py`**

```python
from dataclasses import dataclass
from typing import Tuple

# A box detected by YOLO. cls is "tank" or "wire".
@dataclass
class Detection:
    cls: str
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2 (pixels)
    conf: float = 1.0

# One defect decision made by the CV layer.
@dataclass
class DefectFinding:
    component: str          # "tank" or "wire"
    bbox: Tuple[int, int, int, int]
    severity: str           # Normal / Watch / Investigate / Critical
    relative_delta: float   # how far above reference, in 0..1 intensity units
```

- [ ] **Step 2: Verify it imports**

Run:
```bash
python -c "from thermal.schema import Detection, DefectFinding; print('ok')"
```
Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add src/thermal/schema.py
git commit -m "feat: shared Detection and DefectFinding types"
```

---

## Task 3: Colormap inversion (`colormap.py`)

This turns the colored image into a 0–1 heat map. Cold colors → near 0, hot colors → near 1. We approximate the iron/ironbow palette with matplotlib's `inferno` (black→purple→orange→yellow), matching each pixel to the nearest palette color in perceptual Lab space.

**Files:**
- Create: `src/thermal/colormap.py`
- Test: `tests/test_colormap.py`

- [ ] **Step 1: Write the failing test**

```python
import numpy as np
from thermal.colormap import build_lut, ColorToHeat


def test_intensity_monotonic_on_palette_gradient():
    lut = build_lut("inferno", n=256)
    c2h = ColorToHeat(lut)
    # Build a 1x256 image that IS the palette, cold->hot left to right.
    img = (lut * 255).astype(np.uint8).reshape(1, -1, 3)
    intensity, dist = c2h.to_intensity(img)
    vals = intensity[0]
    assert vals[0] < 0.05          # coldest color -> ~0
    assert vals[-1] > 0.95         # hottest color -> ~1
    assert np.all(np.diff(vals) >= -1e-6)   # monotonically increasing


def test_calibration_flags_wrong_palette():
    lut = build_lut("inferno", n=256)
    c2h = ColorToHeat(lut)
    # Pure green is far from any inferno color -> poor match.
    green = np.zeros((4, 4, 3), dtype=np.uint8)
    green[..., 1] = 255
    _, dist = c2h.to_intensity(green)
    assert float(np.median(dist)) > 20.0   # high distance = bad calibration
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
pytest tests/test_colormap.py -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'thermal.colormap'`

- [ ] **Step 3: Write `src/thermal/colormap.py`**

```python
import numpy as np
from matplotlib import colormaps
from skimage import color
from scipy.spatial import cKDTree


def build_lut(name: str = "inferno", n: int = 256) -> np.ndarray:
    """Return an (n, 3) array of RGB values in 0..1 for the named colormap,
    ordered cold (index 0) to hot (index n-1)."""
    cmap = colormaps[name].resampled(n)
    return np.array([cmap(i)[:3] for i in range(n)], dtype=np.float64)


class ColorToHeat:
    """Inverts a colorized thermal image back to a 0..1 heat-intensity map."""

    def __init__(self, lut_rgb: np.ndarray):
        self.lut_rgb = lut_rgb
        self.n = lut_rgb.shape[0]
        lab = color.rgb2lab(lut_rgb.reshape(-1, 1, 3)).reshape(-1, 3)
        self._tree = cKDTree(lab)

    def to_intensity(self, img_rgb: np.ndarray):
        """img_rgb: HxWx3 uint8. Returns (intensity HxW float 0..1, dist HxW)."""
        arr = img_rgb.astype(np.float64) / 255.0
        lab = color.rgb2lab(arr).reshape(-1, 3)
        dist, idx = self._tree.query(lab)
        intensity = (idx / (self.n - 1)).reshape(img_rgb.shape[:2])
        return intensity.astype(np.float32), dist.reshape(img_rgb.shape[:2])
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
pytest tests/test_colormap.py -v
```
Expected: PASS (both tests)

- [ ] **Step 5: Commit**

```bash
git add src/thermal/colormap.py tests/test_colormap.py
git commit -m "feat: invert thermal palette to 0..1 heat map"
```

---

## Task 4: Defect logic (`defects.py`)

The brain. Two checks: (1) compare each wire to its sibling wires — the outlier is the defect; (2) find a localized hotspot inside the tank. Severity scales with how far above reference the heat sits.

**Files:**
- Create: `src/thermal/defects.py`
- Test: `tests/test_defects.py`

- [ ] **Step 1: Write the failing test**

```python
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
    intensity[45:55, 45:55] = 0.95                          # tiny hotspot
    finding = analyze_tank(intensity, Detection("tank", (0, 0, 100, 100)))
    assert finding.component == "tank"
    assert finding.severity in ("Investigate", "Critical")
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
pytest tests/test_defects.py -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'thermal.defects'`

- [ ] **Step 3: Write `src/thermal/defects.py`**

```python
import numpy as np
from typing import List
from thermal.schema import Detection, DefectFinding

# Severity thresholds in 0..1 intensity units (tunable starting values).
_WATCH, _INVESTIGATE, _CRITICAL = 0.10, 0.20, 0.35


def severity_from_delta(delta: float) -> str:
    if delta < _WATCH:
        return "Normal"
    if delta < _INVESTIGATE:
        return "Watch"
    if delta < _CRITICAL:
        return "Investigate"
    return "Critical"


def _crop(intensity: np.ndarray, bbox) -> np.ndarray:
    h, w = intensity.shape
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(int(x1), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    x2 = max(x1 + 1, min(int(x2), w))
    y2 = max(y1 + 1, min(int(y2), h))
    return intensity[y1:y2, x1:x2]


def _wire_heat(intensity: np.ndarray, bbox, pct: int = 90) -> float:
    return float(np.percentile(_crop(intensity, bbox), pct))


def analyze_wires(intensity: np.ndarray,
                  wires: List[Detection]) -> List[DefectFinding]:
    """Compare each wire's heat to the median of all wires. Needs >= 2 wires."""
    if len(wires) < 2:
        return []
    heats = [_wire_heat(intensity, d.bbox) for d in wires]
    reference = float(np.median(heats))
    findings = []
    for d, h in zip(wires, heats):
        delta = h - reference
        findings.append(DefectFinding("wire", d.bbox,
                                       severity_from_delta(delta), delta))
    return findings


def analyze_tank(intensity: np.ndarray, tank: Detection) -> DefectFinding:
    """Flag a localized hotspot relative to the tank's own body temperature."""
    crop = _crop(intensity, tank.bbox)
    body = float(np.median(crop))
    hot = float(np.percentile(crop, 99))
    delta = hot - body
    return DefectFinding("tank", tank.bbox, severity_from_delta(delta), delta)
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
pytest tests/test_defects.py -v
```
Expected: PASS (all four tests)

- [ ] **Step 5: Commit**

```bash
git add src/thermal/defects.py tests/test_defects.py
git commit -m "feat: relative wire and tank defect detection with severity"
```

---

## Task 5: Report output (`report.py`)

**Files:**
- Create: `src/thermal/report.py`
- Test: `tests/test_report.py`

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
pytest tests/test_report.py -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'thermal.report'`

- [ ] **Step 3: Write `src/thermal/report.py`**

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
pytest tests/test_report.py -v
```
Expected: PASS (all three tests)

- [ ] **Step 5: Commit**

```bash
git add src/thermal/report.py tests/test_report.py
git commit -m "feat: annotated image and JSON report output"
```

---

## Task 6: Pipeline orchestration (`pipeline.py`)

Ties detector + colormap + defects together. The detector is passed in (dependency injection) so we can test with a fake one — **no trained model required here.**

**Files:**
- Create: `src/thermal/pipeline.py`
- Test: `tests/test_pipeline.py`

- [ ] **Step 1: Write the failing test**

```python
import numpy as np
from thermal.schema import Detection
from thermal.colormap import build_lut, ColorToHeat
from thermal.pipeline import analyze_image


class FakeDetector:
    """Returns fixed detections regardless of input."""
    def __init__(self, dets):
        self._dets = dets
    def detect(self, img_rgb, conf=0.25):
        return self._dets


def test_analyze_image_returns_findings_and_calibration():
    img = np.zeros((100, 100, 3), dtype=np.uint8)  # all black -> low intensity
    detector = FakeDetector([
        Detection("tank", (0, 0, 100, 100)),
        Detection("wire", (10, 10, 20, 90)),
        Detection("wire", (40, 10, 50, 90)),
    ])
    c2h = ColorToHeat(build_lut("inferno"))
    findings, intensity, calib_ok = analyze_image(img, detector, c2h)
    assert intensity.shape == (100, 100)
    components = sorted({f.component for f in findings})
    assert components == ["tank", "wire"]
    assert isinstance(calib_ok, bool)
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
pytest tests/test_pipeline.py -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'thermal.pipeline'`

- [ ] **Step 3: Write `src/thermal/pipeline.py`**

```python
import numpy as np
from thermal.defects import analyze_wires, analyze_tank

# Median Lab-distance above which we consider the palette mismatched.
CALIBRATION_DIST_THRESHOLD = 20.0


def analyze_image(img_rgb: np.ndarray, detector, c2h):
    """Run the full pipeline. Returns (findings, intensity_map, calibration_ok)."""
    dets = detector.detect(img_rgb)
    intensity, dist = c2h.to_intensity(img_rgb)
    calibration_ok = bool(np.median(dist) < CALIBRATION_DIST_THRESHOLD)

    wires = [d for d in dets if d.cls == "wire"]
    tanks = [d for d in dets if d.cls == "tank"]

    findings = analyze_wires(intensity, wires)
    findings += [analyze_tank(intensity, t) for t in tanks]
    return findings, intensity, calibration_ok
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
pytest tests/test_pipeline.py -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/thermal/pipeline.py tests/test_pipeline.py
git commit -m "feat: pipeline orchestrating detector, colormap, defects"
```

---

## Task 7: Detector wrapper (`detector.py`)

Thin wrapper over Ultralytics YOLO. It needs `best.pt` to actually run, so its unit test only checks the class translates YOLO output into our `Detection` type using a fake YOLO model (no weights needed).

**Files:**
- Create: `src/thermal/detector.py`
- Test: `tests/test_detector.py`

- [ ] **Step 1: Write the failing test**

```python
import numpy as np
from thermal.detector import TransformerDetector


class _FakeBox:
    def __init__(self, cls_id, xyxy, conf):
        self.cls = [cls_id]
        self.xyxy = [xyxy]
        self.conf = [conf]


class _FakeResult:
    def __init__(self, boxes):
        self.boxes = boxes


class _FakeYOLO:
    names = {0: "tank", 1: "wire"}
    def __call__(self, img, conf=0.25):
        return [_FakeResult([
            _FakeBox(0, [1.0, 2.0, 30.0, 40.0], 0.9),
            _FakeBox(1, [5.0, 6.0, 10.0, 50.0], 0.7),
        ])]


def test_detector_translates_yolo_output():
    det = TransformerDetector.__new__(TransformerDetector)  # skip __init__
    det.model = _FakeYOLO()
    det.names = _FakeYOLO.names
    img = np.zeros((60, 60, 3), dtype=np.uint8)
    dets = det.detect(img)
    assert dets[0].cls == "tank"
    assert dets[0].bbox == (1, 2, 30, 40)
    assert dets[1].cls == "wire"
    assert abs(dets[1].conf - 0.7) < 1e-6
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
pytest tests/test_detector.py -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'thermal.detector'`

- [ ] **Step 3: Write `src/thermal/detector.py`**

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
pytest tests/test_detector.py -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/thermal/detector.py tests/test_detector.py
git commit -m "feat: YOLO detector wrapper translating to Detection type"
```

---

## Task 8: FastAPI service (`api.py`)

A `create_app(detector, c2h)` factory so tests inject a fake detector. The module also builds a default `app` for `uvicorn`, but that line is guarded so importing the module in tests (without `best.pt`) never fails.

**Files:**
- Create: `api.py`
- Test: `tests/test_api.py`

- [ ] **Step 1: Write the failing test**

```python
import base64
import numpy as np
import cv2
from fastapi.testclient import TestClient
from thermal.schema import Detection
from thermal.colormap import build_lut, ColorToHeat
from api import create_app


class FakeDetector:
    def detect(self, img_rgb, conf=0.25):
        return [
            Detection("tank", (0, 0, 100, 100)),
            Detection("wire", (10, 10, 20, 90)),
            Detection("wire", (40, 10, 50, 90)),
        ]


def _png_bytes():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".png", img)
    return buf.tobytes()


def test_analyze_endpoint_returns_report():
    app = create_app(FakeDetector(), ColorToHeat(build_lut("inferno")))
    client = TestClient(app)
    resp = client.post("/analyze", files={"file": ("t.png", _png_bytes(), "image/png")})
    assert resp.status_code == 200
    body = resp.json()
    assert "defects" in body
    assert "calibration_ok" in body
    assert "annotated_image_png_b64" in body
    # annotated image must be valid base64-decodable PNG bytes
    raw = base64.b64decode(body["annotated_image_png_b64"])
    assert raw[:8] == b"\x89PNG\r\n\x1a\n"
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
pytest tests/test_api.py -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'api'`

- [ ] **Step 3: Write `api.py`**

```python
import os
import base64
import numpy as np
import cv2
from fastapi import FastAPI, UploadFile, File

from thermal.colormap import build_lut, ColorToHeat
from thermal.pipeline import analyze_image
from thermal.report import annotate, to_json


def create_app(detector, c2h: ColorToHeat) -> FastAPI:
    app = FastAPI(title="Transformer Thermal Defect Classifier")

    @app.post("/analyze")
    async def analyze(file: UploadFile = File(...)):
        data = await file.read()
        img_bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if img_bgr is None:
            return {"error": "could not decode image"}
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        findings, _intensity, calib_ok = analyze_image(img_rgb, detector, c2h)
        annotated = annotate(img_bgr, findings)
        ok, buf = cv2.imencode(".png", annotated)
        b64 = base64.b64encode(buf.tobytes()).decode()

        return {
            "calibration_ok": calib_ok,
            "defects": to_json(findings),
            "annotated_image_png_b64": b64,
        }

    return app


# Default app for `uvicorn api:app`. Guarded so tests can import without weights.
_weights = os.environ.get("THERMAL_WEIGHTS", "models/best.pt")
if os.path.exists(_weights):
    from thermal.detector import TransformerDetector
    app = create_app(TransformerDetector(_weights), ColorToHeat(build_lut("inferno")))
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
pytest tests/test_api.py -v
```
Expected: PASS

- [ ] **Step 5: Run the full suite**

Run:
```bash
pytest -q
```
Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add api.py tests/test_api.py
git commit -m "feat: FastAPI /analyze endpoint with injectable detector"
```

---

## Task 9: Annotation guide (`docs/annotation-guide.md`)

A manual procedure (no code to test). Produces the labeled dataset that Task 10 trains on.

**Files:**
- Create: `docs/annotation-guide.md`

- [ ] **Step 1: Write `docs/annotation-guide.md`**

````markdown
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
````

- [ ] **Step 2: Commit**

```bash
git add docs/annotation-guide.md
git commit -m "docs: Roboflow annotation guide (tank, wire classes)"
```

---

## Task 10: Colab training notebook (`notebooks/train_yolo.ipynb`)

Generated deterministically by a script so it's version-controlled.

**Files:**
- Create: `scripts/make_notebook.py`
- Create (generated): `notebooks/train_yolo.ipynb`

- [ ] **Step 1: Write `scripts/make_notebook.py`**

```python
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
```

- [ ] **Step 2: Generate the notebook**

Run:
```bash
python scripts/make_notebook.py
```
Expected: `wrote notebooks/train_yolo.ipynb`

- [ ] **Step 3: Verify the notebook is valid**

Run:
```bash
python -c "import nbformat; nbformat.read('notebooks/train_yolo.ipynb', as_version=4); print('valid')"
```
Expected: `valid`

- [ ] **Step 4: Commit**

```bash
git add scripts/make_notebook.py notebooks/train_yolo.ipynb
git commit -m "feat: Colab YOLOv8 training notebook + generator"
```

---

## Task 11: End-to-end run guide (`README.md`)

**Files:**
- Create: `README.md`

- [ ] **Step 1: Write `README.md`**

````markdown
# Transformer Thermal Defect Classifier

Detects overheating wires and tank hotspots in colorized thermal images.
A YOLOv8 detector crops the parts; a CV layer reads relative heat and flags defects.

> Output is **relative severity** (Normal/Watch/Investigate/Critical), not absolute °C.
> The input is colorized screenshots with no embedded temperature.

## Setup
```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
pytest -q          # all CV + API tests pass without a model
```

## Get a model
1. Label images in Roboflow — see `docs/annotation-guide.md`.
2. Train on Colab — open `notebooks/train_yolo.ipynb`, run all cells.
3. Download `best.pt` into `models/`.

## Run the API
```bash
uv pip install -e ".[inference]"   # installs ultralytics/torch for the real detector
THERMAL_WEIGHTS=models/best.pt uvicorn api:app --reload
```
Then POST an image:
```bash
curl -F "file=@your_image.jpg" http://127.0.0.1:8000/analyze
```
Returns JSON: `calibration_ok`, `defects[]` (component, bbox, severity,
relative_delta), and `annotated_image_png_b64` (base64 PNG with boxes).

## How it works
1. **Detector** (`src/thermal/detector.py`) finds `tank` and `wire` boxes.
2. **Colormap** (`src/thermal/colormap.py`) inverts the iron palette to a 0–1 heat map.
3. **Defects** (`src/thermal/defects.py`): a wire much hotter than its siblings,
   or a localized tank hotspot, is flagged; severity scales with the gap.
4. **Report** (`src/thermal/report.py`) draws boxes + emits JSON.

## Tuning
Severity thresholds live in `src/thermal/defects.py` (`_WATCH`, `_INVESTIGATE`,
`_CRITICAL`). If a different camera palette is used, change the colormap name in
`api.py` / `build_lut(...)` and re-check `calibration_ok`.
````

- [ ] **Step 2: Verify the full suite once more**

Run:
```bash
pytest -q
```
Expected: all tests pass.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: end-to-end setup and run guide"
```

---

## Self-Review Notes

- **Spec coverage:** colormap inversion (Task 3), auto-localization via detector
  (Tasks 7, 10 + annotation Task 9), relative hotspot + wire-to-wire + severity
  (Task 4), report (Task 5), FastAPI inference (Task 8), classes `tank`/`wire`
  (Task 9), Colab training (Task 10), calibration-warning for wrong palette
  (Tasks 3, 6), validation approach (README/annotation guide). All covered.
- **Relative-severity-only** constraint surfaced in README and report semantics.
- **Type consistency:** `Detection(cls, bbox, conf)` and
  `DefectFinding(component, bbox, severity, relative_delta)` used identically
  across schema, defects, pipeline, detector, report, and api.
- **No model needed for Tasks 1–8:** detector is injected; tests use fakes.
````
