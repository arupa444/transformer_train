import numpy as np
from thermal.schema import Detection, DefectFinding

# Severity thresholds in 0..1 intensity units (tunable starting values).
_WATCH, _INVESTIGATE, _CRITICAL = 0.10, 0.20, 0.35

# Percentile used as a region's representative "hot" level (tunable).
_WIRE_HEAT_PERCENTILE = 90
_TANK_HOTSPOT_PERCENTILE = 99


def severity_from_delta(delta: float) -> str:
    if delta < _WATCH:
        return "Normal"
    if delta < _INVESTIGATE:
        return "Watch"
    if delta < _CRITICAL:
        return "Investigate"
    return "Critical"


def _crop(intensity: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    h, w = intensity.shape
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(int(x1), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    x2 = max(x1 + 1, min(int(x2), w))
    y2 = max(y1 + 1, min(int(y2), h))
    return intensity[y1:y2, x1:x2]


def _wire_heat(intensity: np.ndarray, bbox: tuple[int, int, int, int],
               pct: int = _WIRE_HEAT_PERCENTILE) -> float:
    return float(np.percentile(_crop(intensity, bbox), pct))


def analyze_wires(intensity: np.ndarray,
                  wires: list[Detection]) -> list[DefectFinding]:
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
    # method="higher" makes p99 land on a real pixel value, not an interpolated
    # one, so a tiny hotspot (~1% of the crop) is not diluted away.
    hot = float(np.percentile(crop, _TANK_HOTSPOT_PERCENTILE, method="higher"))
    delta = hot - body
    return DefectFinding("tank", tank.bbox, severity_from_delta(delta), delta)
