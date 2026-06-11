import numpy as np
from skimage.filters import threshold_otsu
from thermal.schema import Detection, DefectFinding

# Severity thresholds in 0..1 intensity units (tunable starting values).
_WATCH, _INVESTIGATE, _CRITICAL = 0.10, 0.20, 0.35

# Percentile used as a region's representative "hot" level (tunable).
_WIRE_HEAT_PERCENTILE = 90
_TRANSFORMER_HOTSPOT_PERCENTILE = 99

# A trustworthy warm/background split needs at least this many warm pixels.
_MIN_WARM_FRACTION = 0.05
_MIN_WARM_PIXELS = 16


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


def _body_reference(flat: np.ndarray) -> float:
    """Median intensity of the warm (equipment) pixels.

    The transformer detection box usually contains cold background (foliage,
    sky) around the unit. We split warm equipment from cold background with an
    Otsu threshold and take the median of the warm side, so background cannot
    drag the reference down and inflate the hotspot delta into a false alarm.
    Falls back to the overall median when the crop is not clearly bimodal.
    """
    if flat.size == 0:
        return 0.0
    if np.ptp(flat) == 0:  # single value: nothing to split
        return float(np.median(flat))
    try:
        threshold = threshold_otsu(flat)
    except ValueError:
        return float(np.median(flat))
    warm = flat[flat >= threshold]
    if warm.size < max(_MIN_WARM_PIXELS, int(flat.size * _MIN_WARM_FRACTION)):
        return float(np.median(flat))
    return float(np.median(warm))


def analyze_transformer(intensity: np.ndarray,
                        transformer: Detection) -> DefectFinding:
    """Flag a localized hotspot on the transformer relative to its body.

    The box often includes cold background, so the body reference is estimated
    from the warm equipment pixels only (see _body_reference)."""
    crop = _crop(intensity, transformer.bbox)
    body = _body_reference(crop.ravel())
    # method="higher" makes p99 land on a real pixel value, not an interpolated
    # one, so a tiny hotspot (~1% of the crop) is not diluted away.
    hot = float(np.percentile(crop, _TRANSFORMER_HOTSPOT_PERCENTILE, method="higher"))
    delta = hot - body
    return DefectFinding("transformer", transformer.bbox,
                         severity_from_delta(delta), delta)
