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
