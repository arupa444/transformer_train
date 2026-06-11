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
