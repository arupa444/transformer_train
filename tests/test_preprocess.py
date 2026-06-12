import numpy as np
import pytest
from thermal.preprocess import preprocess


def _low_contrast_image():
    # A smooth low-contrast gradient (values 100..140) CLAHE should expand.
    row = np.linspace(100, 140, 64, dtype=np.uint8)
    gray = np.tile(row, (64, 1))
    return np.stack([gray, gray, gray], axis=-1)


def test_preprocess_preserves_shape_and_dtype():
    img = _low_contrast_image()
    out = preprocess(img)
    assert out.shape == img.shape
    assert out.dtype == np.uint8


def test_preprocess_increases_local_contrast():
    img = _low_contrast_image()
    out = preprocess(img)
    # CLAHE should widen the intensity spread of a low-contrast image.
    assert out.std() > img.std()


def test_preprocess_is_deterministic():
    img = _low_contrast_image()
    assert np.array_equal(preprocess(img), preprocess(img))


def test_preprocess_rejects_non_rgb():
    with pytest.raises(ValueError):
        preprocess(np.zeros((10, 10), dtype=np.uint8))
