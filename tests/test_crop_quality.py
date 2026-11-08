"""Unit tests for the plate crop quality helpers in lpr.py.

torch and ultralytics are stubbed out so these run without a GPU or the
heavy ML dependencies (see tests/test_plate_text.py).
"""

import sys
import types
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

for name in ("torch", "ultralytics"):
    if name not in sys.modules:
        stub = types.ModuleType(name)
        if name == "ultralytics":
            stub.YOLO = object
        sys.modules[name] = stub

import lpr  # noqa: E402


def _checkerboard(height, width, square=4, channels=3):
    """A high-contrast, high-frequency image (sharp edges everywhere)."""
    ys, xs = np.indices((height, width))
    board = (((ys // square) + (xs // square)) % 2 * 255).astype(np.uint8)
    if channels == 1:
        return board
    return np.repeat(board[:, :, None], channels, axis=2)


@pytest.mark.parametrize("shape", [(20, 60), (20, 60, 1), (20, 60, 3), (20, 60, 4)])
def test_to_gray_returns_single_channel(shape):
    gray = lpr.to_gray(np.zeros(shape, dtype=np.uint8))
    assert gray.shape == (20, 60)


@pytest.mark.parametrize("image", [None, np.zeros((0, 0, 3), dtype=np.uint8)])
def test_sharpness_of_missing_image_is_zero(image):
    assert lpr.sharpness(image) == 0.0


def test_flat_image_has_zero_sharpness():
    flat = np.full((30, 90, 3), 128, dtype=np.uint8)
    assert lpr.sharpness(flat) == 0.0


def test_sharp_image_scores_higher_than_blurred_copy():
    sharp = _checkerboard(40, 120)
    blurred = lpr.cv2.GaussianBlur(sharp, (9, 9), 3)
    assert lpr.sharpness(sharp) > lpr.sharpness(blurred)


def test_sharpness_works_on_grayscale_crop():
    assert lpr.sharpness(_checkerboard(40, 120, channels=1)) > 0.0


def test_plate_quality_rejects_too_narrow_crops():
    narrow = _checkerboard(20, lpr.MIN_PLATE_PIXEL_WIDTH - 1)
    assert lpr.plate_quality(narrow) == 0.0


@pytest.mark.parametrize("image", [None, np.zeros((0, 0, 3), dtype=np.uint8)])
def test_plate_quality_of_missing_image_is_zero(image):
    assert lpr.plate_quality(image) == 0.0


def test_plate_quality_is_capped():
    # sharpness, contrast and width terms are capped at 5 + 2 + 2.
    huge = _checkerboard(100, 600, square=1)
    assert lpr.plate_quality(huge) == pytest.approx(9.0)


def test_plate_quality_prefers_sharper_crop():
    sharp = _checkerboard(40, 120)
    blurred = lpr.cv2.GaussianBlur(sharp, (15, 15), 5)
    assert lpr.plate_quality(sharp) > lpr.plate_quality(blurred)



@pytest.mark.parametrize("shape", [(20, 60), (20, 60, 1), (20, 60, 3), (20, 60, 4)])
def test_enhance_plate_for_ocr_accepts_any_channel_layout(shape):
    out = lpr.enhance_plate_for_ocr(np.full(shape, 128, dtype=np.uint8))
    assert out.ndim == 3 and out.shape[2] == 3


def test_enhance_plate_for_ocr_upscales_small_plates_and_pads():
    out = lpr.enhance_plate_for_ocr(_checkerboard(20, 60))
    scale = lpr.OCR_TARGET_WIDTH / 60.0
    # Upscaled to the OCR target width, then 8 px of padding on every side.
    assert out.shape[1] == lpr.OCR_TARGET_WIDTH + 16
    assert out.shape[0] == round(20 * scale) + 16


def test_enhance_plate_for_ocr_keeps_wide_plates_at_size():
    width = lpr.OCR_TARGET_WIDTH + 40
    out = lpr.enhance_plate_for_ocr(_checkerboard(30, width))
    assert out.shape[:2] == (30 + 16, width + 16)
