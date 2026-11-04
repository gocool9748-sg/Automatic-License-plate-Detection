"""Unit tests for the plate crop quality helpers in webcam.py.

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

import webcam  # noqa: E402


def _checkerboard(height, width, square=4, channels=3):
    """A high-contrast, high-frequency image (sharp edges everywhere)."""
    ys, xs = np.indices((height, width))
    board = (((ys // square) + (xs // square)) % 2 * 255).astype(np.uint8)
    if channels == 1:
        return board
    return np.repeat(board[:, :, None], channels, axis=2)


@pytest.mark.parametrize("shape", [(20, 60), (20, 60, 1), (20, 60, 3), (20, 60, 4)])
def test_to_gray_returns_single_channel(shape):
    gray = webcam.to_gray(np.zeros(shape, dtype=np.uint8))
    assert gray.shape == (20, 60)


@pytest.mark.parametrize("image", [None, np.zeros((0, 0, 3), dtype=np.uint8)])
def test_sharpness_of_missing_image_is_zero(image):
    assert webcam.sharpness(image) == 0.0


def test_flat_image_has_zero_sharpness():
    flat = np.full((30, 90, 3), 128, dtype=np.uint8)
    assert webcam.sharpness(flat) == 0.0


def test_sharp_image_scores_higher_than_blurred_copy():
    sharp = _checkerboard(40, 120)
    blurred = webcam.cv2.GaussianBlur(sharp, (9, 9), 3)
    assert webcam.sharpness(sharp) > webcam.sharpness(blurred)


def test_sharpness_works_on_grayscale_crop():
    assert webcam.sharpness(_checkerboard(40, 120, channels=1)) > 0.0


def test_plate_quality_rejects_too_narrow_crops():
    narrow = _checkerboard(20, webcam.MIN_PLATE_PIXEL_WIDTH - 1)
    assert webcam.plate_quality(narrow) == 0.0


@pytest.mark.parametrize("image", [None, np.zeros((0, 0, 3), dtype=np.uint8)])
def test_plate_quality_of_missing_image_is_zero(image):
    assert webcam.plate_quality(image) == 0.0


def test_plate_quality_is_capped():
    # sharpness, contrast and width terms are capped at 5 + 2 + 2.
    huge = _checkerboard(100, 600, square=1)
    assert webcam.plate_quality(huge) == pytest.approx(9.0)


def test_plate_quality_prefers_sharper_crop():
    sharp = _checkerboard(40, 120)
    blurred = webcam.cv2.GaussianBlur(sharp, (15, 15), 5)
    assert webcam.plate_quality(sharp) > webcam.plate_quality(blurred)


def test_dark_crop_is_inverted_to_black_text_on_white():
    dark = np.full((10, 10), 30, dtype=np.uint8)
    out = webcam.ensure_black_text_on_white(dark)
    assert np.all(out == 225)


def test_light_crop_is_left_unchanged():
    light = np.full((10, 10), 200, dtype=np.uint8)
    assert webcam.ensure_black_text_on_white(light) is light


def test_ensure_black_text_on_white_passes_none_through():
    assert webcam.ensure_black_text_on_white(None) is None
