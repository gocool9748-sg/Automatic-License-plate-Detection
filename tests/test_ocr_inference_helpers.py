"""Unit tests for the image helpers in test.py (the OCR inference script).

test.py only needs OpenCV and NumPy at import time; Paddle is imported
inside main(), so these tests run without it. The script is loaded by
file path because "test" would otherwise clash with Python's own
standard-library test package.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

_spec = importlib.util.spec_from_file_location("ocr_infer", REPO_ROOT / "test.py")
ocr_infer = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ocr_infer)

SHAPE = ocr_infer.IMAGE_SHAPE  # [3, 48, 320]


def test_resize_pads_short_crop_on_the_right():
    img = np.full((24, 40, 3), 255, dtype=np.uint8)   # scales to 48x80
    out = ocr_infer.resize_for_rec(img, SHAPE)
    assert out.shape == (48, 320, 3)
    assert out[:, :80].min() == 255
    assert out[:, 80:].max() == 0


def test_resize_caps_wide_crop_at_target_width():
    img = np.zeros((48, 1000, 3), dtype=np.uint8)
    assert ocr_infer.resize_for_rec(img, SHAPE).shape == (48, 320, 3)


def test_resize_keeps_tall_narrow_crop_at_least_one_pixel_wide():
    img = np.zeros((500, 3, 3), dtype=np.uint8)
    assert ocr_infer.resize_for_rec(img, SHAPE).shape == (48, 320, 3)


@pytest.mark.parametrize("img", [None, np.zeros((0, 0, 3), dtype=np.uint8)])
def test_resize_rejects_empty_images(img):
    with pytest.raises(ValueError):
        ocr_infer.resize_for_rec(img, SHAPE)


def test_preprocess_returns_chw_normalised_to_minus_one_one():
    img = np.full((48, 320, 3), 255, dtype=np.uint8)
    out = ocr_infer.preprocess(img, SHAPE)
    assert out.shape == (3, 48, 320)
    assert out.dtype == np.float32
    assert np.allclose(out, 1.0)

    black = ocr_infer.preprocess(np.zeros((48, 320, 3), dtype=np.uint8), SHAPE)
    assert np.allclose(black, -1.0)


def test_collect_images_filters_and_sorts(tmp_path):
    for name in ("b.PNG", "a.jpg", "notes.txt", "c.webp"):
        (tmp_path / name).write_bytes(b"")
    (tmp_path / "sub.jpg").mkdir()   # directories are ignored

    found = [p.name for p in ocr_infer.collect_images(str(tmp_path))]
    assert found == ["a.jpg", "b.PNG", "c.webp"]


def test_collect_images_accepts_a_single_file(tmp_path):
    img = tmp_path / "plate.jpg"
    img.write_bytes(b"")
    assert ocr_infer.collect_images(str(img)) == [img]


def test_collect_images_missing_path(tmp_path):
    with pytest.raises(FileNotFoundError):
        ocr_infer.collect_images(str(tmp_path / "missing"))


def test_resolve_path_prefers_existing_user_path(tmp_path):
    weights = tmp_path / "custom.safetensors"
    weights.write_bytes(b"")
    assert ocr_infer.resolve_path(str(weights), "model.safetensors") == str(weights)


def test_resolve_path_falls_back_to_repo_files():
    # en_dict.txt ships next to the script.
    found = ocr_infer.resolve_path("", "en_dict.txt")
    assert Path(found) == REPO_ROOT / "en_dict.txt"


def test_resolve_path_reports_missing_file():
    with pytest.raises(FileNotFoundError):
        ocr_infer.resolve_path("", "definitely_not_here.bin")
