"""Unit tests for AwirosOCR.recognize in lpr.py.

recognize skips crops that are empty, narrower than MIN_PLATE_PIXEL_WIDTH or
blurrier than BLUR_VARIANCE_THRESHOLD without asking the worker. Otherwise it
tries the enhanced crop first and the raw crop second, keeps the reading
with the highest confidence, stops early once a reading reaches 70, and
resets last_junk at the start of every call. _query is replaced by a fake,
so no worker process or socket is used. torch and ultralytics are stubbed
out.
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


class FakeClient:
    """Wraps a bare AwirosOCR whose _query returns scripted answers."""

    def __init__(self, answers):
        self.ocr = object.__new__(lpr.AwirosOCR)   # skip __init__ (spawns worker)
        self.ocr.last_junk = True
        self.answers = list(answers)
        self.queried = []

        def fake_query(image):
            self.queried.append(image)
            return self.answers.pop(0)

        self.ocr._query = fake_query


def sharp_plate(w=80, h=24):
    """A striped crop with plenty of edges, so it passes the blur check."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, ::4] = 255
    return img


@pytest.mark.parametrize("plate", [
    None,
    np.zeros((0, 0, 3), dtype=np.uint8),
])
def test_missing_crop_is_not_sent(plate):
    fake = FakeClient([])
    assert fake.ocr.recognize(plate) == ("", 0.0)
    assert fake.queried == []


def test_too_narrow_crop_is_not_sent():
    fake = FakeClient([])
    narrow = sharp_plate(w=lpr.MIN_PLATE_PIXEL_WIDTH - 1)
    assert fake.ocr.recognize(narrow) == ("", 0.0)
    assert fake.queried == []


def test_blurry_crop_is_not_sent():
    fake = FakeClient([])
    flat = np.full((24, 80, 3), 128, dtype=np.uint8)   # Laplacian variance 0
    assert fake.ocr.recognize(flat) == ("", 0.0)
    assert fake.queried == []


def test_confident_enhanced_read_skips_raw_pass():
    fake = FakeClient([("TN09AB1234", 85.0)])
    plate = sharp_plate()

    assert fake.ocr.recognize(plate) == ("TN09AB1234", 85.0)
    assert len(fake.queried) == 1
    assert fake.queried[0] is not plate   # the enhanced variant went first


def test_low_confidence_tries_raw_crop_and_keeps_better_read():
    fake = FakeClient([("TN09AB1234", 40.0), ("TN09AB1284", 60.0)])
    plate = sharp_plate()

    assert fake.ocr.recognize(plate) == ("TN09AB1284", 60.0)
    assert len(fake.queried) == 2
    assert fake.queried[1] is plate


def test_keeps_enhanced_read_when_raw_is_worse():
    fake = FakeClient([("TN09AB1234", 55.0), ("TN09AB12", 30.0)])
    assert fake.ocr.recognize(sharp_plate()) == ("TN09AB1234", 55.0)


def test_empty_first_read_falls_back_to_raw():
    fake = FakeClient([("", 0.0), ("KA01CD5678", 45.0)])
    assert fake.ocr.recognize(sharp_plate()) == ("KA01CD5678", 45.0)


def test_no_text_from_either_pass():
    fake = FakeClient([("", 0.0), ("", 0.0)])
    assert fake.ocr.recognize(sharp_plate()) == ("", 0.0)


def test_last_junk_is_reset_each_call():
    fake = FakeClient([("TN09AB1234", 90.0)])
    fake.ocr.recognize(sharp_plate())
    assert fake.ocr.last_junk is False
