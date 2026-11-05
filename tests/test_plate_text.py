"""Unit tests for the Indian plate text helpers in lpr.py.

torch and ultralytics are stubbed out so these tests run without a GPU,
model weights or the heavy ML dependencies:

    pip install pytest opencv-python numpy
    python -m pytest tests
"""

import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

for name in ("torch", "ultralytics"):
    if name not in sys.modules:
        stub = types.ModuleType(name)
        if name == "ultralytics":
            stub.YOLO = object
        sys.modules[name] = stub

import lpr  # noqa: E402


@pytest.mark.parametrize("raw, expected", [
    ("tn 09 ab 1234", "TN09AB1234"),
    ("KA-01-MJ-2023", "KA01MJ2023"),
    ("", ""),
    (None, ""),
])
def test_clean_text_keeps_only_letters_and_digits(raw, expected):
    assert lpr.clean_text(raw) == expected


@pytest.mark.parametrize("plate", [
    "TN09AB1234",   # 2-digit district, 2-letter series
    "DL3CAB1234",   # 1-digit district, 3-letter series
    "MH12A1234",    # 1-letter series
])
def test_valid_plates(plate):
    assert lpr.is_valid_plate(plate)


@pytest.mark.parametrize("plate", ["", "TN09AB123", "1N09AB1234", "TN09AB12345"])
def test_invalid_plates(plate):
    assert not lpr.is_valid_plate(plate)


@pytest.mark.parametrize("raw, expected", [
    ("TN09AB1234", "TN09AB1234"),   # already correct
    ("TN09AB12S4", "TN09AB1254"),   # S read in the last four digits
    ("TNO9AB1234", "TN09AB1234"),   # O read in a digit slot
    ("TN098B1234", "TN09BB1234"),   # 8 read in the series
    ("TN09ABI234", "TN09AB1234"),   # I read in the last four digits
])
def test_normalize_plate_fixes_ocr_confusions(raw, expected):
    assert lpr.normalize_plate(raw) == expected


def test_normalize_plate_finds_plate_inside_extra_characters():
    assert lpr.normalize_plate("XTN09AB1234") == "TN09AB1234"


def test_normalize_plate_rejects_garbage():
    assert lpr.normalize_plate("HELLO") == ""


def test_character_vote_outvotes_a_bad_reading():
    history = ["TN09AB1234", "TN09AB1234", "TN09A81234"]
    confidences = [90.0, 85.0, 60.0]
    assert lpr.character_vote(history, confidences) == "TN09AB1234"
