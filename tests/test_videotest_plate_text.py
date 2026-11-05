"""Unit tests for the plate text and voting helpers in videotest.py.

videotest.py keeps its own copy of the text helpers (separate from
webcam.py), so they are tested here too. torch and ultralytics are stubbed
out so these run without a GPU or the heavy ML dependencies.
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

import videotest  # noqa: E402


@pytest.mark.parametrize("raw, expected", [
    ("TN 09 AB 1234", "TN09AB1234"),   # spaces removed
    ("tn09ab1234", "TN09AB1234"),      # lower case accepted
    ("TNO9AB1234", "TN09AB1234"),      # O misread for 0 in the district
    ("TN09A81234", "TN09AB1234"),      # 8 misread for B in the series
    ("XTN09AB1234Y", "TN09AB1234"),    # extra OCR characters around the plate
    ("", ""),
    (None, ""),
    ("HELLO", ""),
])
def test_normalize_plate(raw, expected):
    assert videotest.normalize_plate(raw) == expected


@pytest.mark.parametrize("raw, expected", [
    ("RJ14CV0002", "RJ14CV0002"),
    ("RJ14CVOOO2", "RJ14CV0002"),   # O -> 0 in the last four
    ("8R14CV0002", "BR14CV0002"),   # 8 -> B in the state code
    ("RJ1SCV0002", "RJ15CV0002"),   # S -> 5 in the district
    ("RJ14CV002", ""),              # not 10 characters
    ("", ""),
])
def test_correct_10_char_plate(raw, expected):
    assert videotest.correct_10_char_plate(raw) == expected


@pytest.mark.parametrize("length, pattern", [
    (10, "LLDDLLDDDD"),
    (9, "LLDLLDDDD"),
    (8, "LLDLDDDD"),
    (7, ""),
])
def test_plate_pattern_for_length(length, pattern):
    assert videotest.plate_pattern_for_length(length) == pattern


def test_character_vote_prefers_majority_per_position():
    history = ["TN09AB1234", "TN09AB1234", "TN09AB1284"]
    confidences = [90, 90, 90]
    assert videotest.character_vote(history, confidences) == "TN09AB1234"


def test_character_vote_weights_by_confidence():
    history = ["TN09AB1234", "TN09AB1284"]
    assert videotest.character_vote(history, [40, 95]) == "TN09AB1284"


def test_character_vote_uses_most_common_length():
    history = ["TN09AB1234", "TN09AB1234", "DL3CA1234"]
    assert videotest.character_vote(history, [80, 80, 99]) == "TN09AB1234"


@pytest.mark.parametrize("history, confidences", [
    ([], []),
    (["ABC", "1234567"], [90, 90]),   # no usable plate lengths
])
def test_character_vote_returns_empty_without_usable_readings(history, confidences):
    assert videotest.character_vote(history, confidences) == ""
