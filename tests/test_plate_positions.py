"""Unit tests for the position-aware plate correction helpers in webcam.py.

torch and ultralytics are stubbed out so these run without a GPU or the
heavy ML dependencies (see tests/test_plate_text.py).
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

import webcam  # noqa: E402


@pytest.mark.parametrize("length, pattern", [
    (10, "LLDDLLDDDD"),
    (9, "LLDLLDDDD"),
    (8, "LLDLDDDD"),
    (7, ""),
    (11, ""),
])
def test_plate_pattern_for_length(length, pattern):
    assert webcam.plate_pattern_for_length(length) == pattern


@pytest.mark.parametrize("ch, expected_type, allowed", [
    ("A", "L", True),
    ("7", "L", False),
    ("7", "D", True),
    ("A", "D", False),
    ("A", "", True),     # no expectation -> anything goes
])
def test_character_is_allowed(ch, expected_type, allowed):
    assert webcam.character_is_allowed(ch, expected_type) is allowed


@pytest.mark.parametrize("ch, expected_type, corrected", [
    ("0", "L", "O"),
    ("8", "L", "B"),
    ("o", "D", "0"),     # lower-case input is upper-cased first
    ("S", "D", "5"),
    ("7", "L", "7"),     # no known confusion -> left alone
    ("A", "L", "A"),
    ("", "D", ""),
])
def test_position_correct(ch, expected_type, corrected):
    assert webcam.position_correct(ch, expected_type) == corrected


@pytest.mark.parametrize("raw, expected", [
    ("RJ14CV0002", "RJ14CV0002"),   # already valid
    ("RJI4CV0OO2", "RJ14CV0002"),   # letters read in digit slots
    ("8J14CV0002", "BJ14CV0002"),   # digit read in a letter slot
    ("rj 14 cv 0002", "RJ14CV0002"),  # spacing and case are cleaned
])
def test_correct_10_char_plate_fixes_confusions(raw, expected):
    assert webcam.correct_10_char_plate(raw) == expected


@pytest.mark.parametrize("raw", [
    "RJ14C0002",    # 9 characters: not this format
    "RJ14C70002",   # 7 in a letter slot has no letter substitute
    "",
])
def test_correct_10_char_plate_rejects(raw):
    assert webcam.correct_10_char_plate(raw) == ""
