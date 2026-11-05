"""Unit tests for the position-aware plate correction helpers in lpr.py.

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

import lpr  # noqa: E402


@pytest.mark.parametrize("length, pattern", [
    (10, "LLDDLLDDDD"),
    (9, "LLDLLDDDD"),
    (8, "LLDLDDDD"),
    (7, ""),
    (11, ""),
])
def test_plate_pattern_for_length(length, pattern):
    assert lpr.plate_pattern_for_length(length) == pattern


@pytest.mark.parametrize("ch, expected_type, allowed", [
    ("A", "L", True),
    ("7", "L", False),
    ("7", "D", True),
    ("A", "D", False),
    ("A", "", True),     # no expectation -> anything goes
])
def test_character_is_allowed(ch, expected_type, allowed):
    assert lpr.character_is_allowed(ch, expected_type) is allowed


@pytest.mark.parametrize("ch, expected_type, corrected", [
    ("0", "L", "O"),
    ("8", "L", "B"),
    ("o", "D", "0"),     # lower-case input is upper-cased first
    ("S", "D", "5"),
    ("7", "L", "T"),     # 7 in a letter slot is read as T
    ("9", "L", "9"),     # no known confusion -> left alone
    ("A", "L", "A"),
    ("", "D", ""),
])
def test_position_correct(ch, expected_type, corrected):
    assert lpr.position_correct(ch, expected_type) == corrected

