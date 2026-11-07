"""Unit tests for the plate structure patterns used by character voting.

plate_pattern_for_texts picks the most common letter/digit layout among a
track's readings; plate_pattern_for_length is the fallback for a single
reading. torch and ultralytics are stubbed out.
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


@pytest.mark.parametrize("texts, expected", [
    (["TN48BK2170"], "LLDDLLDDDD"),     # 2-digit RTO, 2-letter series
    (["TN01A1234"], "LLDDLDDDD"),       # 1-letter series
    (["DL3CAB1234"], "LLDLLLDDDD"),     # 1-digit RTO, 3-letter series
    (["TN48BK2170", "TN01A1234", "TN48BK2171"], "LLDDLLDDDD"),   # majority
])
def test_pattern_follows_the_most_common_structure(texts, expected):
    assert lpr.plate_pattern_for_texts(texts) == expected


def test_readings_that_are_not_plates_are_ignored():
    assert lpr.plate_pattern_for_texts(["UNREADABLE", "TN01A1234", "1234"]) == "LLDDLDDDD"


@pytest.mark.parametrize("texts", [[], ["UNREADABLE"], ["22BH1234AB"]])
def test_no_plate_structure_gives_empty_pattern(texts):
    # BH-series plates do not follow the state/RTO layout.
    assert lpr.plate_pattern_for_texts(texts) == ""


@pytest.mark.parametrize("length, expected", [
    (10, "LLDDLLDDDD"),
    (9, "LLDLLDDDD"),
    (8, "LLDLDDDD"),
    (7, ""),
    (11, ""),
])
def test_pattern_for_length(length, expected):
    assert lpr.plate_pattern_for_length(length) == expected


@pytest.mark.parametrize("length", [8, 9, 10])
def test_length_patterns_match_their_length(length):
    assert len(lpr.plate_pattern_for_length(length)) == length
