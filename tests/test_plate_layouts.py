"""Unit tests for the plate layout fixer behind normalize_plate in lpr.py.

_best_layout_fix tries every Indian plate layout of the reading's length and
coerces each character to a letter or digit, counting look-alike fixes.
torch and ultralytics are stubbed out.
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


@pytest.mark.parametrize("raw", ["UNREADABLE", "COLLEGE BUS", "tn 09 unread"])
def test_is_junk_text_flags_banner_and_unreadable_text(raw):
    assert lpr.is_junk_text(raw)


@pytest.mark.parametrize("raw", ["TN09AB1234", "", None])
def test_is_junk_text_passes_plates_and_empty_text(raw):
    assert not lpr.is_junk_text(raw)


@pytest.mark.parametrize("ch, kind, expected", [
    ("A", "L", ("A", 0)),     # already a letter
    ("8", "L", ("B", 1)),     # digit that looks like a letter
    ("3", "L", (None, 99)),   # no letter look-alike
    ("5", "D", ("5", 0)),     # already a digit
    ("O", "D", ("0", 1)),     # letter that looks like a digit
    ("X", "D", (None, 99)),   # no digit look-alike
])
def test_coerce_char(ch, kind, expected):
    assert lpr._coerce_char(ch, kind) == expected


@pytest.mark.parametrize("length, expected", [
    (7, []),
    (8, ["LLDLDDDD"]),
    (9, ["LLDDLDDDD", "LLDLLDDDD", "DDLLDDDDL"]),
    (10, ["LLDDLLDDDD", "LLDLLLDDDD", "DDLLDDDDLL"]),
    (11, ["LLDDLLLDDDD"]),
])
def test_plate_layouts_lists_common_layouts_first(length, expected):
    assert lpr._plate_layouts(length) == expected


def test_best_layout_fix_keeps_a_correct_plate_at_zero_cost():
    assert lpr._best_layout_fix("TN09AB1234") == ("TN09AB1234", 0.0)


def test_best_layout_fix_counts_each_look_alike_fix():
    assert lpr._best_layout_fix("TN09AB12S4") == ("TN09AB1254", 1.0)


def test_best_layout_fix_snaps_a_near_miss_state_to_home_state(monkeypatch):
    monkeypatch.setattr(lpr, "HOME_STATE", "TN")
    assert lpr._best_layout_fix("TM09AB1234") == ("TN09AB1234", 1.0)


def test_best_layout_fix_reads_bh_series_plates():
    plate, _cost = lpr._best_layout_fix("22BH1234AB")
    assert plate == "22BH1234AB"


@pytest.mark.parametrize("raw", ["TN0XAB1234", "QQ09AB1234", "TN09AB"])
def test_best_layout_fix_gives_up_on_impossible_readings(raw):
    assert lpr._best_layout_fix(raw) == ("", 99.0)
