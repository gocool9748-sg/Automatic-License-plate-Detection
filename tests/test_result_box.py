"""Unit tests for the console result box printed by lpr.py and lpr.py.

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


@pytest.fixture
def module():
    return lpr


def test_box_contains_all_fields(module):
    box = module.format_result_box("TN09AB1234", "CAR", 87.4, "12:00:00")
    lines = box.splitlines()

    assert len(lines) == 6
    assert "Plate: TN09AB1234" in lines[1]
    assert "Vehicle: CAR" in lines[2]
    assert "Confidence: 87%" in lines[3]
    assert "Time: 12:00:00" in lines[4]


def test_box_lines_have_equal_width(module):
    box = module.format_result_box("TN09AB1234", "CAR", 87.4, "12:00:00")
    widths = {len(line) for line in box.splitlines()}
    assert len(widths) == 1


def test_box_has_minimum_width(module):
    box = module.format_result_box("A", "B", 1, "t")
    top = box.splitlines()[0]
    # Two corner characters plus the 26-character minimum interior.
    assert len(top) == 28
    assert top.startswith("┌") and top.endswith("┐")


def test_box_grows_for_long_values(module):
    timestamp = "2026-11-04 22:15:00.123456 IST"
    box = module.format_result_box("TN09AB1234", "MOTORCYCLE", 99.6, timestamp)
    lines = box.splitlines()

    assert len({len(line) for line in lines}) == 1
    assert "Confidence: 100%" in lines[3]
    assert lines[4].startswith("│ Time: " + timestamp)
    assert lines[4].endswith("│")
