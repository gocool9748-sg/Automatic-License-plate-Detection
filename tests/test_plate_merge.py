"""Unit tests for merge_plate_dets in lpr.py.

The tracker and the fallback plate search can both find the same physical
plate; merge_plate_dets keeps the first box and drops overlapping later ones.
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


def det(x1, y1, x2, y2, track_id=1, suspect=False):
    return (x1, y1, x2, y2, track_id, suspect)


def test_empty_input():
    assert lpr.merge_plate_dets([]) == []


def test_separate_plates_are_all_kept():
    dets = [det(0, 0, 100, 30, 1), det(300, 0, 400, 30, 2)]
    assert lpr.merge_plate_dets(dets) == dets


def test_same_plate_found_twice_keeps_first_box():
    first = det(0, 0, 100, 30, 1)
    second = det(5, 2, 105, 32, 2, True)
    assert lpr.merge_plate_dets([first, second]) == [first]


def test_small_box_inside_a_large_one_is_a_duplicate():
    # IoU is low, but the small box lies entirely inside the first one.
    big = det(0, 0, 200, 60, 1)
    small = det(10, 10, 60, 30, 2)
    assert lpr.box_iou(big[:4], small[:4]) < 0.30
    assert lpr.merge_plate_dets([big, small]) == [big]


def test_slight_overlap_keeps_both():
    a = det(0, 0, 100, 30, 1)
    b = det(80, 0, 180, 30, 2)   # 20% of either box overlaps
    assert lpr.merge_plate_dets([a, b]) == [a, b]


def test_order_decides_which_duplicate_survives():
    a = det(0, 0, 100, 30, 1)
    b = det(2, 1, 102, 31, 2)
    assert lpr.merge_plate_dets([b, a]) == [b]


def test_zero_size_box_does_not_crash():
    a = det(0, 0, 100, 30, 1)
    empty = det(50, 10, 50, 10, 2)
    # A degenerate box has no overlap area, so it is simply kept.
    assert lpr.merge_plate_dets([a, empty]) == [a, empty]
