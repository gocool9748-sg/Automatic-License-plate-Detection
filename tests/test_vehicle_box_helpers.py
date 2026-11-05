"""Unit tests for the vehicle box helpers in lpr.py.

lpr.py uses box_iou and a threshold-configurable
suppress_duplicate_vehicle_detections, separate from lpr.py. torch and
ultralytics are stubbed out so these run without a GPU.
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


@pytest.mark.parametrize("point, inside", [
    ((50, 50), True),
    ((10, 10), True),     # corners count as inside
    ((100, 80), True),
    ((9, 50), False),
    ((50, 81), False),
])
def test_point_inside_box(point, inside):
    assert lpr.point_inside_box(*point, (10, 10, 100, 80)) is inside


def test_box_iou_identical_and_disjoint():
    assert lpr.box_iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)
    assert lpr.box_iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_box_iou_partial_overlap():
    assert lpr.box_iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(50 / 150)


def test_box_iou_of_zero_area_boxes_is_zero():
    assert lpr.box_iou((5, 5, 5, 5), (5, 5, 5, 5)) == 0.0


def test_suppress_empty_input():
    assert lpr.suppress_duplicate_vehicle_detections([]) == []
    assert lpr.suppress_duplicate_vehicle_detections(None) == []


def test_suppress_keeps_highest_confidence_of_overlapping_pair():
    weak = (0, 0, 100, 100, "CAR", 0.5)
    strong = (2, 2, 102, 102, "TRUCK", 0.9)
    kept = lpr.suppress_duplicate_vehicle_detections([weak, strong])
    assert kept == [strong]


def test_suppress_removes_box_contained_in_another():
    outer = (0, 0, 200, 200, "BUS", 0.8)
    inner = (10, 10, 60, 60, "CAR", 0.7)   # low IoU, but fully contained
    kept = lpr.suppress_duplicate_vehicle_detections([outer, inner])
    assert kept == [outer]


def test_suppress_keeps_separate_vehicles_sorted_by_confidence():
    a = (0, 0, 50, 50, "CAR", 0.6)
    b = (100, 0, 150, 50, "BIKE", 0.9)
    kept = lpr.suppress_duplicate_vehicle_detections([a, b])
    assert kept == [b, a]


def test_suppress_respects_custom_thresholds():
    a = (0, 0, 100, 100, "CAR", 0.9)
    b = (50, 0, 150, 100, "CAR", 0.8)   # IoU 1/3, half of each box overlaps
    assert len(lpr.suppress_duplicate_vehicle_detections([a, b])) == 2
    strict = lpr.suppress_duplicate_vehicle_detections(
        [a, b], iou_threshold=0.3)
    assert strict == [a]
