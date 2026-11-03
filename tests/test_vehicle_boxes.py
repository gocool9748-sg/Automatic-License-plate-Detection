"""Unit tests for the vehicle box helpers in webcam.py.

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


@pytest.mark.parametrize("original, expected", [
    ("Sedan", "CAR"),
    ("Truck", "TRUCK"),
    ("Mini-bus", "BUS"),
    ("Three-wheeler", "AUTO"),
    ("Two-wheeler", "BIKE"),
    ("Spaceship", "OTHER"),   # unknown classes fall back to OTHER
])
def test_map_vehicle_class(original, expected):
    assert webcam.map_vehicle_class(original) == expected


def test_iou_of_identical_boxes_is_one():
    box = (10, 10, 50, 50)
    assert webcam.vehicle_box_iou(box, box) == pytest.approx(1.0)


def test_iou_of_disjoint_boxes_is_zero():
    assert webcam.vehicle_box_iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_iou_of_half_overlap():
    # Two 10x10 boxes overlapping in a 5x10 strip: 50 / (100 + 100 - 50).
    iou = webcam.vehicle_box_iou((0, 0, 10, 10), (5, 0, 15, 10))
    assert iou == pytest.approx(50 / 150)


def test_iou_ignores_extra_fields_and_degenerate_boxes():
    assert webcam.vehicle_box_iou((0, 0, 10, 10, "CAR", 0.9),
                                  (0, 0, 10, 10, "CAR", 0.8)) == pytest.approx(1.0)
    assert webcam.vehicle_box_iou((5, 5, 5, 5), (5, 5, 5, 5)) == 0.0


def test_containment_of_box_inside_another_is_one():
    outer = (0, 0, 100, 100)
    inner = (10, 10, 30, 30)
    assert webcam.vehicle_containment_ratio(outer, inner) == pytest.approx(1.0)
    assert webcam.vehicle_containment_ratio(inner, outer) == pytest.approx(1.0)


@pytest.mark.parametrize("point, inside", [
    ((5, 5), True),
    ((0, 0), True),      # edges count as inside
    ((10, 10), True),
    ((11, 5), False),
    ((5, -1), False),
])
def test_point_inside_box(point, inside):
    assert webcam.point_inside_box(*point, (0, 0, 10, 10)) is inside


def test_suppress_keeps_highest_confidence_duplicate():
    weak = (0, 0, 100, 100, "CAR", 0.50)
    strong = (2, 2, 102, 102, "CAR", 0.90)
    assert webcam.suppress_duplicate_vehicle_detections([weak, strong]) == [strong]


def test_suppress_treats_contained_box_as_duplicate_across_classes():
    big = (0, 0, 200, 200, "BUS", 0.80)
    inside = (20, 20, 80, 80, "CAR", 0.60)
    assert webcam.suppress_duplicate_vehicle_detections([inside, big]) == [big]


def test_suppress_keeps_separate_vehicles_sorted_left_to_right():
    right = (300, 0, 400, 100, "CAR", 0.95)
    left = (0, 0, 100, 100, "BIKE", 0.40)
    assert webcam.suppress_duplicate_vehicle_detections([right, left]) == [left, right]


def test_suppress_passes_through_zero_or_one_detection():
    assert webcam.suppress_duplicate_vehicle_detections([]) == []
    one = [(0, 0, 10, 10, "CAR", 0.5)]
    assert webcam.suppress_duplicate_vehicle_detections(one) == one
