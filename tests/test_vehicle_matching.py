"""Unit tests for find_vehicle_for_plate in lpr.py.

lpr.py matches a plate to a vehicle in order of preference: plate
centre inside the box, inside a padded box, large plate overlap, then a
nearby-vehicle fallback. torch and ultralytics are stubbed out.
"""

import sys
import types
from collections import defaultdict
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
def detections(monkeypatch):
    dets = []
    monkeypatch.setattr(lpr, "vehicle_detections", dets)
    return dets


def test_no_vehicles_returns_unknown(detections):
    assert lpr.find_vehicle_for_plate((10, 10, 20, 20)) == ("UNKNOWN", 0.0)


def test_far_away_vehicle_is_not_matched(detections):
    detections.append((0, 0, 100, 100, "CAR", 0.9))
    assert lpr.find_vehicle_for_plate((500, 500, 520, 510)) == ("UNKNOWN", 0.0)


def test_containing_vehicle_is_matched(detections):
    detections.append((0, 0, 100, 100, "CAR", 0.9))
    assert lpr.find_vehicle_for_plate((40, 70, 60, 80)) == ("CAR", 0.9)


def test_plate_just_outside_box_matches_padded_vehicle(detections):
    # Plate centre (105, 50) is 5px past the right edge, within the 18% pad.
    detections.append((0, 0, 100, 100, "CAR", 0.8))
    assert lpr.find_vehicle_for_plate((95, 45, 115, 55)) == ("CAR", 0.8)


def test_containment_beats_padded_match(detections):
    # The plate centre is inside the truck and only in the car's padding,
    # so the truck wins even though the car is more confident.
    detections.append((0, 0, 100, 100, "CAR", 0.99))
    detections.append((105, 0, 300, 100, "TRUCK", 0.5))
    assert lpr.find_vehicle_for_plate((100, 45, 120, 55)) == ("TRUCK", 0.5)


def test_higher_confidence_wins_between_equal_matches(detections):
    detections.append((0, 0, 100, 100, "CAR", 0.4))
    detections.append((0, 0, 100, 100, "TAXI", 0.7))
    assert lpr.find_vehicle_for_plate((40, 40, 60, 50)) == ("TAXI", 0.7)


def test_returns_plain_types(detections):
    detections.append((0, 0, 100, 100, "CAR", 1))
    vehicle, confidence = lpr.find_vehicle_for_plate((40, 40, 60, 50))
    assert vehicle == "CAR" and isinstance(confidence, float)
