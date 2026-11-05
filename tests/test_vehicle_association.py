"""Unit tests for find_vehicle_for_plate in webcam.py.

webcam.py links a plate to the smallest vehicle box that contains the plate
centre. torch and ultralytics are stubbed out so these run without a GPU.
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

import webcam  # noqa: E402


@pytest.fixture
def detections(monkeypatch):
    dets = []
    monkeypatch.setattr(webcam, "vehicle_detections", dets)
    return dets


def test_no_vehicles_returns_unknown(detections):
    assert webcam.find_vehicle_for_plate((10, 10, 20, 20)) == ("UNKNOWN", 0.0)


def test_plate_outside_every_vehicle_returns_unknown(detections):
    detections.append((0, 0, 100, 100, "CAR", 0.9))
    assert webcam.find_vehicle_for_plate((200, 200, 220, 210)) == ("UNKNOWN", 0.0)


def test_plate_centre_inside_vehicle(detections):
    detections.append((0, 0, 100, 100, "CAR", 0.9))
    assert webcam.find_vehicle_for_plate((40, 70, 60, 80)) == ("CAR", 0.9)


def test_smallest_containing_vehicle_wins(detections):
    # A bus box that also covers a bike in front of it: the plate centre is
    # inside both, but the tighter bike box is the right owner.
    detections.append((0, 0, 400, 300, "BUS", 0.95))
    detections.append((150, 150, 210, 250, "BIKE", 0.6))
    assert webcam.find_vehicle_for_plate((170, 220, 190, 230)) == ("BIKE", 0.6)


def test_only_the_containing_vehicle_is_used(detections):
    detections.append((0, 0, 100, 100, "CAR", 0.5))
    detections.append((300, 0, 400, 100, "TRUCK", 0.99))
    assert webcam.find_vehicle_for_plate((40, 40, 60, 50)) == ("CAR", 0.5)
