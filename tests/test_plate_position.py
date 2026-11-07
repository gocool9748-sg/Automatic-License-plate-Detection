"""Unit tests for plate_position_ok in lpr.py.

A plate detection whose centre sits too high on a bus, truck or car (a
windshield banner, destination board or sticker) is rejected; plates with no
surrounding vehicle are kept. torch and ultralytics are stubbed out.
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


BUS = (100, 100, 300, 500, "BUS", 0.9)   # 400 px tall


def plate_at(rel_y, x1=180, x2=220, h=20):
    """Plate box whose centre is rel_y of the way down the BUS box."""
    cy = 100 + rel_y * 400
    return (x1, cy - h / 2, x2, cy + h / 2)


@pytest.fixture
def vehicles(monkeypatch):
    def set_vehicles(*dets):
        monkeypatch.setattr(lpr, "vehicle_detections", list(dets))
    monkeypatch.setattr(lpr, "PLATE_MIN_REL_Y", 0.65)
    return set_vehicles


def test_plate_low_on_the_vehicle_is_kept(vehicles):
    vehicles(BUS)
    assert lpr.plate_position_ok(plate_at(0.9))


def test_banner_high_on_the_vehicle_is_rejected(vehicles):
    vehicles(BUS)
    assert not lpr.plate_position_ok(plate_at(0.5))


def test_threshold_is_inclusive(vehicles):
    vehicles(BUS)
    assert lpr.plate_position_ok(plate_at(0.65))


def test_plate_without_a_vehicle_is_kept(vehicles):
    vehicles()
    assert lpr.plate_position_ok(plate_at(0.1))


def test_plate_outside_every_vehicle_is_kept(vehicles):
    vehicles(BUS)
    assert lpr.plate_position_ok((600, 100, 640, 120))


def test_vehicle_types_without_position_rules_are_ignored(vehicles):
    vehicles((100, 100, 300, 500, "MOTORCYCLE", 0.9))
    assert lpr.plate_position_ok(plate_at(0.2))


def test_smallest_containing_vehicle_decides(vehicles):
    # A car in front of a bus: the plate is low on the car but high on the bus.
    car = (150, 120, 250, 220, "CAR", 0.8)
    vehicles(BUS, car)
    assert lpr.plate_position_ok((190, 200, 210, 210))
