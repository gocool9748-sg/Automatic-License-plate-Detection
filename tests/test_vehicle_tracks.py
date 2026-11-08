"""Unit tests for persistent vehicle box tracking (update_vehicle_tracks) in lpr.py.

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


@pytest.fixture(autouse=True)
def fresh_tracks(monkeypatch):
    monkeypatch.setattr(lpr, "vehicle_tracks", [])


def test_new_detection_starts_a_track():
    out = lpr.update_vehicle_tracks([(0, 0, 100, 100, "CAR", 0.8)])
    assert out == [(0, 0, 100, 100, "CAR", 0.8)]
    assert len(lpr.vehicle_tracks) == 1
    assert lpr.vehicle_tracks[0]["age"] == 0


def test_overlapping_detection_updates_the_same_track():
    lpr.update_vehicle_tracks([(0, 0, 100, 100, "CAR", 0.8)])
    out = lpr.update_vehicle_tracks([(5, 5, 105, 105, "CAR", 0.9)])
    assert out == [(5, 5, 105, 105, "CAR", 0.9)]
    assert len(lpr.vehicle_tracks) == 1


def test_grown_box_still_matches_by_containment():
    # A vehicle coming close to the camera: IoU is low, but the old box
    # sits inside the new one, so it is still the same track.
    lpr.update_vehicle_tracks([(40, 40, 60, 60, "BUS", 0.7)])
    lpr.update_vehicle_tracks([(0, 0, 200, 200, "BUS", 0.7)])
    assert len(lpr.vehicle_tracks) == 1
    assert lpr.vehicle_tracks[0]["box"] == (0, 0, 200, 200)


def test_type_is_decided_by_accumulated_votes():
    lpr.update_vehicle_tracks([(0, 0, 100, 100, "BUS", 0.9)])
    lpr.update_vehicle_tracks([(0, 0, 100, 100, "BUS", 0.9)])
    # One stronger TRUCK reading does not outvote two BUS readings.
    out = lpr.update_vehicle_tracks([(0, 0, 100, 100, "TRUCK", 0.95)])
    assert out[0][4] == "BUS"


def test_old_votes_decay_each_match():
    lpr.update_vehicle_tracks([(0, 0, 100, 100, "CAR", 1.0)])
    lpr.update_vehicle_tracks([(0, 0, 100, 100, "CAR", 1.0)])
    votes = lpr.vehicle_tracks[0]["votes"]
    assert votes["CAR"] == pytest.approx(1.0 * lpr.VEHICLE_VOTE_DECAY + 1.0)


def test_lost_track_is_held_then_dropped():
    lpr.update_vehicle_tracks([(0, 0, 100, 100, "CAR", 0.8)])
    for _ in range(lpr.VEHICLE_HOLD_CYCLES):
        out = lpr.update_vehicle_tracks([])
        assert len(out) == 1
    assert lpr.update_vehicle_tracks([]) == []


def test_each_detection_matches_at_most_one_track():
    lpr.update_vehicle_tracks([(0, 0, 100, 100, "CAR", 0.8)])
    out = lpr.update_vehicle_tracks([
        (0, 0, 100, 100, "CAR", 0.8),
        (300, 300, 400, 400, "BIKE", 0.6),
    ])
    assert sorted(o[4] for o in out) == ["BIKE", "CAR"]
    assert len(lpr.vehicle_tracks) == 2
