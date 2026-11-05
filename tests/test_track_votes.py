"""Unit tests for the per-track vehicle and confidence helpers in webcam.py.

Covers vote_vehicle, get_track_vehicle and get_track_confidence. torch and
ultralytics are stubbed out so these run without a GPU.
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


@pytest.fixture(autouse=True)
def fresh_state(monkeypatch):
    monkeypatch.setattr(
        webcam, "track_vehicle_votes", defaultdict(lambda: defaultdict(float))
    )
    monkeypatch.setattr(webcam, "track_history", defaultdict(list))
    monkeypatch.setattr(webcam, "track_confidence", defaultdict(list))


def test_unseen_track_is_unknown():
    assert webcam.get_track_vehicle(7) == "UNKNOWN"


def test_unknown_votes_are_ignored():
    webcam.vote_vehicle(1, "UNKNOWN", 0.9)
    assert webcam.get_track_vehicle(1) == "UNKNOWN"


def test_votes_accumulate_by_confidence():
    webcam.vote_vehicle(1, "CAR", 0.6)
    webcam.vote_vehicle(1, "CAR", 0.6)
    webcam.vote_vehicle(1, "TRUCK", 0.9)   # one confident bad frame
    assert webcam.get_track_vehicle(1) == "CAR"


def test_tracks_are_independent():
    webcam.vote_vehicle(1, "CAR", 0.8)
    webcam.vote_vehicle(2, "BIKE", 0.8)
    assert webcam.get_track_vehicle(1) == "CAR"
    assert webcam.get_track_vehicle(2) == "BIKE"


def test_confidence_averages_readings_matching_the_plate():
    webcam.track_history[1] = ["TN09AB1234", "TN09AB1284", "TN09AB1234"]
    webcam.track_confidence[1] = [80.0, 40.0, 90.0]
    assert webcam.get_track_confidence(1, "TN09AB1234") == pytest.approx(85.0)


def test_confidence_falls_back_to_all_readings():
    webcam.track_history[1] = ["TN09AB1234", "TN09AB1284"]
    webcam.track_confidence[1] = [80.0, 60.0]
    assert webcam.get_track_confidence(1, "KA01XY0001") == pytest.approx(70.0)


def test_confidence_without_readings_is_zero():
    assert webcam.get_track_confidence(99, "TN09AB1234") == 0.0
