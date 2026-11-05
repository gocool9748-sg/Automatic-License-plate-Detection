"""Unit tests for the per-track vehicle and confidence helpers in lpr.py.

Covers vote_vehicle, get_track_vehicle and decide_vehicle_type. torch and
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

import lpr  # noqa: E402


@pytest.fixture(autouse=True)
def fresh_state(monkeypatch):
    monkeypatch.setattr(
        lpr, "track_vehicle_votes", defaultdict(lambda: defaultdict(float))
    )
    monkeypatch.setattr(lpr, "track_history", defaultdict(list))
    monkeypatch.setattr(lpr, "track_confidence", defaultdict(list))


def test_unseen_track_is_unknown():
    assert lpr.get_track_vehicle(7) == "UNKNOWN"


def test_unknown_votes_are_ignored():
    lpr.vote_vehicle(1, "UNKNOWN", 0.9)
    assert lpr.get_track_vehicle(1) == "UNKNOWN"


def test_votes_accumulate_by_confidence():
    for _ in range(4):
        lpr.vote_vehicle(1, "CAR", 0.6)
    lpr.vote_vehicle(1, "TRUCK", 0.5)   # one weaker bad frame
    assert lpr.get_track_vehicle(1) == "CAR"


def test_recent_votes_dominate():
    # The per-track type follows the box the plate is on now (old votes
    # decay); the final type is decided per vehicle by decide_vehicle_type.
    lpr.vote_vehicle(1, "CAR", 0.6)
    lpr.vote_vehicle(1, "TRUCK", 0.9)
    lpr.vote_vehicle(1, "TRUCK", 0.9)
    assert lpr.get_track_vehicle(1) == "TRUCK"


def test_bus_wins_with_enough_share():
    assert lpr.decide_vehicle_type({"TRUCK": 2.0, "BUS": 1.0}) == "BUS"
    assert lpr.decide_vehicle_type({"TRUCK": 9.0, "BUS": 1.0}) == "TRUCK"
    assert lpr.decide_vehicle_type({}) == "UNKNOWN"


def test_tracks_are_independent():
    lpr.vote_vehicle(1, "CAR", 0.8)
    lpr.vote_vehicle(2, "BIKE", 0.8)
    assert lpr.get_track_vehicle(1) == "CAR"
    assert lpr.get_track_vehicle(2) == "BIKE"

