"""Unit tests for the per-vehicle cluster helpers in lpr.py.

clusters_concurrent keeps two vehicles apart when their tracks were seen in
the same frame, best_evidence_for picks the sharpest proof images across a
vehicle's tracks and cluster_consensus votes the final plate.
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


def cluster(tracks=(), readings=(), weights=None):
    return {
        "tracks": set(tracks),
        "readings": list(readings),
        "weights": dict(weights or {}),
    }


@pytest.fixture
def frames(monkeypatch):
    table = {}
    monkeypatch.setattr(lpr, "track_frames", table)
    return table


def test_clusters_seen_in_the_same_frame_are_concurrent(frames):
    frames.update({1: {10, 11}, 2: {11, 12}})
    assert lpr.clusters_concurrent(cluster([1]), cluster([2]))


def test_clusters_seen_in_different_frames_are_not_concurrent(frames):
    frames.update({1: {10, 11}, 2: {20, 21}})
    assert not lpr.clusters_concurrent(cluster([1]), cluster([2]))


def test_any_overlapping_track_pair_makes_clusters_concurrent(frames):
    frames.update({1: {1}, 2: {2}, 3: {5}, 4: {2}})
    assert lpr.clusters_concurrent(cluster([1, 2]), cluster([3, 4]))


def test_tracks_without_frame_history_are_not_concurrent(frames):
    frames.update({1: {10}})
    assert not lpr.clusters_concurrent(cluster([1]), cluster([2]))
    assert not lpr.clusters_concurrent(cluster([]), cluster([1]))


def test_best_evidence_picks_the_highest_quality_track(monkeypatch):
    monkeypatch.setattr(lpr, "best_evidence", {
        1: (0.4, "crop1", "snap1"),
        2: (0.9, "crop2", "snap2"),
        3: (0.7, "crop3", "snap3"),
    })
    assert lpr.best_evidence_for(cluster([1, 2])) == ("crop2", "snap2")


def test_best_evidence_without_any_crops(monkeypatch):
    monkeypatch.setattr(lpr, "best_evidence", {})
    assert lpr.best_evidence_for(cluster([1, 2])) == (None, None)


def test_consensus_votes_characters_across_readings(monkeypatch):
    monkeypatch.setattr(lpr, "known_plates", set())
    c = cluster(readings=[
        ("TN09AB1234", 90, 100),
        ("TN09AB1234", 85, 100),
        ("TN09AB1284", 40, 60),   # one weak misread
    ])
    assert lpr.cluster_consensus(c) == "TN09AB1234"


def test_consensus_without_readings_is_empty(monkeypatch):
    monkeypatch.setattr(lpr, "known_plates", set())
    assert lpr.cluster_consensus(cluster()) == ""


def test_consensus_falls_back_to_the_heaviest_exact_reading(monkeypatch):
    monkeypatch.setattr(lpr, "known_plates", set())
    # BH plates have no state/RTO structure, so the character vote is empty.
    c = cluster(
        readings=[("22BH1234AB", 90, 100), ("22BH1234AA", 50, 100)],
        weights={"22BH1234AB": 2.0, "22BH1234AA": 0.5},
    )
    assert lpr.cluster_consensus(c) == "22BH1234AB"
