"""Unit tests for fleet-plate snapping and cluster confidence in lpr.py.

snap_to_known corrects a voted plate that is 1-2 characters off to the one
known fleet plate that best explains all readings of the vehicle.
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


def cluster(*readings):
    """Cluster with (text, confidence, box_width) readings."""
    return {"readings": list(readings)}


@pytest.mark.parametrize("a, b, expected", [
    ("TN09AB1234", "TN09AB1234", 0),
    ("TN09AB1234", "TN09AB1235", 1),
    ("TN09AB1234", "TN09A81204", 2),
    ("TN09AB1234", "TN09AB123", 99),   # different lengths
])
def test_hamming(a, b, expected):
    assert lpr.hamming(a, b) == expected


def test_no_known_plates_leaves_plate_alone(monkeypatch):
    monkeypatch.setattr(lpr, "known_plates", set())
    c = cluster(("TN48BK2145", 90.0, 100))
    assert lpr.snap_to_known("TN48BK2145", c) == "TN48BK2145"


def test_plate_that_is_already_known_is_kept(monkeypatch):
    monkeypatch.setattr(lpr, "known_plates", {"TN48BK2144", "TN48BK2145"})
    c = cluster(("TN48BK2145", 90.0, 100))
    assert lpr.snap_to_known("TN48BK2145", c) == "TN48BK2145"


def test_one_character_off_snaps_to_known_plate(monkeypatch):
    monkeypatch.setattr(lpr, "known_plates", {"TN48BK2144"})
    c = cluster(("TN48BK2145", 90.0, 100), ("TN48BK2145", 80.0, 100))
    assert lpr.snap_to_known("TN48BK2145", c) == "TN48BK2144"


def test_three_characters_off_is_not_snapped(monkeypatch):
    monkeypatch.setattr(lpr, "known_plates", {"TN48BK2144"})
    c = cluster(("TN48BK2777", 90.0, 100))
    assert lpr.snap_to_known("TN48BK2777", c) == "TN48BK2777"


def test_readings_too_far_on_average_are_not_snapped(monkeypatch):
    monkeypatch.setattr(lpr, "known_plates", {"TN48BK2144"})
    # Two characters off in every reading: average difference 2 > 1.5.
    c = cluster(("TN48BK2155", 90.0, 100), ("TN48BK2155", 90.0, 100))
    assert lpr.snap_to_known("TN48BK2155", c) == "TN48BK2155"


def test_ambiguous_between_two_known_plates_is_not_snapped(monkeypatch):
    monkeypatch.setattr(lpr, "known_plates", {"TN48BK2144", "TN48BK2146"})
    c = cluster(("TN48BK2145", 90.0, 100))
    assert lpr.snap_to_known("TN48BK2145", c) == "TN48BK2145"


def test_readings_break_a_tie_between_known_plates(monkeypatch):
    monkeypatch.setattr(lpr, "known_plates", {"TN48BK2144", "TN48BK2146"})
    # The voted plate is one off from both, but the individual readings
    # mostly say ...2144.
    c = cluster(
        ("TN48BK2144", 90.0, 100),
        ("TN48BK2144", 85.0, 100),
        ("TN48BK2145", 60.0, 100),
    )
    assert lpr.snap_to_known("TN48BK2145", c) == "TN48BK2144"


def test_cluster_confidence_averages_agreeing_readings():
    c = cluster(
        ("TN09AB1234", 90.0, 100),
        ("TN09AB1235", 80.0, 100),   # one character off still agrees
        ("KA01XY9999", 10.0, 100),   # a different plate is ignored
    )
    assert lpr.cluster_confidence(c, "TN09AB1234") == pytest.approx(85.0)


def test_cluster_confidence_falls_back_to_all_readings():
    c = cluster(("KA01XY9999", 40.0, 100), ("KA01XY9998", 60.0, 100))
    assert lpr.cluster_confidence(c, "TN09AB1234") == pytest.approx(50.0)


def test_cluster_confidence_of_empty_cluster_is_zero():
    assert lpr.cluster_confidence(cluster(), "TN09AB1234") == 0.0
