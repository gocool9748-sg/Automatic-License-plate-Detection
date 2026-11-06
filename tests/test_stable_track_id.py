"""Unit tests for stable_track_id in lpr.py.

The tracker sometimes hands a plate id over to a different vehicle; a box
that jumps far away or changes size a lot gets a fresh pipeline id so the
two vehicles' readings are not mixed. torch and ultralytics are stubbed out.
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
def fresh_state(monkeypatch):
    monkeypatch.setattr(lpr, "track_last_box", {})
    monkeypatch.setattr(lpr, "track_alias", {})
    monkeypatch.setattr(lpr, "next_alias_id", 500000)
    monkeypatch.setattr(lpr, "frame_count", 10)
    monkeypatch.setattr(lpr, "DEBUG_TRACK_DECISIONS", False)


def at_frame(monkeypatch, n):
    monkeypatch.setattr(lpr, "frame_count", n)


def test_first_sighting_keeps_tracker_id():
    assert lpr.stable_track_id(7, (0, 0, 100, 30)) == 7


def test_small_move_keeps_id(monkeypatch):
    lpr.stable_track_id(7, (0, 0, 100, 30))
    at_frame(monkeypatch, 11)
    assert lpr.stable_track_id(7, (20, 5, 120, 35)) == 7


def test_big_jump_gets_new_id_and_keeps_it(monkeypatch):
    lpr.stable_track_id(7, (0, 0, 100, 30))
    at_frame(monkeypatch, 11)
    new_id = lpr.stable_track_id(7, (1000, 0, 1100, 30))
    assert new_id == 500000
    assert lpr.next_alias_id == 500001

    at_frame(monkeypatch, 12)
    assert lpr.stable_track_id(7, (1010, 0, 1110, 30)) == new_id


def test_large_size_change_gets_new_id(monkeypatch):
    lpr.stable_track_id(7, (0, 0, 100, 30))
    at_frame(monkeypatch, 11)
    assert lpr.stable_track_id(7, (0, 0, 250, 75)) != 7


def test_longer_gap_allows_a_bigger_move(monkeypatch):
    # 300 px in one frame is a jump (limit 2.5 * 100 px) ...
    lpr.stable_track_id(7, (0, 0, 100, 30))
    at_frame(monkeypatch, 11)
    assert lpr.stable_track_id(7, (300, 0, 400, 30)) != 7


def test_same_move_after_a_long_gap_keeps_id(monkeypatch):
    # ... but after 20 frames the limit grows to 4 * 250 px.
    lpr.stable_track_id(7, (0, 0, 100, 30))
    at_frame(monkeypatch, 30)
    assert lpr.stable_track_id(7, (300, 0, 400, 30)) == 7


def test_ids_are_independent(monkeypatch):
    lpr.stable_track_id(1, (0, 0, 100, 30))
    lpr.stable_track_id(2, (1000, 0, 1100, 30))
    at_frame(monkeypatch, 11)
    assert lpr.stable_track_id(1, (10, 0, 110, 30)) == 1
    assert lpr.stable_track_id(2, (1010, 0, 1110, 30)) == 2
