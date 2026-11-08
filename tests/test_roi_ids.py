"""Unit tests for stable IDs of plates found by the vehicle ROI search.

_assign_roi_id in lpr.py gives each ROI-detected plate the ID of the nearest
recent box, so plates without a tracker ID still collect OCR votes.

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
def fresh_roi_state(monkeypatch):
    monkeypatch.setattr(lpr, "roi_tracks", {})
    monkeypatch.setattr(lpr, "roi_next_id", 100000)
    monkeypatch.setattr(lpr, "frame_count", 0)


def test_first_plate_gets_a_new_id():
    assert lpr._assign_roi_id((100, 100, 160, 120)) == 100000
    assert lpr.roi_next_id == 100001


def test_nearby_plate_in_a_later_frame_keeps_its_id():
    first = lpr._assign_roi_id((100, 100, 160, 120))
    lpr.frame_count = 5
    assert lpr._assign_roi_id((110, 102, 170, 122)) == first
    assert lpr.roi_tracks[first] == ((110, 102, 170, 122), 5)


def test_far_away_plate_gets_a_different_id():
    first = lpr._assign_roi_id((100, 100, 160, 120))
    second = lpr._assign_roi_id((600, 400, 660, 420))
    assert second != first
    assert set(lpr.roi_tracks) == {first, second}


def test_plate_goes_to_the_nearest_previous_box():
    a = lpr._assign_roi_id((100, 100, 160, 120))
    b = lpr._assign_roi_id((300, 100, 360, 120))
    # Between the two boxes, but much closer to b.
    assert lpr._assign_roi_id((240, 100, 300, 120)) == b
    assert a != b


def test_stale_ids_expire_after_20_frames():
    first = lpr._assign_roi_id((100, 100, 160, 120))
    lpr.frame_count = 21
    again = lpr._assign_roi_id((100, 100, 160, 120))
    assert again != first
    assert first not in lpr.roi_tracks
