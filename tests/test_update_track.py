"""Unit tests for update_track in lpr.py.

update_track normalises each OCR reading, drops low-confidence ones, keeps a
bounded history per track and publishes a stable plate once enough readings
agree. torch and ultralytics are stubbed out so these run without a GPU.
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

PLATE = "TN09AB1234"


@pytest.fixture(autouse=True)
def fresh_state(monkeypatch):
    for name in ("track_history", "track_confidence", "track_width"):
        monkeypatch.setattr(lpr, name, defaultdict(list))
    monkeypatch.setattr(lpr, "stable_text", {})
    monkeypatch.setattr(lpr, "latest_ocr", {})


@pytest.mark.parametrize("text", ["", None, "HELLO"])
def test_empty_or_invalid_text_is_ignored(text):
    lpr.update_track(1, text, 90.0, 120)
    assert lpr.track_history[1] == []
    assert 1 not in lpr.latest_ocr


def test_low_confidence_reading_is_ignored():
    lpr.update_track(1, PLATE, lpr.MIN_OCR_CONFIDENCE - 1, 120)
    assert lpr.track_history[1] == []


def test_reading_is_normalised_and_recorded():
    lpr.update_track(1, "tn 09 ab 1234", 80.0, 120)
    assert lpr.track_history[1] == [PLATE]
    assert lpr.latest_ocr[1] == (PLATE, 80.0)


def test_single_reading_is_not_yet_stable():
    lpr.update_track(1, PLATE, 95.0, 120)
    assert 1 not in lpr.stable_text


def test_agreeing_readings_become_stable():
    for _ in range(lpr.MIN_VOTES):
        lpr.update_track(1, PLATE, 80.0, 120)
    assert lpr.stable_text[1] == PLATE


def test_history_is_capped():
    for _ in range(lpr.MAX_HISTORY + 5):
        lpr.update_track(1, PLATE, 80.0, 120)
    assert len(lpr.track_history[1]) == lpr.MAX_HISTORY
    assert len(lpr.track_confidence[1]) == lpr.MAX_HISTORY
    assert len(lpr.track_width[1]) == lpr.MAX_HISTORY
