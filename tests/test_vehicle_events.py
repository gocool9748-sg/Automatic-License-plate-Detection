"""Unit tests for register_clean_vehicle_event in videotest.py.

Each stable plate should produce one vehicle record (and one RESULTS_LOG
line), a later vehicle classification should fill in an UNKNOWN type, and
invalid plates should be ignored. torch and ultralytics are stubbed out.
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

import videotest  # noqa: E402

PLATE = "TN09AB1234"


@pytest.fixture(autouse=True)
def fresh_state(monkeypatch, tmp_path):
    monkeypatch.setattr(videotest, "confirmed_vehicle_events", {})
    monkeypatch.setattr(videotest, "recent_vehicle_event_time", {})
    monkeypatch.setattr(videotest, "track_history", defaultdict(list))
    monkeypatch.setattr(videotest, "track_confidence", defaultdict(list))
    monkeypatch.setattr(videotest, "last_vehicle_event", None, raising=False)
    log = tmp_path / "results.log"
    monkeypatch.setattr(videotest, "RESULTS_LOG", str(log))
    return log


def register(plate=PLATE, vehicle="CAR", conf=0.8, track_id=1):
    videotest.register_clean_vehicle_event(plate, vehicle, conf, track_id, None)


@pytest.mark.parametrize("plate", ["", None, "HELLO"])
def test_invalid_plate_is_ignored(plate, fresh_state):
    register(plate=plate)
    assert videotest.confirmed_vehicle_events == {}
    assert not fresh_state.exists()


def test_first_event_is_recorded_and_logged(fresh_state, capsys):
    videotest.track_history[1] = [PLATE, PLATE]
    videotest.track_confidence[1] = [80.0, 90.0]
    register()
    record = videotest.confirmed_vehicle_events[PLATE]
    assert record["vehicle_type"] == "CAR"
    assert record["ocr_confidence"] == pytest.approx(85.0)
    assert record["track_id"] == 1
    assert videotest.last_vehicle_event == record
    line = fresh_state.read_text(encoding="utf-8").strip()
    assert line.endswith(f",{PLATE},CAR,85")
    assert PLATE in capsys.readouterr().out


def test_same_plate_is_recorded_once(fresh_state):
    register()
    register(track_id=2)
    assert len(fresh_state.read_text(encoding="utf-8").splitlines()) == 1
    assert videotest.confirmed_vehicle_events[PLATE]["track_id"] == 1


def test_late_classification_fills_unknown_type():
    register(vehicle="UNKNOWN", conf=0.0)
    register(vehicle="BUS", conf=0.7)
    record = videotest.confirmed_vehicle_events[PLATE]
    assert record["vehicle_type"] == "BUS"
    assert record["vehicle_confidence"] == 0.7


def test_known_type_is_not_overwritten():
    register(vehicle="CAR")
    register(vehicle="TRUCK")
    assert videotest.confirmed_vehicle_events[PLATE]["vehicle_type"] == "CAR"


def test_missing_log_setting_skips_file(monkeypatch, fresh_state):
    monkeypatch.setattr(videotest, "RESULTS_LOG", "")
    register()
    assert PLATE in videotest.confirmed_vehicle_events
    assert not fresh_state.exists()
