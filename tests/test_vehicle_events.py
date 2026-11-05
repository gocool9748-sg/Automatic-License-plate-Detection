"""Unit tests for one-record-per-vehicle event logging in lpr.py.

All OCR readings of one physical vehicle are pooled in a "cluster";
register_clean_vehicle_event marks the vehicle as confirmed and
flush_pending_events logs it ONCE, with the plate decided from all its
readings. torch and ultralytics are stubbed out.
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
def fresh_state(monkeypatch, tmp_path):
    for attr, value in (
        ("confirmed_vehicle_events", {}),
        ("recent_vehicle_event_time", {}),
        ("plate_clusters", []),
        ("track_to_cluster", {}),
        ("logged_events", []),
        ("track_frames", defaultdict(set)),
        ("known_plates", set()),
    ):
        monkeypatch.setattr(lpr, attr, value)
    monkeypatch.setattr(lpr, "last_vehicle_event", None, raising=False)
    log = tmp_path / "results.log"
    monkeypatch.setattr(lpr, "RESULTS_LOG", str(log))
    return log


def read(plate=PLATE, track_id=1, conf=95.0, times=4, width=120):
    for _ in range(times):
        lpr._cluster_add_locked(track_id, plate, conf, width)


def confirm(plate=PLATE, vehicle="CAR", track_id=1):
    lpr.register_clean_vehicle_event(plate, vehicle, 0.8, track_id, None)


def flush():
    lpr.flush_pending_events(force=True)


@pytest.mark.parametrize("plate", ["", None, "HELLO"])
def test_invalid_plate_is_ignored(plate, fresh_state):
    read()
    confirm(plate=plate)
    flush()
    assert lpr.confirmed_vehicle_events == {}
    assert not fresh_state.exists()


def test_first_event_is_recorded_and_logged(fresh_state, capsys):
    read(conf=80.0, times=2)
    read(conf=90.0, times=2)
    confirm()
    flush()
    record = lpr.confirmed_vehicle_events[PLATE]
    assert record["vehicle_type"] == "CAR"
    assert record["ocr_confidence"] == pytest.approx(85.0)
    assert record["track_id"] == 1
    assert lpr.last_vehicle_event == record
    line = fresh_state.read_text(encoding="utf-8").strip()
    assert line.endswith(f",{PLATE},CAR,85")
    assert PLATE in capsys.readouterr().out


def test_same_vehicle_is_recorded_once(fresh_state):
    read(track_id=1)
    read(track_id=2)          # tracker gave the same plate a new id
    confirm(track_id=1)
    confirm(track_id=2)
    flush()
    flush()
    assert len(fresh_state.read_text(encoding="utf-8").splitlines()) == 1


def test_one_character_misread_joins_the_vehicle(fresh_state):
    read(plate=PLATE, times=4)
    read(plate="TN09AB1284", track_id=2, times=1)   # 1 character off
    confirm()
    flush()
    assert list(lpr.confirmed_vehicle_events) == [PLATE]


def test_fleet_buses_two_characters_apart_are_separate(fresh_state):
    # TN48BK2144 and TN48BK2113 are two different buses of the same fleet.
    read(plate="TN48BK2144", track_id=1)
    read(plate="TN48BK2113", track_id=2)
    confirm(plate="TN48BK2144", vehicle="BUS", track_id=1)
    confirm(plate="TN48BK2113", vehicle="BUS", track_id=2)
    flush()
    assert set(lpr.confirmed_vehicle_events) == {"TN48BK2144", "TN48BK2113"}


def test_plate_without_enough_support_is_not_logged(fresh_state):
    read(times=lpr.FINAL_MIN_SUPPORT - 1)
    confirm()
    flush()
    assert lpr.confirmed_vehicle_events == {}


def test_low_confidence_plate_is_not_logged(fresh_state):
    read(conf=lpr.MIN_LOG_CONFIDENCE - 20)
    confirm()
    flush()
    assert lpr.confirmed_vehicle_events == {}


def test_bus_votes_win_over_truck(fresh_state):
    read()
    confirm(vehicle="TRUCK")
    confirm(vehicle="TRUCK")
    confirm(vehicle="BUS")
    flush()
    assert lpr.confirmed_vehicle_events[PLATE]["vehicle_type"] == "BUS"


def test_missing_log_setting_skips_file(monkeypatch, fresh_state):
    monkeypatch.setattr(lpr, "RESULTS_LOG", "")
    read()
    confirm()
    flush()
    assert PLATE in lpr.confirmed_vehicle_events
    assert not fresh_state.exists()
