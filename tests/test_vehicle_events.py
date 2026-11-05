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
    monkeypatch.setattr(lpr, "best_evidence", {})
    monkeypatch.setattr(lpr, "known_plate_labels", {})
    monkeypatch.setattr(lpr, "OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(lpr, "EVIDENCE_DIR", str(tmp_path / "evidence"))
    monkeypatch.setattr(lpr, "REGISTER_DIR", str(tmp_path / "register"))
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


def _register_rows(tmp_dir):
    import csv
    files = list((tmp_dir / "register").glob("*.csv"))
    assert len(files) == 1
    with open(files[0], newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_register_row_and_proof_images_are_saved(fresh_state):
    import numpy as np
    tmp_dir = fresh_state.parent
    plate_crop = np.full((30, 120, 3), 200, np.uint8)
    snapshot = np.full((300, 400, 3), 100, np.uint8)
    lpr.best_evidence[1] = (5.0, plate_crop, snapshot, lpr.time.time())
    lpr.known_plates.add(PLATE)
    lpr.known_plate_labels[PLATE] = "bus 27"
    read()
    confirm(vehicle="BUS")
    flush()

    rows = _register_rows(tmp_dir)
    assert len(rows) == 1
    row = rows[0]
    assert row["plate"] == PLATE
    assert row["bus"] == "bus 27"
    assert row["vehicle"] == "BUS"
    assert row["in_fleet"] == "yes"
    assert row["plate_image"].endswith(f"_{PLATE}_plate.jpg")
    assert row["vehicle_image"].endswith(f"_{PLATE}_vehicle.jpg")
    assert (tmp_dir / row["plate_image"]).is_file()
    assert (tmp_dir / row["vehicle_image"]).is_file()


def test_best_quality_crop_of_the_vehicle_is_used(fresh_state):
    import numpy as np
    import cv2
    tmp_dir = fresh_state.parent
    blurry = np.full((30, 120, 3), 10, np.uint8)
    sharp = np.full((30, 120, 3), 250, np.uint8)
    lpr.best_evidence[1] = (1.0, blurry, None, lpr.time.time())
    lpr.best_evidence[2] = (9.0, sharp, None, lpr.time.time())
    read(track_id=1)
    read(track_id=2)
    confirm(track_id=1)
    flush()

    row = _register_rows(tmp_dir)[0]
    saved = cv2.imread(str(tmp_dir / row["plate_image"]))
    assert saved.mean() > 200          # the sharp crop of track 2 was saved
    assert row["vehicle_image"] == ""  # no snapshot available -> empty, no crash


def test_register_without_fleet_list_leaves_fleet_column_empty(fresh_state):
    read()
    confirm()
    flush()
    row = _register_rows(fresh_state.parent)[0]
    assert row["in_fleet"] == ""
    assert row["plate_image"] == ""     # no proof images collected


def test_known_plate_labels_are_read_from_the_fleet_file(tmp_path, monkeypatch):
    fleet = tmp_path / "known_plates.txt"
    fleet.write_text("# fleet\nTN48W9259   # bus 27\nTN48BK2144\n", encoding="utf-8")
    monkeypatch.setattr(lpr, "KNOWN_PLATES_FILE", str(fleet))
    lpr.load_known_plates()
    assert lpr.known_plates == {"TN48W9259", "TN48BK2144"}
    assert lpr.known_plate_labels == {"TN48W9259": "bus 27"}


def test_vehicle_snapshot_boxes_the_plate(monkeypatch):
    import numpy as np
    frame = np.zeros((1080, 1920, 3), np.uint8)
    monkeypatch.setattr(lpr, "vehicle_detections", [(500, 300, 1500, 1000, "BUS", 0.9)])
    snap = lpr.make_vehicle_snapshot(frame, (900, 850, 1050, 890))
    assert snap is not None
    assert snap.shape[1] <= lpr.SNAPSHOT_MAX_WIDTH
    assert snap[..., 1].max() == 255     # green plate box drawn
