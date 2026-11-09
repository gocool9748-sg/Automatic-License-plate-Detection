"""Unit tests for write_register in lpr.py.

Every logged vehicle appends one row to outputs/register/<date>.csv; the
header is written only when the file is new, "bus" comes from the
known_plates.txt comment and "in_fleet" is filled only when a known-plates
list is loaded. torch and ultralytics are stubbed out.
"""

import csv
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


def record(plate="TN09AB1234", first_seen="2026-11-09 21:05:33", conf=87.6):
    return {
        "first_seen": first_seen,
        "plate": plate,
        "vehicle_type": "BUS",
        "ocr_confidence": conf,
    }


@pytest.fixture(autouse=True)
def register_dir(monkeypatch, tmp_path):
    folder = tmp_path / "register"
    monkeypatch.setattr(lpr, "REGISTER_DIR", str(folder))
    monkeypatch.setattr(lpr, "known_plates", set())
    monkeypatch.setattr(lpr, "known_plate_labels", {})
    return folder


def read_rows(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.reader(f))


def test_new_file_gets_header_and_row(register_dir):
    lpr.write_register(record(), "evidence/p.jpg", "evidence/v.jpg")

    rows = read_rows(register_dir / "2026-11-09.csv")
    assert rows[0] == list(lpr.REGISTER_COLUMNS)
    assert rows[1] == [
        "2026-11-09", "21:05:33", "TN09AB1234", "", "BUS", "88", "",
        "evidence/p.jpg", "evidence/v.jpg",
    ]


def test_second_vehicle_appends_without_repeating_header(register_dir):
    lpr.write_register(record(), "", "")
    lpr.write_register(record(plate="KA01MN0001", first_seen="2026-11-09 21:07:00"), "", "")

    rows = read_rows(register_dir / "2026-11-09.csv")
    assert len(rows) == 3
    assert rows.count(list(lpr.REGISTER_COLUMNS)) == 1
    assert [r[2] for r in rows[1:]] == ["TN09AB1234", "KA01MN0001"]


def test_each_day_gets_its_own_file(register_dir):
    lpr.write_register(record(first_seen="2026-11-09 23:59:59"), "", "")
    lpr.write_register(record(first_seen="2026-11-10 00:00:01"), "", "")
    assert sorted(p.name for p in register_dir.iterdir()) == [
        "2026-11-09.csv", "2026-11-10.csv",
    ]


def test_known_plate_fills_bus_and_in_fleet(monkeypatch, register_dir):
    monkeypatch.setattr(lpr, "known_plates", {"TN09AB1234"})
    monkeypatch.setattr(lpr, "known_plate_labels", {"TN09AB1234": "bus 27"})

    lpr.write_register(record(), "", "")
    lpr.write_register(record(plate="KA01MN0001"), "", "")

    rows = read_rows(register_dir / "2026-11-09.csv")[1:]
    bus, in_fleet = lpr.REGISTER_COLUMNS.index("bus"), lpr.REGISTER_COLUMNS.index("in_fleet")
    assert (rows[0][bus], rows[0][in_fleet]) == ("bus 27", "yes")
    assert (rows[1][bus], rows[1][in_fleet]) == ("", "no")


def test_disabled_register_writes_nothing(monkeypatch, register_dir):
    monkeypatch.setattr(lpr, "REGISTER_DIR", "")
    lpr.write_register(record(), "", "")
    assert not register_dir.exists()


def test_unwritable_register_folder_is_reported(monkeypatch, tmp_path, capsys):
    blocker = tmp_path / "blocked"
    blocker.write_text("x", encoding="utf-8")
    monkeypatch.setattr(lpr, "REGISTER_DIR", str(blocker))

    lpr.write_register(record(), "", "")
    assert "[REGISTER] Could not write" in capsys.readouterr().out
