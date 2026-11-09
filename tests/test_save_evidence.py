"""Unit tests for save_evidence in lpr.py.

Each logged vehicle gets a plate crop and a vehicle snapshot written to
outputs/evidence/<date>/<time>_<plate>_{plate,vehicle}.jpg; the returned
paths are relative to OUTPUT_DIR with forward slashes, and an image that is
missing or cannot be written yields "". torch and ultralytics are stubbed out.
"""

import sys
import types
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

for name in ("torch", "ultralytics"):
    if name not in sys.modules:
        stub = types.ModuleType(name)
        if name == "ultralytics":
            stub.YOLO = object
        sys.modules[name] = stub

import lpr  # noqa: E402

RECORD = {"first_seen": "2026-11-09 21:05:33", "plate": "TN09AB1234"}


@pytest.fixture(autouse=True)
def evidence_dirs(monkeypatch, tmp_path):
    out = tmp_path / "outputs"
    monkeypatch.setattr(lpr, "OUTPUT_DIR", str(out))
    monkeypatch.setattr(lpr, "EVIDENCE_DIR", str(out / "evidence"))
    monkeypatch.setattr(lpr, "SAVE_EVIDENCE", True)
    return out


def image(h=20, w=60):
    return np.full((h, w, 3), 128, dtype=np.uint8)


def test_writes_plate_and_vehicle_images(evidence_dirs):
    plate_path, vehicle_path = lpr.save_evidence(RECORD, image(), image(120, 200))

    assert plate_path == "evidence/2026-11-09/210533_TN09AB1234_plate.jpg"
    assert vehicle_path == "evidence/2026-11-09/210533_TN09AB1234_vehicle.jpg"
    assert (evidence_dirs / plate_path).is_file()
    assert (evidence_dirs / vehicle_path).is_file()


def test_disabled_saves_nothing(monkeypatch, evidence_dirs):
    monkeypatch.setattr(lpr, "SAVE_EVIDENCE", False)
    assert lpr.save_evidence(RECORD, image(), image()) == ("", "")
    assert not (evidence_dirs / "evidence").exists()


@pytest.mark.parametrize("missing", [None, np.zeros((0, 0, 3), dtype=np.uint8)])
def test_missing_snapshot_gives_empty_path(missing, evidence_dirs):
    plate_path, vehicle_path = lpr.save_evidence(RECORD, image(), missing)
    assert plate_path.endswith("_plate.jpg")
    assert vehicle_path == ""
    assert sorted(p.name for p in (evidence_dirs / "evidence" / "2026-11-09").iterdir()) == [
        "210533_TN09AB1234_plate.jpg"
    ]


def test_missing_plate_crop_still_saves_snapshot():
    plate_path, vehicle_path = lpr.save_evidence(RECORD, None, image())
    assert plate_path == ""
    assert vehicle_path.endswith("_vehicle.jpg")


def test_unwritable_folder_returns_empty_paths(monkeypatch, tmp_path, capsys):
    blocker = tmp_path / "blocked"
    blocker.write_text("x", encoding="utf-8")
    monkeypatch.setattr(lpr, "EVIDENCE_DIR", str(blocker))

    assert lpr.save_evidence(RECORD, image(), image()) == ("", "")
    assert "[PROOF] Could not save" in capsys.readouterr().out
