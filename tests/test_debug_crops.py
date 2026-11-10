"""Unit tests for the debug crop helpers in lpr.py.

clean_debug_crop_folder removes the crop_*, latest_id_* and best_plate_*
JPEGs a previous run left in DEBUG_CROPS_DIR and keeps everything else.
save_all_best_crops writes one best_plate_<PLATE>.jpg per recognised plate,
picking the highest-quality crop when several track ids read the same plate
and skipping tracks whose text is missing or not a valid plate. Both do
nothing when DEBUG_SAVE_CROPS is off. torch and ultralytics are stubbed out.
"""

import sys
import types
from collections import defaultdict
from pathlib import Path

import cv2
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

PLATE = "TN09AB1234"
OTHER = "KA01CD5678"


@pytest.fixture
def crops_dir(monkeypatch, tmp_path):
    folder = tmp_path / "debug_crops"
    monkeypatch.setattr(lpr, "DEBUG_CROPS_DIR", str(folder))
    monkeypatch.setattr(lpr, "DEBUG_SAVE_CROPS", True)
    monkeypatch.setattr(lpr, "best_crop", {})
    monkeypatch.setattr(lpr, "best_crop_quality", defaultdict(float))
    monkeypatch.setattr(lpr, "confirmed_text", {})
    monkeypatch.setattr(lpr, "stable_text", {})
    return folder


def crop(value=128, h=20, w=60):
    return np.full((h, w, 3), value, dtype=np.uint8)


def add_track(track_id, text, image, quality, confirmed=True):
    lpr.best_crop[track_id] = image
    lpr.best_crop_quality[track_id] = quality
    (lpr.confirmed_text if confirmed else lpr.stable_text)[track_id] = text


# ---------------- clean_debug_crop_folder ----------------

def test_clean_removes_only_old_crop_jpegs(crops_dir):
    crops_dir.mkdir()
    old = ["crop_1.jpg", "latest_id_7.jpg", "best_plate_TN09AB1234.JPG"]
    kept = ["notes.txt", "crop_1.png", "frame_0001.jpg"]
    for name in old + kept:
        (crops_dir / name).write_bytes(b"x")

    lpr.clean_debug_crop_folder()

    assert sorted(p.name for p in crops_dir.iterdir()) == sorted(kept)


def test_clean_creates_missing_folder(crops_dir):
    lpr.clean_debug_crop_folder()
    assert crops_dir.is_dir()


def test_clean_does_nothing_when_disabled(crops_dir, monkeypatch):
    monkeypatch.setattr(lpr, "DEBUG_SAVE_CROPS", False)
    lpr.clean_debug_crop_folder()
    assert not crops_dir.exists()


# ---------------- save_all_best_crops ----------------

def test_saves_one_file_per_plate(crops_dir):
    add_track(1, PLATE, crop(), 5.0)
    add_track(2, OTHER, crop(), 5.0)

    lpr.save_all_best_crops()

    assert sorted(p.name for p in crops_dir.iterdir()) == [
        f"best_plate_{OTHER}.jpg",
        f"best_plate_{PLATE}.jpg",
    ]


def test_keeps_highest_quality_crop_when_ids_share_a_plate(crops_dir):
    add_track(1, PLATE, crop(value=40), 2.0)
    add_track(2, PLATE, crop(value=220), 9.0)   # re-ID of the same vehicle
    add_track(3, PLATE, crop(value=120), 4.0)

    lpr.save_all_best_crops()

    files = list(crops_dir.iterdir())
    assert [p.name for p in files] == [f"best_plate_{PLATE}.jpg"]
    saved = cv2.imread(str(files[0]))
    assert abs(float(saved.mean()) - 220) < 5


def test_falls_back_to_stable_text(crops_dir):
    add_track(4, PLATE, crop(), 3.0, confirmed=False)
    lpr.save_all_best_crops()
    assert (crops_dir / f"best_plate_{PLATE}.jpg").is_file()


def test_confirmed_text_wins_over_stable_text(crops_dir):
    add_track(5, OTHER, crop(), 3.0, confirmed=False)
    lpr.confirmed_text[5] = PLATE
    lpr.save_all_best_crops()
    assert [p.name for p in crops_dir.iterdir()] == [f"best_plate_{PLATE}.jpg"]


def test_skips_empty_crops_and_invalid_or_missing_text(crops_dir):
    add_track(1, PLATE, np.zeros((0, 0, 3), dtype=np.uint8), 5.0)
    add_track(2, "HELLO", crop(), 5.0)
    lpr.best_crop[3] = crop()          # no text for this track at all
    lpr.best_crop[4] = None

    lpr.save_all_best_crops()

    assert list(crops_dir.iterdir()) == []


def test_save_does_nothing_when_disabled(crops_dir, monkeypatch):
    monkeypatch.setattr(lpr, "DEBUG_SAVE_CROPS", False)
    add_track(1, PLATE, crop(), 5.0)
    lpr.save_all_best_crops()
    assert not crops_dir.exists()
