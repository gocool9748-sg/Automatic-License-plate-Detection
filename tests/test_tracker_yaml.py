"""Unit tests for ensure_tracker_yaml in lpr.py.

The loose ByteTrack config lives in config/bytetrack_lpr.yaml. An existing
file is used as-is (so hand-tuned settings survive), a missing one is written
with the defaults, and any failure falls back to Ultralytics' stock
bytetrack.yaml. torch and ultralytics are stubbed out.
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
def tracker_state(monkeypatch, tmp_path):
    monkeypatch.setattr(lpr, "CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setattr(lpr, "TRACKER_CFG", "bytetrack.yaml")
    monkeypatch.setattr(lpr, "USE_LOOSE_TRACKER", True)


def tracker_path():
    return Path(lpr.CONFIG_DIR) / lpr.TRACKER_YAML


def test_stock_tracker_when_loose_tracker_is_off(monkeypatch):
    monkeypatch.setattr(lpr, "USE_LOOSE_TRACKER", False)
    lpr.ensure_tracker_yaml()
    assert lpr.TRACKER_CFG == "bytetrack.yaml"
    assert not tracker_path().exists()


def test_writes_default_config_when_missing():
    lpr.ensure_tracker_yaml()
    path = tracker_path()
    assert lpr.TRACKER_CFG == str(path)
    text = path.read_text(encoding="utf-8")
    assert "tracker_type: bytetrack" in text
    assert "track_buffer: 60" in text


def test_keeps_existing_hand_tuned_config():
    path = tracker_path()
    path.parent.mkdir(parents=True)
    path.write_text("tracker_type: bytetrack\ntrack_buffer: 120\n", encoding="utf-8")

    lpr.ensure_tracker_yaml()

    assert lpr.TRACKER_CFG == str(path)
    assert path.read_text(encoding="utf-8") == (
        "tracker_type: bytetrack\ntrack_buffer: 120\n"
    )


def test_default_matches_the_shipped_config():
    lpr.ensure_tracker_yaml()
    shipped = Path(__file__).resolve().parent.parent / "config" / lpr.TRACKER_YAML
    assert tracker_path().read_text(encoding="utf-8") == shipped.read_text(
        encoding="utf-8"
    )


def test_falls_back_to_stock_tracker_when_config_cannot_be_written(
    monkeypatch, tmp_path
):
    blocker = tmp_path / "not_a_dir"
    blocker.write_text("x", encoding="utf-8")
    monkeypatch.setattr(lpr, "CONFIG_DIR", str(blocker))

    lpr.ensure_tracker_yaml()

    assert lpr.TRACKER_CFG == "bytetrack.yaml"
