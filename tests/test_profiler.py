"""Unit tests for the detector step Profiler in lpr.py.

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



def test_add_accumulates_sums_and_counts_calls():
    prof = lpr.Profiler()
    prof.add(vehicle=0.010, plate=0.020)
    prof.add(vehicle=0.030)
    assert prof.count == 2
    assert prof.sums["vehicle"] == pytest.approx(0.040)
    assert prof.sums["plate"] == pytest.approx(0.020)


def test_maybe_print_waits_for_the_print_interval(monkeypatch, capsys):
    monkeypatch.setattr(lpr, "PROFILE", True)
    prof = lpr.Profiler()
    prof.add(vehicle=0.010)
    prof.maybe_print()   # interval has not passed yet
    assert capsys.readouterr().out == ""
    assert prof.count == 1


def test_maybe_print_reports_averages_and_resets(monkeypatch, capsys):
    monkeypatch.setattr(lpr, "PROFILE", True)
    prof = lpr.Profiler()
    prof.add(vehicle=0.010, plate=0.020, parse=0.0, roi=0.0, post=0.0)
    prof.add(vehicle=0.030, plate=0.020, parse=0.0, roi=0.0, post=0.0)
    prof.last -= lpr.PROFILE_PRINT_SECONDS + 1

    prof.maybe_print()
    out = capsys.readouterr().out

    # Averages over 2 steps: vehicle 20 ms, plate 20 ms, total 40 ms = 25 fps.
    assert out.startswith("[PROFILE]")
    assert "vehicle model + draw 20 ms" in out
    assert "plate model 20 ms" in out
    assert "total 40 ms (max 25 fps)" in out
    assert prof.count == 0
    assert not prof.sums


def test_maybe_print_is_silent_when_profiling_is_off(monkeypatch, capsys):
    monkeypatch.setattr(lpr, "PROFILE", False)
    prof = lpr.Profiler()
    prof.add(vehicle=0.010)
    prof.last -= lpr.PROFILE_PRINT_SECONDS + 1
    prof.maybe_print()
    assert capsys.readouterr().out == ""
    assert prof.count == 1


def test_maybe_print_with_no_steps_prints_nothing(monkeypatch, capsys):
    monkeypatch.setattr(lpr, "PROFILE", True)
    prof = lpr.Profiler()
    prof.last -= lpr.PROFILE_PRINT_SECONDS + 1
    prof.maybe_print()
    assert capsys.readouterr().out == ""
