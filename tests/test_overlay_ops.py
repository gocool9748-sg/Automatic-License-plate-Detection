"""Unit tests for the recorded overlay draw commands in lpr.py.

torch and ultralytics are stubbed out so these run without a GPU or the
heavy ML dependencies (see tests/test_plate_text.py).
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



def test_ov_rectangle_records_command_on_overlay():
    ov = lpr.OverlayOps((50, 50, 3))
    lpr.ov_rectangle(ov, (1, 2), (10, 20), (0, 255, 0), 2)
    assert ov.ops == [("r", (1, 2), (10, 20), (0, 255, 0), 2)]


def test_ov_puttext_records_command_on_overlay():
    ov = lpr.OverlayOps((50, 50, 3))
    lpr.ov_putText(ov, "TN01", (5, 30), lpr.cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                   (255, 255, 255), 1)
    assert ov.ops == [("t", "TN01", (5, 30), lpr.cv2.FONT_HERSHEY_SIMPLEX,
                       0.5, (255, 255, 255), 1, lpr.cv2.LINE_8)]


def test_recorded_rectangle_matches_direct_drawing():
    ov = lpr.OverlayOps((40, 40, 3))
    lpr.ov_rectangle(ov, (5, 5), (30, 30), (0, 0, 255), 2)

    replayed = np.zeros((40, 40, 3), dtype=np.uint8)
    ov.apply(replayed)

    direct = np.zeros((40, 40, 3), dtype=np.uint8)
    lpr.ov_rectangle(direct, (5, 5), (30, 30), (0, 0, 255), 2)

    assert replayed.any()
    assert np.array_equal(replayed, direct)


def test_recorded_text_matches_direct_drawing():
    args = ("AB12", (2, 30), lpr.cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    ov = lpr.OverlayOps((40, 120, 3))
    lpr.ov_putText(ov, *args)

    replayed = np.zeros((40, 120, 3), dtype=np.uint8)
    ov.apply(replayed)

    direct = np.zeros((40, 120, 3), dtype=np.uint8)
    lpr.ov_putText(direct, *args)

    assert replayed.any()
    assert np.array_equal(replayed, direct)


def test_apply_replays_commands_in_order():
    ov = lpr.OverlayOps((20, 20, 3))
    # A filled red square, then a filled green one on top of it.
    lpr.ov_rectangle(ov, (0, 0), (19, 19), (0, 0, 255), -1)
    lpr.ov_rectangle(ov, (0, 0), (19, 19), (0, 255, 0), -1)

    frame = np.zeros((20, 20, 3), dtype=np.uint8)
    ov.apply(frame)
    assert tuple(frame[10, 10]) == (0, 255, 0)


def test_empty_overlay_leaves_frame_untouched():
    frame = np.full((10, 10, 3), 7, dtype=np.uint8)
    lpr.OverlayOps(frame.shape).apply(frame)
    assert (frame == 7).all()
