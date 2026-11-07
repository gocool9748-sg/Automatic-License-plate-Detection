"""Unit tests for the colour checks in lpr.py.

plate_is_dark_banner rejects light-text-on-dark regions (college banners,
destination boards) and is_yellow_body spots yellow-painted buses that the
vehicle model calls TRUCK. Synthetic images are used, so no video is needed.
torch and ultralytics are stubbed out.
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

import numpy as np  # noqa: E402

YELLOW = (0, 220, 255)   # BGR
GREY = (128, 128, 128)


def crop(background, text, h=40, w=120):
    """h x w BGR crop with a text-like bar in the middle."""
    img = np.full((h, w, 3), background, dtype=np.uint8)
    img[h // 3: 2 * h // 3, w // 6: 5 * w // 6] = text
    return img


def test_light_text_on_dark_background_is_a_banner():
    assert lpr.plate_is_dark_banner(crop(background=30, text=240))


def test_dark_text_on_white_plate_is_not_a_banner():
    assert not lpr.plate_is_dark_banner(crop(background=230, text=20))


def test_light_text_on_mid_grey_background_is_not_a_banner():
    # Mostly dark-ish, but the background is brighter than DARK_BANNER_MAX_MEAN.
    assert not lpr.plate_is_dark_banner(crop(background=150, text=250))


@pytest.mark.parametrize("image", [
    None,
    np.zeros((0, 0, 3), dtype=np.uint8),
    np.zeros((5, 120, 3), dtype=np.uint8),    # too short
    np.zeros((40, 10, 3), dtype=np.uint8),    # too narrow
])
def test_missing_or_tiny_crops_are_not_banners(image):
    assert not lpr.plate_is_dark_banner(image)


def frame_with_body(top, bottom):
    """200x200 frame; the vehicle box (40,40)-(160,160) has its top and
    bottom halves painted the given colours."""
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    frame[40:100, 40:160] = top
    frame[100:160, 40:160] = bottom
    return frame


BOX = (40, 40, 160, 160)


def test_yellow_lower_body_is_detected():
    assert lpr.is_yellow_body(frame_with_body(GREY, YELLOW), BOX)


def test_grey_body_is_not_yellow():
    assert not lpr.is_yellow_body(frame_with_body(GREY, GREY), BOX)


def test_only_the_lower_half_of_the_box_counts():
    # Yellow windshield sticker / sky glare in the top half must not count.
    assert not lpr.is_yellow_body(frame_with_body(YELLOW, GREY), BOX)


def test_box_outside_the_frame_is_not_yellow():
    frame = frame_with_body(YELLOW, YELLOW)
    assert not lpr.is_yellow_body(frame, (300, 300, 400, 400))


def test_box_with_extra_fields_is_accepted():
    frame = frame_with_body(GREY, YELLOW)
    assert lpr.is_yellow_body(frame, (*BOX, "TRUCK", 0.9))
