"""Unit tests for classical_plate_candidates and to_gray3 in lpr.py.

classical_plate_candidates finds dark, text-like blocks of plate proportions
in a vehicle ROI without any model, as a fallback when YOLO misses the plate.
torch and ultralytics are stubbed out.
"""

import sys
import types
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

for name in ("torch", "ultralytics"):
    if name not in sys.modules:
        stub = types.ModuleType(name)
        if name == "ultralytics":
            stub.YOLO = object
        sys.modules[name] = stub

import lpr  # noqa: E402


def roi_with_text(origins, size=(240, 480), shades=None):
    """Light grey ROI with a row of dark characters at each origin
    (about 100 x 24 px each, i.e. plate-like proportions)."""
    roi = np.full((*size, 3), 200, dtype=np.uint8)
    shades = shades or [20] * len(origins)
    for (x, y), shade in zip(origins, shades):
        cv2.putText(roi, "AB1234", (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (shade, shade, shade), 2, cv2.LINE_AA)
    return roi


def contains(box, x1, y1, x2, y2):
    return box[0] <= x1 and box[1] <= y1 and box[2] >= x2 and box[3] >= y2


def test_blank_roi_has_no_candidates():
    roi = np.full((240, 480, 3), 200, dtype=np.uint8)
    assert lpr.classical_plate_candidates(roi) == []


def test_finds_the_text_block():
    roi = roi_with_text([(150, 130)])
    cands = lpr.classical_plate_candidates(roi)

    assert cands
    x1, y1, x2, y2, score = cands[0]
    # The characters span roughly x 150..250, y 110..131.
    assert contains((x1, y1, x2, y2), 155, 112, 245, 128)
    assert 0 <= x1 < x2 <= roi.shape[1] and 0 <= y1 < y2 <= roi.shape[0]
    assert score > 0
    w, h = x2 - x1, y2 - y1
    assert 2.0 <= w / h <= 6.5


def test_candidates_are_sorted_by_darkness_and_capped():
    # Three text blocks; the darkest one should score highest.
    roi = roi_with_text([(20, 50), (240, 130), (60, 210)], shades=[120, 20, 70])

    assert len(lpr.classical_plate_candidates(roi, max_n=3)) == 3
    cands = lpr.classical_plate_candidates(roi, max_n=2)
    assert len(cands) == 2
    scores = [c[4] for c in cands]
    assert scores == sorted(scores, reverse=True)
    assert contains(cands[0], 245, 112, 335, 128)   # the darkest block (240, 130)
    assert len(lpr.classical_plate_candidates(roi, max_n=1)) == 1


def test_to_gray3_keeps_shape_and_equalises_channels():
    img = np.zeros((4, 5, 3), dtype=np.uint8)
    img[..., 2] = 255  # pure red in BGR
    out = lpr.to_gray3(img)

    assert out.shape == img.shape and out.dtype == np.uint8
    assert np.array_equal(out[..., 0], out[..., 1])
    assert np.array_equal(out[..., 1], out[..., 2])
    assert int(out[0, 0, 0]) == int(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[0, 0])
