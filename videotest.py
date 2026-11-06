"""LPR launcher: REAL-TIME PLAYBACK (video at a fixed speed).

The video always plays at PLAYBACK_SPEED x its real speed; detection keeps
up as best it can. All logic lives in lpr.py; this file only picks options.

    python videotest.py
"""

import os
import sys

# The pipeline code lives in src/.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import lpr  # noqa: E402

SETTINGS = dict(
    VIDEO_PATH=os.path.join(lpr.BASE_DIR, "videos", "srm6.mp4"),
    ACCURACY_MODE=False,
    PLAYBACK_SPEED=0.4,          # 0.4 = slow motion, more frames read per bus
    VEHICLE_INFER_EVERY_N_FRAMES=2,
    ROI_MAX_VEHICLES=4,
    USE_HALF_PRECISION=False,
    MAX_DET_PLATES=300,          # YOLO default (no cap)
    MAX_DET_VEHICLES=300,
    PROFILE=False,
)


if __name__ == "__main__":
    lpr.configure(**SETTINGS)
    lpr.main()
