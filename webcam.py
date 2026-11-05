"""LPR launcher: LIVE WEBCAM.

Reads your webcam (or an IP camera) instead of a video file. The camera is
read in the background and the newest frame is always analysed, so results
never fall behind the real scene. All logic lives in lpr.py; this file only
picks options.

    python webcam.py
"""

import os
import sys

# The pipeline code lives in src/.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import lpr  # noqa: E402

SETTINGS = dict(
    # 0 = first webcam, 1 = second webcam, ...
    # or an IP camera address, e.g. "rtsp://user:password@192.168.1.20:554/stream1"
    CAMERA_INDEX=0,
    CAMERA_WIDTH=1920,
    CAMERA_HEIGHT=1080,
    CAMERA_FPS=30,
    MIRROR_CAMERA=False,         # True = selfie-style mirror (don't use for plates)

    # Live camera: show the camera at full speed; detection uses the newest
    # frame (frames it can't keep up with are skipped, never queued).
    ACCURACY_MODE=False,
    VEHICLE_INFER_EVERY_N_FRAMES=3,
    ROI_MAX_VEHICLES=3,
    USE_HALF_PRECISION=True,     # FP16 on an NVIDIA GPU (ignored on CPU)
    MAX_DET_PLATES=20,
    MAX_DET_VEHICLES=40,
    PROFILE=True,                # print where the time goes every few seconds
)


if __name__ == "__main__":
    lpr.configure(**SETTINGS)
    lpr.main()
