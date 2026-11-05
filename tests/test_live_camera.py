"""LiveCamera (lpr.py) with a fake cv2.VideoCapture: it must hand out the
newest frame, survive a short camera drop-out and stop cleanly when the
camera is gone for good. torch and ultralytics are stubbed out."""

import sys
import threading
import time
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


class FakeCapture:
    """Plays a script: a list of booleans, True = deliver a frame."""

    script = []
    opened = []

    def __init__(self, source, backend=None):
        FakeCapture.opened.append((source, backend))
        self.n = 0
        self.lock = threading.Lock()

    def isOpened(self):
        return True

    def set(self, *args):
        return True

    def get(self, prop):
        return 30.0

    def read(self):
        time.sleep(0.002)
        with FakeCapture.lock:
            if not FakeCapture.script:
                return False, None
            ok = FakeCapture.script.pop(0)
            FakeCapture.counter += 1
            value = FakeCapture.counter
        if not ok:
            return False, None
        return True, np.full((4, 4, 3), value % 256, np.uint8)

    def release(self):
        pass


FakeCapture.lock = threading.Lock()
FakeCapture.counter = 0


@pytest.fixture
def fake_camera(monkeypatch):
    FakeCapture.script = []
    FakeCapture.opened = []
    FakeCapture.counter = 0
    monkeypatch.setattr(lpr.cv2, "VideoCapture", FakeCapture)
    monkeypatch.setattr(lpr, "CAMERA_RECONNECT_SECONDS", 0.3)
    return FakeCapture


def test_reads_newest_frame_not_a_backlog(fake_camera):
    fake_camera.script = [True] * 50
    cam = lpr.LiveCamera(0)
    time.sleep(0.3)                  # camera delivered many frames meanwhile
    ok, frame = cam.read()
    assert ok
    assert cam.taken == cam.seq      # we got the latest one, not frame 1
    assert cam.seq > 5
    cam.release()


def test_webcam_index_uses_a_camera_backend(fake_camera):
    fake_camera.script = [True]
    cam = lpr.LiveCamera(0)
    cam.release()
    assert fake_camera.opened[0][0] == 0
    assert fake_camera.opened[0][1] is not None


def test_stream_address_is_opened_as_is(fake_camera):
    fake_camera.script = [True]
    cam = lpr.LiveCamera("rtsp://camera/stream")
    cam.release()
    assert fake_camera.opened[0] == ("rtsp://camera/stream", None)


def test_short_dropout_reconnects(fake_camera, monkeypatch):
    monkeypatch.setattr(lpr, "CAMERA_RECONNECT_SECONDS", 5.0)
    fake_camera.script = [True, False] + [True] * 5
    cam = lpr.LiveCamera(0)
    ok1, _ = cam.read()
    deadline = time.time() + 4
    while cam.seq < 3 and time.time() < deadline:
        time.sleep(0.05)
    ok2, _ = cam.read()
    assert ok1 and ok2
    assert len(fake_camera.opened) >= 2   # it reopened the camera
    cam.release()


def test_lost_camera_stops_reading(fake_camera):
    fake_camera.script = [True]          # one frame, then nothing
    cam = lpr.LiveCamera(0)
    assert cam.read()[0] is True
    start = time.time()
    ok, frame = cam.read()
    assert ok is False and frame is None
    assert time.time() - start < 5
    cam.release()
