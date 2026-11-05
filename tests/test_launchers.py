"""The launcher scripts only pass settings to lpr.configure(). Make sure every
setting they use exists, and that configure() keeps derived settings in step.
torch and ultralytics are stubbed out."""

import importlib
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


@pytest.fixture
def restore_settings():
    saved = {k: v for k, v in vars(lpr).items() if k.isupper()}
    yield
    for k, v in saved.items():
        setattr(lpr, k, v)


@pytest.mark.parametrize("launcher", ["webcam", "videotest"])
def test_launcher_settings_exist_and_apply(launcher, restore_settings):
    module = importlib.import_module(launcher)
    lpr.configure(**module.SETTINGS)
    for name, value in module.SETTINGS.items():
        assert getattr(lpr, name) == value
    assert lpr.USE_LOOSE_TRACKER == (not lpr.ACCURACY_MODE)


def test_unknown_setting_is_rejected(restore_settings):
    with pytest.raises(KeyError):
        lpr.configure(PLAYBACK_SPEEED=0.5)


def test_explicit_tracker_choice_is_kept(restore_settings):
    lpr.configure(ACCURACY_MODE=True, USE_LOOSE_TRACKER=True)
    assert lpr.USE_LOOSE_TRACKER is True


def test_project_files_are_where_the_code_expects_them():
    import awiros_worker
    import os

    assert os.path.isfile(lpr.AWIROS_WORKER)
    assert os.path.isfile(lpr.YOLO_WEIGHTS)
    assert os.path.isdir(lpr.CONFIG_DIR)
    assert os.path.isfile(os.path.join(lpr.CONFIG_DIR, lpr.TRACKER_YAML))
    assert os.path.isfile(os.path.join(lpr.CONFIG_DIR, "known_plates.example.txt"))
    assert awiros_worker.DICT_PATH.is_file()
    assert os.path.dirname(lpr.RESULTS_LOG) == lpr.OUTPUT_DIR
