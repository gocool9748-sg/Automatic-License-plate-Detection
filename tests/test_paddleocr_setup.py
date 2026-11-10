"""Unit tests for the PaddleOCR path setup in ocr_image.py.

_find_paddleocr returns the first candidate folder that contains
ppocr/__init__.py (an explicit path first, then third_party/PaddleOCR and
the older layouts, then the current directory), or None. _ensure_paddleocr
puts that folder on sys.path exactly once and only clones PaddleOCR into
third_party/ when nothing was found. The git clone is replaced by a fake,
so these tests need no network or Paddle.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

_spec = importlib.util.spec_from_file_location(
    "ocr_infer_setup", REPO_ROOT / "src" / "ocr_image.py"
)
ocr_infer = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ocr_infer)


def make_ppocr(root):
    (root / "ppocr").mkdir(parents=True)
    (root / "ppocr" / "__init__.py").write_text("")
    return root


@pytest.fixture
def layout(monkeypatch, tmp_path):
    """A fake project root with an empty src/ and a separate working dir."""
    root = tmp_path / "project"
    src = root / "src"
    src.mkdir(parents=True)
    cwd = tmp_path / "elsewhere"
    cwd.mkdir()
    monkeypatch.setattr(ocr_infer, "_ROOT_DIR", root)
    monkeypatch.setattr(ocr_infer, "_SCRIPT_DIR", src)
    monkeypatch.chdir(cwd)
    monkeypatch.setattr(sys, "path", list(sys.path))
    return root, src, cwd


# ---------------- _find_paddleocr ----------------

def test_returns_none_when_ppocr_is_nowhere(layout):
    assert ocr_infer._find_paddleocr() is None


def test_finds_third_party_clone(layout):
    root, _, _ = layout
    target = make_ppocr(root / "third_party" / "PaddleOCR")
    assert ocr_infer._find_paddleocr() == target


def test_explicit_path_wins_over_third_party(layout, tmp_path):
    root, _, _ = layout
    make_ppocr(root / "third_party" / "PaddleOCR")
    mine = make_ppocr(tmp_path / "my_paddleocr")
    assert ocr_infer._find_paddleocr(str(mine)) == mine


def test_bad_explicit_path_falls_back_to_defaults(layout, tmp_path):
    root, _, _ = layout
    target = make_ppocr(root / "PaddleOCR")   # older layout
    assert ocr_infer._find_paddleocr(str(tmp_path / "missing")) == target


def test_folder_without_ppocr_init_is_ignored(layout):
    root, _, _ = layout
    (root / "third_party" / "PaddleOCR" / "ppocr").mkdir(parents=True)
    assert ocr_infer._find_paddleocr() is None


def test_finds_clone_under_current_directory(layout):
    _, _, cwd = layout
    target = make_ppocr(cwd / "PaddleOCR")
    assert ocr_infer._find_paddleocr() == target


# ---------------- _ensure_paddleocr ----------------

def test_ensure_adds_found_root_to_sys_path_once(layout, monkeypatch):
    root, _, _ = layout
    target = make_ppocr(root / "third_party" / "PaddleOCR")
    monkeypatch.setattr(
        ocr_infer.subprocess, "check_call",
        lambda *a, **k: pytest.fail("should not clone when ppocr exists"),
    )

    assert ocr_infer._ensure_paddleocr() == target
    assert ocr_infer._ensure_paddleocr() == target
    assert sys.path.count(str(target)) == 1
    assert sys.path[0] == str(target)


def test_ensure_clones_into_third_party_when_missing(layout, monkeypatch):
    root, _, _ = layout
    calls = []
    monkeypatch.setattr(
        ocr_infer.subprocess, "check_call", lambda cmd: calls.append(cmd)
    )

    result = ocr_infer._ensure_paddleocr()

    expected = root / "third_party" / "PaddleOCR"
    assert result == expected
    assert len(calls) == 1
    cmd = calls[0]
    assert cmd[:2] == ["git", "clone"]
    assert "--depth" in cmd
    assert cmd[-1] == str(expected)
    assert str(expected) in sys.path
