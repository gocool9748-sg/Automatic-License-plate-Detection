"""Unit tests for the framing and preprocessing helpers in awiros_worker.py.

Paddle is only imported when AwirosEngine is created, so these tests need
just OpenCV and NumPy.
"""

import json
import socket
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import awiros_worker  # noqa: E402


@pytest.fixture
def sock_pair():
    a, b = socket.socketpair()
    yield a, b
    a.close()
    b.close()


def test_send_message_prefixes_big_endian_length(sock_pair):
    a, b = sock_pair
    payload = json.dumps({"text": "TN09AB1234"}).encode("utf-8")
    awiros_worker.send_message(a, payload)

    header = awiros_worker.recv_exact(b, 4)
    assert struct.unpack(">I", header)[0] == len(payload)
    assert awiros_worker.recv_exact(b, len(payload)) == payload


def test_recv_exact_joins_partial_chunks(sock_pair):
    a, b = sock_pair
    a.sendall(b"ab")
    a.sendall(b"cd")
    a.sendall(b"ef")
    assert awiros_worker.recv_exact(b, 6) == b"abcdef"


def test_recv_exact_raises_when_peer_closes_early(sock_pair):
    a, b = sock_pair
    a.sendall(b"abc")
    a.close()
    with pytest.raises(ConnectionError):
        awiros_worker.recv_exact(b, 10)


def test_preprocess_output_shape_and_range():
    img = np.full((24, 60, 3), 255, dtype=np.uint8)   # scales to 48x120
    out = awiros_worker.preprocess(img)
    assert out.shape == (3, 48, 320)
    assert np.allclose(out[:, :, :120], 1.0)
    assert np.allclose(out[:, :, 120:], -1.0)       # zero padding -> -1


def test_preprocess_caps_wide_crop():
    img = np.zeros((10, 2000, 3), dtype=np.uint8)
    assert awiros_worker.preprocess(img).shape == (3, 48, 320)


def test_preprocess_rejects_crop_that_would_be_zero_wide():
    img = np.zeros((1000, 2, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        awiros_worker.preprocess(img)
