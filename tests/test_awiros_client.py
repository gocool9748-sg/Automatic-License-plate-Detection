"""Unit tests for the AwirosOCR client's socket protocol in lpr.py.

The client sends each plate crop to the OCR worker as a 4-byte big-endian
length followed by a JPEG, and reads back a length-prefixed JSON reply.
_recv_exact must keep reading until it has exactly n bytes and raise if the
worker disconnects; _query must decode and normalise a good reply, flag
banner text, and turn worker errors, junk text, bad sizes and dropped
connections into ("", 0.0) instead of raising. A socketpair stands in for
the worker, so no subprocess or Paddle is needed. torch and ultralytics are
stubbed out.
"""

import json
import socket
import struct
import sys
import threading
import types
from pathlib import Path

import cv2
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


@pytest.fixture
def pair():
    """(client, worker_socket) wired together without starting a process."""
    client_sock, worker_sock = socket.socketpair()
    client_sock.settimeout(2.0)
    worker_sock.settimeout(2.0)

    client = object.__new__(lpr.AwirosOCR)   # skip __init__ (it spawns the worker)
    client.sock = client_sock
    client.socket_lock = threading.Lock()
    client.last_junk = False
    client.process = None

    yield client, worker_sock

    client_sock.close()
    worker_sock.close()


def reply(worker_sock, payload):
    body = json.dumps(payload).encode("utf-8")
    worker_sock.sendall(struct.pack(">I", len(body)) + body)


def read_request(worker_sock):
    size = struct.unpack(">I", worker_sock.recv(4, socket.MSG_WAITALL))[0]
    return worker_sock.recv(size, socket.MSG_WAITALL)


def plate_image():
    return np.full((24, 80, 3), 200, dtype=np.uint8)


# ---------------- _recv_exact ----------------

def test_recv_exact_joins_partial_chunks(pair):
    client, worker = pair

    def send_slowly():
        for part in (b"ab", b"cde", b"f"):
            worker.sendall(part)

    t = threading.Thread(target=send_slowly)
    t.start()
    assert client._recv_exact(6) == b"abcdef"
    t.join()


def test_recv_exact_leaves_extra_bytes_for_next_read(pair):
    client, worker = pair
    worker.sendall(b"1234XY")
    assert client._recv_exact(4) == b"1234"
    assert client._recv_exact(2) == b"XY"


def test_recv_exact_raises_when_worker_disconnects(pair):
    client, worker = pair
    worker.sendall(b"ab")
    worker.close()
    with pytest.raises(ConnectionError):
        client._recv_exact(4)


# ---------------- _query ----------------

def test_query_sends_length_prefixed_jpeg(pair):
    client, worker = pair
    reply(worker, {"text": "TN09AB1234", "confidence": 88.5})

    client._query(plate_image())

    jpeg = read_request(worker)
    decoded = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape == (24, 80, 3)


def test_query_normalises_text_and_returns_confidence(pair):
    client, worker = pair
    reply(worker, {"text": " tn 09 ab 1234 ", "confidence": 88.5})

    assert client._query(plate_image()) == ("TN09AB1234", 88.5)
    assert client.last_junk is False


def test_query_worker_error_gives_empty_result(pair):
    client, worker = pair
    reply(worker, {"error": "inference failed"})
    assert client._query(plate_image()) == ("", 0.0)


def test_query_banner_text_sets_last_junk(pair):
    client, worker = pair
    reply(worker, {"text": "SRM COLLEGE", "confidence": 95.0})

    assert client._query(plate_image()) == ("", 0.0)
    assert client.last_junk is True


def test_query_unreadable_is_junk_but_not_banner(pair):
    client, worker = pair
    reply(worker, {"text": "UNREADABLE", "confidence": 60.0})

    assert client._query(plate_image()) == ("", 0.0)
    assert client.last_junk is False


@pytest.mark.parametrize("size", [0, 1_000_001])
def test_query_rejects_bad_reply_size(pair, size):
    client, worker = pair
    worker.sendall(struct.pack(">I", size))
    assert client._query(plate_image()) == ("", 0.0)


def test_query_survives_worker_disconnect(pair):
    client, worker = pair
    worker.shutdown(socket.SHUT_WR)   # worker stops replying mid-request
    assert client._query(plate_image()) == ("", 0.0)
