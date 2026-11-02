
"""
Persistent Awiros ANPR-OCR worker.

This process intentionally loads Paddle/Awiros but does NOT import PyTorch.
The main webcam process can therefore use PyTorch/YOLO without sharing
Paddle/PyTorch CUDA DLLs in the same Windows process.

Protocol over localhost TCP:
  request  = 4-byte big-endian JPEG length + JPEG bytes
  response = 4-byte big-endian JSON length + JSON bytes

Run directly only for debugging:
    python awiros_worker.py
"""

import copy
import json
import socket
import struct
import sys
from pathlib import Path

import cv2
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
HOST = "127.0.0.1"
PORT = 48731

WEIGHTS = SCRIPT_DIR / "models" / "model.safetensors"
DICT_PATH = SCRIPT_DIR / "en_dict.txt"
PADDLEOCR_DIR = SCRIPT_DIR / "PaddleOCR"

CTC_NUM_CLASSES = 64
NRTR_NUM_CLASSES = 67

MODEL_CONFIG = {
    "Architecture": {
        "model_type": "rec",
        "algorithm": "SVTR_HGNet",
        "Transform": None,
        "Backbone": {"name": "PPHGNetV2_B4", "text_rec": True},
        "Head": {
            "name": "MultiHead",
            "out_channels_list": {
                "CTCLabelDecode": CTC_NUM_CLASSES,
                "NRTRLabelDecode": NRTR_NUM_CLASSES,
            },
            "head_list": [
                {
                    "CTCHead": {
                        "Neck": {
                            "name": "svtr",
                            "dims": 120,
                            "depth": 2,
                            "hidden_dims": 120,
                            "kernel_size": [1, 3],
                            "use_guide": True,
                        },
                        "Head": {"fc_decay": 1e-05},
                    }
                },
                {"NRTRHead": {"nrtr_dim": 384, "max_text_length": 25}},
            ],
        },
    },
}

IMAGE_SHAPE = [3, 48, 320]


def recv_exact(conn, n):
    chunks = []
    remaining = n
    while remaining:
        chunk = conn.recv(remaining)
        if not chunk:
            raise ConnectionError("Client disconnected.")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def send_message(conn, payload):
    conn.sendall(struct.pack(">I", len(payload)))
    conn.sendall(payload)


def preprocess(img_bgr):
    _, target_h, target_w = IMAGE_SHAPE
    img_h, img_w = img_bgr.shape[:2]

    if img_h <= 0 or img_w <= 0:
        raise ValueError("Invalid crop.")

    ratio = target_h / img_h
    new_w = min(int(img_w * ratio), target_w)

    if new_w <= 0:
        raise ValueError("Invalid resized width.")

    resized = cv2.resize(
        img_bgr,
        (new_w, target_h),
        interpolation=cv2.INTER_LINEAR,
    )

    if new_w < target_w:
        padded = np.zeros(
            (target_h, target_w, 3),
            dtype=np.uint8,
        )
        padded[:, :new_w, :] = resized
        resized = padded

    img = resized.astype(np.float32) / 255.0
    img = (img - 0.5) / 0.5
    return img.transpose((2, 0, 1))


class AwirosEngine:
    def __init__(self):
        print("[WORKER] Initializing Paddle/Awiros...", flush=True)

        # Importing test.py is safe: test.py does not import torch.
        sys.path.insert(0, str(SCRIPT_DIR))
        from test import _ensure_paddleocr, load_safetensors_to_paddle

        _ensure_paddleocr(str(PADDLEOCR_DIR))

        import paddle
        from ppocr.modeling.architectures import build_model
        from ppocr.postprocess import build_post_process

        self.paddle = paddle

        if paddle.is_compiled_with_cuda():
            paddle.set_device("gpu")
            print("[WORKER] Paddle GPU enabled.", flush=True)
        else:
            # CPU-only Paddle (e.g. plain `paddlepaddle` package) still
            # works, just slower, so don't refuse to start.
            paddle.set_device("cpu")
            print(
                "[WORKER] Paddle has no CUDA support - running OCR on CPU.",
                flush=True,
            )

        self.post_process = build_post_process({
            "name": "CTCLabelDecode",
            "character_dict_path": str(DICT_PATH),
            "use_space_char": True,
        })

        self.model = build_model(
            copy.deepcopy(MODEL_CONFIG)["Architecture"]
        )
        self.model.eval()

        state_dict = load_safetensors_to_paddle(
            paddle,
            str(WEIGHTS),
        )
        self.model.set_state_dict(state_dict)

        print("[WORKER] Awiros weights loaded.", flush=True)
        print("[WORKER] READY", flush=True)

    def recognize(self, image_bytes):
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

        if img is None or img.size == 0:
            return {"text": "", "confidence": 0.0, "error": "decode_failed"}

        tensor = self.paddle.to_tensor(
            np.expand_dims(preprocess(img), axis=0)
        )

        with self.paddle.no_grad():
            preds = self.model(tensor)

        if isinstance(preds, dict):
            pred_tensor = preds.get(
                "ctc",
                next(iter(preds.values())),
            )
        elif isinstance(preds, (list, tuple)):
            pred_tensor = preds[0]
        else:
            pred_tensor = preds

        post_result = self.post_process(
            pred_tensor.numpy()
        )

        if (
            isinstance(post_result, (list, tuple))
            and len(post_result) > 0
        ):
            text, confidence = post_result[0]
        else:
            return {"text": "", "confidence": 0.0}

        return {
            "text": str(text).strip().upper(),
            "confidence": float(confidence) * 100.0,
        }


def main():
    engine = AwirosEngine()

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((HOST, PORT))
    server.listen(1)

    print(f"[WORKER] Listening on {HOST}:{PORT}", flush=True)

    while True:
        conn, addr = server.accept()
        print(f"[WORKER] Client connected: {addr}", flush=True)

        with conn:
            while True:
                try:
                    header = recv_exact(conn, 4)
                    size = struct.unpack(">I", header)[0]

                    if size <= 0 or size > 10_000_000:
                        raise ValueError(f"Invalid image size: {size}")

                    payload = recv_exact(conn, size)
                    result = engine.recognize(payload)

                    encoded = json.dumps(result).encode("utf-8")
                    send_message(conn, encoded)

                except (ConnectionError, OSError):
                    break
                except Exception as exc:
                    result = {
                        "text": "",
                        "confidence": 0.0,
                        "error": repr(exc),
                    }
                    try:
                        send_message(
                            conn,
                            json.dumps(result).encode("utf-8"),
                        )
                    except Exception:
                        break

        print("[WORKER] Client disconnected.", flush=True)


if __name__ == "__main__":
    main()
