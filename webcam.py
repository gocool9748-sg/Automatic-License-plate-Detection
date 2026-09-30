import copy
import os
import re
import cv2
import time
import threading
import socket
import struct
import json
import subprocess
import sys
import numpy as np
import torch
from collections import Counter, defaultdict
from ultralytics import YOLO

# ============================================================
# REAL-TIME LPR V2
# Smooth webcam + background OCR
# ============================================================

YOLO_WEIGHTS = r"G:\LPR\models\best.pt"

CAMERA_INDEX = 0

# YOLO - EXISTING LICENSE PLATE DETECTOR (UNCHANGED)
YOLO_CONFIDENCE = 0.45
YOLO_IMGSZ = 512

# ============================================================
# VEHICLE CLASSIFICATION - UVH-26 INDIAN TRAFFIC MODEL
# ============================================================
# Separate model: does NOT replace or modify the existing plate YOLO.
VEHICLE_WEIGHTS = r"G:\LPR\models\uvh26_vehicle.pt"
VEHICLE_CONFIDENCE = 0.40
VEHICLE_IMGSZ = 640
VEHICLE_INFER_EVERY_N_FRAMES = 2

# Explicit duplicate suppression for the vehicle-classification model.
# UVH-26/Ultralytics already applies NMS internally, but this extra pass
# guarantees that heavily overlapping detections are not drawn as separate
# vehicle boxes in the webcam output.
VEHICLE_DUPLICATE_IOU = 0.50
VEHICLE_DUPLICATE_CONTAINMENT = 0.80

# UVH-26 fine-grained Indian vehicle classes -> project classes.
VEHICLE_CLASS_MAP = {
    "Hatchback": "CAR",
    "Sedan": "CAR",
    "SUV": "CAR",
    "MUV": "CAR",
    "Van": "CAR",
    "LCV": "TRUCK",
    "Truck": "TRUCK",
    "Bus": "BUS",
    "Mini-bus": "BUS",
    "Tempo-traveller": "BUS",
    "Three-wheeler": "AUTO",
    "Two-wheeler": "BIKE",
    "Bicycle": "BIKE",
    "Others": "OTHER",
}

# OCR
OCR_INTERVAL = 0.14          # seconds between OCR jobs
MAX_HISTORY = 18
MIN_VOTES = 2
MIN_OCR_CONFIDENCE = 45
MIN_PLATE_PIXEL_WIDTH = 35
BLUR_VARIANCE_THRESHOLD = 20.0

DEBUG_OCR = False

# Save the exact plate crops that are sent to Awiros OCR.
# These are saved automatically so you can inspect what YOLO is giving OCR.
DEBUG_SAVE_CROPS = True
DEBUG_CROPS_DIR = r"G:\LPR\debug_crops_webcam"
MAX_SAVED_CROPS_PER_TRACK = 20

# OpenCV normally gives the camera image in its real orientation.
# Keep this False for ANPR. Set True only if you want a selfie-style mirror.
MIRROR_CAMERA = False

# Final per-vehicle results are appended here as CSV
# (time,plate,vehicle,confidence). Set to None to disable file logging.
RESULTS_LOG = r"G:\LPR\results.log"

# ============================================================
# INDIAN PLATE FORMAT
# ============================================================

PLATE_REGEX = re.compile(
    r"^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{4}$"
)

LETTER_FOR_DIGIT = {
    "0": "O",
    "1": "I",
    "2": "Z",
    "5": "S",
    "6": "G",
    "8": "B",
}

DIGIT_FOR_LETTER = {
    "O": "0",
    "D": "0",
    "Q": "0",
    "I": "1",
    "L": "1",
    "Z": "2",
    "S": "5",
    "B": "8",
    "G": "6",
    "T": "1",
}

# ============================================================
# SHARED STATE
# ============================================================

# Latest-only OCR request. The camera loop never waits for Awiros.
ocr_request_lock = threading.Lock()
ocr_request = None
ocr_request_event = threading.Event()

result_lock = threading.Lock()

# Latest OCR result for each track.
latest_ocr = {}

# Stable plate for each track.
stable_text = {}

# Historical OCR readings.
track_history = defaultdict(list)
track_confidence = defaultdict(list)
track_width = defaultdict(list)

# Avoid sending the same track to OCR repeatedly.
last_ocr_time = defaultdict(float)

# Keep the sharpest plate crop seen since the last OCR job.
best_crop = {}
best_crop_quality = defaultdict(float)


stop_event = threading.Event()

frame_count = 0

# Latest vehicle detections from the separate UVH-26 model.
# This is independent of the existing plate YOLO state.
vehicle_detections = []

# Debug crop bookkeeping.
saved_crop_count = defaultdict(int)
crop_save_sequence = 0

# Vehicle type votes per plate track (weighted by detection confidence).
track_vehicle_votes = defaultdict(lambda: defaultdict(float))

# Tracks whose final result box has already been printed.
reported_tracks = set()


# ============================================================
# TEXT / PLATE HELPERS
# ============================================================

def clean_text(text):
    if not text:
        return ""

    return re.sub(
        r"[^A-Z0-9]",
        "",
        text.upper()
    )


def is_valid_plate(text):
    return bool(
        text and PLATE_REGEX.fullmatch(text)
    )


def fix_position_characters(text):
    """
    Conservative correction using the Indian plate structure:

        AA 0-99 A-AA 0000
    """

    text = clean_text(text)

    if len(text) not in (8, 9, 10):
        return text

    chars = list(text)

    # State code.
    for i in range(2):
        if chars[i].isdigit():
            chars[i] = LETTER_FOR_DIGIT.get(
                chars[i],
                chars[i]
            )

    # Try both 1-digit and 2-digit registration numbers
    # and 1-3 character series.
    for number_len in (1, 2):
        series_start = 2 + number_len

        for series_len in (1, 2, 3):
            digit_start = series_start + series_len

            if digit_start + 4 != len(chars):
                continue

            candidate = chars[:]

            # Registration number -> digits.
            for i in range(2, series_start):
                if candidate[i].isalpha():
                    candidate[i] = DIGIT_FOR_LETTER.get(
                        candidate[i],
                        candidate[i]
                    )

            # Series -> letters.
            for i in range(series_start, digit_start):
                if candidate[i].isdigit():
                    candidate[i] = LETTER_FOR_DIGIT.get(
                        candidate[i],
                        candidate[i]
                    )

            # Last four -> digits.
            for i in range(digit_start, len(candidate)):
                if candidate[i].isalpha():
                    candidate[i] = DIGIT_FOR_LETTER.get(
                        candidate[i],
                        candidate[i]
                    )

            result = "".join(candidate)

            if is_valid_plate(result):
                return result

    return text


def normalize_plate(raw):
    raw = clean_text(raw)

    if not raw:
        return ""

    fixed = fix_position_characters(raw)

    if is_valid_plate(fixed):
        return fixed

    # Search windows because OCR may add extra characters.
    for length in (10, 9, 8):

        if len(raw) < length:
            continue

        for start in range(
            len(raw) - length + 1
        ):
            candidate = raw[
                start:start + length
            ]

            fixed = fix_position_characters(
                candidate
            )

            if is_valid_plate(fixed):
                return fixed

    return ""


def sharpness(image):
    if image is None or image.size == 0:
        return 0.0

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    return float(
        cv2.Laplacian(
            gray,
            cv2.CV_64F
        ).var()
    )


def sharpen(image):
    kernel = np.array(
        [
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ],
        dtype=np.float32
    )

    return cv2.filter2D(
        image,
        -1,
        kernel
    )


def ensure_black_text_on_white(image):
    if image is None:
        return image

    if np.mean(image) < 127:
        return cv2.bitwise_not(image)

    return image


def plate_quality(image):
    """Score a plate crop so OCR gets the sharpest useful frame."""
    if image is None or image.size == 0:
        return 0.0

    h, w = image.shape[:2]
    if w < MIN_PLATE_PIXEL_WIDTH:
        return 0.0

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sharp = cv2.Laplacian(gray, cv2.CV_64F).var()
    contrast = float(gray.std())

    return (
        min(sharp / 100.0, 5.0)
        + min(contrast / 50.0, 2.0)
        + min(w / 120.0, 2.0)
    )


def correct_10_char_plate(text):
    """
    Correct common OCR confusion for the most common 10-character
    Indian format: AA DD LL DDDD, e.g. RJ14CV0002.
    """
    text = clean_text(text)
    if len(text) != 10:
        return ""

    chars = list(text)
    pattern = "LLDDLLDDDD"

    for i, expected in enumerate(pattern):
        if expected == "L" and chars[i].isdigit():
            chars[i] = LETTER_FOR_DIGIT.get(chars[i], chars[i])
        elif expected == "D" and chars[i].isalpha():
            chars[i] = DIGIT_FOR_LETTER.get(chars[i], chars[i])

    candidate = "".join(chars)
    return candidate if is_valid_plate(candidate) else ""

# ============================================================
# DEBUG CROP SAVING
# ============================================================

def save_best_crop(crop, track_id, quality):
    """
    Save ONE best crop per tracked vehicle.

    The file is overwritten whenever a better crop is selected, so
    the debug folder never fills with hundreds of nearly identical
    images from the same vehicle.
    """
    if not DEBUG_SAVE_CROPS:
        return

    if crop is None or crop.size == 0:
        return

    try:
        os.makedirs(DEBUG_CROPS_DIR, exist_ok=True)

        path = os.path.join(
            DEBUG_CROPS_DIR,
            f"best_plate_id_{track_id:03d}.jpg"
        )

        ok = cv2.imwrite(path, crop)

        if ok:
            print(
                f"[BEST CROP] ID {track_id} | quality={quality:.2f} -> {path}"
            )
        else:
            print(f"[CROP ERROR] Could not write: {path}")

    except Exception as e:
        print(f"[CROP ERROR] {e}")


def save_all_best_crops():
    """Save the final best crop for every track when the program exits."""
    if not DEBUG_SAVE_CROPS:
        return

    try:
        os.makedirs(DEBUG_CROPS_DIR, exist_ok=True)

        saved = 0
        for track_id, crop in best_crop.items():
            if crop is None or crop.size == 0:
                continue

            path = os.path.join(
                DEBUG_CROPS_DIR,
                f"best_plate_id_{track_id:03d}.jpg"
            )

            if cv2.imwrite(path, crop):
                saved += 1
                print(
                    f"[FINAL BEST CROP] ID {track_id} -> {path}"
                )

        print(f"[DEBUG] Saved {saved} final best plate crop(s).")

    except Exception as e:
        print(f"[CROP ERROR] {e}")


def clean_debug_crop_folder():
    """Remove old debug crops so each run starts clean."""
    if not DEBUG_SAVE_CROPS:
        return

    try:
        os.makedirs(DEBUG_CROPS_DIR, exist_ok=True)

        for name in os.listdir(DEBUG_CROPS_DIR):
            if (
                name.startswith("crop_")
                or name.startswith("latest_id_")
                or name.startswith("best_plate_")
            ) and name.lower().endswith(".jpg"):
                try:
                    os.remove(os.path.join(DEBUG_CROPS_DIR, name))
                except OSError:
                    pass

    except Exception as e:
        print(f"[CROP CLEANUP WARNING] {e}")


# ============================================================
# AWIROS ANPR-OCR - SEPARATE PROCESS
# ============================================================
#
# IMPORTANT:
# YOLO/PyTorch and Paddle/Awiros cannot safely share the same
# Windows CUDA/cuDNN DLLs in this environment.
#
# Therefore Awiros runs in a completely separate Python process.
# The main process remains PyTorch/YOLO-only.
# Plate crops are sent over localhost TCP.
# ============================================================

AWIROS_WORKER = r"G:\LPR\awiros_worker.py"
AWIROS_HOST = "127.0.0.1"
AWIROS_PORT = 48731
AWIROS_START_TIMEOUT = 30.0


class AwirosOCR:
    """Client for the separate Paddle/Awiros worker process."""

    def __init__(
        self,
        worker_path=AWIROS_WORKER,
        host=AWIROS_HOST,
        port=AWIROS_PORT,
    ):
        self.worker_path = worker_path
        self.host = host
        self.port = port
        self.process = None
        self.sock = None
        self.socket_lock = threading.Lock()

        print("[Awiros] Starting separate OCR worker...")
        self.start()
        print("[Awiros] Worker connected.")

    def _recv_exact(self, n):
        chunks = []
        remaining = n

        while remaining:
            chunk = self.sock.recv(remaining)

            if not chunk:
                raise ConnectionError(
                    "Awiros worker disconnected."
                )

            chunks.append(chunk)
            remaining -= len(chunk)

        return b"".join(chunks)

    def _connect(self):
        sock = socket.socket(
            socket.AF_INET,
            socket.SOCK_STREAM
        )

        sock.settimeout(1.0)

        try:
            sock.connect(
                (self.host, self.port)
            )
        except Exception:
            sock.close()
            raise

        # No long blocking timeout for OCR itself.
        sock.settimeout(5.0)
        self.sock = sock

    def start(self):
        if not os.path.isfile(self.worker_path):
            raise FileNotFoundError(
                f"Awiros worker not found: {self.worker_path}"
            )

        # Start a completely separate Python process.
        # It imports Paddle but never imports PyTorch.
        self.process = subprocess.Popen(
            [
                sys.executable,
                self.worker_path,
            ],
            cwd=os.path.dirname(self.worker_path),
        )

        deadline = time.perf_counter() + AWIROS_START_TIMEOUT

        last_error = None

        while time.perf_counter() < deadline:

            if self.process.poll() is not None:
                raise RuntimeError(
                    "Awiros worker exited during startup "
                    f"with code {self.process.returncode}."
                )

            try:
                self._connect()
                return
            except Exception as e:
                last_error = e
                time.sleep(0.25)

        self.process.terminate()

        raise TimeoutError(
            "Timed out waiting for Awiros OCR worker. "
            f"Last error: {last_error}"
        )

    def recognize(self, plate):
        """
        Send one OpenCV BGR crop to the Paddle/Awiros process.

        Returns:
            (text, confidence_0_to_100)
        """

        if plate is None or plate.size == 0:
            return "", 0.0

        h, w = plate.shape[:2]

        if w < MIN_PLATE_PIXEL_WIDTH:
            return "", 0.0

        # Don't waste OCR time on very blurred crops.
        if sharpness(plate) < BLUR_VARIANCE_THRESHOLD:
            return "", 0.0

        ok, encoded = cv2.imencode(
            ".jpg",
            plate,
            [
                cv2.IMWRITE_JPEG_QUALITY,
                95,
            ],
        )

        if not ok:
            return "", 0.0

        payload = encoded.tobytes()

        if len(payload) > 10_000_000:
            return "", 0.0

        with self.socket_lock:

            try:
                self.sock.sendall(
                    struct.pack(
                        ">I",
                        len(payload)
                    )
                )

                self.sock.sendall(payload)

                header = self._recv_exact(4)

                response_size = struct.unpack(
                    ">I",
                    header
                )[0]

                if (
                    response_size <= 0 or
                    response_size > 1_000_000
                ):
                    raise ValueError(
                        f"Invalid worker response size: "
                        f"{response_size}"
                    )

                response = self._recv_exact(
                    response_size
                )

                data = json.loads(
                    response.decode("utf-8")
                )

                if data.get("error"):
                    if DEBUG_OCR:
                        print(
                            "[Awiros worker error]",
                            data["error"]
                        )
                    return "", 0.0

                raw_text = str(
                    data.get("text", "")
                ).strip().upper()

                confidence = float(
                    data.get("confidence", 0.0)
                )

                # Parent process owns plate normalization and temporal
                # stabilization.
                plate_text = normalize_plate(
                    raw_text
                )

                return (
                    plate_text,
                    confidence
                )

            except Exception as e:

                if DEBUG_OCR:
                    print(
                        "[Awiros client ERROR]",
                        repr(e)
                    )

                return "", 0.0

    def close(self):
        try:
            if self.sock is not None:
                self.sock.close()
        except Exception:
            pass

        self.sock = None

        if self.process is not None:

            if self.process.poll() is None:

                try:
                    self.process.terminate()
                    self.process.wait(
                        timeout=3
                    )
                except Exception:

                    try:
                        self.process.kill()
                    except Exception:
                        pass

            self.process = None


# Global OCR client.
awiros_ocr = None


def perform_ocr(plate):
    if awiros_ocr is None:
        return "", 0.0

    return awiros_ocr.recognize(plate)

# ============================================================
# TEMPORAL VOTING
# ============================================================

def plate_pattern_for_length(length):
    if length == 10:
        return "LLDDLLDDDD"
    if length == 9:
        return "LLDLLDDDD"
    if length == 8:
        return "LLDLDDDD"
    return ""


def character_is_allowed(ch, expected_type):
    if expected_type == "L":
        return ch.isalpha()
    if expected_type == "D":
        return ch.isdigit()
    return True


def position_correct(ch, expected_type):
    if not ch:
        return ch

    ch = ch.upper()

    if expected_type == "L" and ch.isdigit():
        return LETTER_FOR_DIGIT.get(ch, ch)

    if expected_type == "D" and ch.isalpha():
        return DIGIT_FOR_LETTER.get(ch, ch)

    return ch


def character_vote(history, confidences):
    """
    Combine character positions across recent complete OCR
    readings instead of requiring the entire string to match.
    """

    if not history:
        return ""

    lengths = Counter(
        len(x) for x in history
        if len(x) in (8, 9, 10)
    )

    if not lengths:
        return ""

    target_len = lengths.most_common(1)[0][0]

    candidates = [
        (text, conf)
        for text, conf in zip(history, confidences)
        if len(text) == target_len
    ]

    if not candidates:
        return ""

    pattern = plate_pattern_for_length(target_len)

    if not pattern:
        return ""

    result = []

    for pos in range(target_len):
        votes = defaultdict(float)
        expected = pattern[pos]

        for text, conf in candidates:
            ch = text[pos]
            ch = position_correct(ch, expected)

            if not character_is_allowed(ch, expected):
                continue

            weight = (
                0.5 +
                min(max(conf, 0.0), 100.0) / 100.0
            )

            votes[ch] += weight

        if not votes:
            return ""

        result.append(
            max(votes, key=votes.get)
        )

    result = "".join(result)

    return result if is_valid_plate(result) else ""


def update_track(
    track_id,
    text,
    confidence,
    box_width
):
    if not text:
        return

    text = normalize_plate(text)

    if not text:
        return

    if confidence < MIN_OCR_CONFIDENCE:
        return

    with result_lock:
        history = track_history[track_id]
        confs = track_confidence[track_id]
        widths = track_width[track_id]

        history.append(text)
        confs.append(confidence)
        widths.append(max(box_width, 1))

        if len(history) > MAX_HISTORY:
            history.pop(0)
            confs.pop(0)
            widths.pop(0)

        # Exact whole-string voting is retained as a fallback.
        exact_scores = defaultdict(float)

        for t, conf, width in zip(
            history,
            confs,
            widths
        ):
            width_weight = min(
                max(width / 100.0, 0.5),
                3.0
            )

            conf_weight = (
                0.5 +
                min(max(conf, 0.0), 100.0) / 100.0
            )

            exact_scores[t] += (
                width_weight * conf_weight
            )

        exact_best = (
            max(exact_scores, key=exact_scores.get)
            if exact_scores
            else ""
        )

        # Character-level voting is the main improvement.
        # Example:
        # RJ12CV0002
        # RJ14CV0002
        # RJ14CV0002
        # -> RJ14CV0002
        char_best = character_vote(
            history,
            confs
        )

        if len(history) >= MIN_VOTES and char_best:
            stable_text[track_id] = char_best

        elif exact_best:
            support = sum(
                1 for t in history
                if t == exact_best
            )

            if support >= MIN_VOTES:
                stable_text[track_id] = exact_best

        latest_ocr[track_id] = (
            text,
            confidence
        )


# ============================================================
# BACKGROUND OCR WORKER
# ============================================================

def ocr_worker():
    """
    Latest-frame OCR worker.

    If the bus moves while Awiros is working, old OCR work is
    discarded and the newest plate crop is processed next.
    """

    global ocr_request

    while not stop_event.is_set():

        ocr_request_event.wait(
            timeout=0.05
        )

        if stop_event.is_set():
            break

        with ocr_request_lock:
            item = ocr_request
            ocr_request = None
            ocr_request_event.clear()

        if item is None:
            continue

        track_id, crop, box_width = item

        try:
            text, confidence = perform_ocr(crop)

            if text:
                update_track(
                    track_id,
                    text,
                    confidence,
                    box_width
                )

        except Exception as e:
            if DEBUG_OCR:
                print(
                    "[OCR WORKER ERROR]",
                    e
                )


# ============================================================
# VEHICLE CLASSIFICATION HELPERS
# ============================================================

def map_vehicle_class(original_class):
    """Map UVH-26's detailed Indian vehicle classes to project classes."""
    return VEHICLE_CLASS_MAP.get(str(original_class), "OTHER")


def vehicle_box_iou(box_a, box_b):
    """Calculate IoU between two xyxy vehicle boxes."""
    ax1, ay1, ax2, ay2 = box_a[:4]
    bx1, by1, bx2, by2 = box_b[:4]

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection

    return intersection / union if union > 0.0 else 0.0


def vehicle_containment_ratio(box_a, box_b):
    """
    Return the larger fraction of one box covered by the other.

    This catches duplicate boxes where one prediction is mostly contained
    inside another even when their IoU is slightly below the NMS threshold.
    """
    ax1, ay1, ax2, ay2 = box_a[:4]
    bx1, by1, bx2, by2 = box_b[:4]

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    smaller_area = min(area_a, area_b)
    return intersection / smaller_area if smaller_area > 0.0 else 0.0


def suppress_duplicate_vehicle_detections(detections):
    """
    Keep one highest-confidence box for each physical vehicle.

    Detections are compared across all vehicle classes because the project
    classes are coarser than UVH-26's original classes. For example, two
    overlapping predictions such as Sedan/CAR and SUV/CAR are still treated
    as duplicates.
    """
    if len(detections) <= 1:
        return detections

    # Highest confidence first so the surviving box is always the strongest
    # prediction for an overlapping vehicle.
    candidates = sorted(
        detections,
        key=lambda d: float(d[5]),
        reverse=True,
    )

    kept = []

    for candidate in candidates:
        candidate_box = candidate[:4]
        is_duplicate = False

        for existing in kept:
            existing_box = existing[:4]
            iou = vehicle_box_iou(candidate_box, existing_box)
            containment = vehicle_containment_ratio(
                candidate_box,
                existing_box,
            )

            if (
                iou >= VEHICLE_DUPLICATE_IOU
                or containment >= VEHICLE_DUPLICATE_CONTAINMENT
            ):
                is_duplicate = True
                break

        if not is_duplicate:
            kept.append(candidate)

    # Keep drawing order stable and left-to-right for a cleaner display.
    kept.sort(key=lambda d: (d[0], d[1]))
    return kept


def point_inside_box(px, py, box):
    """Return True when a point lies inside an xyxy vehicle box."""
    vx1, vy1, vx2, vy2 = box
    return vx1 <= px <= vx2 and vy1 <= py <= vy2


def find_vehicle_for_plate(plate_box):
    """
    Associate an existing license-plate detection with the vehicle
    containing the plate center. Existing plate detection/OCR is untouched.
    """
    if not vehicle_detections:
        return "UNKNOWN", 0.0

    x1, y1, x2, y2 = plate_box
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    best = None
    best_score = -1.0

    for vx1, vy1, vx2, vy2, vehicle_type, confidence in vehicle_detections:
        if point_inside_box(cx, cy, (vx1, vy1, vx2, vy2)):
            # Prefer the smallest containing vehicle box.
            area = max((vx2 - vx1) * (vy2 - vy1), 1.0)
            score = confidence + 1000000.0 / area
            if score > best_score:
                best_score = score
                best = (vehicle_type, confidence)

    if best is None:
        return "UNKNOWN", 0.0

    return best


def draw_vehicle_box(frame, box, vehicle_type, confidence):
    """Draw the Indian vehicle classification without changing plate drawing."""
    x1, y1, x2, y2 = map(int, box)

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        (255, 140, 0),
        2
    )

    label = f"{vehicle_type} {confidence:.2f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.60
    thickness = 2

    (tw, th), _ = cv2.getTextSize(
        label, font, scale, thickness
    )

    top = max(0, y1 - th - 8)

    cv2.rectangle(
        frame,
        (x1, top),
        (x1 + tw + 8, y1),
        (255, 140, 0),
        -1
    )

    cv2.putText(
        frame,
        label,
        (x1 + 4, y1 - 5),
        font,
        scale,
        (0, 0, 0),
        thickness,
        cv2.LINE_AA
    )


# ============================================================
# FINAL RESULT REPORTING (one clean box per vehicle)
# ============================================================

def vote_vehicle(track_id, vehicle_type, confidence):
    """Accumulate vehicle-type votes so one bad frame can't flip the label."""
    if vehicle_type != "UNKNOWN":
        track_vehicle_votes[track_id][vehicle_type] += confidence


def get_track_vehicle(track_id):
    """Vehicle type with the highest accumulated vote for this track."""
    votes = track_vehicle_votes.get(track_id)
    if not votes:
        return "UNKNOWN"
    return max(votes, key=votes.get)


def get_track_confidence(track_id, plate):
    """Average OCR confidence (0-100) of readings that support the stable plate."""
    with result_lock:
        confs = [
            c for t, c in zip(
                track_history[track_id],
                track_confidence[track_id]
            )
            if t == plate
        ]

        if not confs:
            confs = list(track_confidence[track_id])

    return sum(confs) / len(confs) if confs else 0.0


def format_result_box(plate, vehicle, confidence, timestamp):
    lines = [
        f"Plate: {plate}",
        f"Vehicle: {vehicle}",
        f"Confidence: {confidence:.0f}%",
        f"Time: {timestamp}",
    ]
    inner = max(max(len(l) for l in lines) + 2, 26)

    out = ["┌" + "─" * inner + "┐"]
    for l in lines:
        out.append("│ " + l.ljust(inner - 1) + "│")
    out.append("└" + "─" * inner + "┘")
    return "\n".join(out)


def report_vehicle(track_id):
    """Print the final result box once per track, when the plate is stable."""
    if track_id in reported_tracks:
        return

    with result_lock:
        plate = stable_text.get(track_id, "")

    vehicle = get_track_vehicle(track_id)

    # Wait until both plate and vehicle type are known.
    if not plate or vehicle == "UNKNOWN":
        return

    confidence = get_track_confidence(track_id, plate)
    timestamp = time.strftime("%H:%M:%S")

    box = format_result_box(plate, vehicle, confidence, timestamp)
    print("\n" + box + "\n")

    if RESULTS_LOG:
        try:
            with open(RESULTS_LOG, "a", encoding="utf-8") as f:
                f.write(
                    f"{timestamp},{plate},{vehicle},{confidence:.0f}\n"
                )
        except OSError:
            pass

    reported_tracks.add(track_id)


# ============================================================
# DRAW
# ============================================================

def draw_result(
    frame,
    x1,
    y1,
    x2,
    y2,
    track_id,
    vehicle_type="UNKNOWN",
    vehicle_confidence=0.0
):

    with result_lock:

        stable = stable_text.get(
            track_id,
            ""
        )

        latest = latest_ocr.get(
            track_id
        )

    # Green box when stable, yellow otherwise.
    if stable:
        color = (0, 255, 0)
        plate_label = (
            f"{stable} | ID {track_id}"
        )
    else:
        color = (0, 220, 255)

        if latest:
            plate_label = (
                f"Reading: {latest[0]}"
            )
        else:
            plate_label = (
                f"Reading... | ID {track_id}"
            )

    # Add vehicle type to the existing plate label.
    if vehicle_type != "UNKNOWN":
        label = f"{vehicle_type} | {plate_label}"
    else:
        label = plate_label

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        color,
        2
    )

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.65
    thickness = 2

    (tw, th), _ = cv2.getTextSize(
        label,
        font,
        scale,
        thickness
    )

    top = max(
        0,
        y1 - th - 12
    )

    cv2.rectangle(
        frame,
        (x1, top),
        (x1 + tw + 8, y1),
        color,
        -1
    )

    cv2.putText(
        frame,
        label,
        (x1 + 4, y1 - 5),
        font,
        scale,
        (0, 0, 0),
        thickness,
        cv2.LINE_AA
    )


# ============================================================
# MAIN
# ============================================================

def main():

    global frame_count
    global awiros_ocr

    # Make sure box-drawing characters print correctly on Windows.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    stop_event.clear()

    # Start each run with a clean debug folder.
    if DEBUG_SAVE_CROPS:
        clean_debug_crop_folder()
        print(f"[DEBUG] ONE best crop per vehicle will be saved to: {DEBUG_CROPS_DIR}")

    print("=" * 65)
    print("REAL-TIME INDIAN LICENSE PLATE RECOGNITION - AWIROS")
    print("=" * 65)

    print("[1] Loading YOLO...")

    # EXISTING PLATE YOLO - UNCHANGED
    model = YOLO(YOLO_WEIGHTS)

    print("[1A] Loading UVH-26 Indian vehicle classification model...")

    if not os.path.isfile(VEHICLE_WEIGHTS):
        print(f"[ERROR] Vehicle model not found: {VEHICLE_WEIGHTS}")
        print("[ERROR] Download/copy uvh26_vehicle.pt to G:\\LPR\\models first.")
        return

    vehicle_model = YOLO(VEHICLE_WEIGHTS)
    print(f"[Vehicle] Weights: {VEHICLE_WEIGHTS}")
    print(f"[Vehicle] Classes: {vehicle_model.names}")

    if torch.cuda.is_available():
        device = 0
        print("[GPU] CUDA available - using GPU 0")
    else:
        device = "cpu"
        print("[GPU] CUDA unavailable - using CPU")

    print("[2] Starting Awiros ANPR-OCR worker...")
    try:
        awiros_ocr = AwirosOCR(
            worker_path=AWIROS_WORKER,
            host=AWIROS_HOST,
            port=AWIROS_PORT,
        )
    except Exception as e:
        print()
        print("[ERROR] Awiros OCR worker could not be started.")
        print(f"[ERROR] {e}")
        print()
        print("Check:")
        print("  - G:\\LPR\\awiros_worker.py exists")
        print("  - models\\model.safetensors exists")
        print("  - en_dict.txt exists")
        print("  - PaddleOCR folder exists")
        print("  - test.py is present in G:\\LPR")
        return

    print("[3] Opening laptop camera...")

    cap = cv2.VideoCapture(
        CAMERA_INDEX,
        cv2.CAP_MSMF
    )

    if not cap.isOpened():
        cap = cv2.VideoCapture(
            CAMERA_INDEX
        )

    if not cap.isOpened():
        print()
        print("[ERROR] Camera could not be opened.")
        print("Try CAMERA_INDEX = 1.")
        return

    # Lower resolution keeps webcam inference responsive.
    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        960
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        540
    )

    # Ask the camera for MJPG when supported.
    cap.set(
        cv2.CAP_PROP_FOURCC,
        cv2.VideoWriter_fourcc(*"MJPG")
    )

    # Start OCR worker.
    worker = threading.Thread(
        target=ocr_worker,
        daemon=True
    )

    worker.start()

    print("[4] Camera started.")
    print()
    print("Controls:")
    print("  Q / ESC -> quit")
    print("  R       -> reset OCR history")
    print()
    print("Awiros OCR runs in a background thread.")
    print("The camera loop should remain smooth.")
    print()

    previous_time = time.perf_counter()
    fps = 0.0

    try:

        while True:

            ret, frame = cap.read()

            if not ret:
                print(
                    "[ERROR] Failed to read camera frame."
                )
                break

            if MIRROR_CAMERA:
                frame = cv2.flip(frame, 1)

            frame_count += 1

            # ------------------------------------------------
            # UVH-26 VEHICLE CLASSIFICATION
            # ------------------------------------------------
            # Runs separately from the existing plate YOLO.
            # It is intentionally throttled to every N frames for speed.
            global vehicle_detections

            if frame_count % VEHICLE_INFER_EVERY_N_FRAMES == 0:
                vehicle_results = vehicle_model.predict(
                    frame,
                    conf=VEHICLE_CONFIDENCE,
                    imgsz=VEHICLE_IMGSZ,
                    device=device,
                    verbose=False
                )

                new_vehicle_detections = []
                vehicle_result = vehicle_results[0]

                if vehicle_result.boxes is not None:
                    for vbox in vehicle_result.boxes:
                        vxyxy = (
                            vbox.xyxy[0]
                            .cpu()
                            .numpy()
                        )

                        vx1, vy1, vx2, vy2 = map(int, vxyxy)

                        vx1 = max(0, vx1)
                        vy1 = max(0, vy1)
                        vx2 = min(frame.shape[1] - 1, vx2)
                        vy2 = min(frame.shape[0] - 1, vy2)

                        if vx2 <= vx1 or vy2 <= vy1:
                            continue

                        vclass_id = int(vbox.cls[0])
                        vconfidence = float(vbox.conf[0])
                        original_class = vehicle_model.names[vclass_id]
                        vehicle_type = map_vehicle_class(original_class)

                        new_vehicle_detections.append(
                            (
                                vx1, vy1, vx2, vy2,
                                vehicle_type,
                                vconfidence
                            )
                        )

                # UVH-26 can occasionally return two highly overlapping
                # boxes for the same physical vehicle. Remove those before
                # storing/drawing them and before plate-to-vehicle matching.
                before_count = len(new_vehicle_detections)
                vehicle_detections = suppress_duplicate_vehicle_detections(
                    new_vehicle_detections
                )
                after_count = len(vehicle_detections)

                if after_count < before_count:
                    print(
                        f"[Vehicle NMS] Removed {before_count - after_count} "
                        f"duplicate box(es). Kept {after_count}."
                    )

            # Draw the latest vehicle detections.
            for (
                vx1, vy1, vx2, vy2,
                vehicle_type, vconfidence
            ) in vehicle_detections:
                draw_vehicle_box(
                    frame,
                    (vx1, vy1, vx2, vy2),
                    vehicle_type,
                    vconfidence
                )

            # ------------------------------------------------
            # EXISTING PLATE YOLO / tracking - UNCHANGED
            # ------------------------------------------------

            results = model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                conf=YOLO_CONFIDENCE,
                imgsz=YOLO_IMGSZ,
                device=device,
                verbose=False
            )

            result = results[0]

            if result.boxes is not None:

                boxes = result.boxes

                for i in range(
                    len(boxes)
                ):

                    xyxy = (
                        boxes.xyxy[i]
                        .cpu()
                        .numpy()
                    )

                    x1, y1, x2, y2 = map(
                        int,
                        xyxy
                    )

                    x1 = max(
                        0,
                        x1
                    )

                    y1 = max(
                        0,
                        y1
                    )

                    x2 = min(
                        frame.shape[1] - 1,
                        x2
                    )

                    y2 = min(
                        frame.shape[0] - 1,
                        y2
                    )

                    if (
                        x2 <= x1 or
                        y2 <= y1
                    ):
                        continue

                    if boxes.id is not None:
                        track_id = int(
                            boxes.id[i].item()
                        )
                    else:
                        track_id = i

                    box_width = x2 - x1

                    # Associate this existing plate detection with
                    # the vehicle box from UVH-26.
                    vehicle_type, vehicle_confidence = find_vehicle_for_plate(
                        (x1, y1, x2, y2)
                    )

                    # Accumulate vehicle-type votes, print the final
                    # result box once the plate is stable, and use the
                    # voted type for the on-screen label.
                    vote_vehicle(
                        track_id,
                        vehicle_type,
                        vehicle_confidence
                    )
                    report_vehicle(track_id)
                    vehicle_type = get_track_vehicle(track_id)

                    # ------------------------------------------------
                    # OCR scheduling
                    #
                    # IMPORTANT:
                    # We DO NOT wait for Awiros here.
                    # ------------------------------------------------

                    now = time.perf_counter()

                    should_ocr = (
                        now -
                        last_ocr_time[track_id]
                        >= OCR_INTERVAL
                    )

                    crop = frame[
                        y1:y2,
                        x1:x2
                    ].copy()

                    # Keep the sharpest crop for this track.
                    quality = plate_quality(crop)

                    if quality > best_crop_quality[track_id]:
                        best_crop[track_id] = crop
                        best_crop_quality[track_id] = quality

                    if should_ocr:

                        selected_crop = best_crop.get(track_id)

                        if selected_crop is not None:

                            # Send the best crop seen so far to Awiros.
                            # Keep it in memory so it can continue improving
                            # until the vehicle leaves the camera view.
                            global ocr_request

                            with ocr_request_lock:
                                ocr_request = (
                                    track_id,
                                    selected_crop.copy(),
                                    box_width
                                )
                                ocr_request_event.set()

                        # Do NOT reset best_crop here. We want the best
                        # image across the entire lifetime of this track.

                        last_ocr_time[
                            track_id
                        ] = now

                    draw_result(
                        frame,
                        x1,
                        y1,
                        x2,
                        y2,
                        track_id,
                        vehicle_type,
                        vehicle_confidence
                    )

            # ------------------------------------------------
            # FPS
            # ------------------------------------------------

            current_time = (
                time.perf_counter()
            )

            delta = (
                current_time -
                previous_time
            )

            previous_time = current_time

            if delta > 0:
                instant_fps = 1.0 / delta

                # Smooth FPS.
                fps = (
                    0.9 * fps +
                    0.1 * instant_fps
                )

            cv2.rectangle(
                frame,
                (0, 0),
                (310, 45),
                (0, 0, 0),
                -1
            )

            cv2.putText(
                frame,
                f"Real-Time LPR | Awiros | FPS: {fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

            cv2.imshow(
                "Real-Time LPR - Awiros OCR",
                frame
            )

            key = cv2.waitKey(
                1
            ) & 0xFF

            if key in (
                ord("q"),
                27
            ):
                break

            if key == ord("r"):

                with result_lock:
                    track_history.clear()
                    track_confidence.clear()
                    track_width.clear()
                    stable_text.clear()
                    latest_ocr.clear()

                last_ocr_time.clear()
                best_crop.clear()
                best_crop_quality.clear()
                saved_crop_count.clear()
                track_vehicle_votes.clear()
                reported_tracks.clear()

                print(
                    "[RESET] OCR history cleared."
                )

    finally:

        # Save exactly ONE best crop for every tracked vehicle seen
        # during this run, before clearing/closing anything.
        if DEBUG_SAVE_CROPS:
            save_all_best_crops()

        stop_event.set()

        ocr_request_event.set()

        if awiros_ocr is not None:
            awiros_ocr.close()
            awiros_ocr = None

        cap.release()

        cv2.destroyAllWindows()

        print()
        print("LPR stopped.")


if __name__ == "__main__":
    main()