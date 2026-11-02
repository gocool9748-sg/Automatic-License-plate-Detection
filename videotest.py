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

# Paths are resolved relative to this script, so the project works from
# wherever it is cloned.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

YOLO_WEIGHTS = os.path.join(BASE_DIR, "models", "best.pt")

# Test video path - change this to any video you want to test.
VIDEO_PATH = os.path.join(BASE_DIR, "videos", "srm5.mp4")

# Keep video in its original orientation.
MIRROR_CAMERA = False

# YOLO - EXISTING LICENSE PLATE DETECTOR (UNCHANGED)
YOLO_CONFIDENCE = 0.45
YOLO_IMGSZ = 512

# ============================================================
# VEHICLE CLASSIFICATION - UVH-26 INDIAN TRAFFIC MODEL
# ============================================================
# Separate model: does NOT replace or modify the existing plate YOLO.
VEHICLE_WEIGHTS = os.path.join(BASE_DIR, "models", "uvh26_vehicle.pt")
VEHICLE_CONFIDENCE = 0.25
VEHICLE_IMGSZ = 640
VEHICLE_INFER_EVERY_N_FRAMES = 2

# Clean one final record per physical vehicle/plate instead of repeating
# the same vehicle on every video frame.
VEHICLE_EVENT_COOLDOWN = 10.0  # seconds before the same plate can be recorded again

# Final per-vehicle results are appended here as CSV
# (timestamp,plate,vehicle,confidence). Set to None to disable file logging.
RESULTS_LOG = os.path.join(BASE_DIR, "results.log")

# Print the per-detection "[VEHICLE DEBUG] raw=... kept=..." line every
# inference. Keep False so the console only shows the clean result boxes.
VEHICLE_DEBUG_PRINT = False

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
    "tempo-traveller": "BUS",
    "Three-wheeler": "AUTO",
    "Two-wheeler": "BIKE",
    "Bicycle": "BIKE",
    "bicycle": "BIKE",
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
DEBUG_CROPS_DIR = os.path.join(BASE_DIR, "debug_crops_webcam")
MAX_SAVED_CROPS_PER_TRACK = 20


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

# Final clean vehicle events for this run.
# Keyed by plate number; values contain the latest confirmed vehicle data.
confirmed_vehicle_events = {}
recent_vehicle_event_time = {}
last_vehicle_event = None

# Vehicle type votes per plate track (weighted by detection confidence).
track_vehicle_votes = defaultdict(lambda: defaultdict(float))


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

AWIROS_WORKER = os.path.join(BASE_DIR, "awiros_worker.py")
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


def point_inside_box(px, py, box):
    """Return True when a point lies inside an xyxy vehicle box."""
    vx1, vy1, vx2, vy2 = box
    return vx1 <= px <= vx2 and vy1 <= py <= vy2


def box_iou(box_a, box_b):
    """Compute IoU for two xyxy boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    intersection = iw * ih

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0


def suppress_duplicate_vehicle_detections(detections, iou_threshold=0.35, containment_threshold=0.65):
    """Keep one strongest UVH-26 detection for each physical vehicle.

    UVH-26 can occasionally return several overlapping boxes for the same
    vehicle, sometimes even with different fine-grained classes.  This
    class-agnostic suppression keeps the highest-confidence box and removes
    strong geometric duplicates before anything is drawn or associated with
    a plate.
    """
    if not detections:
        return []

    # Highest confidence first. Each item is:
    # (x1, y1, x2, y2, vehicle_type, confidence)
    ordered = sorted(detections, key=lambda d: float(d[5]), reverse=True)
    kept = []

    for candidate in ordered:
        cx1, cy1, cx2, cy2, ctype, cconf = candidate
        cbox = (float(cx1), float(cy1), float(cx2), float(cy2))
        c_area = max(1.0, (cx2 - cx1) * (cy2 - cy1))
        is_duplicate = False

        for existing in kept:
            ex1, ey1, ex2, ey2, etype, econf = existing
            ebox = (float(ex1), float(ey1), float(ex2), float(ey2))

            # Standard overlap test.
            iou = box_iou(cbox, ebox)

            # Also catch boxes where one prediction is mostly contained inside
            # another but their IoU is not large enough because of very
            # different box sizes.
            ix1 = max(float(cx1), float(ex1))
            iy1 = max(float(cy1), float(ey1))
            ix2 = min(float(cx2), float(ex2))
            iy2 = min(float(cy2), float(ey2))
            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)

            ex_area = max(1.0, (ex2 - ex1) * (ey2 - ey1))
            overlap_candidate = inter / c_area
            overlap_existing = inter / ex_area

            if (iou >= iou_threshold or
                    overlap_candidate >= containment_threshold or
                    overlap_existing >= containment_threshold):
                is_duplicate = True
                break

        if not is_duplicate:
            kept.append(candidate)

    return kept


def find_vehicle_for_plate(plate_box):
    """
    Robustly associate an existing plate detection with a UVH-26
    vehicle detection. Plate/OCR models are not changed.

    Matching order:
      1. plate center inside vehicle box
      2. plate center inside a padded vehicle box
      3. meaningful plate/vehicle overlap
      4. nearby vehicle fallback when geometry is slightly imperfect
    """
    if not vehicle_detections:
        return "UNKNOWN", 0.0

    px1, py1, px2, py2 = map(float, plate_box)
    pcx = (px1 + px2) / 2.0
    pcy = (py1 + py2) / 2.0
    pw = max(px2 - px1, 1.0)
    ph = max(py2 - py1, 1.0)

    candidates = []

    for vx1, vy1, vx2, vy2, vehicle_type, confidence in vehicle_detections:
        vx1, vy1, vx2, vy2 = map(float, (vx1, vy1, vx2, vy2))
        vw = max(vx2 - vx1, 1.0)
        vh = max(vy2 - vy1, 1.0)

        # 1) Exact containment.
        inside = point_inside_box(pcx, pcy, (vx1, vy1, vx2, vy2))

        # 2) Pad the vehicle box. This handles small detector box errors.
        pad_x = vw * 0.18
        pad_y = vh * 0.18
        expanded = (
            vx1 - pad_x, vy1 - pad_y,
            vx2 + pad_x, vy2 + pad_y
        )
        inside_expanded = point_inside_box(pcx, pcy, expanded)

        # 3) IoU is normally tiny because the plate is much smaller than
        # the whole vehicle, so use intersection-over-plate-area instead.
        ix1 = max(px1, vx1)
        iy1 = max(py1, vy1)
        ix2 = min(px2, vx2)
        iy2 = min(py2, vy2)
        intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        plate_overlap = intersection / (pw * ph)

        # 4) Distance from plate center to vehicle center, normalized by
        # vehicle dimensions. Lower is better.
        vcx = (vx1 + vx2) / 2.0
        vcy = (vy1 + vy2) / 2.0
        distance = ((pcx - vcx) / vw) ** 2 + ((pcy - vcy) / vh) ** 2

        if inside:
            match_score = 100.0 + float(confidence)
        elif inside_expanded:
            match_score = 70.0 + float(confidence)
        elif plate_overlap >= 0.50:
            match_score = 50.0 + float(confidence) + 10.0 * plate_overlap
        elif distance <= 1.25:
            # Conservative fallback for slightly misaligned boxes.
            match_score = 20.0 + float(confidence) - 5.0 * distance
        else:
            continue

        candidates.append((
            match_score,
            float(confidence),
            str(vehicle_type)
        ))

    if not candidates:
        return "UNKNOWN", 0.0

    candidates.sort(key=lambda item: item[0], reverse=True)
    _, confidence, vehicle_type = candidates[0]
    return vehicle_type, confidence


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
# CLEAN VEHICLE EVENT + RESULT BOX
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


def register_clean_vehicle_event(
    plate,
    vehicle_type,
    vehicle_confidence,
    track_id,
    frame,
):
    """
    Create one clean vehicle record when a stable plate is available,
    print the result box once, and append it to RESULTS_LOG.

    This does NOT change plate detection or Awiros OCR. It only consumes
    their already-produced stable result and prevents the same vehicle
    from being recorded on every frame.
    """
    global last_vehicle_event

    if not plate or not is_valid_plate(plate):
        return

    # Computed BEFORE taking result_lock (the lock is not re-entrant).
    ocr_confidence = get_track_confidence(track_id, plate)

    now = time.time()

    # Prefer the classified type when available.
    clean_type = vehicle_type if vehicle_type != "UNKNOWN" else "UNKNOWN"

    with result_lock:
        previous_time = recent_vehicle_event_time.get(plate, 0.0)
        existing = confirmed_vehicle_events.get(plate)

        # Update an existing record if the vehicle type becomes available
        # after the OCR result was stabilized.
        if existing is not None:
            if existing["vehicle_type"] == "UNKNOWN" and clean_type != "UNKNOWN":
                existing["vehicle_type"] = clean_type
                existing["vehicle_confidence"] = vehicle_confidence
            return

        # Same plate appearing again shortly after its first confirmation
        # is treated as the same physical vehicle event.
        if now - previous_time < VEHICLE_EVENT_COOLDOWN:
            return

        record = {
            "plate": plate,
            "vehicle_type": clean_type,
            "vehicle_confidence": float(vehicle_confidence),
            "ocr_confidence": float(ocr_confidence),
            "track_id": int(track_id),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        confirmed_vehicle_events[plate] = record
        recent_vehicle_event_time[plate] = now
        last_vehicle_event = record.copy()

    # Print the clean result box (outside the lock).
    print(
        "\n" + format_result_box(
            record["plate"],
            record["vehicle_type"],
            record["ocr_confidence"],
            time.strftime("%H:%M:%S"),
        ) + "\n"
    )

    if RESULTS_LOG:
        try:
            with open(RESULTS_LOG, "a", encoding="utf-8") as f:
                f.write(
                    f"{record['timestamp']},"
                    f"{record['plate']},"
                    f"{record['vehicle_type']},"
                    f"{record['ocr_confidence']:.0f}\n"
                )
        except OSError:
            pass


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
    global last_vehicle_event

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
        print(f"[ERROR] Download/copy uvh26_vehicle.pt to {os.path.join(BASE_DIR, 'models')} first.")
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
        print(f"  - {AWIROS_WORKER} exists")
        print("  - models\\model.safetensors exists")
        print("  - en_dict.txt exists")
        print("  - PaddleOCR folder exists")
        print(f"  - test.py is present in {BASE_DIR}")
        return

    print("[3] Opening test video...")
    print(f"[VIDEO] {VIDEO_PATH}")

    if not os.path.isfile(VIDEO_PATH):
        print()
        print("[ERROR] Video file not found.")
        print(f"[ERROR] {VIDEO_PATH}")
        print("Change VIDEO_PATH at the top of webcam.py.")
        return

    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print()
        print("[ERROR] Video could not be opened.")
        print(f"[ERROR] Check this file: {VIDEO_PATH}")
        return

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if not video_fps or video_fps <= 0:
        video_fps = 30.0

    print(f"[VIDEO] Resolution: {video_width}x{video_height}")
    print(f"[VIDEO] FPS: {video_fps:.2f}")
    print(f"[VIDEO] Total frames: {total_frames}")

    # Start OCR worker.
    worker = threading.Thread(
        target=ocr_worker,
        daemon=True
    )

    worker.start()

    print("[4] Video started.")
    print()
    print("Controls:")
    print("  Q / ESC -> quit")
    print("  R       -> reset OCR history")
    print("  SPACE   -> pause / resume video")
    print()
    print("Awiros OCR runs in a background thread.")
    print("Video playback is paced to the source video's FPS.")
    print()

    previous_time = time.perf_counter()
    fps = 0.0
    paused = False
    frame_delay_ms = max(1, int(round(1000.0 / video_fps)))

    try:

        while True:

            if not paused:
                ret, frame = cap.read()

                if not ret:
                    print("[VIDEO] Reached end of video.")
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

                # Remove overlapping duplicate predictions before drawing or
                # associating a vehicle with a license plate.
                vehicle_detections = suppress_duplicate_vehicle_detections(
                    new_vehicle_detections,
                    iou_threshold=0.35,
                    containment_threshold=0.65,
                )

                # Diagnostic: show raw vs kept detections so we can verify the
                # duplicate suppression without changing the plate/OCR path.
                if VEHICLE_DEBUG_PRINT:
                    if vehicle_detections:
                        debug_items = [
                            f"{t}:{c:.2f}"
                            for _, _, _, _, t, c in vehicle_detections
                        ]
                        print(
                            f"[VEHICLE DEBUG] raw={len(new_vehicle_detections)} "
                            f"kept={len(vehicle_detections)} | "
                            + ", ".join(debug_items)
                        )
                    else:
                        print(
                            f"[VEHICLE DEBUG] raw={len(new_vehicle_detections)} kept=0"
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

                    # Accumulate vehicle-type votes for this track and use
                    # the voted type from here on (stable label + result).
                    vote_vehicle(
                        track_id,
                        vehicle_type,
                        vehicle_confidence
                    )
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

                    # Once the existing OCR logic has produced a stable
                    # plate AND the vehicle type is known, turn it into
                    # ONE clean vehicle-level event (prints the result box).
                    with result_lock:
                        confirmed_plate = stable_text.get(track_id, "")

                    if confirmed_plate and vehicle_type != "UNKNOWN":
                        register_clean_vehicle_event(
                            confirmed_plate,
                            vehicle_type,
                            vehicle_confidence,
                            track_id,
                            frame
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
                f"Video LPR | Awiros | FPS: {fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

            # ------------------------------------------------
            # CLEAN VEHICLE STATUS
            # ------------------------------------------------
            with result_lock:
                current_event = last_vehicle_event.copy() if last_vehicle_event else None

            if current_event:
                status = (
                    f"LAST: {current_event['vehicle_type']} | "
                    f"{current_event['plate']}"
                )
                cv2.rectangle(
                    frame,
                    (0, 50),
                    (430, 90),
                    (0, 0, 0),
                    -1
                )
                cv2.putText(
                    frame,
                    status,
                    (10, 78),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.62,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA
                )

            cv2.imshow(
                "Video LPR - Awiros OCR",
                frame
            )

            key = cv2.waitKey(
                frame_delay_ms if not paused else 50
            ) & 0xFF

            if key in (
                ord("q"),
                27
            ):
                break

            if key == ord(" "):
                paused = not paused
                print("[VIDEO] PAUSED" if paused else "[VIDEO] RESUMED")
                continue

            if key == ord("r"):

                with result_lock:
                    track_history.clear()
                    track_confidence.clear()
                    track_width.clear()
                    stable_text.clear()
                    latest_ocr.clear()
                    confirmed_vehicle_events.clear()
                    recent_vehicle_event_time.clear()
                    last_vehicle_event = None

                last_ocr_time.clear()
                best_crop.clear()
                best_crop_quality.clear()
                saved_crop_count.clear()
                track_vehicle_votes.clear()

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
        print("Video LPR stopped.")


if __name__ == "__main__":
    main()