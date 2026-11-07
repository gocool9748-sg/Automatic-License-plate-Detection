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
# REAL-TIME LPR - the pipeline engine
# Smooth video + background OCR, tuned for small / dusty /
# yellow commercial plates (college buses etc.)
#
# Run it through one of the launchers, which only set a few options:
#   python webcam.py     -> every frame analysed (ACCURACY_MODE)
#   python videotest.py  -> video at true speed / slow motion
# The settings below are the defaults; a launcher overrides some of them
# with configure().
# ============================================================

# Paths are resolved relative to the project folder, so the project works
# from wherever it is cloned:
#   src/      code          config/   settings files     models/  weights
#   outputs/  results.log + debug crops (created on first run)
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SRC_DIR)               # project root
CONFIG_DIR = os.path.join(BASE_DIR, "config")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

YOLO_WEIGHTS = os.path.join(BASE_DIR, "models", "best.pt")

# Test video path - change this to any video you want to test.
VIDEO_PATH = os.path.join(BASE_DIR, "videos", "cb3.mp4")

# Keep video in its original orientation.
MIRROR_CAMERA = False

# LIVE CAMERA: set CAMERA_INDEX to a webcam number (0 = first webcam, 1 = the
# next one, ...) or to a stream address ("rtsp://user:pass@ip/stream").
# None = read VIDEO_PATH instead.
CAMERA_INDEX = None
CAMERA_WIDTH = 1920          # requested resolution (the camera may pick the nearest)
CAMERA_HEIGHT = 1080
CAMERA_FPS = 30
CAMERA_FOURCC = "MJPG"       # most USB webcams only give 1080p at full speed as MJPG
CAMERA_RECONNECT_SECONDS = 15.0   # keep retrying a lost camera this long, then stop

# YOLO - LICENSE PLATE DETECTOR
YOLO_CONFIDENCE = 0.25   # was 0.45; ByteTrack tolerates low-conf boxes
YOLO_IMGSZ = 1280        # was 512; plate was only ~37px after downscale

# ------------------------------------------------------------
# ACCURACY FIXES (Nov 2026)
# ------------------------------------------------------------
# The plate model was trained on bordered white/yellow plates. On the college
# buses the plate is yellow paint on a yellow body, and in COLOUR the model
# misses it and fires on the "COLLEGE BUS" banner instead. Fed a GRAYSCALE
# copy of the frame, it finds those plates reliably (e.g. cb7.mp4 frame 720:
# no plate box in colour, 0.78 confidence in grayscale). OCR still gets the
# colour crop.
PLATE_DETECT_GRAYSCALE = True

# Plates are read for this state first: an unknown state code one character
# away from it ("IN48..", "TM48..", "1N48..") is corrected to it. "" = off.
HOME_STATE = "TN"

# The vehicle model often calls a close-up bus "Truck". A vehicle is logged as
# BUS when at least this share of its class votes were BUS.
BUS_MIN_SHARE = 0.30

# Indian school/college buses are painted yellow. A vehicle the model calls
# TRUCK whose box is mostly yellow is counted as a BUS. Set False if yellow
# trucks also pass this camera.
YELLOW_BODY_IS_BUS = True
YELLOW_BODY_MIN_FRACTION = 0.08   # yellow share of the LOWER half of the box (cars ~0.0, buses 0.10-0.40)

# A vehicle is only logged when its final plate is backed by real readings:
# at least FINAL_MIN_SUPPORT reads within 1 character of it, with this
# average OCR confidence. Stops plates being 'voted together' out of junk.
MIN_LOG_CONFIDENCE = 70

# OCR text that is clearly not a plate (windshield banner "COLLEGE BUS", the
# OCR model's own "UNREADABLE" output). A track that reads like this is
# ignored from then on.
JUNK_OCR_WORDS = (
    "UNREAD", "READAB", "READBL", "COLLEG", "OLLEG", "LLEGE", "EGE", "LLE",
)
# Only BANNER text blocks a track for good. "UNREADABLE" is what the OCR
# says for a plate that is still too far / blurred - that plate must keep
# being read as the vehicle comes closer.
BANNER_OCR_WORDS = ("COLLEG", "OLLEG", "LLEGE", "EGE", "LLE")

# A track that gave up after MAX_OCR_FAILS is tried again once its plate box
# has grown by this factor (the vehicle came closer, the plate got readable).
OCR_RETRY_GROWTH = 1.25

# ============================================================
# VEHICLE CLASSIFICATION - UVH-26 INDIAN TRAFFIC MODEL
# ============================================================
VEHICLE_WEIGHTS = os.path.join(BASE_DIR, "models", "uvh26_vehicle.pt")
VEHICLE_CONFIDENCE = 0.25
VEHICLE_IMGSZ = 640
VEHICLE_INFER_EVERY_N_FRAMES = 3   # vehicle boxes are tracked + held, so every 3rd frame is enough

# ------------------------------------------------------------
# SPEED
# ------------------------------------------------------------
# FP16 on an NVIDIA GPU: roughly 1.5-2x faster, practically the same accuracy.
# (Ignored automatically on CPU, and switched off if the GPU refuses it.)
USE_HALF_PRECISION = True
HALF = False                 # set at startup from USE_HALF_PRECISION + the device

# Cap how many boxes the models may return (less work in NMS / post-processing).
MAX_DET_PLATES = 20
MAX_DET_VEHICLES = 40

# Print where the time goes every few seconds, e.g.
#   [PROFILE] vehicle 6 ms | plate model 31 ms | ROI search 14 ms | ...
# Use it to see which step to tune. Set False to silence it.
PROFILE = True
PROFILE_PRINT_SECONDS = 5.0

# The blue boxes are tracked over time. Each physical vehicle keeps its own
# class votes, so its type (BUS / TRUCK / ...) stays stable while it approaches
# the camera, and the box is held for a few cycles if the model briefly loses it
# (big, close vehicles). The type shown on the blue box is the SAME type that is
# shown on the plate label, "LAST:" status and written to the log.
VEHICLE_TRACK_IOU = 0.30
VEHICLE_HOLD_CYCLES = 6        # inference cycles a lost vehicle box is kept
VEHICLE_VOTE_DECAY = 0.97      # slow decay: type is decided over the whole approach

# Reject "plates" that are really light text on a dark background (windshield
# banners like COLLEGE BUS, destination boards). Real plates are dark text on a
# light/yellow background. Set False if you need black-background plates.
REJECT_DARK_BACKGROUND_PLATES = True
DARK_BANNER_MAX_MEAN = 100


# Final per-vehicle results appended as CSV (timestamp,plate,vehicle,confidence).
# Set to None to disable file logging.
RESULTS_LOG = os.path.join(OUTPUT_DIR, "results.log")

VEHICLE_DEBUG_PRINT = False

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
OCR_INTERVAL = 0.03   # was 0.08: at ~15 fps that skipped every 2nd frame
MAX_HISTORY = 24

MIN_VOTES = 5
FINAL_MIN_READINGS = 5
FINAL_MIN_SUPPORT = 3
MIN_OCR_CONFIDENCE = 30        # was 45
MIN_PLATE_PIXEL_WIDTH = 20     # was 35; small/far plates are upscaled before OCR
BLUR_VARIANCE_THRESHOLD = 3.0  # was 20 (dusty / motion-blurred footage)
OCR_TARGET_WIDTH = 300         # upscale small plates to this width before OCR

# A plate on a bus/truck/car sits low on the vehicle. Detections whose center
# is higher than this fraction of the vehicle box (0 = top, 1 = bottom) are
# treated as false positives (e.g. the "COLLEGE BUS" windshield banner).
PLATE_MIN_REL_Y = 0.65   # was 0.60. Real plates sit ~0.9, the banner ~0.5
PLATE_REL_Y_TYPES = {"BUS", "TRUCK", "CAR"}

# Detections that sit too high on a vehicle (banner, stickers, destination
# board) are discarded outright instead of being "suspect but still OCR'd".
DROP_SUSPECT_PLATES = True

# ------------------------------------------------------------
# Near-duplicate handling: the same vehicle read with 1 wrong
# character (TN48BK2170 / TN43BK2170 / TN41BK2170) is ONE vehicle.
# ------------------------------------------------------------

# ------------------------------------------------------------
# Vehicle identity: ALL readings of one physical vehicle are pooled in a
# "cluster" (even when the tracker changes its ID) and the plate is decided
# by character-level voting over the whole cluster.
# ------------------------------------------------------------
# The fleet is mostly TN48BK21xx: two DIFFERENT buses can be only 2 characters
# apart (TN48BK2144 / TN48BK2113), so only 1-character look-alikes are merged,
# and only for a short time (the old 2 chars / 30 s swallowed the next bus).
CLUSTER_MAX_DIFF = 1          # a reading within this many chars of a cluster joins it
CLUSTER_MATCH_WINDOW = 8.0    # seconds a cluster stays open for look-alike readings
CLUSTER_LOGGED_REJOIN = 2.0   # an already-logged vehicle only absorbs reads this soon after
CLUSTER_IDLE_FINALIZE = 2.0   # log a vehicle this long after its plate was last read...
EVENT_FINALIZE_DELAY = 8.0    # ...or at most this long after it was first confirmed
CLUSTER_PRUNE_AFTER = 300.0   # forget clusters idle this long

# Optional: one plate per line (e.g. your college bus fleet, '#' comments ok).
# When the file exists, a decided plate that is a near-miss of exactly one known
# plate is snapped to it. Plates not in the file are still logged as read.
KNOWN_PLATES_FILE = os.path.join(CONFIG_DIR, "known_plates.txt")
KNOWN_SNAP_MAX_AVG_DIFF = 1.5

# Fallback: if the full-frame plate detector finds nothing valid, search only
# the bottom part of each vehicle box, upscaled, at a lower confidence.
ROI_FALLBACK = True
ROI_BOTTOM_FRACTION = 0.35
ROI_UPSCALE_WIDTH = 1280
ROI_CONF = 0.25   # 0.12 produced junk boxes on grille/headlights
ROI_MAX_VEHICLES = 3
ROI_MIN_VEHICLE_WIDTH = 120   # skip vehicles too far away to read a plate

# Last-resort plate finder that does NOT use the YOLO plate model: looks for
# dark-text-on-light blocks in the bottom of the vehicle. Used when the plate
# model finds nothing there (e.g. yellow plates on yellow buses).
ROI_CLASSICAL_FALLBACK = True

# Video is always shown at its real speed. If the display loop itself falls
# behind (slow PC), it skips up to this many frames at a time to catch up.
MAX_SKIP_FRAMES = 5

# ACCURACY_MODE = True: every frame is analysed before the next one is shown,
# so the video plays at whatever speed detection can manage (like the original
# script) and no OCR chances are lost. False = video at true speed, detection
# keeps up as best it can.
ACCURACY_MODE = True

# (Only used when ACCURACY_MODE = False)
# 1.0 = the video plays at its true speed. Detection analyses fewer frames of
# each vehicle at true speed than in slow motion, so if plates stop reading,
# try 0.5 (half speed, twice as many frames analysed per vehicle).
PLAYBACK_SPEED = 1.0

# Only drop frames when the display is clearly behind (avoids a jumpy,
# "fast-forward" look from skipping a frame here and there).
SKIP_IF_BEHIND_FRAMES = 8

# The detector now sees every 2nd-4th frame, so the default ByteTrack (which
# needs boxes to overlap a lot between frames) loses plates and hands out new
# IDs, which throws away the OCR votes collected so far. This looser tracker
# config fixes that. Set False to use the stock bytetrack.yaml.
USE_LOOSE_TRACKER = not ACCURACY_MODE
TRACKER_YAML = "bytetrack_lpr.yaml"
TRACKER_CFG = "bytetrack.yaml"

# Stop OCR (and hide the box) for a track that never produced a valid plate.
MAX_OCR_FAILS = 50

# (Only used when DROP_SUSPECT_PLATES = False)
# A detection that sits high on a vehicle is "suspect": it is still tried with
# OCR, but hidden until it produces a valid plate reading, and given up on sooner.
SUSPECT_MAX_OCR_FAILS = 12

# Draw EVERY raw plate-model detection with its confidence, before any
# filtering: MAGENTA = full-frame detector, CYAN rectangle = ROI search area,
# ORANGE = ROI detections. Use this to see where the model is really firing.
DEBUG_DRAW_RAW = True

# Real Indian state / UT codes. Garbage like "DR" is rejected.
VALID_STATE_CODES = {
    "AN", "AP", "AR", "AS", "BR", "CG", "CH", "DD", "DL", "DN", "GA", "GJ",
    "HP", "HR", "JH", "JK", "KA", "KL", "LA", "LD", "MH", "ML", "MN", "MP",
    "MZ", "NL", "OD", "OR", "PB", "PY", "RJ", "SK", "TG", "TN", "TR", "TS",
    "UK", "UP", "WB",
}

DEBUG_OCR = False  # set True to print "[OCR RAW] ..." lines

DEBUG_SAVE_CROPS = True
DEBUG_CROPS_DIR = os.path.join(OUTPUT_DIR, "debug_crops")

# ------------------------------------------------------------
# PROOF + GATE REGISTER
# ------------------------------------------------------------
# Every logged vehicle gets proof images, so a disputed entry can be checked
# in seconds:
#   outputs/evidence/<date>/<time>_<plate>_plate.jpg     sharpest plate crop
#   outputs/evidence/<date>/<time>_<plate>_vehicle.jpg   vehicle snapshot, plate boxed
# and one row in the day's register (opens in Excel):
#   outputs/register/<date>.csv
#   date,time,plate,bus,vehicle,confidence,in_fleet,plate_image,vehicle_image
# "time" is when the vehicle was FIRST seen; "bus" is the comment after the
# plate in config/known_plates.txt (e.g. "TN48W9259  # bus 27").
SAVE_EVIDENCE = True
EVIDENCE_DIR = os.path.join(OUTPUT_DIR, "evidence")
REGISTER_DIR = os.path.join(OUTPUT_DIR, "register")
SNAPSHOT_MAX_WIDTH = 640         # vehicle snapshots are scaled down to this width
REGISTER_COLUMNS = (
    "date", "time", "plate", "bus", "vehicle", "confidence", "in_fleet",
    "plate_image", "vehicle_image",
)


# ============================================================
# INDIAN PLATE FORMAT
# ============================================================

PLATE_REGEX = re.compile(r"^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{4}$")

LETTER_FOR_DIGIT = {
    "0": "O", "1": "I", "2": "Z", "4": "A", "5": "S", "6": "G", "7": "T",
    "8": "B",
}

DIGIT_FOR_LETTER = {
    "O": "0", "D": "0", "Q": "0", "U": "0", "I": "1", "L": "1", "J": "1",
    "T": "1", "Z": "2", "A": "4", "S": "5", "B": "8", "G": "6",
}

# These states still issue 1-digit RTO codes (DL 1C AB 1234). Everywhere else
# the RTO code has 2 digits, so "TN4BW9259" is a misread of TN48W9259.
ONE_DIGIT_RTO_STATES = {"DL"}

BH_SERIES_REGEX = re.compile(r"^[0-9]{2}BH[0-9]{4}[A-Z]{1,2}$")

# ============================================================
# SHARED STATE
# ============================================================

ocr_request_lock = threading.Lock()
ocr_request_event = threading.Event()

# One pending crop PER TRACK (newest wins), served round-robin. The old single
# shared slot let whichever plate was handled last each frame win every OCR
# turn, so other plates on screen were starved.
ocr_pending = {}
ocr_last_served = defaultdict(float)

result_lock = threading.Lock()

latest_ocr = {}
stable_text = {}
track_history = defaultdict(list)
track_confidence = defaultdict(list)
track_width = defaultdict(list)

last_ocr_time = defaultdict(float)

best_crop = {}
# track id -> (quality, plate crop, vehicle snapshot, time) for the proof images
best_evidence = {}
best_crop_quality = defaultdict(float)

confirmed_text = {}
last_vote_winner = {}
winner_streak = defaultdict(int)

stop_event = threading.Event()

frame_count = 0

vehicle_detections = []
vehicle_tracks = []   # persistent blue boxes with class votes


confirmed_vehicle_events = {}
recent_vehicle_event_time = {}
last_vehicle_event = None

# One cluster per physical vehicle (pooled OCR readings across track IDs).
plate_clusters = []
track_to_cluster = {}
track_frames = defaultdict(set)   # track id -> detector frame numbers it appeared in
logged_events = []                # [(plate, time, cluster)]
known_plates = set()
known_plate_labels = {}           # plate -> bus name from known_plates.txt

track_vehicle_votes = defaultdict(lambda: defaultdict(float))

# Consecutive OCR attempts with no valid plate, per track.
ocr_fail_count = defaultdict(int)
ocr_block_width = {}   # track id -> plate width when OCR gave up on it

# Lightweight tracks for plates found by the ROI fallback detector.
roi_tracks = {}
roi_next_id = 100000


# ------------------------------------------------------------
# TRACK IDENTITY GUARD
# ------------------------------------------------------------
# The loose ByteTrack config (match_thresh 0.95) can hand an OLD track id to a
# NEW plate, e.g. the next bus's plate inherits the id of the previous bus's
# blocked banner or already-logged plate. That plate is then never read, or
# its readings are added to the previous bus. A box that suddenly jumps far
# away or changes size a lot gets a fresh id with a clean history.
TRACK_JUMP_FACTOR = 2.5          # max centre jump, in plate widths (per ~5 frames)
TRACK_SIZE_CHANGE = 2.0          # max width ratio between two sightings
DEBUG_TRACK_DECISIONS = True     # print why a plate box is not being read

track_last_box = {}              # tracker id -> (box, detector frame)
track_alias = {}                 # tracker id -> id used by the pipeline
next_alias_id = 500000


def stable_track_id(raw_id, box):
    global next_alias_id

    cur = track_alias.get(raw_id, raw_id)
    last = track_last_box.get(raw_id)

    if last is not None:
        (lx1, ly1, lx2, ly2), last_frame = last
        lw = max(lx2 - lx1, 1.0)
        w = max(box[2] - box[0], 1.0)
        dx = (box[0] + box[2]) / 2.0 - (lx1 + lx2) / 2.0
        dy = (box[1] + box[3]) / 2.0 - (ly1 + ly2) / 2.0
        gap = max(1, frame_count - last_frame)
        limit = TRACK_JUMP_FACTOR * max(lw, w) * min(4.0, max(1.0, gap / 5.0))
        ratio = w / lw

        if (
            (dx * dx + dy * dy) ** 0.5 > limit
            or ratio > TRACK_SIZE_CHANGE
            or ratio < 1.0 / TRACK_SIZE_CHANGE
        ):
            cur = next_alias_id
            next_alias_id += 1
            track_alias[raw_id] = cur
            if DEBUG_TRACK_DECISIONS:
                print(f"[TRACK] tracker id {raw_id} jumped -> new plate id {cur}")

    track_last_box[raw_id] = (tuple(float(v) for v in box), frame_count)
    return cur


# ============================================================
# TEXT / PLATE HELPERS
# ============================================================

def clean_text(text):
    if not text:
        return ""
    return re.sub(r"[^A-Z0-9]", "", text.upper())


def is_junk_text(raw):
    """True for OCR output that is clearly not a plate (banner / UNREADABLE)."""
    r = clean_text(raw)
    return bool(r) and any(w in r for w in JUNK_OCR_WORDS)


def is_valid_plate(text):
    if not text:
        return False

    if BH_SERIES_REGEX.fullmatch(text):
        return True

    if not PLATE_REGEX.fullmatch(text) or text[:2] not in VALID_STATE_CODES:
        return False

    # 1-digit RTO codes only exist in a few states (see ONE_DIGIT_RTO_STATES).
    if not text[3].isdigit() and text[:2] not in ONE_DIGIT_RTO_STATES:
        return False

    return True


def _coerce_char(ch, kind):
    """Force ch to a letter ("L") or digit ("D"). Returns (char, cost)."""
    if kind == "L":
        if ch.isalpha():
            return ch, 0
        return (LETTER_FOR_DIGIT[ch], 1) if ch in LETTER_FOR_DIGIT else (None, 99)

    if ch.isdigit():
        return ch, 0
    return (DIGIT_FOR_LETTER[ch], 1) if ch in DIGIT_FOR_LETTER else (None, 99)


def _plate_layouts(length):
    """Every Indian plate layout of this length, common ones first."""
    layouts = []
    for rto in (2, 1):
        for series in (2, 1, 3):
            if 2 + rto + series + 4 == length:
                layouts.append("LL" + "D" * rto + "L" * series + "DDDD")
    if length in (9, 10):
        layouts.append("DDLLDDDD" + "L" * (length - 8))   # 22 BH 1234 AB
    return layouts


def _best_layout_fix(text):
    """Best valid plate with the same length as text -> (plate, cost)."""
    best = ("", 99.0)

    for layout in _plate_layouts(len(text)):
        chars, cost = [], 0.0

        for ch, kind in zip(text, layout):
            c, k = _coerce_char(ch, kind)
            if c is None:
                cost = 99.0
                break
            chars.append(c)
            cost += k

        if cost >= 99:
            continue

        cand = "".join(chars)

        if layout.startswith("DD"):
            if cand[2:4] != "BH":
                continue
        else:
            if HOME_STATE and cand[:2] not in VALID_STATE_CODES:
                if sum(a != b for a, b in zip(cand[:2], HOME_STATE)) == 1:
                    cand = HOME_STATE + cand[2:]
                    cost += 1

            # 1-digit RTO code is the rarer layout. BH-series layouts
            # ("DDLL...") also have a letter at index 3 but no RTO code.
            if layout[3] == "L":
                cost += 0.5

        if is_valid_plate(cand) and cost < best[1]:
            best = (cand, cost)

    return best


def normalize_plate(raw, max_cost=3.0):
    """OCR text -> a valid Indian plate, or "" if it can't be one."""
    raw = clean_text(raw)

    if not raw or is_junk_text(raw):
        return ""

    best = _best_layout_fix(raw)

    # A same-length reading that needs at most one look-alike fix is
    # trusted over dropping characters (TN09A81234 -> TN09AB1234, not
    # TN09A8123).
    if best[0] and best[1] <= 1.0:
        return best[0]

    # OCR may add stray characters (plate border, "IND", dots, a doubled
    # character). Try dropping some; every dropped character costs a little.
    variants = []
    for length in (10, 9, 8):
        for start in range(len(raw) - length + 1):
            if length < len(raw):
                variants.append((raw[start:start + length], 0.5 * (len(raw) - length)))
    if 9 <= len(raw) <= 11:
        for i in range(len(raw)):
            variants.append((raw[:i] + raw[i + 1:], 1.0))

    seen = set()
    for text, extra in variants:
        if text in seen:
            continue
        seen.add(text)
        plate, cost = _best_layout_fix(text)
        if plate and cost + extra < best[1]:
            best = (plate, cost + extra)

    return best[0] if best[1] <= max_cost else ""


def to_gray3(image):
    """3-channel grayscale copy (what the plate detector is fed)."""
    return cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)


def is_yellow_body(frame, box):
    """True if most of the vehicle box is yellow paint."""
    x1, y1, x2, y2 = [int(v) for v in box[:4]]
    y1 = y1 + (y2 - y1) // 2   # lower half = painted body (no windshield / sky)
    roi = frame[max(0, y1):y2, max(0, x1):x2]
    if roi.size == 0:
        return False
    roi = cv2.resize(roi, (64, 64), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    h, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    yellow = (h >= 18) & (h <= 38) & (sat >= 90) & (val >= 90)
    return float(yellow.mean()) >= YELLOW_BODY_MIN_FRACTION


def decide_vehicle_type(votes):
    """Vehicle type from class votes. Buses are often called TRUCK up close,
    so a real share of BUS votes wins."""
    if not votes:
        return "UNKNOWN"
    total = sum(v for v in votes.values() if v > 0)
    if total > 0 and votes.get("BUS", 0.0) / total >= BUS_MIN_SHARE:
        return "BUS"
    return max(votes, key=votes.get)


def to_gray(image):
    """Return a single-channel copy of a BGR, BGRA or already-gray crop."""
    if image.ndim == 2:
        return image
    if image.shape[2] == 1:
        return image[:, :, 0]
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def sharpness(image):
    if image is None or image.size == 0:
        return 0.0

    gray = to_gray(image)

    return float(
        cv2.Laplacian(
            gray,
            cv2.CV_64F
        ).var()
    )


def plate_quality(image):
    """Score a plate crop so OCR gets the sharpest useful frame."""
    if image is None or image.size == 0:
        return 0.0

    h, w = image.shape[:2]
    if w < MIN_PLATE_PIXEL_WIDTH:
        return 0.0

    gray = to_gray(image)
    sharp = cv2.Laplacian(gray, cv2.CV_64F).var()
    contrast = float(gray.std())

    return (
        min(sharp / 100.0, 5.0)
        + min(contrast / 50.0, 2.0)
        + min(w / 80.0, 2.0)   # was w/120 so small plates still score
    )


def enhance_plate_for_ocr(plate):
    """Upscale + local contrast + border padding for small, dusty plates."""
    h, w = plate.shape[:2]

    if w < OCR_TARGET_WIDTH:
        scale = OCR_TARGET_WIDTH / float(w)
        plate = cv2.resize(
            plate, None, fx=scale, fy=scale,
            interpolation=cv2.INTER_CUBIC
        )

    # CLAHE on lightness only, so yellow plates keep their colour.
    lab = cv2.cvtColor(plate, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(l)
    plate = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

    # Padding helps recognizers whose text touches the crop edge.
    return cv2.copyMakeBorder(plate, 8, 8, 8, 8, cv2.BORDER_REPLICATE)


# ============================================================
# DEBUG CROP SAVING
# ============================================================

def save_all_best_crops():
    """
    Save only ONE final best crop per recognized plate. Plate text, not the
    tracker ID, is the identity, since ByteTrack may re-ID the same vehicle.
    """
    if not DEBUG_SAVE_CROPS:
        return

    try:
        os.makedirs(DEBUG_CROPS_DIR, exist_ok=True)

        best_crop_by_plate = {}

        with result_lock:
            stable_snapshot = dict(confirmed_text)
            for tid, plate in stable_text.items():
                stable_snapshot.setdefault(tid, plate)

        for track_id, crop in best_crop.items():
            if crop is None or crop.size == 0:
                continue

            plate = normalize_plate(stable_snapshot.get(track_id, ""))

            if not plate or not is_valid_plate(plate):
                continue

            quality = float(best_crop_quality.get(track_id, plate_quality(crop)))
            previous = best_crop_by_plate.get(plate)

            if previous is None or quality > previous[0]:
                best_crop_by_plate[plate] = (quality, track_id, crop)

        saved = 0

        for plate, (quality, track_id, crop) in sorted(best_crop_by_plate.items()):
            safe_plate = re.sub(r"[^A-Z0-9]", "", plate)
            path = os.path.join(DEBUG_CROPS_DIR, f"best_plate_{safe_plate}.jpg")

            if cv2.imwrite(path, crop):
                saved += 1
                print(
                    f"[FINAL BEST CROP] {plate} | "
                    f"source ID {track_id} | quality={quality:.2f} -> {path}"
                )

        print(
            f"[DEBUG] Saved {saved} final best plate crop(s) "
            f"for {len(best_crop_by_plate)} unique vehicle(s)."
        )

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
# YOLO/PyTorch and Paddle/Awiros cannot safely share the same
# Windows CUDA/cuDNN DLLs, so Awiros runs in its own Python process
# and plate crops are sent over localhost TCP.
# ============================================================

AWIROS_WORKER = os.path.join(SRC_DIR, "awiros_worker.py")
AWIROS_HOST = "127.0.0.1"
AWIROS_PORT = 48731
AWIROS_START_TIMEOUT = 30.0


class AwirosOCR:
    """Client for the separate Paddle/Awiros worker process."""

    def __init__(self, worker_path=AWIROS_WORKER, host=AWIROS_HOST, port=AWIROS_PORT):
        self.worker_path = worker_path
        self.host = host
        self.port = port
        self.process = None
        self.sock = None
        self.socket_lock = threading.Lock()
        self.last_junk = False

        print("[Awiros] Starting separate OCR worker...")
        self.start()
        print("[Awiros] Worker connected.")

    def _recv_exact(self, n):
        chunks = []
        remaining = n

        while remaining:
            chunk = self.sock.recv(remaining)

            if not chunk:
                raise ConnectionError("Awiros worker disconnected.")

            chunks.append(chunk)
            remaining -= len(chunk)

        return b"".join(chunks)

    def _connect(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1.0)

        try:
            sock.connect((self.host, self.port))
        except Exception:
            sock.close()
            raise

        sock.settimeout(5.0)
        self.sock = sock

    def start(self):
        if not os.path.isfile(self.worker_path):
            raise FileNotFoundError(f"Awiros worker not found: {self.worker_path}")

        self.process = subprocess.Popen(
            [sys.executable, self.worker_path],
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

    def _query(self, image):
        """Send one BGR image to the worker. Returns (normalized_text, conf)."""
        ok, encoded = cv2.imencode(
            ".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95]
        )

        if not ok:
            return "", 0.0

        payload = encoded.tobytes()

        if len(payload) > 10_000_000:
            return "", 0.0

        with self.socket_lock:
            try:
                self.sock.sendall(struct.pack(">I", len(payload)))
                self.sock.sendall(payload)

                size = struct.unpack(">I", self._recv_exact(4))[0]

                if size <= 0 or size > 1_000_000:
                    raise ValueError(f"Invalid worker response size: {size}")

                data = json.loads(self._recv_exact(size).decode("utf-8"))

                if data.get("error"):
                    if DEBUG_OCR:
                        print("[Awiros worker error]", data["error"])
                    return "", 0.0

                raw = str(data.get("text", "")).strip().upper()
                conf = float(data.get("confidence", 0.0))

                if DEBUG_OCR:
                    print(f"[OCR RAW] '{raw}' conf={conf:.1f}")

                # Parent process owns plate normalization and voting.
                if is_junk_text(raw):
                    r = clean_text(raw)
                    if any(w in r for w in BANNER_OCR_WORDS):
                        self.last_junk = True
                    return "", 0.0
                return normalize_plate(raw), conf

            except Exception as e:
                if DEBUG_OCR:
                    print("[Awiros client ERROR]", repr(e))
                return "", 0.0

    def recognize(self, plate):
        """
        Try the enhanced crop first, then the raw crop if that fails.

        Returns:
            (text, confidence_0_to_100)
        """
        if plate is None or plate.size == 0:
            return "", 0.0

        h, w = plate.shape[:2]

        if w < MIN_PLATE_PIXEL_WIDTH:
            return "", 0.0

        if sharpness(plate) < BLUR_VARIANCE_THRESHOLD:
            return "", 0.0

        best = ("", 0.0)
        self.last_junk = False

        for variant in (enhance_plate_for_ocr(plate), plate):
            text, conf = self._query(variant)

            if text and conf > best[1]:
                best = (text, conf)

            if text and conf >= 70:  # good enough, skip second pass
                break

        return best

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
                    self.process.wait(timeout=3)
                except Exception:
                    try:
                        self.process.kill()
                    except Exception:
                        pass

            self.process = None


awiros_ocr = None


def perform_ocr(plate):
    if awiros_ocr is None:
        return "", 0.0

    return awiros_ocr.recognize(plate)


# ============================================================
# TEMPORAL VOTING
# ============================================================

PLATE_STRUCT = re.compile(r"^[A-Z]{2}([0-9]{1,2})([A-Z]{1,3})[0-9]{4}$")


def plate_pattern_for_texts(texts):
    """Letter/digit pattern of the most common plate structure in `texts`,
    e.g. TN48BK2170 -> LLDDLLDDDD, TN01A1234 -> LLDDLDDDD."""
    structures = Counter()

    for x in texts:
        m = PLATE_STRUCT.fullmatch(x)

        if m:
            structures[(len(m.group(1)), len(m.group(2)))] += 1

    if not structures:
        return ""

    (digits, letters), _ = structures.most_common(1)[0]
    return "LL" + "D" * digits + "L" * letters + "DDDD"


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


def character_vote(history, confidences, widths=None):
    """Combine character positions across complete OCR readings, weighted by
    OCR confidence and plate width. The plate structure (digits in the RTO
    code, letters in the series) is the most common one among the readings."""
    if not history:
        return ""

    if widths is None:
        widths = [100.0] * len(history)

    pattern = plate_pattern_for_texts(history)

    if not pattern:
        return ""

    target_len = len(pattern)

    candidates = [
        (text, conf, width)
        for text, conf, width in zip(history, confidences, widths)
        if len(text) == target_len
    ]

    if not candidates:
        return ""

    result = []

    for pos in range(target_len):
        votes = defaultdict(float)
        expected = pattern[pos]

        for text, conf, width in candidates:
            ch = position_correct(text[pos], expected)

            if not character_is_allowed(ch, expected):
                continue

            votes[ch] += (
                (0.5 + min(max(conf, 0.0), 100.0) / 100.0)
                * min(max(width / 100.0, 0.5), 3.0)
            )

        if not votes:
            return ""

        result.append(max(votes, key=votes.get))

    result = "".join(result)

    return result if is_valid_plate(result) else ""


def update_track(track_id, text, confidence, box_width):
    """
    Add one OCR reading and recompute the plate using temporal +
    character-level voting. A single bad frame can never finalize a vehicle.
    """
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

        exact_scores = defaultdict(float)
        exact_counts = defaultdict(int)

        for t, conf, width in zip(history, confs, widths):
            width_weight = min(max(width / 100.0, 0.5), 3.0)
            conf_weight = 0.5 + min(max(conf, 0.0), 100.0) / 100.0

            exact_scores[t] += width_weight * conf_weight
            exact_counts[t] += 1

        exact_best = (
            max(exact_scores, key=exact_scores.get) if exact_scores else ""
        )

        char_best = character_vote(history, confs)

        candidate = char_best or exact_best

        if len(history) >= MIN_VOTES and candidate:
            stable_text[track_id] = candidate

        latest_ocr[track_id] = (text, confidence)

        # Pool this reading with every other reading of the same vehicle.
        _cluster_add_locked(track_id, text, confidence, box_width)

        # ---------------- FINAL CONFIRMATION ----------------
        support = exact_counts.get(candidate, 0)

        # Readings that differ by at most one character still count as support.
        if candidate and len(candidate) in (8, 9, 10):
            pattern = (
                plate_pattern_for_texts([candidate])
                or plate_pattern_for_length(len(candidate))
            )
            char_support = 0

            for reading in history:
                if len(reading) != len(candidate):
                    continue

                matches = 0
                for pos, expected in enumerate(pattern):
                    a = position_correct(reading[pos], expected)
                    b = position_correct(candidate[pos], expected)
                    if a == b:
                        matches += 1

                if matches >= len(candidate) - 1:
                    char_support += 1

            support = max(support, char_support)

        if (
            len(history) >= FINAL_MIN_READINGS
            and candidate
            and support >= FINAL_MIN_SUPPORT
        ):
            previous_winner = last_vote_winner.get(track_id)

            if previous_winner == candidate:
                winner_streak[track_id] += 1
            else:
                last_vote_winner[track_id] = candidate
                winner_streak[track_id] = 1

            if winner_streak[track_id] >= 2:
                confirmed_text[track_id] = candidate
                stable_text[track_id] = candidate
        else:
            if track_id not in confirmed_text:
                stable_text.pop(track_id, None)


# ============================================================
# BACKGROUND OCR WORKER
# ============================================================

def ocr_worker():
    """Latest-frame OCR worker: stale requests are discarded."""

    while not stop_event.is_set():

        ocr_request_event.wait(timeout=0.05)

        if stop_event.is_set():
            break

        with ocr_request_lock:
            if ocr_pending:
                track_id = min(
                    ocr_pending, key=lambda t: ocr_last_served[t]
                )
                crop, box_width, _q = ocr_pending.pop(track_id)
                ocr_last_served[track_id] = time.perf_counter()
                item = True
            else:
                item = None

            if not ocr_pending:
                ocr_request_event.clear()

        if item is None:
            continue

        try:
            text, confidence = perform_ocr(crop)
            junk = bool(awiros_ocr is not None and awiros_ocr.last_junk and not text)

            if text and confidence >= MIN_OCR_CONFIDENCE:
                with result_lock:
                    ocr_fail_count[track_id] = 0
                update_track(track_id, text, confidence, box_width)
            else:
                with result_lock:
                    ocr_fail_count[track_id] += 1
                    # Banner / sticker text: stop reading this track unless
                    # it already produced real plate readings.
                    if junk and not track_history.get(track_id):
                        ocr_fail_count[track_id] = max(
                            ocr_fail_count[track_id], MAX_OCR_FAILS
                        )

        except Exception as e:
            if DEBUG_OCR:
                print("[OCR WORKER ERROR]", e)


# ============================================================
# VEHICLE CLASSIFICATION HELPERS
# ============================================================

def map_vehicle_class(original_class):
    return VEHICLE_CLASS_MAP.get(str(original_class), "OTHER")


def point_inside_box(px, py, box):
    vx1, vy1, vx2, vy2 = box
    return vx1 <= px <= vx2 and vy1 <= py <= vy2


def box_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = iw * ih

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0


def suppress_duplicate_vehicle_detections(
    detections, iou_threshold=0.35, containment_threshold=0.65
):
    """Keep one strongest UVH-26 detection per physical vehicle."""
    if not detections:
        return []

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

            iou = box_iou(cbox, ebox)

            ix1 = max(float(cx1), float(ex1))
            iy1 = max(float(cy1), float(ey1))
            ix2 = min(float(cx2), float(ex2))
            iy2 = min(float(cy2), float(ey2))
            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)

            ex_area = max(1.0, (ex2 - ex1) * (ey2 - ey1))

            if (
                iou >= iou_threshold
                or inter / c_area >= containment_threshold
                or inter / ex_area >= containment_threshold
            ):
                is_duplicate = True
                break

        if not is_duplicate:
            kept.append(candidate)

    return kept


def _overlap(a, b):
    """(iou, intersection / smaller-box-area) of two xyxy boxes."""
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    area_a = max(1.0, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(1.0, (b[2] - b[0]) * (b[3] - b[1]))
    union = area_a + area_b - inter
    return inter / union, inter / min(area_a, area_b)


def update_vehicle_tracks(new_dets):
    """Keep persistent vehicle boxes. Each has its own class votes, so the
    type drawn on the blue box is stable and is the one used everywhere.
    new_dets: [(x1, y1, x2, y2, type, conf)]. Returns the same format."""
    global vehicle_tracks

    pairs = []
    for ti, tr in enumerate(vehicle_tracks):
        for di, d in enumerate(new_dets):
            iou, contain = _overlap(tr["box"], d[:4])
            pairs.append((iou, contain, ti, di))

    used_t, used_d = set(), set()

    # Pass 1: normal IoU match. Pass 2: a vehicle that grew/shrank a lot
    # (coming close to the camera) still sits inside / around its old box.
    for key_index, threshold in ((0, VEHICLE_TRACK_IOU), (1, 0.6)):
        for pair in sorted(pairs, key=lambda p: p[key_index], reverse=True):
            if pair[key_index] < threshold:
                break
            ti, di = pair[2], pair[3]
            if ti in used_t or di in used_d:
                continue
            used_t.add(ti)
            used_d.add(di)

            tr = vehicle_tracks[ti]
            d = new_dets[di]
            tr["box"] = tuple(d[:4])
            tr["conf"] = float(d[5])
            tr["age"] = 0
            for k in list(tr["votes"]):
                tr["votes"][k] *= VEHICLE_VOTE_DECAY
            tr["votes"][d[4]] += float(d[5])

    kept = []

    for ti, tr in enumerate(vehicle_tracks):
        if ti in used_t:
            kept.append(tr)
        else:
            tr["age"] += 1
            if tr["age"] <= VEHICLE_HOLD_CYCLES:
                kept.append(tr)

    for di, d in enumerate(new_dets):
        if di not in used_d:
            kept.append({
                "box": tuple(d[:4]),
                "conf": float(d[5]),
                "age": 0,
                "votes": defaultdict(float, {d[4]: float(d[5])}),
            })

    vehicle_tracks = kept

    out = []
    for tr in vehicle_tracks:
        vtype = max(tr["votes"], key=tr["votes"].get)
        out.append((*tr["box"], vtype, tr["conf"]))

    return out


def plate_is_dark_banner(crop):
    """True for light-text-on-dark-background regions (banners, boards)."""
    if crop is None or crop.size == 0:
        return False

    h, w = crop.shape[:2]
    if h < 8 or w < 20:
        return False

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    thr, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    dark = gray <= thr

    if dark.mean() < 0.55:       # background is light: a normal plate
        return False

    return float(gray[dark].mean()) < DARK_BANNER_MAX_MEAN


def find_vehicle_for_plate(plate_box):
    """
    Associate a plate detection with a UVH-26 vehicle detection.

    Matching order:
      1. plate center inside vehicle box
      2. plate center inside a padded vehicle box
      3. meaningful plate/vehicle overlap
      4. nearby vehicle fallback
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

        inside = point_inside_box(pcx, pcy, (vx1, vy1, vx2, vy2))

        pad_x = vw * 0.18
        pad_y = vh * 0.18
        expanded = (vx1 - pad_x, vy1 - pad_y, vx2 + pad_x, vy2 + pad_y)
        inside_expanded = point_inside_box(pcx, pcy, expanded)

        ix1 = max(px1, vx1)
        iy1 = max(py1, vy1)
        ix2 = min(px2, vx2)
        iy2 = min(py2, vy2)
        intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        plate_overlap = intersection / (pw * ph)

        vcx = (vx1 + vx2) / 2.0
        vcy = (vy1 + vy2) / 2.0
        distance = ((pcx - vcx) / vw) ** 2 + ((pcy - vcy) / vh) ** 2

        if inside:
            # Tightest containing box wins (the vehicle itself, e.g. a bike in
            # front of a bus), confidence only breaks ties. Area is in units
            # of 100x100 px, so this always stays above the other cases.
            match_score = 200.0 - min(vw * vh / 1e4, 99.0) + 0.01 * float(confidence)
        elif inside_expanded:
            match_score = 70.0 + float(confidence)
        elif plate_overlap >= 0.50:
            match_score = 50.0 + float(confidence) + 10.0 * plate_overlap
        elif distance <= 1.25:
            match_score = 20.0 + float(confidence) - 5.0 * distance
        else:
            continue

        candidates.append((match_score, float(confidence), str(vehicle_type)))

    if not candidates:
        return "UNKNOWN", 0.0

    candidates.sort(key=lambda item: item[0], reverse=True)
    _, confidence, vehicle_type = candidates[0]
    return vehicle_type, confidence


def plate_position_ok(plate_box):
    """
    Reject plate detections that sit too high on a bus/truck/car
    (windshield banners, destination boards, stickers).
    """
    px1, py1, px2, py2 = plate_box
    pcx = (px1 + px2) / 2.0
    pcy = (py1 + py2) / 2.0

    best = None  # smallest containing vehicle box

    for vx1, vy1, vx2, vy2, vtype, vconf in vehicle_detections:
        if vtype not in PLATE_REL_Y_TYPES:
            continue
        if not point_inside_box(pcx, pcy, (vx1, vy1, vx2, vy2)):
            continue

        area = (vx2 - vx1) * (vy2 - vy1)
        if best is None or area < best[0]:
            best = (area, vy1, vy2)

    if best is None:
        return True  # no vehicle context, don't reject

    _, vy1, vy2 = best
    rel_y = (pcy - vy1) / max(vy2 - vy1, 1)
    return rel_y >= PLATE_MIN_REL_Y


def _assign_roi_id(box):
    """Give ROI-detected plates a stable ID by nearest previous box."""
    global roi_next_id

    cx = (box[0] + box[2]) / 2.0
    cy = (box[1] + box[3]) / 2.0
    w = max(box[2] - box[0], 1)

    best_id, best_d = None, None

    for tid, (b, last) in list(roi_tracks.items()):
        if frame_count - last > 20:
            del roi_tracks[tid]
            continue

        bcx = (b[0] + b[2]) / 2.0
        bcy = (b[1] + b[3]) / 2.0
        d = ((cx - bcx) ** 2 + (cy - bcy) ** 2) ** 0.5

        if d < 2.0 * max(w, b[2] - b[0]) and (best_d is None or d < best_d):
            best_id, best_d = tid, d

    if best_id is None:
        best_id = roi_next_id
        roi_next_id += 1

    roi_tracks[best_id] = (box, frame_count)
    return best_id


def classical_plate_candidates(roi, max_n=3):
    """Find plate-like regions (dark text blocks) without any ML model.
    Returns [(x1,y1,x2,y2,score)] in roi pixel coords."""
    h, w = roi.shape[:2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    k = max(1.0, w / 1280.0)
    bh = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT,
        cv2.getStructuringElement(cv2.MORPH_RECT, (int(21*k)|1, int(9*k)|1)))
    gx = np.abs(cv2.Sobel(bh, cv2.CV_32F, 1, 0, ksize=3))
    gx = cv2.normalize(gx, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
    gx = cv2.GaussianBlur(gx, (3, 3), 0)
    _, th = cv2.threshold(gx, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (int(25*k)|1, int(7*k)|1)))
    th = cv2.erode(th, None, iterations=1)
    th = cv2.dilate(th, None, iterations=2)
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cnts:
        x, y, bw, bhh = cv2.boundingRect(c)
        if bhh < 12*k or bw < 45*k or bw > 420*k or bw / float(bhh) < 2.0 or bw / float(bhh) > 6.5:
            continue
        dens = float(bh[y:y+bhh, x:x+bw].mean())
        out.append((x, y, x+bw, y+bhh, dens))
    out.sort(key=lambda t: t[4], reverse=True)
    return out[:max_n]


def detect_plates_in_vehicle_roi(frame, model, device, canvas=None,
                                 existing_boxes=()):
    """
    Fallback plate search: crop the bottom of each vehicle box, upscale it,
    and run the plate model there. Returns [(x1, y1, x2, y2, track_id)].
    """
    H, W = frame.shape[:2]
    found = []
    if canvas is None:
        canvas = frame

    vehicles = [v for v in vehicle_detections if v[4] != "BIKE"]
    vehicles.sort(
        key=lambda v: (v[2] - v[0]) * (v[3] - v[1]), reverse=True
    )

    # Only vehicles that don't already have a good plate, and are big enough
    # for a plate to be readable.
    def has_plate(v):
        for (ex1, ey1, ex2, ey2) in existing_boxes:
            if point_inside_box(
                (ex1 + ex2) / 2.0, (ey1 + ey2) / 2.0, (v[0], v[1], v[2], v[3])
            ):
                return True
        return False

    vehicles = [
        v for v in vehicles
        if (v[2] - v[0]) >= ROI_MIN_VEHICLE_WIDTH and not has_plate(v)
    ]

    for vx1, vy1, vx2, vy2, vtype, vconf in vehicles[:ROI_MAX_VEHICLES]:
        vw = vx2 - vx1
        vh = vy2 - vy1

        rx1 = max(0, int(vx1 - 0.10 * vw))
        rx2 = min(W, int(vx2 + 0.10 * vw))
        ry1 = max(0, int(vy2 - ROI_BOTTOM_FRACTION * vh))
        ry2 = min(H, int(vy2 + 0.05 * vh))

        if rx2 - rx1 < 40 or ry2 - ry1 < 20:
            continue

        roi = frame[ry1:ry2, rx1:rx2]

        scale = min(max(1.0, ROI_UPSCALE_WIDTH / float(rx2 - rx1)), 4.0)
        if scale > 1.0:
            roi = cv2.resize(
                roi, None, fx=scale, fy=scale,
                interpolation=cv2.INTER_CUBIC
            )

        res = model.predict(
            to_gray3(roi) if PLATE_DETECT_GRAYSCALE else roi,
            conf=ROI_CONF, imgsz=ROI_UPSCALE_WIDTH,
            device=device, half=HALF, max_det=MAX_DET_PLATES, verbose=False
        )[0]

        if DEBUG_DRAW_RAW:
            ov_rectangle(canvas, (rx1, ry1), (rx2, ry2), (255, 255, 0), 2)

        best = None

        for b in (res.boxes if res.boxes is not None else []):
            bx1, by1, bx2, by2 = (b.xyxy[0].cpu().numpy() / scale)

            gx1 = max(0, int(bx1 + rx1))
            gy1 = max(0, int(by1 + ry1))
            gx2 = min(W - 1, int(bx2 + rx1))
            gy2 = min(H - 1, int(by2 + ry1))

            bw = gx2 - gx1
            bh = gy2 - gy1

            if bw <= 0 or bh <= 0:
                continue

            if DEBUG_DRAW_RAW:
                ov_rectangle(canvas, (gx1, gy1), (gx2, gy2), (0, 165, 255), 1)
                ov_putText(
                    canvas, f"roi {float(b.conf[0]):.2f}",
                    (gx1, max(12, gy1 - 4)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 165, 255), 1, cv2.LINE_AA
                )

            aspect = bw / float(bh)
            if aspect < 1.2 or aspect > 7.0:
                continue

            if not plate_position_ok((gx1, gy1, gx2, gy2)):
                continue

            conf = float(b.conf[0])
            if best is None or conf > best[0]:
                best = (conf, gx1, gy1, gx2, gy2)

        if best is not None:
            _, gx1, gy1, gx2, gy2 = best
            found.append(
                (gx1, gy1, gx2, gy2, _assign_roi_id((gx1, gy1, gx2, gy2)))
            )
        elif ROI_CLASSICAL_FALLBACK:
            # Model found nothing: use the first non-junk classical candidate.
            for cx1, cy1, cx2, cy2, _score in classical_plate_candidates(roi):
                cw = cx2 - cx1
                ch = cy2 - cy1
                gx1 = max(0, int((cx1 - 0.08 * cw) / scale + rx1))
                gx2 = min(W - 1, int((cx2 + 0.08 * cw) / scale + rx1))
                gy1 = max(0, int((cy1 - 0.25 * ch) / scale + ry1))
                gy2 = min(H - 1, int((cy2 + 0.25 * ch) / scale + ry1))

                if gx2 <= gx1 or gy2 <= gy1:
                    continue
                if not plate_position_ok((gx1, gy1, gx2, gy2)):
                    continue

                tid = _assign_roi_id((gx1, gy1, gx2, gy2))

                if ocr_fail_count[tid] >= MAX_OCR_FAILS and tid not in stable_text:
                    continue  # this one never reads, try the next candidate

                if DEBUG_DRAW_RAW:
                    ov_rectangle(canvas, (gx1, gy1), (gx2, gy2), (0, 0, 255), 1)
                    ov_putText(
                        canvas, "classical", (gx1, max(12, gy1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1,
                        cv2.LINE_AA
                    )

                found.append((gx1, gy1, gx2, gy2, tid))
                break

    return found


def merge_plate_dets(dets):
    """
    dets: [(x1, y1, x2, y2, track_id, suspect)] in priority order.
    Drops boxes that overlap an earlier one (same physical plate seen by the
    tracker and the fallback search).
    """
    kept = []

    for d in dets:
        box = d[:4]
        area = max(1, (box[2] - box[0]) * (box[3] - box[1]))
        duplicate = False

        for k in kept:
            kb = k[:4]
            iw = max(0, min(box[2], kb[2]) - max(box[0], kb[0]))
            ih = max(0, min(box[3], kb[3]) - max(box[1], kb[1]))
            k_area = max(1, (kb[2] - kb[0]) * (kb[3] - kb[1]))

            if (
                box_iou(box, kb) > 0.30
                or iw * ih / float(min(area, k_area)) > 0.50
            ):
                duplicate = True
                break

        if not duplicate:
            kept.append(d)

    return kept


class LiveCamera:
    """Reads a webcam / IP stream in its own thread and always hands out the
    NEWEST frame. Reading a camera frame by frame would let OpenCV's buffer
    fall seconds behind the real scene while detection is busy. A lost camera
    is reopened for up to CAMERA_RECONNECT_SECONDS before giving up."""

    def __init__(self, source):
        self.source = source
        self.cap = None
        self.cond = threading.Condition()
        self.frame = None
        self.seq = 0          # frames received
        self.taken = 0        # last frame handed out
        self.stopped = False
        self._open()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _open(self):
        if isinstance(self.source, int):
            backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
            cap = cv2.VideoCapture(self.source, backend)
            if CAMERA_FOURCC:
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*CAMERA_FOURCC))
            if CAMERA_WIDTH:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
            if CAMERA_HEIGHT:
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
            if CAMERA_FPS:
                cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
        else:
            cap = cv2.VideoCapture(self.source)

        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        self.cap = cap
        return cap.isOpened()

    def isOpened(self):
        return self.cap is not None and self.cap.isOpened()

    def get(self, prop):
        return self.cap.get(prop) if self.cap is not None else 0.0

    def _loop(self):
        lost_since = None

        while not self.stopped:
            ok, frame = (False, None)
            if self.cap is not None:
                try:
                    ok, frame = self.cap.read()
                except Exception:
                    ok = False

            if ok and frame is not None:
                if lost_since is not None:
                    print("[CAMERA] Camera is back.")
                lost_since = None
                with self.cond:
                    self.frame = frame
                    self.seq += 1
                    self.cond.notify_all()
                continue

            now = time.perf_counter()
            if lost_since is None:
                lost_since = now
                print("[CAMERA] No frame from the camera - reconnecting...")

            if now - lost_since > CAMERA_RECONNECT_SECONDS:
                print("[CAMERA] Camera lost - giving up.")
                with self.cond:
                    self.stopped = True
                    self.cond.notify_all()
                break

            try:
                self.cap.release()
            except Exception:
                pass
            time.sleep(1.0)
            self._open()

    def read(self):
        """Newest frame not handed out yet: (True, frame), or (False, None)
        once the camera is gone for good."""
        with self.cond:
            self.cond.wait_for(
                lambda: self.seq != self.taken or self.stopped,
                timeout=CAMERA_RECONNECT_SECONDS + 5.0,
            )
            if self.seq == self.taken:
                return False, None
            self.taken = self.seq
            return True, self.frame

    def release(self):
        with self.cond:
            self.stopped = True
            self.cond.notify_all()
        self.thread.join(timeout=3)
        try:
            self.cap.release()
        except Exception:
            pass


class OverlayOps:
    """The detector does NOT paint a full-frame image any more. It records a
    short list of draw commands, and the display loop replays them straight
    onto the picture. That removes ~6 MB of per-frame masking work, and text
    keeps its true colours (the old mask dropped pure-black text pixels, which
    made labels look hollow)."""

    def __init__(self, shape):
        self.shape = shape
        self.ops = []

    def apply(self, frame):
        for op in self.ops:
            if op[0] == "r":
                cv2.rectangle(frame, op[1], op[2], op[3], op[4])
            else:
                cv2.putText(frame, op[1], op[2], op[3], op[4], op[5], op[6], op[7])


def ov_rectangle(target, pt1, pt2, color, thickness=1):
    if isinstance(target, OverlayOps):
        target.ops.append(("r", pt1, pt2, color, thickness))
    else:
        cv2.rectangle(target, pt1, pt2, color, thickness)


def ov_putText(target, text, org, font, scale, color, thickness=1,
               line_type=cv2.LINE_8):
    if isinstance(target, OverlayOps):
        target.ops.append(("t", text, org, font, scale, color, thickness, line_type))
    else:
        cv2.putText(target, text, org, font, scale, color, thickness, line_type)


class Profiler:
    """Average milliseconds per detector step, printed every few seconds."""

    LABELS = (
        ("vehicle", "vehicle model + draw"),
        ("plate", "plate model"),
        ("parse", "parse boxes"),
        ("roi", "ROI search"),
        ("post", "OCR sched + draw"),
    )

    def __init__(self):
        self.sums = defaultdict(float)
        self.count = 0
        self.last = time.perf_counter()

    def add(self, **parts):
        for k, v in parts.items():
            self.sums[k] += v
        self.count += 1

    def maybe_print(self):
        now = time.perf_counter()

        if (
            not PROFILE
            or self.count == 0
            or now - self.last < PROFILE_PRINT_SECONDS
        ):
            return

        n = self.count
        total = sum(self.sums.values()) / n
        parts = " | ".join(
            f"{label} {self.sums[key] / n * 1000:.0f} ms"
            for key, label in self.LABELS
        )
        fps = (1.0 / total) if total > 0 else 0.0
        print(f"[PROFILE] {parts} | total {total * 1000:.0f} ms (max {fps:.0f} fps)")

        self.sums.clear()
        self.count = 0
        self.last = now


PROF = Profiler()


def draw_vehicle_box(frame, box, vehicle_type, confidence):
    x1, y1, x2, y2 = map(int, box)

    ov_rectangle(frame, (x1, y1), (x2, y2), (255, 140, 0), 2)

    label = f"{vehicle_type} {confidence:.2f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.60
    thickness = 2

    (tw, th), _ = cv2.getTextSize(label, font, scale, thickness)
    top = max(0, y1 - th - 8)

    ov_rectangle(frame, (x1, top), (x1 + tw + 8, y1), (255, 140, 0), -1)
    ov_putText(
        frame, label, (x1 + 4, y1 - 5), font, scale,
        (0, 0, 0), thickness, cv2.LINE_AA
    )


# ============================================================
# CLEAN VEHICLE EVENT + RESULT BOX
# ============================================================

def vote_vehicle(track_id, vehicle_type, confidence):
    """Follow the blue box the plate is on (recent frames dominate)."""
    if vehicle_type != "UNKNOWN":
        votes = track_vehicle_votes[track_id]
        for k in list(votes):
            votes[k] *= 0.6
        votes[vehicle_type] += max(float(confidence), 0.05)


def get_track_vehicle(track_id):
    votes = track_vehicle_votes.get(track_id)
    if not votes:
        return "UNKNOWN"
    return max(votes, key=votes.get)


def hamming(a, b):
    """Character differences between two plates (99 if lengths differ)."""
    if len(a) != len(b):
        return 99
    return sum(x != y for x, y in zip(a, b))


def make_vehicle_snapshot(frame, plate_box):
    """Small picture of the vehicle with its plate boxed in green: the
    smallest vehicle box containing the plate, or the area around the plate
    when no vehicle box is known."""
    H, W = frame.shape[:2]
    px1, py1, px2, py2 = [int(v) for v in plate_box]
    pcx, pcy = (px1 + px2) / 2.0, (py1 + py2) / 2.0
    pw, ph = max(px2 - px1, 1), max(py2 - py1, 1)

    region = None
    for vx1, vy1, vx2, vy2, _vt, _vc in vehicle_detections:
        if point_inside_box(pcx, pcy, (vx1, vy1, vx2, vy2)):
            area = (vx2 - vx1) * (vy2 - vy1)
            if region is None or area < region[0]:
                region = (area, int(vx1), int(vy1), int(vx2), int(vy2))

    if region is not None:
        _a, x1, y1, x2, y2 = region
    else:
        x1, x2 = px1 - 3 * pw, px2 + 3 * pw
        y1, y2 = py1 - 8 * ph, py2 + 2 * ph

    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W, max(x2, px2 + 2)), min(H, max(y2, py2 + 2))
    snap = frame[y1:y2, x1:x2].copy()

    if snap.size == 0:
        return None

    cv2.rectangle(snap, (px1 - x1, py1 - y1), (px2 - x1, py2 - y1), (0, 255, 0), 3)

    if snap.shape[1] > SNAPSHOT_MAX_WIDTH:
        scale = SNAPSHOT_MAX_WIDTH / float(snap.shape[1])
        snap = cv2.resize(snap, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    return snap


def best_evidence_for(cluster):
    """(plate crop, vehicle snapshot) with the best quality over all tracks of
    a vehicle, or (None, None). Caller must hold result_lock."""
    best = None
    for tid in cluster["tracks"]:
        ev = best_evidence.get(tid)
        if ev is not None and (best is None or ev[0] > best[0]):
            best = ev
    return (best[1], best[2]) if best else (None, None)


def save_evidence(record, plate_crop, snapshot):
    """Write the proof images; returns their paths relative to OUTPUT_DIR."""
    if not SAVE_EVIDENCE:
        return "", ""

    day, clock = record["first_seen"].split(" ")
    stem = f"{clock.replace(':', '')}_{record['plate']}"
    folder = os.path.join(EVIDENCE_DIR, day)
    saved = []

    for suffix, image in (("plate", plate_crop), ("vehicle", snapshot)):
        if image is None or image.size == 0:
            saved.append("")
            continue
        path = os.path.join(folder, f"{stem}_{suffix}.jpg")
        try:
            os.makedirs(folder, exist_ok=True)
            if cv2.imwrite(path, image, [cv2.IMWRITE_JPEG_QUALITY, 90]):
                saved.append(os.path.relpath(path, OUTPUT_DIR).replace(os.sep, "/"))
                continue
        except (OSError, cv2.error) as e:
            print(f"[PROOF] Could not save {path}: {e}")
        saved.append("")

    return saved[0], saved[1]


def write_register(record, plate_image, vehicle_image):
    """Append one row to outputs/register/<date>.csv (header on a new file)."""
    if not REGISTER_DIR:
        return

    import csv

    day, clock = record["first_seen"].split(" ")
    path = os.path.join(REGISTER_DIR, f"{day}.csv")
    in_fleet = ""
    if known_plates:
        in_fleet = "yes" if record["plate"] in known_plates else "no"

    row = {
        "date": day,
        "time": clock,
        "plate": record["plate"],
        "bus": known_plate_labels.get(record["plate"], ""),
        "vehicle": record["vehicle_type"],
        "confidence": f"{record['ocr_confidence']:.0f}",
        "in_fleet": in_fleet,
        "plate_image": plate_image,
        "vehicle_image": vehicle_image,
    }

    try:
        os.makedirs(REGISTER_DIR, exist_ok=True)
        new_file = not os.path.isfile(path)
        with open(path, "a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=REGISTER_COLUMNS)
            if new_file:
                writer.writeheader()
            writer.writerow(row)
    except OSError as e:
        print(f"[REGISTER] Could not write {path}: {e}")


def load_known_plates():
    global known_plates, known_plate_labels
    known_plates = set()
    known_plate_labels = {}

    if not KNOWN_PLATES_FILE or not os.path.isfile(KNOWN_PLATES_FILE):
        print("[Known] No known_plates.txt found - plates are decided by voting only.")
        return

    try:
        with open(KNOWN_PLATES_FILE, "r", encoding="utf-8") as f:
            for line in f:
                p = clean_text(line.split("#")[0])
                if is_valid_plate(p):
                    known_plates.add(p)
                    if "#" in line:
                        label = line.split("#", 1)[1].strip()
                        if label:
                            known_plate_labels[p] = label
    except OSError as e:
        print(f"[Known] Could not read {KNOWN_PLATES_FILE}: {e}")
        return

    print(f"[Known] Loaded {len(known_plates)} known plate(s).")


def snap_to_known(plate, cluster):
    """Snap a voted plate to the one known plate that best explains ALL the
    readings, if it is clearly the best match. Otherwise leave it alone."""
    if not known_plates or plate in known_plates:
        return plate

    scored = []

    for k in known_plates:
        if len(k) != len(plate) or hamming(k, plate) > 2:
            continue

        total = 0.0
        wsum = 0.0

        for text, conf, _w in cluster["readings"]:
            wt = 0.5 + min(max(conf, 0.0), 100.0) / 100.0
            total += wt * min(hamming(k, text), 4)
            wsum += wt

        if wsum > 0:
            scored.append((total / wsum, k))

    if not scored:
        return plate

    scored.sort()

    if scored[0][0] > KNOWN_SNAP_MAX_AVG_DIFF:
        return plate

    if len(scored) > 1 and scored[1][0] - scored[0][0] < 0.3:
        return plate  # ambiguous between two known plates

    return scored[0][1]


def cluster_consensus(cluster):
    """Best plate for a vehicle: character vote over every reading of it."""
    readings = cluster["readings"]

    if not readings:
        return ""

    plate = character_vote(
        [r[0] for r in readings],
        [r[1] for r in readings],
        [r[2] for r in readings],
    )

    if not plate and cluster["weights"]:
        plate = max(cluster["weights"], key=cluster["weights"].get)

    if plate and known_plates:
        plate = snap_to_known(plate, cluster)

    return plate


def cluster_confidence(cluster, plate):
    """Average OCR confidence (0-100) of readings that agree with the plate."""
    confs = [c for t, c, _w in cluster["readings"] if hamming(t, plate) <= 1]

    if not confs:
        confs = [c for _t, c, _w in cluster["readings"]]

    return sum(confs) / len(confs) if confs else 0.0


def clusters_concurrent(a, b):
    """True if the two clusters were ever visible in the same frame, which
    means they are different physical vehicles."""
    for ta in a["tracks"]:
        fa = track_frames.get(ta)
        if not fa:
            continue
        for tb in b["tracks"]:
            fb = track_frames.get(tb)
            if fb and (fa & fb):
                return True
    return False


def _new_cluster(now):
    c = {
        "readings": [],                      # (text, confidence, box_width)
        "weights": defaultdict(float),       # exact text -> weight
        "tracks": set(),
        "vehicle_votes": defaultdict(float),
        "first_seen": now,
        "last_seen": now,
        "ready": False,
        "ready_since": 0.0,
        "logged": False,
    }
    plate_clusters.append(c)
    return c


def _cluster_add_locked(track_id, text, conf, width):
    """Add one valid reading to the vehicle it belongs to.
    Caller must hold result_lock."""
    now = time.time()
    weight = (
        (0.5 + min(max(conf, 0.0), 100.0) / 100.0)
        * min(max(width / 100.0, 0.5), 3.0)
    )

    cluster = track_to_cluster.get(track_id)

    if cluster is None:
        my_frames = track_frames.get(track_id, set())
        best = None

        for c in plate_clusters:
            if now - c["last_seen"] > CLUSTER_MATCH_WINDOW or not c["weights"]:
                continue

            # Already logged and gone for a while: this is the NEXT vehicle.
            if c["logged"] and now - c["last_seen"] > CLUSTER_LOGGED_REJOIN:
                continue

            top = max(c["weights"], key=c["weights"].get)
            d = hamming(top, text)

            if d > CLUSTER_MAX_DIFF:
                continue

            # Seen in the same frame as one of its tracks = another vehicle.
            if any(my_frames & track_frames.get(tr, set()) for tr in c["tracks"]):
                continue

            if best is None or d < best[0]:
                best = (d, c)

        cluster = best[1] if best else _new_cluster(now)
        track_to_cluster[track_id] = cluster

    cluster["readings"].append((text, float(conf), float(max(width, 1))))

    if len(cluster["readings"]) > 80:
        cluster["readings"].pop(0)

    cluster["weights"][text] += weight
    cluster["tracks"].add(track_id)
    cluster["last_seen"] = now


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
    plate, vehicle_type, vehicle_confidence, track_id, frame
):
    """Mark this track's vehicle as confirmed. The final plate is decided
    later (flush_pending_events) from ALL readings of the vehicle."""
    if not plate or not is_valid_plate(plate):
        return

    with result_lock:
        c = track_to_cluster.get(track_id)

        if c is None:
            return

        if vehicle_type != "UNKNOWN":
            votes = c["vehicle_votes"]
            for k in list(votes):
                votes[k] *= 0.9
            votes[vehicle_type] += 1.0

        if not c["ready"]:
            c["ready"] = True
            c["ready_since"] = time.time()


def flush_pending_events(force=False):
    """Log each confirmed vehicle ONCE, when its plate has stopped being read
    (or after EVENT_FINALIZE_DELAY), using the pooled-vote plate."""
    global last_vehicle_event

    now = time.time()
    done = []

    with result_lock:
        for c in plate_clusters:
            if c["logged"] or not c["ready"]:
                continue

            idle = now - c["last_seen"]
            waited = now - c["ready_since"]

            if not (
                force
                or idle >= CLUSTER_IDLE_FINALIZE
                or waited >= EVENT_FINALIZE_DELAY
            ):
                continue

            plate = cluster_consensus(c)

            if not plate or not is_valid_plate(plate):
                continue

            # The voted plate must be backed by real readings.
            support = sum(
                1 for t, _c, _w in c["readings"] if hamming(t, plate) <= 1
            )
            if (
                support < FINAL_MIN_SUPPORT
                or cluster_confidence(c, plate) < MIN_LOG_CONFIDENCE
            ):
                if force or idle > CLUSTER_MATCH_WINDOW:
                    c["logged"] = True   # never became a trustworthy plate
                continue

            c["logged"] = True

            if plate in confirmed_vehicle_events:
                continue

            # Same vehicle split into two clusters by a tracker/ID switch?
            duplicate = False
            for p, t_logged, other in logged_events:
                if (
                    now - t_logged < CLUSTER_MATCH_WINDOW
                    and hamming(p, plate) <= CLUSTER_MAX_DIFF
                    and not clusters_concurrent(other, c)
                ):
                    duplicate = True
                    break

            if duplicate:
                continue

            votes = c["vehicle_votes"]
            vtype = decide_vehicle_type(votes)

            record = {
                "plate": plate,
                "vehicle_type": vtype,
                "vehicle_confidence": 0.0,
                "ocr_confidence": float(cluster_confidence(c, plate)),
                "track_id": int(next(iter(c["tracks"]))),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "first_seen": time.strftime(
                    "%Y-%m-%d %H:%M:%S", time.localtime(c["first_seen"])
                ),
            }

            confirmed_vehicle_events[plate] = record
            recent_vehicle_event_time[plate] = now
            logged_events.append((plate, now, c))
            last_vehicle_event = record.copy()
            done.append((record, best_evidence_for(c)))

        # Housekeeping: forget old clusters.
        stale = [
            c for c in plate_clusters
            if now - c["last_seen"] > CLUSTER_PRUNE_AFTER
            and (c["logged"] or not c["ready"])
        ]

        for c in stale:
            plate_clusters.remove(c)

            for tr in list(c["tracks"]):
                if track_to_cluster.get(tr) is c:
                    del track_to_cluster[tr]

        logged_events[:] = [
            e for e in logged_events if now - e[1] < CLUSTER_PRUNE_AFTER
        ]

        # Proof images of tracks not seen for a while are no longer needed.
        for tid in [
            t for t, ev in best_evidence.items()
            if now - ev[3] > CLUSTER_PRUNE_AFTER
        ]:
            del best_evidence[tid]

    for rec, (plate_crop, snapshot) in done:
        print(
            "\n" + format_result_box(
                rec["plate"],
                rec["vehicle_type"],
                rec["ocr_confidence"],
                time.strftime("%H:%M:%S"),
            ) + "\n"
        )

        if RESULTS_LOG:
            try:
                with open(RESULTS_LOG, "a", encoding="utf-8") as f:
                    f.write(
                        f"{rec['timestamp']},"
                        f"{rec['plate']},"
                        f"{rec['vehicle_type']},"
                        f"{rec['ocr_confidence']:.0f}\n"
                    )
            except OSError:
                pass

        plate_image, vehicle_image = save_evidence(rec, plate_crop, snapshot)
        write_register(rec, plate_image, vehicle_image)

        if plate_image or vehicle_image:
            print(f"[PROOF] {rec['plate']}: outputs/{plate_image or vehicle_image}")


# ============================================================
# DRAW
# ============================================================

def draw_result(
    frame, x1, y1, x2, y2, track_id,
    vehicle_type="UNKNOWN", vehicle_confidence=0.0
):
    with result_lock:
        stable = stable_text.get(track_id, "")
        latest = latest_ocr.get(track_id)

        # Show the pooled, multi-track vote once the vehicle is confirmed.
        c = track_to_cluster.get(track_id)
        if stable and c is not None and c["ready"]:
            stable = cluster_consensus(c) or stable

    # Green box when stable, yellow otherwise.
    if stable:
        color = (0, 255, 0)
        plate_label = f"{stable} | ID {track_id}"
    else:
        color = (0, 220, 255)

        if latest:
            plate_label = f"Reading: {latest[0]}"
        else:
            plate_label = f"Reading... | ID {track_id}"

    if vehicle_type != "UNKNOWN":
        label = f"{vehicle_type} | {plate_label}"
    else:
        label = plate_label

    ov_rectangle(frame, (x1, y1), (x2, y2), color, 2)

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.65
    thickness = 2

    (tw, th), _ = cv2.getTextSize(label, font, scale, thickness)
    top = max(0, y1 - th - 12)

    ov_rectangle(frame, (x1, top), (x1 + tw + 8, y1), color, -1)
    ov_putText(
        frame, label, (x1 + 4, y1 - 5), font, scale,
        (0, 0, 0), thickness, cv2.LINE_AA
    )


# ============================================================
# MAIN
# ============================================================

def ensure_tracker_yaml():
    """Write a ByteTrack config tolerant of frame gaps; return its path."""
    global TRACKER_CFG

    if not USE_LOOSE_TRACKER:
        TRACKER_CFG = "bytetrack.yaml"
        return

    try:
        path = os.path.join(
            CONFIG_DIR, TRACKER_YAML
        )

        with open(path, "w", encoding="utf-8") as f:
            f.write(
                "tracker_type: bytetrack\n"
                "track_high_thresh: 0.25\n"
                "track_low_thresh: 0.1\n"
                "new_track_thresh: 0.25\n"
                "track_buffer: 60\n"
                "match_thresh: 0.95\n"
                "fuse_score: True\n"
            )

        TRACKER_CFG = path
        print(f"[Tracker] Using loose ByteTrack config: {path}")

    except Exception as e:
        TRACKER_CFG = "bytetrack.yaml"
        print(f"[Tracker] Could not write {TRACKER_YAML} ({e}); using default.")


# ============================================================
# DETECTION THREAD
# ============================================================
# The display loop (main thread) plays the video at its real speed and just
# draws the latest results. Detection runs here, on the newest frame only,
# with exactly the same logic as before. If detection is slower than the
# video it simply updates less often; it never slows the video down.

frame_cond = threading.Condition()
latest_frame = [-1, None]        # [frame index, clean BGR frame]
overlay_lock = threading.Lock()
latest_overlay = None            # (shape, OverlayOps, timestamp) of the last analysed frame
det_fps = 0.0
det_done_idx = -1                # last frame index the detector finished


def detector_loop(model, vehicle_model, device):
    global frame_count
    global vehicle_detections
    global latest_overlay
    global det_fps
    global det_done_idx

    import traceback

    last_idx = -1
    last_t = time.perf_counter()
    last_error = ""

    while not stop_event.is_set():

        with frame_cond:
            frame_cond.wait_for(
                lambda: stop_event.is_set() or latest_frame[0] != last_idx,
                timeout=0.2
            )

            if stop_event.is_set():
                break

            if latest_frame[0] == last_idx or latest_frame[1] is None:
                continue

            last_idx = latest_frame[0]
            clean_frame = latest_frame[1]

        try:
            layer = OverlayOps(clean_frame.shape)

            frame_count += 1
            _p0 = time.perf_counter()

            # ------------------------------------------------
            # UVH-26 VEHICLE CLASSIFICATION (every N frames)
            # ------------------------------------------------
            if frame_count == 1 or frame_count % VEHICLE_INFER_EVERY_N_FRAMES == 0:
                vehicle_results = vehicle_model.predict(
                    clean_frame,
                    conf=VEHICLE_CONFIDENCE,
                    imgsz=VEHICLE_IMGSZ,
                    device=device,
                    half=HALF,
                    max_det=MAX_DET_VEHICLES,
                    verbose=False
                )

                new_vehicle_detections = []
                vehicle_result = vehicle_results[0]

                if vehicle_result.boxes is not None:
                    for vbox in vehicle_result.boxes:
                        vxyxy = vbox.xyxy[0].cpu().numpy()
                        vx1, vy1, vx2, vy2 = map(int, vxyxy)

                        vx1 = max(0, vx1)
                        vy1 = max(0, vy1)
                        vx2 = min(layer.shape[1] - 1, vx2)
                        vy2 = min(layer.shape[0] - 1, vy2)

                        if vx2 <= vx1 or vy2 <= vy1:
                            continue

                        vclass_id = int(vbox.cls[0])
                        vconfidence = float(vbox.conf[0])
                        original_class = vehicle_model.names[vclass_id]
                        vehicle_type = map_vehicle_class(original_class)

                        if (
                            YELLOW_BODY_IS_BUS
                            and vehicle_type == "TRUCK"
                            and is_yellow_body(clean_frame, (vx1, vy1, vx2, vy2))
                        ):
                            vehicle_type = "BUS"

                        new_vehicle_detections.append(
                            (vx1, vy1, vx2, vy2, vehicle_type, vconfidence)
                        )

                vehicle_detections = update_vehicle_tracks(
                    suppress_duplicate_vehicle_detections(
                        new_vehicle_detections,
                        iou_threshold=0.35,
                        containment_threshold=0.65,
                    )
                )

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

            for (vx1, vy1, vx2, vy2, vehicle_type, vconfidence) in vehicle_detections:
                draw_vehicle_box(
                    layer, (vx1, vy1, vx2, vy2), vehicle_type, vconfidence
                )

            _p1 = time.perf_counter()

            # ------------------------------------------------
            # PLATE YOLO / tracking
            # ------------------------------------------------
            det_frame = to_gray3(clean_frame) if PLATE_DETECT_GRAYSCALE else clean_frame

            results = model.track(
                det_frame,
                persist=True,
                tracker=TRACKER_CFG,
                conf=YOLO_CONFIDENCE,
                imgsz=YOLO_IMGSZ,
                device=device,
                half=HALF,
                max_det=MAX_DET_PLATES,
                verbose=False
            )

            result = results[0]
            _p2 = time.perf_counter()

            # Collect plate detections, dropping ones that sit too high on
            # the vehicle (windshield banners etc.).
            plate_dets = []

            if result.boxes is not None:
                boxes = result.boxes

                for i in range(len(boxes)):
                    x1, y1, x2, y2 = map(int, boxes.xyxy[i].cpu().numpy())

                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(layer.shape[1] - 1, x2)
                    y2 = min(layer.shape[0] - 1, y2)

                    if x2 <= x1 or y2 <= y1:
                        continue

                    if DEBUG_DRAW_RAW:
                        ov_rectangle(layer, (x1, y1), (x2, y2), (255, 0, 255), 1)
                        ov_putText(
                            layer, f"raw {float(boxes.conf[i]):.2f}",
                            (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (255, 0, 255), 1, cv2.LINE_AA
                        )

                    # Sits high on a vehicle (windshield banner, sticker,
                    # destination board)? Not a plate: discard it.
                    suspect = not plate_position_ok((x1, y1, x2, y2))

                    if suspect and DROP_SUSPECT_PLATES:
                        continue

                    tid = (
                        int(boxes.id[i].item())
                        if boxes.id is not None else i
                    )
                    tid = stable_track_id(tid, (x1, y1, x2, y2))
                    plate_dets.append((x1, y1, x2, y2, tid, suspect))

            # Search the lower part of every vehicle that has no good plate
            # yet (not only when the whole frame is empty).
            _p3 = time.perf_counter()

            if ROI_FALLBACK:
                good_boxes = [
                    d[:4] for d in plate_dets
                    if not d[5] or d[4] in stable_text or d[4] in latest_ocr
                ]
                roi_dets = detect_plates_in_vehicle_roi(
                    clean_frame, model, device, canvas=layer,
                    existing_boxes=good_boxes
                )
                plate_dets = merge_plate_dets(
                    [d for d in plate_dets if not d[5]]
                    + [(d[0], d[1], d[2], d[3], d[4], False) for d in roi_dets]
                    + [d for d in plate_dets if d[5]]
                )

            _p4 = time.perf_counter()

            if plate_dets:

                for (x1, y1, x2, y2, track_id, suspect) in plate_dets:

                    box_width = x2 - x1

                    # Remember which frames each track appeared in: two tracks
                    # in the same frame are two different vehicles.
                    with result_lock:
                        track_frames[track_id].add(frame_count)

                    # Light text on a dark background = banner, not a plate.
                    if (
                        REJECT_DARK_BACKGROUND_PLATES
                        and track_id not in stable_text
                        and plate_is_dark_banner(clean_frame[y1:y2, x1:x2])
                    ):
                        continue

                    has_reads = (
                        track_id in stable_text or track_id in latest_ocr
                    )

                    # Track never produced a valid plate: stop wasting OCR
                    # on it and don't draw it. Suspect boxes get fewer tries.
                    fail_limit = (
                        SUSPECT_MAX_OCR_FAILS
                        if suspect and not has_reads else MAX_OCR_FAILS
                    )

                    if (
                        ocr_fail_count[track_id] >= fail_limit
                        and track_id not in stable_text
                    ):
                        # Gave up on this plate - but if it has grown a lot
                        # since (vehicle came closer), try reading it again.
                        w0 = ocr_block_width.get(track_id)
                        if w0 is None:
                            ocr_block_width[track_id] = box_width
                            if DEBUG_TRACK_DECISIONS:
                                print(
                                    f"[TRACK] plate id {track_id} paused after "
                                    f"{ocr_fail_count[track_id]} unreadable OCR tries "
                                    f"(width {box_width}px) - retry at "
                                    f"{int(OCR_RETRY_GROWTH * box_width)}px"
                                )
                            continue
                        if box_width < OCR_RETRY_GROWTH * w0:
                            continue
                        with result_lock:
                            ocr_fail_count[track_id] = 0
                        ocr_block_width.pop(track_id, None)

                    vehicle_type, vehicle_confidence = find_vehicle_for_plate(
                        (x1, y1, x2, y2)
                    )

                    vote_vehicle(track_id, vehicle_type, vehicle_confidence)
                    vehicle_type = get_track_vehicle(track_id)

                    # ------------------------------------------------
                    # OCR scheduling - never waits for Awiros.
                    # ------------------------------------------------
                    now = time.perf_counter()

                    should_ocr = (
                        now - last_ocr_time[track_id] >= OCR_INTERVAL
                    )

                    crop = clean_frame[y1:y2, x1:x2].copy()

                    quality = plate_quality(crop)

                    if quality > best_crop_quality[track_id]:
                        best_crop[track_id] = crop
                        best_crop_quality[track_id] = quality

                        if SAVE_EVIDENCE:
                            snapshot = make_vehicle_snapshot(
                                clean_frame, (x1, y1, x2, y2)
                            )
                            with result_lock:
                                best_evidence[track_id] = (
                                    quality, crop, snapshot, time.time()
                                )

                    if should_ocr:
                        best = best_crop.get(track_id)
                        best_q = best_crop_quality[track_id]

                        # Prefer the current frame if it's nearly as good as
                        # the best one, so votes come from independent
                        # readings. Fall back to the best crop otherwise.
                        if quality > 0 and quality >= 0.7 * best_q:
                            selected_crop = crop
                        else:
                            selected_crop = best

                        if selected_crop is not None:
                            sel_q = quality if selected_crop is crop else best_q

                            with ocr_request_lock:
                                # If the OCR is busy, keep the SHARPEST waiting
                                # crop instead of always replacing it by the newest.
                                old = ocr_pending.get(track_id)

                                if old is None or sel_q >= old[2]:
                                    ocr_pending[track_id] = (
                                        selected_crop.copy(),
                                        box_width,
                                        sel_q,
                                    )

                                ocr_request_event.set()

                        last_ocr_time[track_id] = now

                    # Suspect boxes stay invisible until they read a valid plate.
                    if not suspect or has_reads or (
                        track_id in stable_text or track_id in latest_ocr
                    ):
                        draw_result(
                            layer, x1, y1, x2, y2, track_id,
                            vehicle_type, vehicle_confidence
                        )

                    # Only the temporally confirmed plate becomes a final event.
                    with result_lock:
                        confirmed_plate = confirmed_text.get(track_id, "")

                    if confirmed_plate and vehicle_type != "UNKNOWN":
                        register_clean_vehicle_event(
                            confirmed_plate,
                            vehicle_type,
                            vehicle_confidence,
                            track_id,
                            layer
                        )

            if frame_count % 200 == 0:
                cutoff = frame_count - 600
                with result_lock:
                    for s in track_frames.values():
                        s.difference_update([f for f in s if f < cutoff])

            # The overlay is a short list of draw commands; the display loop
            # replays them straight onto the picture.
            _p5 = time.perf_counter()
            PROF.add(
                vehicle=_p1 - _p0, plate=_p2 - _p1, parse=_p3 - _p2,
                roi=_p4 - _p3, post=_p5 - _p4,
            )
            PROF.maybe_print()

            packed = (layer.shape, layer, time.perf_counter())

            with overlay_lock:
                latest_overlay = packed

            now_t = time.perf_counter()
            dt = now_t - last_t
            last_t = now_t
            if dt > 0:
                det_fps = 0.9 * det_fps + 0.1 * (1.0 / dt)

        except Exception:
            err = traceback.format_exc()
            if err != last_error:
                print("[DETECTOR ERROR]\n" + err)
                last_error = err

        finally:
            with frame_cond:
                det_done_idx = last_idx
                frame_cond.notify_all()


def configure(**settings):
    """Override module settings (used by the launcher scripts).

    Only existing UPPER_CASE settings can be set, so a typo fails loudly
    instead of being silently ignored."""
    global USE_LOOSE_TRACKER

    g = globals()
    for name, value in settings.items():
        if not name.isupper() or name not in g:
            raise KeyError(f"Unknown LPR setting: {name}")
        g[name] = value

    # Derived setting: keep it in step unless it was set explicitly.
    if "ACCURACY_MODE" in settings and "USE_LOOSE_TRACKER" not in settings:
        USE_LOOSE_TRACKER = not ACCURACY_MODE


def main():
    # All globals declared once, at the top.
    global last_vehicle_event
    global frame_count
    global awiros_ocr
    global vehicle_detections
    global HALF

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    stop_event.clear()

    if RESULTS_LOG:
        os.makedirs(os.path.dirname(os.path.abspath(RESULTS_LOG)), exist_ok=True)

    load_known_plates()

    if DEBUG_SAVE_CROPS:
        clean_debug_crop_folder()
        print(f"[DEBUG] ONE best crop per recognized plate/vehicle will be saved to: {DEBUG_CROPS_DIR}")

    print("=" * 65)
    print("REAL-TIME INDIAN LICENSE PLATE RECOGNITION - AWIROS")
    print("=" * 65)

    print("[1] Loading YOLO...")
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

    HALF = bool(USE_HALF_PRECISION and device != "cpu")

    print("[GPU] Warming up models...")
    try:
        _dummy = np.zeros((720, 1280, 3), np.uint8)
        model.predict(_dummy, imgsz=YOLO_IMGSZ, device=device, half=HALF, verbose=False)
        vehicle_model.predict(_dummy, imgsz=VEHICLE_IMGSZ, device=device, half=HALF, verbose=False)
    except Exception as e:
        print(f"[GPU] Warm-up failed ({e}); continuing without FP16.")
        HALF = False

    print(f"[GPU] Half precision (FP16): {'ON' if HALF else 'off'}")

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
        print("  - config\\en_dict.txt exists")
        print("  - third_party\\PaddleOCR folder exists (cloned automatically)")
        print(f"  - ocr_image.py is present in {SRC_DIR}")
        return

    live = CAMERA_INDEX is not None

    if live:
        print("[3] Opening live camera...")
        print(f"[CAMERA] {CAMERA_INDEX}")

        cap = LiveCamera(CAMERA_INDEX)

        if not cap.isOpened():
            cap.release()
            print()
            print("[ERROR] Camera could not be opened.")
            print(f"[ERROR] CAMERA_INDEX = {CAMERA_INDEX!r}")
            print("Check the camera is plugged in and not used by another app")
            print("(Zoom, Teams, the Camera app...). Try CAMERA_INDEX = 1 for a")
            print("second webcam, or an rtsp:// address for an IP camera.")
            return
    else:
        print("[3] Opening test video...")
        print(f"[VIDEO] {VIDEO_PATH}")

        if not os.path.isfile(VIDEO_PATH):
            print()
            print("[ERROR] Video file not found.")
            print(f"[ERROR] {VIDEO_PATH}")
            print("Change VIDEO_PATH in the launcher (webcam.py / videotest.py).")
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
    if not live:
        print(f"[VIDEO] Total frames: {total_frames}")

    worker = threading.Thread(target=ocr_worker, daemon=True)
    worker.start()

    print("[4] Camera started." if live else "[4] Video started.")
    print()
    print("Controls:")
    print("  Q / ESC -> quit")
    print("  R       -> reset OCR history")
    print("  SPACE   -> pause / resume video")
    print()
    print("Awiros OCR runs in a background thread.")
    if live:
        print("Live camera: the newest frame is always analysed.")
    else:
        print("Video playback is paced to the source video's FPS.")
    print()

    ensure_tracker_yaml()

    detector = threading.Thread(
        target=detector_loop,
        args=(model, vehicle_model, device),
        daemon=True
    )
    detector.start()

    previous_time = time.perf_counter()
    fps = 0.0
    paused = False
    play_fps = video_fps * PLAYBACK_SPEED
    play_t0 = time.perf_counter()
    read_idx = 0
    skipped_total = 0
    pause_started = 0.0
    base_frame = None
    prefetched = None   # next frame, decoded while the detector works

    try:

        while True:

            if not paused:
                # Behind real time? Skip frames instead of playing slowly.
                # (Never in ACCURACY_MODE: every frame is analysed.)
                due = int((time.perf_counter() - play_t0) * play_fps)
                behind = due - read_idx - 1

                if not live and not ACCURACY_MODE and behind > SKIP_IF_BEHIND_FRAMES:
                    for _ in range(min(behind, MAX_SKIP_FRAMES)):
                        if not cap.grab():
                            break
                        read_idx += 1
                        skipped_total += 1

                if prefetched is not None:
                    ret, new_frame = prefetched
                    prefetched = None
                else:
                    ret, new_frame = cap.read()

                if not ret:
                    if live:
                        print("[CAMERA] No more frames from the camera.")
                    else:
                        print("[VIDEO] Reached end of video.")
                    break

                read_idx += 1

                if MIRROR_CAMERA:
                    new_frame = cv2.flip(new_frame, 1)

                base_frame = new_frame

                # Hand the newest clean frame to the detector.
                with frame_cond:
                    latest_frame[0] = read_idx
                    latest_frame[1] = base_frame
                    frame_cond.notify_all()

                # While the detector works on this frame, decode the next one.
                if ACCURACY_MODE and not live and prefetched is None:
                    prefetched = cap.read()

                # ACCURACY_MODE: show this frame only after it was analysed.
                if ACCURACY_MODE:
                    with frame_cond:
                        frame_cond.wait_for(
                            lambda: det_done_idx >= read_idx
                            or stop_event.is_set(),
                            timeout=15.0
                        )

            if base_frame is None:
                continue

            # Display copy: original pixels + the latest detection overlay.
            frame = base_frame.copy()

            with overlay_lock:
                layer = latest_overlay

            if layer is not None and layer[0] == frame.shape:
                layer[1].apply(frame)

            # ------------------------------------------------
            # FPS (display speed / detection speed)
            # ------------------------------------------------
            current_time = time.perf_counter()
            delta = current_time - previous_time
            previous_time = current_time

            if delta > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / delta)

            ov_rectangle(frame, (0, 0), (640, 45), (0, 0, 0), -1)
            ov_putText(
                frame, f"Video LPR | Awiros | FPS: {fps:.0f} | Detect: {det_fps:.1f} | Skipped: {skipped_total}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                (255, 255, 255), 2, cv2.LINE_AA
            )

            # ------------------------------------------------
            # Print / log events that have waited long enough
            # (near-duplicate plates are merged here).
            # ------------------------------------------------
            flush_pending_events()

            # ------------------------------------------------
            # CLEAN VEHICLE STATUS
            # ------------------------------------------------
            with result_lock:
                current_event = (
                    last_vehicle_event.copy() if last_vehicle_event else None
                )

            if current_event:
                status = (
                    f"LAST: {current_event['vehicle_type']} | "
                    f"{current_event['plate']}"
                )
                ov_rectangle(frame, (0, 50), (430, 90), (0, 0, 0), -1)
                ov_putText(
                    frame, status, (10, 78),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                    (0, 255, 0), 2, cv2.LINE_AA
                )

            cv2.imshow("Video LPR - Awiros OCR", frame)

            # Sleep only if we are ahead of real time.
            if paused:
                wait_ms = 50
            elif ACCURACY_MODE or live:
                wait_ms = 1   # a live camera is paced by the camera itself
            else:
                ahead = (read_idx / play_fps) - (
                    time.perf_counter() - play_t0
                )
                wait_ms = min(100, max(1, int(ahead * 1000)))

            key = cv2.waitKey(wait_ms) & 0xFF

            if key in (ord("q"), 27):
                break

            if key == ord(" "):
                paused = not paused

                if paused:
                    pause_started = time.perf_counter()
                else:
                    play_t0 += time.perf_counter() - pause_started

                print("[VIDEO] PAUSED" if paused else "[VIDEO] RESUMED")
                continue

            if key == ord("r"):

                with result_lock:
                    track_history.clear()
                    track_confidence.clear()
                    track_width.clear()
                    stable_text.clear()
                    confirmed_text.clear()
                    last_vote_winner.clear()
                    winner_streak.clear()
                    latest_ocr.clear()
                    confirmed_vehicle_events.clear()
                    recent_vehicle_event_time.clear()
                    plate_clusters.clear()
                    track_to_cluster.clear()
                    track_frames.clear()
                    track_last_box.clear()
                    track_alias.clear()
                    logged_events.clear()
                    last_vehicle_event = None

                last_ocr_time.clear()
                best_crop.clear()
                best_evidence.clear()
                best_crop_quality.clear()
                track_vehicle_votes.clear()
                ocr_fail_count.clear()
                ocr_block_width.clear()
                roi_tracks.clear()
                vehicle_tracks.clear()

                with ocr_request_lock:
                    ocr_pending.clear()

                print("[RESET] OCR history cleared.")

    finally:

        # Write out anything still staged so the last vehicle isn't lost.
        flush_pending_events(force=True)

        if DEBUG_SAVE_CROPS:
            save_all_best_crops()

        stop_event.set()
        ocr_request_event.set()

        with frame_cond:
            frame_cond.notify_all()

        detector.join(timeout=3)

        if awiros_ocr is not None:
            awiros_ocr.close()
            awiros_ocr = None

        cap.release()
        cv2.destroyAllWindows()

        print()
        print("Video LPR stopped.")


if __name__ == "__main__":
    main()