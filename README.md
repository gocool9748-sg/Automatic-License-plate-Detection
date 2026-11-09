# Automatic License Plate Detection (Indian ANPR)

Real-time automatic number plate recognition for Indian vehicles. A YOLO model
finds license plates, a PaddleOCR-based recogniser (Awiros ANPR-OCR, PP-OCRv5
SVTR_HGNet) reads them, and a separate UVH-26 YOLO model classifies the vehicle
(car, bike, auto, bus, truck).

## How it works

```
camera / video frame
   ├── YOLO plate detector (models/best.pt) + ByteTrack  → plate crops per track
   │       └── sharpest crop → awiros_worker.py (TCP, separate process) → raw text
   │               └── Indian-format correction + per-character voting → stable plate
   └── UVH-26 vehicle model (uvh26_vehicle.pt)           → vehicle type per track
```

- **Plate format correction**: readings are checked against the Indian pattern
  `AA 00 AAA 0000` (regex `^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{4}$`), and common
  OCR mix-ups (`0/O`, `1/I`, `5/S`, `8/B`, …) are fixed by position.
- **Temporal voting**: several readings of the same tracked plate are combined
  character by character, so one bad frame does not change the result.
- **Separate OCR process**: Paddle runs in `awiros_worker.py`, a different Python
  process from PyTorch/YOLO, because the two clash over CUDA/cuDNN DLLs on
  Windows. Plate crops are sent over `127.0.0.1:48731`.

## Project layout

```
LPR/
├── webcam.py                 run this: live webcam / IP camera
├── videotest.py              run this: a video file
├── src/
│   ├── lpr.py                the pipeline engine (all settings at the top)
│   ├── awiros_worker.py      OCR worker, started automatically
│   └── ocr_image.py          OCR on a plate image / folder of crops
├── tools/
│   └── vehicle_test.py       vehicle classification only (annotated video)
├── config/
│   ├── en_dict.txt           OCR character dictionary
│   ├── bytetrack_lpr.yaml    plate tracker settings
│   └── known_plates.example.txt   fleet list template (copy to known_plates.txt)
├── models/                   best.pt, model.safetensors (Git LFS), uvh26_vehicle.pt
├── videos/                   input videos (not in git)
├── images/                   sample images
├── outputs/                  register/, evidence/, results.log, debug_crops/ (created on first run)
├── third_party/PaddleOCR/    cloned automatically on first run (not in git)
├── docs/ACCURACY_FIXES.md    what each accuracy setting fixes and why
├── tests/                    unit tests (no GPU or model weights needed)
└── requirements.txt, requirements-dev.txt
```

The two launchers in the top folder only set a few options (`SETTINGS`) and
then run `src/lpr.py`, so both always use the same pipeline.

## Setup

```bash
git lfs install
git clone https://github.com/gocool9748-sg/Automatic-License-plate-Detection.git
cd Automatic-License-plate-Detection
git lfs pull            # downloads models/model.safetensors (~150 MB)

python -m venv venv
venv\Scripts\activate   # Windows  (Linux/macOS: source venv/bin/activate)
pip install -r requirements.txt
```

`requirements.txt` installs `paddlepaddle-gpu` on Windows/Linux. On a machine
without an NVIDIA GPU, install `paddlepaddle` instead. The OCR worker detects
this and runs on the CPU (slower).

On first run the OCR code clones the PaddleOCR repo into `third_party/PaddleOCR/`
(needed to build the model). Pass `--paddleocr_dir` to `src/ocr_image.py` to use an
existing copy.

You also need the **UVH-26 vehicle model** (`uvh26_vehicle.pt`). It is not in
this repo; place it in `models/`.

### File locations

All paths are relative to the repo folder, so the scripts work wherever you
clone it. Expected layout:

```
models/best.pt               plate detector (in repo)
models/model.safetensors     OCR weights (Git LFS)
models/uvh26_vehicle.pt      vehicle model (add yourself)
videos/*.mp4                 input videos (VIDEO_PATH in webcam.py / videotest.py)
videos/movingcars.mp4        input for tools/vehicle_test.py
outputs/register/<date>.csv  daily gate register: which bus came in when (opens in Excel)
outputs/evidence/<date>/     proof images: plate crop + vehicle snapshot per entry
outputs/results.log          per-vehicle CSV log (simple, all days in one file)
outputs/debug_crops/         best plate crop per vehicle
outputs/vehicle_test/        tools/vehicle_test.py output video
config/known_plates.txt      your fleet list (optional, see Tuning)
```

To use a different video, change `VIDEO_PATH` in `webcam.py` / `videotest.py`
(or at the top of `tools/vehicle_test.py`).

## Usage

Both launchers run the same pipeline (`lpr.py`); they only differ in a few
options set in their `SETTINGS`.

**Live webcam** - reads `CAMERA_INDEX` (0 = first webcam, 1 = second, or an
IP camera address like `"rtsp://user:pass@192.168.1.20:554/stream1"`). The
camera is read in the background and the newest frame is always analysed, so
results never lag behind; a camera that drops out is reconnected for up to
`CAMERA_RECONNECT_SECONDS`:

```bash
python webcam.py
```

**Video file** - set `VIDEO_PATH` in `videotest.py`. The video plays at
`PLAYBACK_SPEED` (0.4 = slow motion) and detection keeps up as best it can.
Set `ACCURACY_MODE=True` there to analyse every frame instead:

```bash
python videotest.py
```

Keys: `q` / `Esc` quit · `r` reset tracked plates · `Space` pause (video only).

**Gate register and proof.** Every vehicle is logged once, with proof:

- `outputs/register/2026-11-06.csv` - one file per day, one row per vehicle:

  | date | time | plate | bus | vehicle | confidence | in_fleet | plate_image | vehicle_image |
  |---|---|---|---|---|---|---|---|---|
  | 2026-11-06 | 08:12:43 | TN48BK2144 | bus 88 | BUS | 99 | yes | evidence/2026-11-06/081243_TN48BK2144_plate.jpg | evidence/.../081243_TN48BK2144_vehicle.jpg |

  `time` is when the vehicle was first seen. `bus` and `in_fleet` come from
  `config/known_plates.txt` (write the bus name after a `#`, e.g.
  `TN48BK2144  # bus 88`); `in_fleet = no` flags a plate that is not in your list.
- `outputs/evidence/<date>/` - the sharpest plate crop and a vehicle snapshot
  with the plate boxed in green, named `<time>_<plate>_plate.jpg` /
  `_vehicle.jpg`, so a disputed entry can be checked in seconds.
- `outputs/results.log` - the same entries as plain CSV
  (`time,plate,vehicle,confidence`), all days in one file.

Set `SAVE_EVIDENCE = False` in `src/lpr.py` to skip the images.

**OCR on plate images**

```bash
python src/ocr_image.py --image_path plate.jpg --weights models/model.safetensors --device cpu
python src/ocr_image.py --image_path crops/ --weights models/model.safetensors --output_json results.json
```

**Vehicle classification only**

```bash
python tools/vehicle_test.py
```

## Tests

The pure helpers in `src/lpr.py`, `src/ocr_image.py` and
`awiros_worker.py` (plate text cleaning and voting, per-track plate history,
crop quality, vehicle box overlap, plate-to-vehicle matching, vehicle type
voting, one-record-per-vehicle event logging, known-plate snapping, duplicate
plate box merging, tracker id jump detection, plate layout fixing, plate
structure patterns, dark banner and yellow bus body checks, plate position on
the vehicle, vehicle cluster overlap and consensus, persistent vehicle box
tracking, ROI plate IDs, OCR crop enhancement, overlay draw commands, the step
profiler, launcher settings, live camera reader, OCR socket framing, the
ByteTrack config loader, proof image saving, the gate register CSV, classical
plate candidates) have unit tests that need no GPU or model
weights. torch and ultralytics are stubbed out, so only a few light packages
are required:

```bash
pip install -r requirements-dev.txt   # pytest, opencv-python, numpy
python -m pytest tests
```

## Tuning

Main settings at the top of `src/lpr.py` (a launcher can override any of them in
its `SETTINGS`):

| Setting | Default | Effect |
| --- | --- | --- |
| `YOLO_CONFIDENCE` | 0.25 | Minimum plate detection confidence |
| `VEHICLE_CONFIDENCE` | 0.25 | Minimum vehicle detection confidence |
| `OCR_INTERVAL` | 0.03 s | Time between OCR requests per plate |
| `MIN_VOTES` | 5 | Readings needed before a plate is shown as stable |
| `MIN_OCR_CONFIDENCE` | 30 | Discard OCR readings below this |
| `MIN_LOG_CONFIDENCE` | 70 | A vehicle is logged only if its plate reads average at least this |
| `MIN_PLATE_PIXEL_WIDTH` | 20 | Ignore plates smaller than this |
| `BLUR_VARIANCE_THRESHOLD` | 3.0 | Skip crops that are too blurry |
| `PLATE_DETECT_GRAYSCALE` | True | Feed the plate detector a grayscale frame (finds yellow-on-yellow bus plates) |
| `HOME_STATE` | `TN` | A state code one character off is corrected to this |
| `CLUSTER_MAX_DIFF` | 1 | Plates this many characters apart are treated as the same vehicle |
| `YELLOW_BODY_IS_BUS` | True | A yellow vehicle the model calls TRUCK is logged as BUS |

See [docs/ACCURACY_FIXES.md](docs/ACCURACY_FIXES.md) for what each accuracy setting fixes. For the
best results, copy `config/known_plates.example.txt` to `config/known_plates.txt` and list every
fleet plate: readings 1-2 characters off are then corrected to the right bus.
