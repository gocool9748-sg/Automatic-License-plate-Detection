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

## Files

| File | Purpose |
| --- | --- |
| `webcam.py` | Real-time ANPR + vehicle classification from a webcam |
| `videotest.py` | Same pipeline on a video file (`VIDEO_PATH`) |
| `vehicle_test.py` | Vehicle classification only, writes an annotated video |
| `awiros_worker.py` | Persistent OCR worker started automatically by the scripts above |
| `test.py` | OCR on a single plate image or a folder of crops |
| `en_dict.txt` | Character dictionary for the OCR model |
| `models/best.pt` | YOLO license plate detector |
| `models/model.safetensors` | Awiros ANPR-OCR weights (stored with Git LFS) |
| `images/` | Sample images |

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
without an NVIDIA GPU, install `paddlepaddle` instead.

On first run the OCR code clones the PaddleOCR repo into `PaddleOCR/` (needed
to build the model). Pass `--paddleocr_dir` to `test.py` to use an existing copy.

You also need the **UVH-26 vehicle model** (`uvh26_vehicle.pt`). It is not in
this repo; place it in `models/`.

### Configure paths

`webcam.py`, `videotest.py` and `vehicle_test.py` use absolute Windows paths
under `G:\LPR\`. Edit the constants at the top of each script to match where
you cloned the repo:

```python
YOLO_WEIGHTS    = r"G:\LPR\models\best.pt"
VEHICLE_WEIGHTS = r"G:\LPR\models\uvh26_vehicle.pt"
AWIROS_WORKER   = r"G:\LPR\awiros_worker.py"
RESULTS_LOG     = r"G:\LPR\results.log"
DEBUG_CROPS_DIR = r"G:\LPR\debug_crops_webcam"
VIDEO_PATH      = r"G:\LPR\videos\srm5.mp4"   # videotest.py only
```

## Usage

**Webcam**

```bash
python webcam.py
```

**Video file** (set `VIDEO_PATH` first)

```bash
python videotest.py
```

Keys: `q` / `Esc` quit · `r` reset tracked plates · `Space` pause (video only).

Final results are appended to `results.log` as CSV
(`time,plate,vehicle,confidence`). The best plate crop for each vehicle is
saved to the debug crops folder while `DEBUG_SAVE_CROPS = True`.

**OCR on plate images**

```bash
python test.py --image_path plate.jpg --weights models/model.safetensors --device cpu
python test.py --image_path crops/ --weights models/model.safetensors --output_json results.json
```

**Vehicle classification only**

```bash
python vehicle_test.py
```

## Tuning

Main settings at the top of `webcam.py` / `videotest.py`:

| Setting | Default | Effect |
| --- | --- | --- |
| `YOLO_CONFIDENCE` | 0.45 | Minimum plate detection confidence |
| `VEHICLE_CONFIDENCE` | 0.40 (webcam), 0.25 (video) | Minimum vehicle detection confidence |
| `OCR_INTERVAL` | 0.14 s | Time between OCR requests |
| `MIN_VOTES` | 2 | Readings needed before a plate is accepted |
| `MIN_OCR_CONFIDENCE` | 45 | Discard OCR readings below this |
| `MIN_PLATE_PIXEL_WIDTH` | 35 | Ignore plates smaller than this |
| `BLUR_VARIANCE_THRESHOLD` | 20.0 | Skip crops that are too blurry |
