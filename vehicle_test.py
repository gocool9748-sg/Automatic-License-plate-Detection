import cv2
from ultralytics import YOLO

# ==============================
# PATHS
# ==============================
MODEL_PATH = r"G:\LPR\models\uvh26_vehicle.pt"
VIDEO_PATH = r"G:\LPR\videos\movingcars.mp4"
OUTPUT_PATH = r"G:\LPR\results\vehicle_classified.mp4"

# ==============================
# LOAD MODEL
# ==============================
print("[1] Loading UVH-26 model...")

model = YOLO(MODEL_PATH)

print("[2] Model classes:")
print(model.names)

# ==============================
# CLASS MAPPING
# ==============================
CLASS_MAP = {
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

    "Others": "OTHER"
}

# ==============================
# OPEN VIDEO
# ==============================
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("[ERROR] Could not open video!")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"[3] Video: {width}x{height} @ {fps:.2f} FPS")

# ==============================
# OUTPUT VIDEO
# ==============================
fourcc = cv2.VideoWriter_fourcc(*"mp4v")

out = cv2.VideoWriter(
    OUTPUT_PATH,
    fourcc,
    fps,
    (width, height)
)

print("[4] Processing video...")
print("[5] Output:", OUTPUT_PATH)

# ==============================
# PROCESS FRAME BY FRAME
# ==============================
frame_count = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1

    # YOLO detection
    results = model.predict(
        frame,
        conf=0.40,
        imgsz=640,
        device=0,
        verbose=False
    )

    result = results[0]

    # Draw detections
    for box in result.boxes:

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        confidence = float(box.conf[0])

        class_id = int(box.cls[0])

        original_class = model.names[class_id]

        # Convert UVH class → our class
        vehicle_type = CLASS_MAP.get(
            original_class,
            "OTHER"
        )

        # Draw box
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        # Label
        label = f"{vehicle_type} {confidence:.2f}"

        cv2.putText(
            frame,
            label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    # Write frame to output video
    out.write(frame)

    # Show live processing window
    cv2.imshow("Indian Vehicle Classification", frame)

    # Press Q to stop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ==============================
# CLEANUP
# ==============================
cap.release()
out.release()
cv2.destroyAllWindows()

print()
print("======================================")
print("       PROCESSING COMPLETED")
print("======================================")
print(f"Output video:")
print(OUTPUT_PATH)