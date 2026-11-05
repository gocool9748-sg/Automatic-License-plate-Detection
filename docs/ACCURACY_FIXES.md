# LPR accuracy fixes (Nov 2026)

These changes are in `src/lpr.py` (the pipeline engine used by both `webcam.py` and
`videotest.py`). Search it for `ACCURACY FIXES` to find the new settings.

## What was wrong (found by testing on `cb7.mp4`)

1. **The plate detector didn't see the bus plates.** The college buses have
   yellow plates painted on a yellow body, with no border. `best.pt` was
   trained on normal bordered plates, so in colour it almost never detected
   them. It kept detecting the **"COLLEGE BUS" banner** on the windshield instead.
   The OCR then read the banner as things like `DL1EGE809` or `DL2CE8805`, and
   some of those reached the results log.
2. **Plate-format correction broke correct reads.** In `videotest.py`, a 9-character
   reading tried the 1-digit RTO layout first, so a correct `TN48W9259` became
   `TN4BW9259`. That's where `TN4BW9259`, `TN4BW4530`, `TN4BW4504` and `TN4BF2177`
   in your log came from.
3. **The next bus could disappear.** Plates up to 2 characters apart, seen
   within 30 seconds, were merged into one vehicle. Your fleet is mostly
   `TN48BK21xx`, so `TN48BK2113` (bus 91) was swallowed by `TN48BK2144` (bus 88)
   and never logged.
4. **Junk reads could be voted into a fake plate.** Character voting over
   several bad reads could build a plate nobody read, e.g. `MH01A1101`.
5. **Buses were logged as TRUCK.** Up close, the vehicle model often calls a
   bus "Truck".

## What changed

| Setting / function | What it does |
|---|---|
| `PLATE_DETECT_GRAYSCALE = True` | The plate detector gets a grayscale copy of the frame; OCR still gets the colour crop. On `cb7.mp4` the yellow plates are now detected from far away. Example: frame 720 had no box in colour and gets 0.78 confidence in grayscale. |
| `normalize_plate` / `fix_position_characters` | Tries every Indian layout (`AA 00 A-AAA 0000` and BH series `00 BH 0000 AA`) and keeps the one that needs the fewest character corrections. Also drops stray characters (`TN48BJ117J5` → `TN48BJ1175`, `INDTN48BK2113` → `TN48BK2113`). |
| `ONE_DIGIT_RTO_STATES = {"DL"}` | Only Delhi has 1-digit RTO codes, so `TN4BW9259` can never be logged. |
| `HOME_STATE = "TN"` | A state code one character off (`TM48…`, `1N48…`, `IN48…`) is corrected to `TN`. |
| Extra look-alike fixes | `4↔A`, `7→T`, `U→0`, `J→1` added. |
| `JUNK_OCR_WORDS` | OCR text like `COLLEGE` or `UNREADABLE` is rejected, and that box isn't read again, so the banner can't become a plate. |
| `CLUSTER_MAX_DIFF = 1`, `CLUSTER_MATCH_WINDOW = 8.0`, `CLUSTER_LOGGED_REJOIN = 2.0` | Only 1-character look-alikes within a few seconds are treated as the same vehicle, so consecutive fleet buses stay separate. |
| `MIN_LOG_CONFIDENCE = 70` + support check | A vehicle is logged only when its final plate has at least `FINAL_MIN_SUPPORT` reads within 1 character of it, at 70%+ average OCR confidence. |
| `BUS_MIN_SHARE = 0.30` | A vehicle with at least 30% of its class votes for BUS is logged as BUS. |
| `YELLOW_BODY_IS_BUS = True` | A "TRUCK" box that is mostly yellow paint counts as a BUS. Set it to False if yellow trucks also use this gate. |

## Result on `cb7.mp4` (offline run of the real pipeline)

| Vehicle | Before | After |
|---|---|---|
| Bus 27 | read once, logged as `TN4BW9259` | `TN48W9259 BUS`, 99% |
| Bus 88 | mostly the "COLLEGE BUS" banner was read | `TN48BK2144 BUS`, 99% |
| Bus 91 | logged as TRUCK, or not at all | `TN48BK2113 BUS`, 99% |
| Car | correct | `TN45AX5428 CAR`, 99% |

Bus 68's rear plate (`TN48BJ1175`) is visible for only about half a second and was
not logged in the offline test (run at 12.5 analysed frames per second).

## For even better accuracy

* **Known fleet list:** copy `config/known_plates.example.txt` to
  `config/known_plates.txt` and add every bus plate. A reading that's 1–2 characters
  off is then corrected to the right bus.
* **Retrain the plate detector** on frames from your own gate videos (yellow
  bus plates), so it doesn't need the grayscale workaround.
