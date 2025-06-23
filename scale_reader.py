#!/usr/bin/env python3
"""
scale_reader.py – Auto-detect a green backlit LCD, calibrate digit positions
with a "heavy" weight, then read via seven-segment masks and log stable changes.
"""

import argparse
from datetime import datetime

import cv2
import numpy as np
import pandas as pd

# --- seven-segment definitions ----------------------------------

# each segment is defined by fractional (x1,y1)->(x2,y2) within a digit ROI
SEGMENTS = {
    0: ((0.15,0.05),(0.85,0.20)),  # top
    1: ((0.80,0.15),(0.95,0.50)),  # top-right
    2: ((0.80,0.50),(0.95,0.85)),  # bottom-right
    3: ((0.15,0.80),(0.85,0.95)),  # bottom
    4: ((0.05,0.50),(0.20,0.85)),  # bottom-left
    5: ((0.05,0.15),(0.20,0.50)),  # top-left
    6: ((0.15,0.45),(0.85,0.55)),  # middle
}

# map the set of "on" segments -> digit character
DIGIT_MAP = {
    frozenset([0,1,2,3,4,5])      : '0',
    frozenset([1,2])              : '1',
    frozenset([0,1,6,4,3])        : '2',
    frozenset([0,1,6,2,3])        : '3',
    frozenset([5,6,1,2])          : '4',
    frozenset([0,5,6,2,3])        : '5',
    frozenset([0,5,6,2,3,4])      : '6',
    frozenset([0,1,2])            : '7',
    frozenset([0,1,2,3,4,5,6])    : '8',
    frozenset([0,1,2,3,5,6])      : '9',
}

def recognize_digit(bin_roi, on_thresh=0.5):
    """Return the digit represented by a binarized ROI."""
    h, w = bin_roi.shape
    on = set()
    for seg_id, ((x1,y1),(x2,y2)) in SEGMENTS.items():
        xa, ya = int(x1*w), int(y1*h)
        xb, yb = int(x2*w), int(y2*h)
        patch = bin_roi[ya:yb, xa:xb]
        if patch.size == 0:
            continue
        if cv2.countNonZero(patch) / float((xb-xa)*(yb-ya)) > on_thresh:
            on.add(seg_id)
    return DIGIT_MAP.get(frozenset(on), '?')


# --- ROI & calibration helpers ------------------------------------

def find_green_roi(frame):
    """Locate the largest greenish region in the frame."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (40,100,100), (90,255,255))
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (5,5)),
        iterations=2
    )
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    x,y,w,h = cv2.boundingRect(max(cnts, key=cv2.contourArea))
    return x, y, w, h


def capture_scale(output_csv, camera_index=0, debounce=3):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
def capture_scale(output_csv, camera_index=0, debounce=3, debug=False):

    print("Detecting screen… hold scale steady")
    roi = None
    while roi is None:
        ret, frame = cap.read()
        if not ret:
            continue
        roi = find_green_roi(frame)
    x0,y0,w0,h0 = roi
    print(f"  → ROI = x:{x0}, y:{y0}, w:{w0}, h:{h0}")

    # --- calibration pass ---
    print("Calibrating digit positions: place HEAVY weight on scale, then press 'c'")
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        crop = frame[y0:y0+h0, x0:x0+w0]
        cv2.imshow("Calibrate (press 'c')", crop)
        if cv2.waitKey(1) & 0xFF == ord('c'):
            break
    cv2.destroyWindow("Calibrate (press 'c')")

    gray0 = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    boxes = calibrate_digit_boxes(gray0)
    if not boxes:
        raise RuntimeError("Failed to find any digit boxes during calibration.")
    widths = [w for _,w in boxes]
    median_w = np.median(widths)
    elems = []
    for x,w in boxes:
        kind = 'dot' if w < median_w*0.6 else 'digit'
        elems.append((x,w,kind))
    elems.sort(key=lambda t: t[0])
    print(f"  → Found {sum(1 for _,_,k in elems if k=='digit')} digits "
          f"and {sum(1 for _,_,k in elems if k=='dot')} dot(s).")

    data, last, cand, streak = [], None, None, 0
    print("Starting capture — press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.rectangle(frame, (x0,y0),(x0+w0,y0+h0),(0,255,0),2)
        crop = frame[y0:y0+h0, x0:x0+w0]

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binary = cv2.morphologyEx(
            binary, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT,(3,3)),
            iterations=1
        )

        reading = ""
        for x,w,kind in elems:
            sub = binary[:, x:x+w]
            if kind == 'digit':
                reading += recognize_digit(sub)
            else:
                if cv2.countNonZero(sub) > 0:
                    reading += '.'

                if debug:
                    cv2.rectangle(frame, (x0+x, y0), (x0+x+w, y0+h0),
                                  (0,0,255), 1)
                    h = sub.shape[0]
                    for ((sx1,sy1),(sx2,sy2)) in SEGMENTS.values():
                        xa, ya = int(sx1*w), int(sy1*h)
                        xb, yb = int(sx2*w), int(sy2*h)
                        cv2.rectangle(frame,
                                      (x0+x+xa, y0+ya),
                                      (x0+x+xb, y0+yb),
                                      (255,0,0), 1)
        if reading == cand:
            streak += 1
        else:
            cand, streak = reading, 1

        if cand and cand != last and streak >= debounce:
            ts = datetime.now().isoformat(timespec="seconds")
            print(f"{ts}: {cand}")
            data.append({"timestamp": ts, "weight": cand})
            last = cand

        cv2.putText(frame, f"Value: {last or '-'}", (10,35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.imshow("Live", frame)
        cv2.imshow("Binary", binary)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if data:
        df = pd.DataFrame(data)
        df.to_csv(output_csv, index=False)
        df.to_excel(output_csv.replace(".csv", ".xlsx"), index=False)
        print(f"Saved {len(data)} readings to {output_csv}")
    else:
        print("No data captured.")


def main():
    p = argparse.ArgumentParser(
        description="Record scale readings via camera + seven-segment masks"
    )
    p.add_argument("output", help="Output CSV path")
    p.add_argument("--camera", type=int, default=0, help="Camera index")
    p.add_argument(
        "--debounce", type=int, default=3,
        help="Frames a reading must persist before logging"
    )
    p.add_argument(
        "--debug", action="store_true",
        help="Display digit and segment bounding boxes"
    )
    args = p.parse_args()
    capture_scale(args.output, camera_index=args.camera, debounce=args.debounce,
                  debug=args.debug)


if __name__ == "__main__":
    main()
