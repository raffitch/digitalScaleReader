#!/usr/bin/env python3
"""
scale_reader.py – Auto-detect a green backlit LCD, prep it for OCR,
then read with Tesseract 4’s LSTM and log stable changes.
"""

import argparse
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import pytesseract


def find_green_roi(frame):
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


def preprocess_for_ocr(crop):
    # 1) Grayscale
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    # 2) Otsu threshold (normal): dark digits → black, bright bg → white
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # 3) Median blur to remove speckles
    binary = cv2.medianBlur(binary, 3)
    # 4) Pad 10px white border so text isn't touching edges
    binary = cv2.copyMakeBorder(binary, 10,10,10,10,
                                cv2.BORDER_CONSTANT, value=255)
    # 5) Upscale 2× for Tesseract
    binary = cv2.resize(binary, None, fx=2, fy=2,
                        interpolation=cv2.INTER_CUBIC)
    return binary


def capture_scale(output_csv, camera_index=0, debounce=3):
    cap = cv2.VideoCapture(camera_index, cv2.CAP_ANY)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {camera_index}")

    # Auto-detect the green backlight region
    print("Detecting screen… hold scale steady")
    roi = None
    while roi is None:
        ret, frame = cap.read()
        if not ret:
            continue
        roi = find_green_roi(frame)
    x0,y0,w0,h0 = roi
    print(f"  → ROI = x:{x0}, y:{y0}, w:{w0}, h:{h0}")

    data, last, cand, streak = [], None, None, 0
    print("Starting capture — press 'q' to quit")

    # Tesseract: LSTM engine, single-line, digits+dot only
    tess_cfg = "--oem 1 --psm 7 -c tessedit_char_whitelist=0123456789."

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Draw ROI
        cv2.rectangle(frame, (x0,y0), (x0+w0, y0+h0), (0,255,0), 2)
        crop = frame[y0:y0+h0, x0:x0+w0]

        # Preprocess & OCR
        proc = preprocess_for_ocr(crop)
        text = pytesseract.image_to_string(proc, config=tess_cfg).strip()
        reading = "".join(ch for ch in text if ch.isdigit() or ch == ".")

        # Debounce: require same reading for N frames
        if reading == cand:
            streak += 1
        else:
            cand, streak = reading, 1

        if cand and cand != last and streak >= debounce:
            ts = datetime.now().isoformat(timespec="seconds")
            print(f"{ts}: {cand}")
            data.append({"timestamp": ts, "weight": cand})
            last = cand

        # Overlay result
        cv2.putText(frame, f"Value: {last or '-'}", (10,35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        # Show windows
        cv2.imshow("Live", frame)
        cv2.imshow("OCR Input", proc)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Save to CSV + XLSX
    if data:
        df = pd.DataFrame(data)
        df.to_csv(output_csv, index=False)
        df.to_excel(output_csv.replace(".csv", ".xlsx"), index=False)
        print(f"Saved {len(data)} readings to {output_csv}")
    else:
        print("No data captured.")


def main():
    p = argparse.ArgumentParser(
        description="Record scale readings via camera + Tesseract LSTM"
    )
    p.add_argument("output", help="Output CSV path")
    p.add_argument("--camera", type=int, default=0, help="Camera index")
    p.add_argument(
        "--debounce", type=int, default=3,
        help="Frames a reading must persist before logging"
    )
    args = p.parse_args()
    capture_scale(args.output, camera_index=args.camera, debounce=args.debounce)


if __name__ == "__main__":
    main()
