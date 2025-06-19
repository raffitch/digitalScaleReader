import cv2
import pytesseract
import pandas as pd
from datetime import datetime
import argparse


def capture_scale(output_csv: str, display_roi=None, camera_index=0):
    """Capture weight readings from a camera pointed at a digital scale.

    Args:
        output_csv: Path to CSV file where readings will be stored.
        display_roi: Tuple (x, y, w, h) for cropping the region of interest containing the LCD display.
        camera_index: Index of the camera to use.
    """
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError("Unable to open camera")

    data = []
    last_value = None

    print("Press 'q' to quit...")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Crop to display ROI if provided
        if display_roi is not None:
            x, y, w, h = display_roi
            frame_roi = frame[y:y+h, x:x+w]
        else:
            frame_roi = frame

        gray = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2GRAY)
        # Threshold to make text more distinct
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Use pytesseract to do OCR on digits only
        config = "--psm 7 -c tessedit_char_whitelist=0123456789.-"
        text = pytesseract.image_to_string(thresh, config=config)
        text = text.strip()

        # Show for debugging
        cv2.imshow('Scale ROI', frame_roi)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # If the OCR result is non-empty and changed, record
        if text and text != last_value:
            timestamp = datetime.now().isoformat()
            print(f"{timestamp}: {text}")
            data.append({'timestamp': timestamp, 'weight': text})
            last_value = text

    cap.release()
    cv2.destroyAllWindows()
    if data:
        df = pd.DataFrame(data)
        df.to_csv(output_csv, index=False)
        # Also save to Excel for convenience
        excel_path = output_csv.rsplit('.', 1)[0] + '.xlsx'
        df.to_excel(excel_path, index=False)
        print(f"Saved {len(data)} readings to {output_csv} and {excel_path}")
    else:
        print("No data captured")


def main():
    parser = argparse.ArgumentParser(description="Record digital scale readings via camera")
    parser.add_argument('output', help='Output CSV file path')
    parser.add_argument('--roi', type=int, nargs=4, metavar=('X','Y','W','H'),
                        help='ROI of LCD display (x y w h)')
    parser.add_argument('--camera', type=int, default=0, help='Camera index (default 0)')
    args = parser.parse_args()

    capture_scale(args.output, display_roi=tuple(args.roi) if args.roi else None,
                  camera_index=args.camera)


if __name__ == '__main__':
    main()
