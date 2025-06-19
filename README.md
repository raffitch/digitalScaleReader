# Digital Scale Reader

This tool captures weight readings from a jewelry scale's LCD display using a webcam. It uses OCR (Tesseract) to read the digits and records each change in weight.

## Requirements

* Python 3.10+
* Tesseract OCR installed and accessible in your `PATH`
* Python packages listed in `requirements.txt`

Install packages via:

```bash
pip install -r requirements.txt
```

On Debian/Ubuntu systems, you can install Tesseract with:

```bash
sudo apt-get install tesseract-ocr
```

## Usage

```bash
python scale_reader.py output.csv --roi X Y W H --camera 0
```

Arguments:

- `output.csv`: Path to a CSV file where readings will be written. A matching Excel file (`.xlsx`) is also generated.
- `--roi X Y W H`: Optional. Defines the region of interest containing the LCD display (top-left `X`,`Y` coordinates and width `W`/height `H`). If not provided, the full frame is used.
- `--camera N`: Optional. Index of the webcam to use (default `0`).

While the program is running, a window shows the cropped display. Press `q` to stop recording. Each time a different weight reading is detected, it is logged with a timestamp.

## Example

```bash
python scale_reader.py weights.csv --roi 100 150 200 80
```

This records scale readings to `weights.csv` and `weights.xlsx`, using the specified ROI.
