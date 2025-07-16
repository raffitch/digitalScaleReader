# Digital Scale Reader

This tool captures weight readings from a jewelry scale's LCD display using a webcam. The program automatically finds the green backlit screen, asks you to place a heavy object so all segments light up, then learns the position of each digit. After calibration it reads the seven-segment display and logs weight changes.

## Requirements

* Python 3.10+
* Python packages listed in `requirements.txt`

Install packages via:

```bash
pip install -r requirements.txt
```

## Usage

```bash
python scale_reader.py output.csv --camera 0
```

Arguments:

- `output.csv`: Path to a CSV file where readings will be written. A matching Excel file (`.xlsx`) is also generated.
- `--camera N`: Optional. Index of the webcam to use (default `0`).
- `--debounce M`: Frames a reading must persist before logging (default `3`).
- `--debug`: Show bounding boxes for digits and segments during capture.

When started, the program detects the LCD region. It then prompts you to place a heavy weight on the scale so every digit shows `8`. Press `c` to capture the digit layout, remove the weight, and the program begins logging. Press `q` to stop.

## Example

```bash
python scale_reader.py weights.csv --camera 0 --debug
```

This records scale readings to `weights.csv` after a one-time calibration with a heavy weight. The `--debug` flag overlays digit and segment boxes for troubleshooting.

## Firmware

The microcontroller firmware resides in `scaleReaderArduino.ino`. It reads the HX711 load cell and streams the weight in grams over the serial port. When using an ESP8266 NodeMCU v2 connect the HX711 pins as follows:

- DT → D6
- SCK → D7

Compile the sketch with the ESP8266 board package and upload it to the NodeMCU. The firmware outputs readings at 20&nbsp;Hz in the format `<millis>\t<grams>`.

## Calibration

Run `python scale_reader.py`, choose your serial port and then select the
*Calibrate* option. Follow the prompts to weigh a known mass. The script prints
the computed `COUNTS_PER_GRAM` constant—update this value in
`scaleReaderArduino.ino` and re-upload the firmware.
