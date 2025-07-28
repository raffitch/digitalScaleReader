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

1. Edit `scaleReaderArduino.ino` and temporarily set
   `constexpr float COUNTS_PER_GRAM = 1.0f;` then upload the sketch. This causes
   the firmware to stream raw HX711 counts.
2. Run `python scale_reader.py`, choose your serial port and select the
   *Calibrate* option. When prompted, remove all weight and then place a known
   mass.
3. Enter the exact weight and note the printed `COUNTS_PER_GRAM` value.
4. Update the constant in `scaleReaderArduino.ino` with this value and upload
   the firmware again.
5. Subsequent runs of the script using the *Weigh* option will now report
   weight in grams.

## Flowmeter Mode

The Python tool also supports a simple pulse-counting mode for flowmeter
experiments. After choosing the serial port, select the *Flowmeter* option.
Commands while running:

- `s` – start counting pulses and open the valve
- `r` – reset the pulse counter and timer
- `q` – stop and close the valve

The console shows the total pulse count and the average frequency in
pulses&nbsp;per&nbsp;second. Resets start a fresh run so averages never go
negative and the valve engages immediately on `s`.
