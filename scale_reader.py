#!/usr/bin/env python3
"""
microfluidic_scale.py
---------------------
• auto-detect serial port
• reads grams directly from the HX711 firmware
• optional density → volume & flow
• live weight plot with 15 min sliding window + on-plot numeric readout
• CSV log: time_s, grams, millilitres, mL_s
"""

import argparse
import csv
import sys
import time
import os
from collections import deque
import math                     # (only needed if you add EMA later)

import numpy as np
import matplotlib.pyplot as plt
import serial
from serial.tools import list_ports
from matplotlib.ticker import ScalarFormatter

WINDOW_SEC = 15 * 60            # 900 s = 15 min


# ---------- helpers -------------------------------------------------------
def median_with_progress(ser, n, label, bar_width=40):
    buf = []
    print(f"{label}:")
    for i in range(n):
        while True:
            pkt = read_line(ser)
            if pkt:
                buf.append(pkt[1])
                break
        filled = int((i + 1) / n * bar_width)
        bar = "#" * filled + "-" * (bar_width - filled)
        sys.stdout.write(f"\r  [{bar}] {i + 1}/{n}")
        sys.stdout.flush()
    print()
    return float(np.median(buf))


def choose_port() -> str:
    ports = list_ports.comports()
    if not ports:
        sys.exit("❌  No serial ports found.")
    for i, p in enumerate(ports):
        print(f"[{i}] {p.device} — {p.description}")
    sel = input(f"Select port [0-{len(ports) - 1}]: ")
    try:
        return ports[int(sel)].device
    except (ValueError, IndexError):
        sys.exit("Invalid selection.")


def read_line(ser: serial.Serial):
    line = ser.readline().decode(errors='ignore').strip()
    if not line:
        return None
    try:
        t_ms, g = line.split('\t')
        return int(t_ms), float(g)
    except ValueError:
        return None


# ---------- main ----------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Gravimetric microfluidics logger")
    parser.add_argument('--density', type=float,
                        help="fluid density in g/mL (e.g. 0.997); omit to skip volume/flow")
    args = parser.parse_args()

    port = choose_port()
    ser = serial.Serial(port, 115200, timeout=1)
    print("🔌  Serial opened, waiting 2 s for first data…")
    time.sleep(2)
    ser.reset_input_buffer()

    # ---------- setup -----------------------------------------------
    print("\n=== READY ===")
    print("Using weight in grams from firmware (tare is handled on boot)")

    # density?
    rho = args.density
    if rho:
        print(f"Density set to {rho} g/mL → enabling volume & flow calc")
    else:
        print("No density given → skipping volume & flow.")

    # ---------- CSV & plotting -------------------------------------------
    fn = os.path.expanduser(f"~/Desktop/flow_{time.strftime('%Y%m%d-%H%M%S')}.csv")
    csv_f = open(fn, 'w', newline='')
    csv_w = csv.writer(csv_f)
    header = ['time_s', 'weight_g']
    if rho:
        header += ['volume_mL', 'flow_mL_s']
    csv_w.writerow(header)

    if rho:
        fig, (ax_w, ax_f) = plt.subplots(
            2, 1, sharex=True, figsize=(8, 6),
            gridspec_kw={'height_ratios': [3, 1]})
    else:
        fig, ax_w = plt.subplots(figsize=(8, 4))
        ax_f = None

    ax_w.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    ax_w.ticklabel_format(axis='y', style='plain')

    xs = deque(maxlen=int(WINDOW_SEC * 20))
    ys = deque(maxlen=int(WINDOW_SEC * 20))
    line_w, = ax_w.plot([], [], lw=1.4)
    ax_w.set_ylim(0, 10)

    if ax_f:
        line_f, = ax_f.plot([], [], lw=1.2, color='tab:red')
        xs_f = deque(maxlen=int(WINDOW_SEC * 20))
        ys_f = deque(maxlen=int(WINDOW_SEC * 20))

    ax_w.set_ylabel("Weight  [g]")
    if ax_f:
        ax_f.set_ylabel("Flow  [mL/s]")
    ax_w.set_xlabel("Time  [s]")
    fig.suptitle(f"Run started: {time.strftime('%Y-%m-%d  %H:%M:%S')}")

    plt.ion()

    # --- ADD 1: numeric on-plot readout ----------------------------------
    txt_w = ax_w.text(0.02, 0.92, '', transform=ax_w.transAxes,
                      ha='left', va='top',
                      fontsize=10, weight='bold',
                      bbox=dict(facecolor='w', alpha=0.65, boxstyle='round'))
    if ax_f:
        txt_f = ax_f.text(0.02, 0.92, '', transform=ax_f.transAxes,
                          ha='left', va='top',
                          fontsize=9,
                          bbox=dict(facecolor='w', alpha=0.65, boxstyle='round'))

    FLOW_WIN_SEC = 2.0
    flow_buf = deque()

    t0 = time.time()
    print("\n📈  Logging…  Ctrl-C to stop.")
    try:
        while True:
            pkt = read_line(ser)
            if not pkt:
                continue
            _, g = pkt

            t_s = time.time() - t0

            if rho:
                vol = g / rho
                flow_buf.append((t_s, vol))
                while flow_buf and (t_s - flow_buf[0][0]) > FLOW_WIN_SEC:
                    flow_buf.popleft()
                if len(flow_buf) > 1:
                    flow = (vol - flow_buf[0][1]) / (t_s - flow_buf[0][0])
                else:
                    flow = 0.0
                csv_w.writerow([f"{t_s:.2f}", f"{g:.4f}",
                                f"{vol:.4f}", f"{flow:.6f}"])
            else:
                csv_w.writerow([f"{t_s:.2f}", f"{g:.4f}"])

            # ----------- plot update ------------------------------------
            xs.append(t_s)
            ys.append(g)
            line_w.set_data(xs, ys)

            if t_s > WINDOW_SEC:
                ax_w.set_xlim(t_s - WINDOW_SEC, t_s)
            else:
                ax_w.set_xlim(0, WINDOW_SEC)

            # --- ADD 2: update numeric readout for weight ---------------
            txt_w.set_text(f"{g:,.3f} g")

            if ax_f:
                xs_f.append(t_s)
                ys_f.append(flow)
                line_f.set_data(xs_f, ys_f)
                ax_f.set_xlim(ax_w.get_xlim())

                # --- ADD 3: update numeric readout for flow -------------
                txt_f.set_text(f"{flow:,.4f} mL/s")

            plt.pause(0.001)

    except KeyboardInterrupt:
        print(f"\n🛑  Stopped. Data saved to {fn}")

    finally:
        csv_f.flush()
        csv_f.close()
        ser.close()
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    main()
