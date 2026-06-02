#!/usr/bin/env python3
"""
run_4dstem_viewer.py
--------------------
Execution script: interactively browse a 4D-STEM dataset.

Usage:
    python run_4dstem_viewer.py [--file PATH] [--empad_shape Ny Nx DPy DPx]

Keyboard shortcuts in viewer:
  Arrow keys / left-click  — move probe position
  l                        — toggle log-scaling on diffraction pattern
  Drag on DP panel         — select virtual-detector aperture
"""

import argparse
import sys
import numpy as np
import matplotlib
matplotlib.use("TkAgg")          # change to "Qt5Agg" if preferred
import matplotlib.pyplot as plt
import tkinter.filedialog as tkf

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from fourdstem import FourDSTEM_process

def main():
    parser = argparse.ArgumentParser(description="Interactive 4D-STEM viewer")
    parser.add_argument("--file", type=str, default=None,
                        help="Path to 4D-STEM data (.tif/.tiff/.raw/.dm3/.dm4)")
    parser.add_argument("--empad_shape", nargs=4, type=int, default=None,
                        metavar=("Ny", "Nx", "DPy", "DPx"),
                        help="Shape of EMPAD raw data (required for .raw files)")
    parser.add_argument("--scan_per_pixel", type=float, default=1.0,
                        help="Real-space calibration in scan units per pixel [nm/pix]")
    parser.add_argument("--dp_per_pixel", type=float, default=1.0,
                        help="Reciprocal-space calibration in mrad or 1/nm per pixel")
    parser.add_argument("--scan_unit", type=str, default="nm")
    parser.add_argument("--k_unit", type=str, default="1/nm")
    args = parser.parse_args()

    file_path = args.file or tkf.askopenfilename(
        title="Select 4D-STEM data file",
        filetypes=[("All supported", "*.tif *.tiff *.raw *.dm3 *.dm4"),
                   ("TIFF stack", "*.tif *.tiff"),
                   ("Raw binary", "*.raw"),
                   ("DM file", "*.dm3 *.dm4")])

    if not file_path:
        print("No file selected. Exiting.")
        sys.exit(0)

    print(f"Loading: {file_path}")

    f_shape = args.empad_shape if args.empad_shape else None
    fd = FourDSTEM_process(
        file_path,
        scan_per_pixel=args.scan_per_pixel,
        dp_per_pixel=args.dp_per_pixel,
        scan_unit=args.scan_unit,
        k_unit=args.k_unit,
        f_shape=f_shape,
    )

    fd.spike_remove(percent_thresh=0.01, mode="lower", apply_remove=True)

    plt.ion()
    fd.show_4d_viewer(fd.original_stack)
    plt.show(block=True)


if __name__ == "__main__":
    main()
