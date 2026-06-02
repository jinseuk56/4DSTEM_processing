#!/usr/bin/env python3
"""
run_spectrum_image_viewer.py
----------------------------
Interactive viewer for 3D spectrum image stacks (e.g., Ny x Nx x N_channels).
Supports scrolling through spatial coordinates and viewing the local spectrum.

Usage:
    python run_spectrum_image_viewer.py [--file PATH] [--x_scale SCALE] [--x_unit UNIT]
"""
import argparse
import sys
import numpy as np
import tifffile
import tkinter.filedialog as tkf

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from fourdstem import launch_pyqt_3d_viewer

def main():
    parser = argparse.ArgumentParser(description="Interactive 3D spectrum image viewer")
    parser.add_argument("--file", type=str, default=None,
                        help="Path to 3D image stack TIFF")
    parser.add_argument("--x_scale", type=float, default=1.0,
                        help="Scale of X axis")
    parser.add_argument("--x_unit", type=str, default="channel",
                        help="Unit of X axis")
    args = parser.parse_args()

    file_path = args.file or tkf.askopenfilename(
        title="Select 3D Spectrum Image stack (TIFF)",
        filetypes=[("TIFF images", "*.tif *.tiff")]
    )
    if not file_path:
        print("No file selected. Exiting.")
        sys.exit(0)

    print(f"Loading {file_path}...")
    data = tifffile.imread(file_path)
    print(f"Loaded stack of shape {data.shape}")
    if data.ndim != 3:
        raise ValueError(f"Expected 3D stack, got {data.ndim}D stack with shape {data.shape}")

    print("Launching PyQt5 + pyqtgraph 3D spectrum image viewer app...")
    launch_pyqt_3d_viewer(data, x_scale=args.x_scale, x_unit=args.x_unit)
    print("Viewer closed.")

if __name__ == "__main__":
    main()
