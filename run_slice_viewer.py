#!/usr/bin/env python3
"""
run_slice_viewer.py
-------------------
Interactive slice viewer for stacks of 2D images (e.g., tilt series or time series).
Allows scrolling through slices using Up/Right (next) and Down/Left (previous) keys.

Usage:
    python run_slice_viewer.py [--file PATH]
"""
import argparse
import sys
import numpy as np
import tifffile
import tkinter.filedialog as tkf

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from fourdstem import launch_pyqt_slice_viewer

def main():
    parser = argparse.ArgumentParser(description="Interactive 2D slice stack viewer")
    parser.add_argument("--file", type=str, default=None,
                        help="Path to 3D slice stack TIFF (shape: N_slices x Ny x Nx)")
    args = parser.parse_args()

    file_path = args.file or tkf.askopenfilename(
        title="Select 3D Image Slice Stack (TIFF)",
        filetypes=[("TIFF images", "*.tif *.tiff")]
    )
    if not file_path:
        print("No file selected. Exiting.")
        sys.exit(0)

    print(f"Loading {file_path}...")
    data = tifffile.imread(file_path)
    print(f"Loaded stack of shape {data.shape}")
    if data.ndim != 3:
        raise ValueError(f"Expected 3D stack (N_slices, Ny, Nx), got {data.ndim}D stack with shape {data.shape}")

    print("Launching PyQt5 + pyqtgraph slice viewer app...")
    launch_pyqt_slice_viewer(data)
    print("Viewer closed.")

if __name__ == "__main__":
    main()
