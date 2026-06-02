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
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import tifffile
import tkinter.filedialog as tkf

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from fourdstem.viewers import slice_viewer

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

    fig, ax = plt.subplots(figsize=(6, 6))
    fig.suptitle("Slice Viewer: Press Up/Right to go forward, Down/Left to go backward")

    viewer = slice_viewer(ax, data)
    fig.canvas.mpl_connect("key_press_event", viewer.on_press)
    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
