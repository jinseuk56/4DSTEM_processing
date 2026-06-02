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
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import tifffile
import tkinter.filedialog as tkf

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from fourdstem.viewers import threed_viewer

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

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("1st figure (intensity): click or arrows to move position | 2nd figure (virtual bandpass image)\n3rd figure: drag to select virtual bandpass / press 'l' to toggle log / 't' for sum spectrum")

    viewer = threed_viewer(fig, ax, data, x_scale=args.x_scale, x_unit=args.x_unit)
    fig.canvas.mpl_connect("key_press_event", viewer.on_press)
    fig.canvas.mpl_connect("button_press_event", viewer.on_pick)
    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
