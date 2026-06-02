#!/usr/bin/env python3
"""
run_cepstral_analysis.py
------------------------
Execution script: Cepstral 4D-STEM transformation pipeline.

Workflow:
  1. Load 4D-STEM dataset.
  2. Compute Cepstral transform (with or without mean-pattern subtraction dCP).
  3. Optionally compute rotational averages and variances of the cepstral patterns.
  4. Save the cepstral and dcp stacks as TIFF stacks.
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
from fourdstem import FourDSTEM_process
from fourdstem.io import save_as_tiff

def main():
    parser = argparse.ArgumentParser(description="Cepstral 4D-STEM transform pipeline")
    parser.add_argument("--file", type=str, default=None,
                        help="Path to 4D-STEM dataset (.tif/.tiff/.raw/.dm3/.dm4)")
    parser.add_argument("--scan_per_pixel", type=float, default=1.0,
                        help="Real-space calibration [nm/pixel]")
    parser.add_argument("--dp_per_pixel", type=float, default=0.2079,
                        help="Reciprocal-space calibration [1/nm per pixel]")
    parser.add_argument("--dCP", action="store_true", default=True,
                        help="Perform mean-subtracted cepstral transform (dCP)")
    parser.add_argument("--rot_average", action="store_true", default=True,
                        help="Compute rotational average of the cepstrum")
    parser.add_argument("--rot_variance", action="store_true", default=True,
                        help="Compute rotational variance of the cepstrum")
    parser.add_argument("--save_prefix", type=str, default=None,
                        help="Prefix for saved output files")
    args = parser.parse_args()

    # --- Select File ---
    file_path = args.file or tkf.askopenfilename(
        title="Select 4D-STEM data file",
        filetypes=[("All supported", "*.tif *.tiff *.raw *.dm3 *.dm4")]
    )
    if not file_path:
        print("No file selected. Exiting.")
        sys.exit(0)

    save_prefix = args.save_prefix or file_path.rsplit(".", 1)[0]

    print(f"\n=== Cepstral Analysis ===")
    print(f"File: {file_path}")

    # --- Process ---
    fd = FourDSTEM_process(file_path, scan_per_pixel=args.scan_per_pixel, dp_per_pixel=args.dp_per_pixel)
    
    print("\nRunning Cepstral Transformation...")
    fd.cepstral(dCP=args.dCP, rot_average=args.rot_average, rot_variance=args.rot_variance)

    # --- Save Outputs ---
    save_as_tiff(save_prefix + "_cepstral.tif", fd.ceps)
    print(f"Saved: {save_prefix}_cepstral.tif (shape: {fd.ceps.shape})")
    
    if args.dCP and hasattr(fd, 'dcp') and fd.dcp is not None:
        save_as_tiff(save_prefix + "_dcp.tif", fd.dcp)
        print(f"Saved: {save_prefix}_dcp.tif (shape: {fd.dcp.shape})")

    if args.rot_average and hasattr(fd, 'ceps_avg_stack'):
        save_as_tiff(save_prefix + "_ceps_avg.tif", fd.ceps_avg_stack)
        print(f"Saved: {save_prefix}_ceps_avg.tif (shape: {fd.ceps_avg_stack.shape})")

    if args.rot_variance and hasattr(fd, 'ceps_var_stack'):
        save_as_tiff(save_prefix + "_ceps_var.tif", fd.ceps_var_stack)
        print(f"Saved: {save_prefix}_ceps_var.tif (shape: {fd.ceps_var_stack.shape})")

    # --- Visualisation ---
    print("\nDisplaying interactive 4D viewer for Cepstrum (dCP)...")
    fd.show_4d_viewer(fd.dcp)
    
    if args.rot_variance:
        print("\nDisplaying 3D viewer for Cepstrum rotational variance stack...")
        fd.show_3d_viewer(fd.ceps_var_stack, fd.real_per_pixel*10, "Å")
        
    plt.show()

if __name__ == "__main__":
    main()
