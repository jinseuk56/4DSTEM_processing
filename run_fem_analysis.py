#!/usr/bin/env python3
"""
run_fem_analysis.py
-------------------
Execution script: Fluctuation Electron Microscopy (FEM) analysis pipeline.

Workflow:
  1. Load 4D-STEM data
  2. Detect diffraction centre (CoM)
  3. Compute radial average and variance profiles for all probe positions
  4. Compute angular correlation spectra at selected scattering vectors
  5. Save radial-average, radial-variance and angular-correlation stacks

Usage:
    python run_fem_analysis.py [--file PATH] [--k_range K_START K_STOP K_STEP]
                               [--angle_sampling N] [--gaussian_sigma SIGMA]
                               [--save_prefix PREFIX]
"""

import argparse
import sys
import time
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import tifffile
import tkinter.filedialog as tkf
from scipy import ndimage

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from fourdstem import FourDSTEM_process
from fourdstem.fem import (
    radial_stats,
    fourd_radial_transformation,
    angular_correlation_fft,
    angular_correlation_direct,
)
from fourdstem.processing import radial_indices, indices_at_r


def main():
    parser = argparse.ArgumentParser(description="FEM analysis pipeline")
    parser.add_argument("--file", type=str, default=None)
    parser.add_argument("--k_range", nargs=3, type=int, default=[15, 30, 1],
                        metavar=("START", "STOP", "STEP"),
                        help="Range of radial k-indices for angular correlation (default: 15 30 1)")
    parser.add_argument("--angle_sampling", type=int, default=361,
                        help="Number of angular bins (0–360°). Default: 361")
    parser.add_argument("--gaussian_sigma", type=float, default=2.0,
                        help="Gaussian smoothing sigma applied to azimuthal profile before correlation. Default: 2.0")
    parser.add_argument("--save_prefix", type=str, default=None,
                        help="Prefix for saved output files. Defaults to input filename stem.")
    parser.add_argument("--dp_per_pixel", type=float, default=1.0,
                        help="Reciprocal-space calibration (mrad or 1/nm per pixel)")
    args = parser.parse_args()

    # --- Load data ---
    file_path = args.file or tkf.askopenfilename(
        title="Select 4D-STEM data file",
        filetypes=[("All supported", "*.tif *.tiff *.raw *.dm3 *.dm4")])
    if not file_path:
        print("No file selected. Exiting.")
        sys.exit(0)

    save_prefix = args.save_prefix or file_path.rsplit(".", 1)[0]

    print(f"\n=== FEM Analysis ===")
    print(f"File: {file_path}")

    fd = FourDSTEM_process(file_path, dp_per_pixel=args.dp_per_pixel)
    fd.spike_remove(percent_thresh=0.01, mode="lower", apply_remove=True)
    c_pos = fd.find_center(visual=False)
    print(f"Diffraction centre: {c_pos}")

    # --- Radial average and variance ---
    print("\nComputing radial average and variance profiles...")
    fd.rotational_average(rot_variance=True)
    tifffile.imwrite(save_prefix + "_radial_avg.tif", fd.radial_avg_stack)
    tifffile.imwrite(save_prefix + "_radial_var.tif", fd.radial_var_stack)
    print(f"  Saved: {save_prefix}_radial_avg.tif")
    print(f"  Saved: {save_prefix}_radial_var.tif")

    # --- Angular correlation ---
    k_range = range(args.k_range[0], args.k_range[1], args.k_range[2])
    angle_sampling = args.angle_sampling
    data_shape = fd.original_stack.shape

    print(f"\nComputing angular correlations for k = {list(k_range)} ...")
    print(f"  Angle sampling: {angle_sampling} bins | Gaussian sigma: {args.gaussian_sigma}")
    start = time.time()

    ac_spectra = []
    ac_fft_stack = []
    for k in k_range:
        k_ind, a_ind = indices_at_r(data_shape[2:], k, c_pos.tolist() if hasattr(c_pos, 'tolist') else list(c_pos))
        temp_spectra = []
        temp_fft = []
        for i in range(data_shape[0]):
            for j in range(data_shape[1]):
                values = np.zeros(angle_sampling)
                values[a_ind.astype(int)] = fd.original_stack[i, j, k_ind[0], k_ind[1]]
                values = ndimage.gaussian_filter(values, sigma=args.gaussian_sigma)

                # Use fast FFT-based angular correlation
                ac = angular_correlation_fft(values)
                ac_fft = np.abs(np.fft.fft(ac))

                temp_spectra.append(ac)
                temp_fft.append(ac_fft)

        ac_spectra.append(np.asarray(temp_spectra).reshape(data_shape[0], data_shape[1], -1))
        ac_fft_stack.append(np.asarray(temp_fft).reshape(data_shape[0], data_shape[1], -1))
        elapsed = time.time() - start
        print(f"  k={k} done  ({elapsed:.1f}s elapsed)")

    ac_spectra = np.asarray(ac_spectra)   # (n_k, Ny, Nx, n_angles)
    ac_fft_stack = np.asarray(ac_fft_stack)

    tifffile.imwrite(save_prefix + "_ang_corr_spectra.tif", ac_spectra.astype(np.float32))
    tifffile.imwrite(save_prefix + "_ang_corr_fft.tif", ac_fft_stack.astype(np.float32))
    print(f"\n  Saved: {save_prefix}_ang_corr_spectra.tif  shape={ac_spectra.shape}")
    print(f"  Saved: {save_prefix}_ang_corr_fft.tif  shape={ac_fft_stack.shape}")

    # --- Quick summary plot ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(
        np.flip(np.mean(ac_spectra, axis=(1, 2)).T, 0), cmap="viridis",
        extent=[k_range[0], k_range[-1], 0, angle_sampling])
    axes[0].set_xlabel("k-index"); axes[0].set_ylabel("Δangle (deg)")
    axes[0].set_title("Mean angular correlation C(k, Δφ)")
    axes[1].imshow(
        np.flip(np.mean(ac_fft_stack, axis=(1, 2)).T[1:11], 0), cmap="viridis",
        extent=[k_range[0], k_range[-1], 0.5, 10.5])
    axes[1].set_xlabel("k-index"); axes[1].set_ylabel("n-fold symmetry")
    axes[1].set_title("FFT of angular correlation (n-fold)")
    fig.tight_layout()
    fig.savefig(save_prefix + "_ang_corr_summary.png", dpi=150)
    print(f"  Saved: {save_prefix}_ang_corr_summary.png")
    plt.show()

    print("\nFEM analysis complete.")


if __name__ == "__main__":
    main()
