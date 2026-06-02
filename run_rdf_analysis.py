#!/usr/bin/env python3
"""
run_rdf_analysis.py
-------------------
Execution script: Radial Distribution Function (RDF) analysis pipeline.

Workflow:
  1. Load 4D-STEM data (and compute rotational average) or load a pre-computed 3D radial average stack.
  2. Compute mean square scattering factor <f^2> and square of mean scattering factor <f>^2.
  3. Fit scale factor N and correction term alpha to match experimental intensity to scattering factors.
  4. Compute the Reduced Intensity Function (RIF) and Radial Distribution Function G(r) for all pixels.
  5. Save the RIF and G(r) stacks, and plot the results.
"""

import argparse
import sys
import os
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import tifffile
import tkinter.filedialog as tkf

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from fourdstem import FourDSTEM_process
from fourdstem.rdf import (
    fit_rdf_parameters, calculate_scattering_factors, calculate_rif, rif_to_rdf,
    calculate_damping_filter, get_filter_type_from_filename, calculate_atomic_form_factors
)
from fourdstem.io import save_as_tiff

def main():
    parser = argparse.ArgumentParser(description="RDF analysis pipeline for 4D-STEM")
    parser.add_argument("--file", type=str, default=None,
                        help="Path to 4D-STEM data or pre-computed 3D radial average TIFF stack")
    parser.add_argument("--is_4d", action="store_true",
                        help="Specify if the input file is a raw 4D-STEM dataset (otherwise assumed 3D)")
    parser.add_argument("--elements", nargs="+", type=int, default=[8, 31],
                        help="Atomic numbers of elements in the sample (default: 8 31 for GaO)")
    parser.add_argument("--compositions", nargs="+", type=float, default=[0.4819, 0.5181],
                        help="Atomic fraction compositions of elements (default: 0.4819 0.5181)")
    parser.add_argument("--k_step", type=float, default=0.02629,
                        help="Reciprocal space step k_step (1/Å per pixel). Default: 0.02629")
    parser.add_argument("--r_max", type=float, default=5.0,
                        help="Maximum radial distance r in Å for G(r). Default: 5.0")
    parser.add_argument("--r_step", type=float, default=0.01,
                        help="Radial step size delta_r in Å for G(r). Default: 0.01")
    parser.add_argument("--method", type=str, choices=["standard", "legacy"], default="legacy",
                        help="Calculation method for scattering factors. Default: legacy")
    parser.add_argument("--damping_filter", type=str, default="4TEM BH",
                        help="Type of damping filter ('boxcar', 'triangular', 'trapezoidal', 'Happ-Genzel', '3TEM BH', '4TEM BH') or path to file. Default: 4TEM BH")
    parser.add_argument("--low_cut", type=float, default=0.05,
                        help="Low-pass cut-off frequency fraction. Default: 0.05")
    parser.add_argument("--high_cut", type=float, default=0.8,
                        help="High-pass cut-off frequency fraction. Default: 0.8")
    parser.add_argument("--save_prefix", type=str, default=None,
                        help="Prefix for saved output files")

    args = parser.parse_args()

    # --- Select File ---
    file_path = args.file or tkf.askopenfilename(
        title="Select 4D-STEM data or 3D radial average TIFF stack",
        filetypes=[("TIFF images", "*.tif *.tiff"), ("All supported", "*.tif *.tiff *.raw *.dm3 *.dm4")]
    )
    if not file_path:
        print("No file selected. Exiting.")
        sys.exit(0)

    save_prefix = args.save_prefix or file_path.rsplit(".", 1)[0]

    # --- Load Data ---
    print(f"\n=== RDF Analysis ===")
    print(f"File: {file_path}")
    
    if args.is_4d or file_path.endswith((".raw", ".dm3", ".dm4")):
        print("Processing raw 4D-STEM dataset...")
        fd = FourDSTEM_process(file_path, visual=False)
        fd.spike_remove(percent_thresh=0.01, mode="lower", apply_remove=True)
        fd.find_center(visual=False)
        print("Computing rotational averages...")
        fd.rotational_average(rot_variance=False)
        rot_dp_data = fd.radial_avg_stack
    else:
        print("Loading pre-computed 3D radial average stack...")
        rot_dp_data = tifffile.imread(file_path)
        if rot_dp_data.ndim == 3 and rot_dp_data.shape[0] > 100:
            print("Note: input has a large number of frames in first dimension. Keeping all frames.")
            
    print(f"Radial average stack shape: {rot_dp_data.shape}")
    Ny, Nx, Nr = rot_dp_data.shape

    # --- Set up k-list and r-list ---
    k_list = np.arange(0, args.k_step * Nr, args.k_step)
    r_list = np.arange(0, args.r_max, args.r_step)
    
    # --- Load or Calculate Damping Filter ---
    damping_filter = None
    if args.damping_filter:
        if os.path.exists(args.damping_filter):
            damping_filter = tifffile.imread(args.damping_filter)
            print(f"Loaded damping filter from file: {args.damping_filter}")
            if damping_filter.ndim > 1:
                damping_filter = damping_filter[5] if len(damping_filter) > 5 else damping_filter[0]
        else:
            filter_type = get_filter_type_from_filename(args.damping_filter)
            damping_filter = calculate_damping_filter(filter_type, Nr)
            print(f"Calculated native damping filter of type: {filter_type}")
    else:
        default_filter = "4TEM BH"
        damping_filter = calculate_damping_filter(default_filter, Nr)
        print(f"No damping filter specified. Calculated default native filter: {default_filter}")


    # --- Calculate Scattering Factors and Fit Parameters ---
    subtracted_term, divisor_term = calculate_scattering_factors(
        args.elements, args.compositions, k_list, method=args.method
    )
    
    mean_intensity = np.mean(rot_dp_data, axis=(0, 1))
    
    print("\nFitting scale factor N and correction alpha to experimental mean intensity...")
    alpha, N = fit_rdf_parameters(mean_intensity, subtracted_term)
    print(f"Fitted parameters: alpha = {alpha:.4f}, N = {N:.4f}")

    # --- Compute RIF and G(r) ---
    print("\nComputing RIF and G(r) for all pixels...")
    RIF_data = []
    Gr_data = []
    
    filt = damping_filter.copy()
    filt[:int(args.low_cut * len(filt))] = 0
    filt[int(args.high_cut * len(filt)):] = 0
    ind = np.linspace(0, len(filt) - 1, Nr).astype(np.int16)
    filt = filt[ind]

    for i in range(Ny):
        for j in range(Nx):
            intensity = rot_dp_data[i, j]
            rif = calculate_rif(intensity, subtracted_term, divisor_term, alpha, N, k_list, filt)
            gr = rif_to_rdf(rif, r_list, k_list)
            RIF_data.append(rif)
            Gr_data.append(gr)

    RIF_data = np.asarray(RIF_data).reshape(Ny, Nx, -1)
    Gr_data = np.asarray(Gr_data).reshape(Ny, Nx, -1)

    # --- Save Outputs ---
    save_as_tiff(save_prefix + "_RIF.tif", RIF_data.astype(np.float32))
    save_as_tiff(save_prefix + f"_Gr_upto_{int(args.r_max)}A.tif", Gr_data.astype(np.float32))
    print(f"Saved RIF stack: {save_prefix}_RIF.tif (shape: {RIF_data.shape})")
    print(f"Saved G(r) stack: {save_prefix}_Gr_upto_{int(args.r_max)}A.tif (shape: {Gr_data.shape})")

    # --- Plot Summary ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0, 0].plot(k_list, mean_intensity, 'k-', label="Experimental Mean")
    axes[0, 0].plot(k_list, (subtracted_term + alpha * (np.mean(mean_intensity) - np.mean(subtracted_term))) * N, 'r-', label="Fit")
    axes[0, 0].set_xlabel("k (1/Å)")
    axes[0, 0].set_ylabel("Intensity")
    axes[0, 0].set_title("Scattering Factor Fit")
    axes[0, 0].legend()
    axes[0, 0].grid()

    mean_rif = np.mean(RIF_data, axis=(0, 1))
    axes[0, 1].plot(k_list, mean_rif, 'g-', label="Mean RIF")
    axes[0, 1].set_xlabel("k (1/Å)")
    axes[0, 1].set_ylabel("RIF")
    axes[0, 1].set_title("Mean Reduced Intensity Function")
    axes[0, 1].grid()

    for i in range(min(Ny, 5)):
        axes[1, 0].plot(r_list, Gr_data[i, 0], label=f"Pixel ({i},0)")
    axes[1, 0].set_xlabel("r (Å)")
    axes[1, 0].set_ylabel("G(r)")
    axes[1, 0].set_title("Sample G(r) Profiles")
    axes[1, 0].set_xlim(1.0, 3.0)
    axes[1, 0].grid()

    ini_ind = np.abs(r_list - 1.50).argmin()
    fin_ind = np.abs(r_list - 2.00).argmin()
    max_peak_ind = np.argmax(Gr_data[:, :, ini_ind:fin_ind], axis=2)
    peak_positions = r_list[ini_ind + max_peak_ind]
    
    axes[1, 1].hist(peak_positions.flatten(), bins=50, color='purple', alpha=0.7)
    axes[1, 1].set_xlabel("Peak Position (Å)")
    axes[1, 1].set_ylabel("Counts")
    axes[1, 1].set_title("Distribution of G(r) First Peak (1.5 - 2.0 Å)")
    axes[1, 1].grid()

    fig.tight_layout()
    fig.savefig(save_prefix + "_rdf_summary.png", dpi=150)
    print(f"Saved summary figure: {save_prefix}_rdf_summary.png")
    plt.show()

    print("\nRDF analysis complete.")

if __name__ == "__main__":
    main()
