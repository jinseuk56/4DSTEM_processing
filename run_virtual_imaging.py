#!/usr/bin/env python3
"""
run_virtual_imaging.py
----------------------
Execution script: virtual STEM imaging, DPC phase integration, and symmetry evaluation.

Workflow:
  1. Load 4D-STEM dataset.
  2. Remove spike pixels.
  3. Locate the diffraction center (CoM).
  4. Extract the Bright Field (BF) disk and set up detector apertures.
  5. Generate Virtual BF and ADF images.
  6. Calculate Differential Phase Contrast (DPC) and reconstruct phase/potential.
  7. (Optional) Evaluate rotational and mirror symmetries.

Usage:
    python run_virtual_imaging.py [--file PATH] [--scan_per_pixel SCALE] [--mrad_per_pixel SCALE]
                                  [--cbox_edge EDGE] [--hpass HP] [--lpass LP] [--symmetry_angle DEG] [--use_gpu]
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

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from fourdstem import FourDSTEM_process, rotation, radial_indices

def main():
    parser = argparse.ArgumentParser(description="Virtual STEM imaging & DPC reconstruction")
    parser.add_argument("--file", type=str, default=None,
                        help="Path to 4D-STEM data")
    parser.add_argument("--scan_per_pixel", type=float, default=0.502,
                        help="Real-space calibration in Å/pixel. Default: 0.502")
    parser.add_argument("--mrad_per_pixel", type=float, default=1.25,
                        help="Reciprocal-space calibration in mrad/pixel. Default: 1.25")
    parser.add_argument("--cbox_edge", type=int, default=64,
                        help="Size of boundary box edge for center finding. Default: 64")
    parser.add_argument("--hpass", type=float, default=0.05,
                        help="High-pass filter parameter for DPC. Default: 0.05")
    parser.add_argument("--lpass", type=float, default=0.05,
                        help="Low-pass filter parameter for DPC. Default: 0.05")
    parser.add_argument("--symmetry_angle", type=float, default=90.0,
                        help="Angle in degrees for symmetry evaluation. Default: 90.0")
    parser.add_argument("--save_prefix", type=str, default=None,
                        help="Prefix for saved output files")
    parser.add_argument("--use_gpu", action="store_true",
                        help="Use GPU acceleration (via CuPy) for DPC and symmetry evaluations")

    args = parser.parse_args()

    # --- Load File ---
    file_path = args.file or tkf.askopenfilename(
        title="Select 4D-STEM data file",
        filetypes=[("All supported", "*.tif *.tiff *.raw *.dm3 *.dm4")]
    )
    if not file_path:
        print("No file selected. Exiting.")
        sys.exit(0)

    save_prefix = args.save_prefix or file_path.rsplit(".", 1)[0]

    print(f"\n=== Virtual STEM & DPC Integration ===")
    print(f"File: {file_path}")

    fd = FourDSTEM_process(file_path, scan_per_pixel=args.scan_per_pixel, dp_per_pixel=args.mrad_per_pixel)

    # --- Preprocessing & Alignment ---
    print("\nRemoving spike pixels...")
    fd.spike_remove(percent_thresh=0.01, mode="lower", apply_remove=True)

    print("\nLocating diffraction center...")
    c_pos = fd.find_center(cbox_edge=args.cbox_edge, visual=True)
    print(f"Diffraction pattern center (y, x): {c_pos}")

    print("\nExtracting central beam disk...")
    fd.disk_extract(buffer_size=1, visual=True)

    # --- Virtual Detectors ---
    semiangle = fd.least_R * args.mrad_per_pixel
    BF_det = np.array([0.0, semiangle])
    ADF_det = np.array([semiangle, semiangle * 2])
    print(f"BF detector: {BF_det} mrad | ADF detector: {ADF_det} mrad")

    print("\nGenerating virtual STEM images...")
    fd.virtual_stem(BF_det, ADF_det, visual=True)
    
    # Save virtual STEM images
    tifffile.imwrite(save_prefix + "_vADF.tif", fd.ADF_stem.astype(np.float32))
    tifffile.imwrite(save_prefix + "_vBF.tif", fd.BF_stem.astype(np.float32))
    print(f"  Saved virtual STEM images with prefix: {save_prefix}")

    # --- DPC Integration ---
    print("\nReconstructing phase/potential via DPC integration...")
    fd.DPC(correct_rotation=True, n_theta=100, hpass=args.hpass, lpass=args.lpass, visual=True, use_gpu=args.use_gpu)

    # Save DPC outputs
    tifffile.imwrite(save_prefix + "_dDPC_charge.tif", fd.charge_density.astype(np.float32))
    tifffile.imwrite(save_prefix + "_iDPC_potential.tif", fd.potential.astype(np.float32))
    print(f"  Saved DPC charge density and integrated potential.")

    # --- Symmetry Evaluation ---
    print(f"\nEvaluating rotational/mirror symmetry at {args.symmetry_angle}°...")
    start = time.process_time()
    fd.symmetry_evaluation(args.symmetry_angle, also_mirror=True, visual=True, use_gpu=args.use_gpu)
    elapsed = time.process_time() - start
    print(f"Symmetry evaluation finished in {elapsed:.2f} seconds.")
    
    plt.show()

if __name__ == "__main__":
    main()
