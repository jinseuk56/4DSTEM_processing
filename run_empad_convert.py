#!/usr/bin/env python3
"""
run_empad_convert.py
--------------------
Execution script: batch-convert EMPAD raw binary files to TIFF stacks.

Usage:
    python run_empad_convert.py [--shape Ny Nx DPy DPx] [--output_suffix SUFFIX]

Supports selecting multiple .raw files via a file dialog.
"""

import argparse
import sys
import numpy as np
import tkinter.filedialog as tkf

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from fourdstem.io import load_empad_raw, save_as_tiff


def main():
    parser = argparse.ArgumentParser(description="Batch convert EMPAD .raw files to TIFF stacks")
    parser.add_argument("--shape", nargs=4, type=int, default=[128, 128, 128, 128],
                        metavar=("Ny", "Nx", "DPy", "DPx"),
                        help="Shape of the 4D-STEM dataset [scanning_y, scanning_x, DP_y, DP_x]. "
                             "Default: 128 128 128 128 (standard EMPAD)")
    parser.add_argument("--output_suffix", type=str, default="_as_tiff_stack",
                        help="Suffix appended to input filename for output TIFF (default: '_as_tiff_stack')")
    parser.add_argument("--datatype", type=str, default="float32",
                        help="NumPy dtype for the raw binary (default: float32)")
    args = parser.parse_args()

    f_shape = args.shape

    raw_adrs = list(tkf.askopenfilenames(
        title="Select EMPAD .raw files",
        filetypes=[("EMPAD raw binary", "*.raw"), ("All files", "*")]))

    if not raw_adrs:
        print("No files selected. Exiting.")
        sys.exit(0)

    print(f"Selected {len(raw_adrs)} file(s):")
    for adr in raw_adrs:
        print(f"  {adr}")

    for adr in raw_adrs:
        print(f"\nProcessing: {adr}")
        stack_4d = load_empad_raw(adr, datatype=args.datatype, f_shape=f_shape)
        print(f"  Shape after loading: {stack_4d.shape}")
        print(f"  Min: {stack_4d.min():.4f}  Max: {stack_4d.max():.4f}")

        out_path = adr[:-4] + args.output_suffix + ".tif"
        save_as_tiff(out_path, stack_4d)
        print(f"  Saved → {out_path}")

    print("\nAll files converted successfully.")


if __name__ == "__main__":
    main()
