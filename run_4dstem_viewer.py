#!/usr/bin/env python3
"""
run_pyqt_4dstem_viewer.py
-------------------------
Interactive 4D-STEM viewer built on PyQt5 and pyqtgraph.
Provides real-time CPU/GPU accelerated image rendering and ROI selection.

Keyboard shortcuts:
  Arrow keys - move probe position
  L          - toggle log-scale on diffraction pattern
"""

import argparse
import sys
import numpy as np
import tkinter.filedialog as tkf

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from fourdstem import FourDSTEM_process, launch_pyqt_viewer

def main():
    parser = argparse.ArgumentParser(description="PyQt + pyqtgraph 4D-STEM viewer")
    parser.add_argument("--file", type=str, default=None,
                        help="Path to 4D-STEM data file (.tif/.tiff/.raw/.dm3/.dm4)")
    args = parser.parse_args()

    file_path = args.file or tkf.askopenfilename(
        title="Select 4D-STEM data file",
        filetypes=[("All supported", "*.tif *.tiff *.raw *.dm3 *.dm4")]
    )
    if not file_path:
        print("No file selected. Exiting.")
        sys.exit(0)

    print(f"Loading dataset: {file_path} ...")
    fd = FourDSTEM_process(file_path, visual=False)
    
    print("Launching PyQt5 + pyqtgraph 4D-STEM viewer app...")
    launch_pyqt_viewer(fd.original_stack)
    print("Viewer closed.")

if __name__ == "__main__":
    main()
