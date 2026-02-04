#!/usr/bin/env python3
"""
VEP Viewer Launcher
===================
Standalone script to launch PyVista viewer from saved checkpoint.
Run separately from Qt to avoid event loop conflicts.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    if len(sys.argv) < 2:
        print("Usage: python viewer.py <checkpoint.npz>")
        sys.exit(1)
        
    checkpoint_path = sys.argv[1]
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        sys.exit(1)
    
    print(f"Loading checkpoint: {checkpoint_path}")
    
    # Load data
    data = np.load(checkpoint_path, allow_pickle=True)
    
    time = data['time']
    activity = data['data']
    onset_times = data['onset_times']
    x0_values = data['x0_values']
    
    # Load anatomy
    atlas = int(data['atlas'])
    cortex = str(data['cortex'])
    
    from vep.anatomy import BrainAnatomy
    anatomy = BrainAnatomy()
    anatomy.load_connectivity(n_regions=atlas)
    anatomy.load_cortex(resolution=cortex)
    
    print(f"Loaded: {atlas} regions, {len(time)} time points")
    
    # Launch native viewer
    from vep.native import NativeVisualizer
    visualizer = NativeVisualizer(anatomy)
    visualizer.show(time, activity, onset_times, x0_values)


if __name__ == "__main__":
    main()
