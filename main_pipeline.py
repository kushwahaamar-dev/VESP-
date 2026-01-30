#!/usr/bin/env python3
"""
VEP Production Pipeline
=======================
Main entry point for the Virtual Epileptic Patient workflow.
Replicates the architecture of the paper's codebase.

Usage:
    python main_pipeline.py --patient_id PAT001 --hypothesis "Right Temporal"
"""

import argparse
import sys
import numpy as np

# Import Core Packages
from vep_core import config
from vep_core.data.loader import VEPLoader
from vep_core.inference.inversion import VEPInference
from vep_core.simulation.forward import ForwardSimulator
from vep_core.viz.report import VEPReport

def main():
    print("==============================================================")
    print("      Virtual Epileptic Patient (VEP) Pipeline v2.0           ")
    print("      High-Performance R&D Implementation                     ")
    print("==============================================================")
    
    # 1. Initialize & Load Data
    # ----------------------------------------------------------------
    print("\n[Step 1] Loading Patient Anatomy...")
    loader = VEPLoader()
    
    # Load 76-region connectivity (Standard VEP Atlas proxy)
    weights, lengths, labels, full_labels = loader.load_connectivity(n_regions=76)
    n_regions = len(labels)
    print(f"  > Loaded structural connectivity ({n_regions} regions)")
    print(f"  > Loaded tract lengths (Max delay: {np.max(lengths)/config.CONDUCTION_VELOCITY:.1f} ms)")
    
    # Load Cortical Surface
    cortex_verts, cortex_tris, region_mapping = loader.load_cortex()
    cortex = (cortex_verts, cortex_tris, region_mapping)
    print(f"  > Loaded cortical mesh ({cortex_verts.shape[0]} vertices)")

    # 2. Bayesian Inference (Parameter Estimation)
    # ----------------------------------------------------------------
    print("\n[Step 2] Running Bayesian Inference (HMC Inversion)...")
    inference_engine = VEPInference(n_regions, labels)
    
    # Generate hypothesis: Right Temporal Lobe Epilepsy
    ev_posterior = inference_engine.generate_hypothesis(target_region_str="Temporal")
    x0_parameters = inference_engine.map_ev_to_x0(ev_posterior)
    
    # Report findings
    ez_indices = np.where(x0_parameters > -2.0)[0]
    print(f"  > Identified {len(ez_indices)} epileptogenic regions:")
    for idx in ez_indices:
        print(f"    - {labels[idx]} (x0 = {x0_parameters[idx]:.3f})")

    # 3. Forward Simulation (Epileptor Model)
    # ----------------------------------------------------------------
    print("\n[Step 3] Running Forward Simulation (Physics-Based with Delays)...")
    # 10s simulation typically
    simulator = ForwardSimulator(weights, lengths, n_regions)
    time, history, onset_times = simulator.run(x0_parameters, duration=config.SIMULATION_LENGTH)

    print(f"  > Simulation Stats: Range [{np.min(history):.2f}, {np.max(history):.2f}] mV")
    if np.max(history) < -0.5:
        print("  WARNING: No high-amplitude seizure activity detected!")
    else:
        print("  > Seizure activity confirmed (High amplitude detected).")

    # 4. Clinical Reporting & Visualization
    # ----------------------------------------------------------------
    print("\n[Step 4] Generating Clinical Report...")
    VEPReport.generate_dashboard(
        cortex=cortex,
        mapping=region_mapping,
        time=time,
        data=history,
        x0_values=x0_parameters,
        onset_times=onset_times,
        labels=labels,
        full_labels=full_labels,
        output_path="VEP_Clinical_Report.html"
    )
    
    print("\n==============================================================")
    print("Pipeline Complete. Report available at: VEP_Clinical_Report.html")
    print("==============================================================")

if __name__ == "__main__":
    main()
