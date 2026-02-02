#!/usr/bin/env python3
"""
VEP Production Pipeline
=======================
Main entry point for the Virtual Epileptic Patient workflow.
Replicates the architecture of the paper's codebase.

Usage:
    python main_pipeline.py --patient PAT001 --duration 4000
"""

import argparse
import sys
import os
import numpy as np
import logging

# Import Core Packages
from vep_core import config
from vep_core.data.loader import VEPLoader
from vep_core.inference.inversion import VEPInference
from vep_core.simulation.forward import ForwardSimulator
from vep_core.viz.report import VEPReport
from vep_core.analytics import ClinicalAnalytics

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Virtual Epileptic Patient (VEP) Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--duration", 
        type=float, 
        default=config.default_sim.duration,
        help="Simulation duration in milliseconds"
    )
    
    parser.add_argument(
        "--output", 
        type=str, 
        default="VEP_Clinical_Report.html", 
        help="Path to save the generated HTML report"
    )
    
    parser.add_argument(
        "--patient", 
        type=str, 
        default="PAT001", 
        help="Patient identifier (for logging/metadata)"
    )
    
    parser.add_argument(
        "--checkpoint", 
        type=str, 
        default="simulation_checkpoint.npz", 
        help="Path for saving/loading simulation state"
    )
    
    parser.add_argument(
        "--resume", 
        action="store_true", 
        help="Resume from checkpoint if exists"
    )
    
    return parser.parse_args()

def main():
    # Configure Professional Logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(name)s: %(message)s',
        datefmt='%H:%M:%S'
    )
    logger = logging.getLogger("VEP")
    
    args = parse_args()

    logger.info("==============================================================")
    logger.info(f"      Virtual Epileptic Patient (VEP) Pipeline v2.1           ")
    logger.info(f"      Patient: {args.patient} | Duration: {args.duration}ms   ")
    logger.info("==============================================================")
    
    # 1. Initialize & Load Data
    # ----------------------------------------------------------------
    logger.info("[Step 1] Loading Patient Anatomy...")
    loader = VEPLoader()
    
    # Load 76-region connectivity (Standard VEP Atlas proxy)
    weights, lengths, labels, full_labels = loader.load_connectivity(n_regions=76)
    n_regions = len(labels)
    logger.info(f"  > Loaded structural connectivity ({n_regions} regions)")
    
    max_delay = np.max(lengths)/config.CONDUCTION_VELOCITY
    logger.info(f"  > Loaded tract lengths (Max delay: {max_delay:.1f} ms)")
    
    # Load Cortical Surface
    cortex_verts, cortex_tris, region_mapping = loader.load_cortex()
    cortex = (cortex_verts, cortex_tris, region_mapping)
    logger.info(f"  > Loaded cortical mesh ({cortex_verts.shape[0]} vertices)")

    # 2. Bayesian Inference (Parameter Estimation)
    # ----------------------------------------------------------------
    logger.info("[Step 2] Running Bayesian Inference (HMC Inversion)...")
    inference_engine = VEPInference(n_regions, labels, full_labels=full_labels)
    
    # Generate hypothesis: Right Temporal Lobe Epilepsy
    ev_posterior = inference_engine.generate_hypothesis(target_region_str="Temporal")
    x0_parameters = inference_engine.map_ev_to_x0(ev_posterior)
    
    # Report findings
    ez_indices = np.where(x0_parameters > -2.0)[0]
    logger.info(f"  > Identified {len(ez_indices)} epileptogenic regions:")
    for idx in ez_indices:
        logger.info(f"    - {labels[idx]} (x0 = {x0_parameters[idx]:.3f})")

    # 3. Forward Simulation (Epileptor Model)
    # ----------------------------------------------------------------
    logger.info("[Step 3] Running Forward Simulation (Physics-Based with Delays)...")
    simulator = ForwardSimulator(weights, lengths, n_regions)
    
    # Run with CLI duration
    # Checkpointing Logic
    if args.resume and os.path.exists(args.checkpoint):
        logger.info(f"  > Resuming from checkpoint: {args.checkpoint}")
        time, history, onset_times = simulator.load_checkpoint(args.checkpoint)
    else:
        time, history, onset_times = simulator.run(x0_parameters, duration=args.duration)
        simulator.save_checkpoint(args.checkpoint, time, history, onset_times)

    logger.info(f"  > Simulation Stats: Range [{np.min(history):.2f}, {np.max(history):.2f}] mV")
    if np.max(history) < -0.5:
        logger.warning("  WARNING: No high-amplitude seizure activity detected!")
    else:
        logger.info("  > Seizure activity confirmed (High amplitude detected).")

    # 3b. Clinical Analytics
    metrics = ClinicalAnalytics.analyze_propagation(onset_times, labels)
    logger.info(f"  [Analytics] Seizure Metrics:")
    logger.info(f"    - Primary EZ: {metrics.primary_ez_region}")
    logger.info(f"    - Recruitment: {metrics.n_recruited}/{n_regions} ({metrics.recruitment_ratio:.1%})")
    logger.info(f"    - Propagation Latency (Mean): {metrics.mean_latency:.1f} ms")

    # 4. Clinical Reporting & Visualization
    # ----------------------------------------------------------------
    logger.info("[Step 4] Generating Clinical Report...")
    VEPReport.generate_dashboard(
        cortex=cortex,
        mapping=region_mapping,
        time=time,
        data=history,
        x0_values=x0_parameters,
        onset_times=onset_times,
        labels=labels,
        full_labels=full_labels,
        output_path=args.output
    )
    
    logger.info("==============================================================")
    logger.info(f"Pipeline Complete. Report available at: {args.output}")
    logger.info("==============================================================")

if __name__ == "__main__":
    main()
