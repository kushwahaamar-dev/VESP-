#!/usr/bin/env python3
"""
VEP Production Pipeline
=======================
Virtual Epileptic Patient simulation and visualization.

Usage:
    python pipeline.py --duration 4000 --output report.html
"""

import argparse
import logging
import numpy as np
import sys

from vep.config import sim_config, physics_config
from vep.anatomy import BrainAnatomy
from vep.simulator import Simulator
from vep.visualizer import BrainVisualizer


def setup_logging():
    """Configure logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    )


def parse_args():
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Virtual Epileptic Patient Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--duration', type=float, default=sim_config.duration,
                        help='Simulation duration (ms)')
    parser.add_argument('--output', type=str, default='vep_report.html',
                        help='Output HTML report path')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Load from checkpoint instead of running simulation')
    parser.add_argument('--save-checkpoint', type=str, default='checkpoint.npz',
                        help='Save simulation results to checkpoint file')
    
    return parser.parse_args()


def main():
    setup_logging()
    args = parse_args()
    logger = logging.getLogger("VEP")
    
    logger.info("=" * 60)
    logger.info("  Virtual Epileptic Patient (VEP) Pipeline")
    logger.info("=" * 60)
    
    # ==================== 1. LOAD ANATOMY ====================
    logger.info("[1/4] Loading brain anatomy...")
    anatomy = BrainAnatomy()
    anatomy.load_connectivity(n_regions=76)
    anatomy.load_cortex()
    
    # ==================== 2. CONFIGURE EZ ====================
    logger.info("[2/4] Configuring epileptogenic zones...")
    
    # Find temporal/limbic regions for EZ
    ez_indices = []
    target_keywords = ['AMYG', 'HC', 'PHC', 'T']  # Amygdala, Hippocampus, Temporal
    
    for i, label in enumerate(anatomy.labels):
        body = label[1:] if label[0] in ['l', 'r'] else label
        if any(kw in body for kw in target_keywords):
            ez_indices.append(i)
    
    # Limit to 3-5 regions
    ez_indices = ez_indices[:5]
    
    if not ez_indices:
        # Fallback: use random regions
        logger.warning("No matching regions found, using indices 40-42")
        ez_indices = [40, 41, 42]
    
    ez_labels = [anatomy.labels[i] for i in ez_indices]
    logger.info(f"  EZ regions: {ez_labels}")
    
    # ==================== 3. RUN SIMULATION ====================
    logger.info("[3/4] Running forward simulation...")
    
    simulator = Simulator(anatomy)
    
    if args.checkpoint:
        logger.info(f"  Loading from checkpoint: {args.checkpoint}")
        time, data, onset_times = simulator.load_checkpoint(args.checkpoint)
        x0_values = simulator.model.x0.copy()
        simulator.model.set_epileptogenic_zones(ez_indices)
        x0_values = simulator.model.x0.copy()
    else:
        time, data, onset_times = simulator.run(ez_indices, duration=args.duration)
        x0_values = simulator.model.x0.copy()
        
        if args.save_checkpoint:
            simulator.save_checkpoint(args.save_checkpoint, time, data, onset_times)
    
    # Stats
    n_recruited = np.sum(onset_times > 0)
    logger.info(f"  Recruited: {n_recruited}/{anatomy.n_regions} regions")
    logger.info(f"  Activity range: [{data.min():.2f}, {data.max():.2f}]")
    
    # ==================== 4. GENERATE REPORT ====================
    logger.info("[4/4] Generating visualization...")
    
    visualizer = BrainVisualizer(anatomy)
    visualizer.create_report(time, data, onset_times, x0_values, args.output)
    
    logger.info("=" * 60)
    logger.info(f"  Complete! Open: {args.output}")
    logger.info("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
