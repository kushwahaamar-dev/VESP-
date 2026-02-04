#!/usr/bin/env python3
"""
VEP Production Pipeline
=======================
Virtual Epileptic Patient simulation and visualization.

Usage:
    python pipeline.py --duration 4000 --atlas 76 --output report.html
    python pipeline.py --atlas 192 --cortex 80k   # High resolution
    python pipeline.py --atlas 84                  # Macaque brain
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
    parser.add_argument('--atlas', type=int, default=76,
                        choices=[66, 68, 76, 80, 96, 192, 998, 84],
                        help='Brain atlas resolution (84=macaque)')
    parser.add_argument('--cortex', type=str, default='16k',
                        choices=['16k', '80k', '120k'],
                        help='Cortical mesh resolution')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Load from checkpoint instead of running simulation')
    parser.add_argument('--save-checkpoint', type=str, default='checkpoint.npz',
                        help='Save simulation results to checkpoint file')
    parser.add_argument('--list-atlases', action='store_true',
                        help='List available atlas options and exit')
    
    return parser.parse_args()


def main():
    setup_logging()
    args = parse_args()
    logger = logging.getLogger("VEP")
    
    # ==================== LIST ATLASES ====================
    if args.list_atlases:
        anatomy = BrainAnatomy()
        available = anatomy.list_available_atlases()
        print("\n📊 Available Brain Atlases:")
        print("-" * 40)
        for n in available:
            species = "🐵 Macaque" if n == 84 else "🧠 Human"
            print(f"  {n:4d} regions  {species}")
        print("-" * 40)
        return 0
    
    logger.info("=" * 60)
    logger.info("  Virtual Epileptic Patient (VEP) Pipeline")
    logger.info(f"  Atlas: {args.atlas} regions | Cortex: {args.cortex}")
    logger.info("=" * 60)
    
    # ==================== 1. LOAD ANATOMY ====================
    logger.info("[1/4] Loading brain anatomy...")
    anatomy = BrainAnatomy()
    anatomy.load_connectivity(n_regions=args.atlas)
    anatomy.load_cortex(resolution=args.cortex)
    
    # ==================== 2. CONFIGURE EZ ====================
    logger.info("[2/4] Configuring epileptogenic zones...")
    
    # Find limbic/temporal regions for EZ
    ez_indices = []
    
    if anatomy.species == "macaque":
        # Macaque: use specific region indices
        ez_indices = [10, 11, 12]  # Example regions
    else:
        # Human: find by label
        target_keywords = ['AMYG', 'HC', 'PHC', 'T']
        for i, label in enumerate(anatomy.labels):
            body = label[1:] if label[0] in ['l', 'r'] else label
            if any(kw in body for kw in target_keywords):
                ez_indices.append(i)
    
    ez_indices = ez_indices[:5]
    
    if not ez_indices:
        logger.warning("No matching regions found, using indices 0-2")
        ez_indices = [0, 1, 2]
    
    ez_labels = [anatomy.labels[i] for i in ez_indices]
    logger.info(f"  EZ regions: {ez_labels}")
    
    # ==================== 3. RUN SIMULATION ====================
    logger.info("[3/4] Running forward simulation...")
    
    simulator = Simulator(anatomy)
    
    if args.checkpoint:
        logger.info(f"  Loading from checkpoint: {args.checkpoint}")
        time, data, onset_times = simulator.load_checkpoint(args.checkpoint)
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
