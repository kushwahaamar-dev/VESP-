"""
Clinical Analytics Module
=========================
Post-simulation analysis of seizure dynamics.
Calculates key biomarkers for surgical planning.
"""

import numpy as np
from dataclasses import dataclass

@dataclass
class SeizureMetrics:
    """Clinical biomarkers derived from simulation."""
    n_recruited: int
    recruitment_ratio: float
    mean_latency: float # ms (Time from first onset to average propagation)
    primary_ez_region: str

class ClinicalAnalytics:
    """Static analysis engine for VEP results."""
    
    @staticmethod
    def analyze_propagation(onset_times, labels, distances=None):
        """
        Compute seizure propagation metrics.
        
        Args:
            onset_times: (N,) array of onset times (ms), -1 if no seizure.
            labels: (N,) list of region names.
        """
        # 1. Identify Recruiting Regions
        recruited_mask = onset_times > 0
        recruited_indices = np.where(recruited_mask)[0]
        n_recruited = len(recruited_indices)
        total_regions = len(onset_times)
        
        if n_recruited == 0:
            return SeizureMetrics(0, 0.0, 0.0, "None")
            
        # 2. Identify Primary Onset (EZ)
        # Region with earliest onset time
        ez_idx = recruited_indices[np.argmin(onset_times[recruited_indices])]
        primary_ez = labels[ez_idx]
        t_start = onset_times[ez_idx]
        
        # 3. Calculate Latency (Propagation Speed Proxy)
        # Time differences relative to primary onset
        relative_times = onset_times[recruited_indices] - t_start
        # Mean latency of propagated regions (excluding EZ itself)
        if n_recruited > 1:
            mean_latency = np.mean(relative_times[relative_times > 0])
        else:
            mean_latency = 0.0
            
        return SeizureMetrics(
            n_recruited=n_recruited,
            recruitment_ratio=n_recruited / total_regions,
            mean_latency=mean_latency,
            primary_ez_region=primary_ez
        )
