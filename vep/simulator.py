"""
Forward Simulator
=================
Time integration with transmission delays and seizure detection.
"""

import numpy as np
from numba import jit
from tqdm import tqdm
import logging

from .config import sim_config, physics_config
from .epileptor import Epileptor

logger = logging.getLogger(__name__)

@jit(nopython=True, fastmath=True, cache=True)
def compute_coupling(history, weights, delays, step, buffer_size, coupling_strength):
    """Compute delayed network coupling for all regions."""
    n = weights.shape[0]
    coupling = np.zeros(n)
    
    curr_idx = step % buffer_size
    current_x1 = history[curr_idx, :]
    
    for i in range(n):
        c = 0.0
        for j in range(n):
            w = weights[i, j]
            if w > 0:
                delay = delays[i, j]
                hist_idx = (step - delay) % buffer_size
                delayed_x1 = history[hist_idx, j]
                c += w * (delayed_x1 - current_x1[i])
        coupling[i] = coupling_strength * c
    
    return coupling


class Simulator:
    """Forward simulation engine with transmission delays."""
    
    def __init__(self, anatomy, sim_cfg=None, phys_cfg=None):
        self.anatomy = anatomy
        self.sim_cfg = sim_cfg or sim_config
        self.phys_cfg = phys_cfg or physics_config
        
        # Compute delay matrix (in steps)
        dt = self.sim_cfg.dt
        v = self.phys_cfg.velocity
        self.delays = (anatomy.distances / (v * dt)).astype(np.int32)
        self.max_delay = np.max(self.delays)
        self.buffer_size = self.max_delay + 10
        
        # Model
        self.model = Epileptor(anatomy.n_regions, self.phys_cfg)
        
        logger.info(f"Simulator ready: max delay {self.max_delay} steps ({self.max_delay * dt:.1f} ms)")
    
    def run(self, ez_indices, duration=None):
        """
        Run simulation with specified epileptogenic zones.
        
        Returns: time, data, onset_times
        """
        duration = duration or self.sim_cfg.duration
        dt = self.sim_cfg.dt
        n_steps = int(duration / dt)
        n_regions = self.anatomy.n_regions
        
        # Set epileptogenic zones
        self.model.set_epileptogenic_zones(ez_indices)
        
        logger.info(f"Running {n_steps} steps ({duration} ms)...")
        
        # Initialize
        state = self.model.initial_state()
        history = np.zeros((self.buffer_size, n_regions), dtype=np.float64)
        history[:] = state[:, 0]  # Fill with initial x1
        
        # Output storage (downsample for memory)
        downsample = max(1, int(1.0 / dt))  # Save every 1ms
        n_saved = n_steps // downsample
        data = np.zeros((n_saved, n_regions), dtype=np.float32)
        time = np.linspace(0, duration, n_saved)
        
        # Onset detection
        onset_times = np.full(n_regions, -1.0)
        onset_threshold = -1.0  # x1 above this = spiking
        
        # Integration loop
        for t in tqdm(range(n_steps), desc="Simulating"):
            # Compute coupling
            coupling = compute_coupling(
                history, self.anatomy.weights, self.delays,
                t, self.buffer_size, self.phys_cfg.coupling
            )
            
            # Step model
            state = self.model.step(state, coupling, dt)
            
            # Update history
            idx = (t + 1) % self.buffer_size
            history[idx, :] = state[:, 0]
            
            # Detect onsets
            spiking = (state[:, 0] > onset_threshold) & (onset_times < 0)
            onset_times[spiking] = t * dt
            
            # Save data
            if t % downsample == 0:
                save_idx = t // downsample
                if save_idx < n_saved:
                    data[save_idx] = state[:, 0].astype(np.float32)
        
        # Stats
        n_onset = np.sum(onset_times > 0)
        logger.info(f"Simulation complete: {n_onset} regions recruited")
        
        return time, data, onset_times
    
    def save_checkpoint(self, path, time, data, onset_times):
        """Save simulation results."""
        np.savez_compressed(path, time=time, data=data, onset_times=onset_times,
                           x0=self.model.x0, labels=self.anatomy.labels)
        logger.info(f"Saved checkpoint: {path}")
    
    def load_checkpoint(self, path):
        """Load simulation results."""
        npz = np.load(path, allow_pickle=True)
        logger.info(f"Loaded checkpoint: {path}")
        return npz['time'], npz['data'], npz['onset_times']
