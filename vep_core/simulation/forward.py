"""
Forward Simulation Engine
=========================
High-Performance Time-Integration with Transmission Delays.
"""

import numpy as np
from tqdm import tqdm
from numba import jit
from .. import config
from ..models.epileptor import Epileptor

@jit(nopython=True, fastmath=True, cache=True)
def compute_delay_coupling(history, weights, delays, current_step, buffer_size, global_coupling):
    """
    Compute delayed input for all regions.
    C_i = a * Sum_j( W_ij * (x1_j(t - d_ij) - x1_i(t)) )
    """
    n_regions = weights.shape[0]
    coupling_vector = np.zeros(n_regions)
    
    # Current state index in buffer
    curr_ptr = current_step % buffer_size
    current_x1 = history[curr_ptr, :] # (N,)
    
    for i in range(n_regions):
        c_i = 0.0
        for j in range(n_regions):
            wij = weights[i, j]
            if wij > 0:
                # Calculate delayed index
                delay_steps = delays[i, j]
                # Circular buffer index
                # If t < delay, we use index 0 (IC) or handle gracefully
                # Since history is initialized with ICs, this is fine
                
                # Formula: (current_t - delay) % size
                # Note: delay_steps can be 0 for self-connection (usually 0 weight anyway)
                hist_ptr = (current_step - delay_steps)
                
                # Careful with negative time in ring buffer logic
                # Since we use modulo, (curr - delay) % L works in Python/Numba 
                # e.g. (5 - 10) % 100 = -5 % 100 = 95. Correct for a ring buffer.
                hist_ptr = hist_ptr % buffer_size
                
                delayed_x1 = history[hist_ptr, j]
                
                c_i += wij * (delayed_x1 - current_x1[i])
        
        coupling_vector[i] = global_coupling * c_i
        
    return coupling_vector

class ForwardSimulator:
    def __init__(self, weights, lengths, n_regions):
        self.weights = weights
        self.lengths = lengths
        self.n_regions = n_regions
        self.model = Epileptor(n_regions)
        
        # Compute Delays (Integer steps)
        # v = 3.0 mm/ms. dt = 0.05 ms.
        # time_delay = L / v
        # steps = time_delay / dt = L / (v * dt)
        velocity = config.CONDUCTION_VELOCITY
        dt = config.DT
        self.delays = (lengths / (velocity * dt)).astype(np.int32)
        
        self.max_delay_steps = np.max(self.delays)
        self.buffer_size = self.max_delay_steps + 10 # Safety margin
        
        print(f"[Simulator] Max Delay: {np.max(lengths)/velocity:.1f}ms ({self.max_delay_steps} steps)")
        
    def run(self, x0_parameters, duration=config.SIMULATION_LENGTH):
        """
        Execute the simulation with delays.
        """
        steps = int(duration / config.DT)
        print(f"[Simulation] Running HPCI kernel (Delays enabled, JIT compiled)...")
        
        # 1. Setup Model
        self.model.set_epileptogenicity(x0_parameters)
        
        # 2. Initialize History Ring Buffer
        # We only need to store 'x1' for coupling. 
        # But the model integration needs full state.
        # Optimization: Store full state only for current, X1 history for delays.
        
        # History of x1 variable (Ring Buffer) -> Shape (Buffer, N)
        history_x1 = np.zeros((self.buffer_size, self.n_regions), dtype=np.float64)
        
        # Initial State (Current)
        current_state = self.model.initial_state()
        
        # Fill buffer with initial condition (stationary assumption t<0)
        history_x1[:] = current_state[:, 0]
        
        # 3. Output Storage (Downsampled)
        downsample = int(2.0 / config.DT)  # Save every 2ms
        n_saved = steps // downsample
        saved_data = np.zeros((n_saved, self.n_regions), dtype=np.float32) # Save memory
        saved_time = np.linspace(0, duration, n_saved)
        
        # Onset Time Tracking
        # Track when Region's x1 amplitude implies seizure.
        # Simple detector: z dropping below 2.5? 
        # Or x1 > -1.0 for extended time?
        # Let's use x1 > -1.2 as "Spiking"
        onset_times = np.full(self.n_regions, -1.0)
        
        # 4. Integration Loop
        for t in tqdm(range(steps)):
            
            # Compute Delayed Coupling (JIT)
            coupling = compute_delay_coupling(
                history_x1, self.weights, self.delays, t, 
                self.buffer_size, config.GLOBAL_COUPLING
            )
            
            # Step Model
            current_state = self.model.integrate_step(current_state, coupling)
            
            # Update History Ring
            ptr = (t + 1) % self.buffer_size
            history_x1[ptr, :] = current_state[:, 0]
            
            # Onset Detection
            # If x1 > -1.2 (Spike threshold) and onset not yet recorded
            spiking_indices = np.where((current_state[:, 0] > -1.2) & (onset_times == -1.0))[0]
            if len(spiking_indices) > 0:
                current_time_ms = t * config.DT
                onset_times[spiking_indices] = current_time_ms
            
            # Save Data
            if t % downsample == 0:
                idx = t // downsample
                if idx < n_saved:
                    saved_data[idx] = current_state[:, 0].astype(np.float32)
                    
        return saved_time, saved_data, onset_times
