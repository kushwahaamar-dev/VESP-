"""
The Epileptor Neural Mass Model (High-Performance Kernel)
=========================================================
JIT-compiled implementation of the 6D system using Numba.
Ensures microsecond-scale execution for clinical-grade simulations.
Refactored for Strict Configuration.
"""

import numpy as np
from numba import jit
from ..config import default_physics, PhysicsConfig

@jit(nopython=True, fastmath=True, cache=True)
def dfun(state, coupling, param_x0, I_ext1, I_ext2, r_timescale, tau0):
    """
    Compute derivatives (deterministic part) for the 6D Epileptor.
    """
    x1 = state[:, 0]
    y1 = state[:, 1]
    z  = state[:, 2]
    x2 = state[:, 3]
    y2 = state[:, 4]
    g  = state[:, 5]
    
    # --- Pop 1: Fast Discharges ---
    # dx1 = y1 - x1^3 + 3x1^2 - z + I_ext1 + coupling
    dx1 = y1 - x1**3 + 3*x1**2 - z + I_ext1 + coupling
    dy1 = 1 - 5*x1**2 - y1
    
    # dz = r(4(x1 - x0) - z)
    dz = r_timescale * (4 * (x1 - param_x0) - z)
    
    # --- Pop 2: Spike-Wave ---
    dx2 = -y2 + x2 - x2**3 + I_ext2 + 0.002*g - 0.3*(z - 3.5)
    dy2 = (-y2 + x2**4) / tau0
    
    # Filter
    dg = -0.01 * (g - 0.1*x1)
    
    # Stack output
    d_state = np.stack((dx1, dy1, dz, dx2, dy2, dg), axis=1)
    return d_state

class Epileptor:
    def __init__(self, n_regions, config: PhysicsConfig = default_physics):
        self.n_regions = n_regions
        self.config = config
        # Initial excitability (default)
        self.param_x0 = np.ones(n_regions) * self.config.x0_critical
        
    def set_epileptogenicity(self, x0):
        self.param_x0 = x0.astype(np.float64)

    def initial_state(self):
        state = np.zeros((self.n_regions, 6), dtype=np.float64)
        state[:, 0] = -1.6 + np.random.normal(0, 0.1, self.n_regions) # x1
        state[:, 1] = -10.0 # y1
        state[:, 2] = 3.0   # z
        state[:, 3] = -1.5  # x2
        state[:, 4] = 0.0   # y2
        state[:, 5] = 0.0   # g
        return state

    def integrate_step(self, state, coupling, dt=None):
        """
        Euler-Maruyama step (Python wrapper around JIT kernel).
        """
        if dt is None:
            # We assume dt is handled by caller (simulator) loop, 
            # but if not provided we might need a default? 
            # Ideally simulator controls dt. 
            # We'll use 0.05 default if None, or raise error?
            dt = 0.05
            
        # 1. Deterministic Drift
        derivs = dfun(state, coupling, self.param_x0, 
                      self.config.Iext1, self.config.Iext2, 
                      self.config.r, self.config.tau0)
        
        # 2. Stochastic Diffusion (Additive Noise on x1 and x2)
        sig = self.config.noise_sig
        
        # In-place update
        state += derivs * dt
        
        # Additive Noise
        noise = np.random.normal(0, sig, (self.n_regions, 2))
        state[:, 0] += noise[:, 0] * np.sqrt(dt)
        state[:, 3] += noise[:, 1] * np.sqrt(dt)
        
        return state
