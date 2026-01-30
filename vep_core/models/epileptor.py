"""
The Epileptor Neural Mass Model (High-Performance Kernel)
=========================================================
JIT-compiled implementation of the 6D system using Numba.
Ensures microsecond-scale execution for clinical-grade simulations.
"""

import numpy as np
from numba import jit
from .. import config

@jit(nopython=True, fastmath=True, cache=True)
def dfun(state, coupling, param_x0, I_ext1, I_ext2, r_timescale, tau0):
    """
    Compute derivatives (deterministic part) for the 6D Epileptor.
    
    Args:
        state: (N, 6) array [x1, y1, z, x2, y2, g]
        coupling: (N,) array of input current from network
        param_x0: (N,) excitability parameters
    Returns:
        d_state: (N, 6) derivatives
    """
    x1 = state[:, 0]
    y1 = state[:, 1]
    z  = state[:, 2]
    x2 = state[:, 3]
    y2 = state[:, 4]
    g  = state[:, 5]
    
    # --- Pop 1: Fast Discharges ---
    # f1(x1) = x1^3 - 3x1^2 if x1 < 0 else (x1 - 0.6(z-4))^2 ...
    # The standard Epileptor uses a cubic approximation f1 = x1^3 - 3x1^2 to maintain smoothness
    # But strictly Jirsa 2014 has a slope change. 
    # For speed/smoothness in VEP, we use the polynomial form from Proix 2017:
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
    def __init__(self, n_regions):
        self.n_regions = n_regions
        self.param_x0 = np.ones(n_regions) * -2.2
        
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

    def integrate_step(self, state, coupling, dt=config.DT):
        """
        Euler-Maruyama step (Python wrapper around JIT kernel).
        """
        # 1. Deterministic Drift
        derivs = dfun(state, coupling, self.param_x0, 
                      config.I_EXT_1, config.I_EXT_2, 
                      config.R_TIMESCALE, config.TAU_0)
        
        # 2. Stochastic Diffusion (Additive Noise on x1 and x2)
        # We add noise to x1 (Pop1) and x2 (Pop2) to trigger transitions
        sig = 0.0002 # Noise std dev
        # Noise shape (N, 6). Only columns 0 (x1) and 3 (x2) get noise? 
        # Usually just x1 or x2. Let's add to both excitation variables.
        
        # In-place update
        state += derivs * dt
        
        # Additive Noise
        noise = np.random.normal(0, sig, (self.n_regions, 2))
        state[:, 0] += noise[:, 0] * np.sqrt(dt)
        state[:, 3] += noise[:, 1] * np.sqrt(dt)
        
        return state
