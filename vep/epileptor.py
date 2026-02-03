"""
Epileptor Neural Mass Model
===========================
6D model of seizure dynamics with Numba JIT acceleration.
Based on Jirsa et al. (2014) and TVB implementation.
"""

import numpy as np
from numba import jit
from .config import physics_config

@jit(nopython=True, fastmath=True, cache=True)
def epileptor_dfun(state, coupling, x0, Iext1, Iext2, tau0, r):
    """
    Compute derivatives for the 6D Epileptor model.
    
    State variables per region: [x1, y1, z, x2, y2, g]
    - x1, y1: Fast population (seizure spikes)
    - z: Slow permittivity variable (seizure termination)
    - x2, y2: Spike-wave population
    - g: Low-pass filter
    """
    n = state.shape[0]
    dx = np.zeros((n, 6))
    
    x1 = state[:, 0]
    y1 = state[:, 1]
    z  = state[:, 2]
    x2 = state[:, 3]
    y2 = state[:, 4]
    g  = state[:, 5]
    
    # Population 1: Fast oscillations
    # dx1 = y1 - f1(x1) - z + Iext1 + coupling
    # where f1(x1) = x1^3 - 3*x1^2 (cubic approximation)
    dx[:, 0] = y1 - x1**3 + 3*x1**2 - z + Iext1 + coupling
    dx[:, 1] = 1.0 - 5*x1**2 - y1
    
    # Slow variable: z (permittivity)
    # dz = r * (4*(x1 - x0) - z)
    dx[:, 2] = r * (4.0 * (x1 - x0) - z)
    
    # Population 2: Spike-wave
    dx[:, 3] = -y2 + x2 - x2**3 + Iext2 + 0.002*g - 0.3*(z - 3.5)
    dx[:, 4] = (-y2 + x2**4) / tau0
    
    # Low-pass filter
    dx[:, 5] = -0.01 * (g - 0.1*x1)
    
    return dx


class Epileptor:
    """Epileptor neural mass model with network coupling."""
    
    def __init__(self, n_regions, config=None):
        self.n_regions = n_regions
        self.config = config or physics_config
        
        # Excitability parameters (x0: lower = more epileptogenic)
        self.x0 = np.ones(n_regions) * self.config.x0_healthy
        
    def set_epileptogenic_zones(self, indices, x0_value=None):
        """Mark regions as epileptogenic."""
        if x0_value is None:
            x0_value = self.config.x0_ez
        self.x0[indices] = x0_value
        
    def initial_state(self):
        """Generate initial conditions (interictal state)."""
        state = np.zeros((self.n_regions, 6), dtype=np.float64)
        
        # Resting state values
        state[:, 0] = -1.6  # x1 (slightly above threshold)
        state[:, 1] = -10.0 # y1
        state[:, 2] = 3.0   # z (permittivity)
        state[:, 3] = -1.0  # x2
        state[:, 4] = 0.0   # y2
        state[:, 5] = 0.0   # g
        
        # Small perturbations for symmetry breaking
        state[:, 0] += np.random.normal(0, 0.05, self.n_regions)
        
        return state
    
    def step(self, state, coupling, dt):
        """Single Euler-Maruyama integration step."""
        # Deterministic drift
        deriv = epileptor_dfun(
            state, coupling, self.x0,
            self.config.Iext1, self.config.Iext2,
            self.config.tau0, self.config.r
        )
        
        # Update state
        new_state = state + deriv * dt
        
        # Additive noise on excitatory variables
        noise_amp = self.config.noise * np.sqrt(dt)
        new_state[:, 0] += np.random.normal(0, noise_amp, self.n_regions)
        new_state[:, 3] += np.random.normal(0, noise_amp, self.n_regions)
        
        return new_state
