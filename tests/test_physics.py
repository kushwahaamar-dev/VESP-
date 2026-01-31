"""
Physics Engine Verification
===========================
Automated tests for the VEP numerical integration kernels.
"""

import pytest
import numpy as np
from vep_core.models.epileptor import Epileptor
from vep_core.config import default_physics

def test_epileptor_initialization():
    """Verify state allocation and sizing."""
    n = 10
    model = Epileptor(n)
    assert model.n_regions == 10
    
    state = model.initial_state()
    assert state.shape == (10, 6)
    # x1 range check
    assert np.all(state[:, 0] >= -2.0)
    assert np.all(state[:, 0] <= -1.0)

def test_epileptor_dynamics():
    """Verify integration steps produce non-NaN changes."""
    n = 2
    model = Epileptor(n)
    state = model.initial_state()
    coupling = np.zeros(n)
    
    # Take one step
    new_state = model.integrate_step(state.copy(), coupling, dt=0.05)
    
    # 1. Stability Check
    assert not np.isnan(new_state).any()
    assert not np.isinf(new_state).any()
    
    # 2. Dynamics Check (x1 should evolve)
    # y1 = -10, x1 = -1.6. dx1 ~ y1 ...
    # State should change
    assert not np.allclose(state, new_state)

def test_coupling_sensitivity():
    """Verify that coupling input affects the next state."""
    n = 2
    model = Epileptor(n)
    state = model.initial_state()
    
    # Case A: Zero coupling
    state_a = model.integrate_step(state.copy(), np.zeros(n), dt=0.05)
    
    # Case B: Strong positive coupling
    state_b = model.integrate_step(state.copy(), np.ones(n) * 2.0, dt=0.05)
    
    # Results should differ
    assert not np.allclose(state_a, state_b)
