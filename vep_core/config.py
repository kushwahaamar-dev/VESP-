"""
VEP Configuration Module
========================
Defines strict configuration schemas using Dataclasses.
"""

from dataclasses import dataclass

@dataclass
class PhysicsConfig:
    """Physical constants for the brain model."""
    conduction_velocity: float = 3.0  # mm/ms
    # Epileptor constants
    Iext1: float = 3.1
    Iext2: float = 0.45
    r: float = 0.00035
    tau0: float = 2857.0
    noise_sig: float = 0.0002
    x0_critical: float = -2.0
    global_coupling: float = 0.05 # Scaling factor for long-range input

@dataclass
class SimConfig:
    """Simulation time integration settings."""
    dt: float = 0.05         # Integration step (ms)
    duration: float = 4000.0 # Total simulation time (ms)
    
    @property
    def steps(self) -> int:
        return int(self.duration / self.dt)

# Global default instance (for backward compatibility if needed, 
# but usage should prefer explicit config passing)
default_physics = PhysicsConfig()
default_sim = SimConfig()

# Legacy constants for compatibility during refactor
CONDUCTION_VELOCITY = default_physics.conduction_velocity
DT = default_sim.dt
SIMULATION_DURATION = default_sim.duration
