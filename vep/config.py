"""
VEP Configuration
=================
Simulation parameters and physical constants.
"""

from dataclasses import dataclass

@dataclass
class SimulationConfig:
    """Time integration settings."""
    dt: float = 0.05          # Integration step (ms)
    duration: float = 4000.0  # Total simulation time (ms)
    
    @property
    def n_steps(self) -> int:
        return int(self.duration / self.dt)

@dataclass  
class PhysicsConfig:
    """Epileptor model parameters."""
    # Conduction
    velocity: float = 3.0     # mm/ms (signal propagation speed)
    coupling: float = 0.1     # Global coupling strength
    
    # Epileptor constants
    Iext1: float = 3.1
    Iext2: float = 0.45
    tau0: float = 2857.0
    r: float = 0.00035
    noise: float = 0.0001
    
    # Excitability threshold (x0 > this = epileptogenic)
    x0_healthy: float = -2.2
    x0_ez: float = -1.6

@dataclass
class VisualizationConfig:
    """Rendering settings."""
    mesh_opacity: float = 0.15
    mesh_color: str = "lightblue"
    ez_color: str = "#ff3333"       # Red
    pz_color: str = "#ffaa00"       # Orange  
    healthy_color: str = "#4488ff"  # Blue
    node_size_min: float = 8.0
    node_size_max: float = 25.0
    animation_fps: int = 30

# Default instances
sim_config = SimulationConfig()
physics_config = PhysicsConfig()
viz_config = VisualizationConfig()
