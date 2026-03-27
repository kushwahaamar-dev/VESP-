"""
Fractional-Order Seizure Dynamics
Implements Grünwald-Letnikov fractional derivative for memory effects
"""
import numpy as np

def grunwald_letnikov_weights(beta, n_steps):
    """
    Compute Grünwald-Letnikov weights for fractional derivative of order beta
    
    Args:
        beta: Fractional order (0 < beta <= 1)
        n_steps: Number of time steps (memory length)
    
    Returns:
        weights: Array of size n_steps with GL weights
    """
    weights = np.zeros(n_steps)
    weights[0] = 1.0
    
    for j in range(1, n_steps):
        weights[j] = weights[j-1] * (1.0 - (1.0 + beta) / j)
    
    return weights


def fractional_derivative(x_history, beta, dt):
    """
    Compute fractional derivative D^beta x / dt^beta using Grünwald-Letnikov
    
    Args:
        x_history: Array of past states [x(t), x(t-dt), x(t-2dt), ...]
        beta: Fractional order (0 < beta <= 1)
        dt: Time step
    
    Returns:
        D^beta x: Fractional derivative at current time
    """
    n_history = len(x_history)
    weights = grunwald_letnikov_weights(beta, n_history)
    
    # Fractional derivative: sum of weighted past states
    frac_deriv = np.sum(weights * x_history) / (dt ** beta)
    
    return frac_deriv


def simulate_fractional_seizure(n_electrodes, ez_indices, connectivity, 
                                beta=0.8, duration=2000, dt=1):
    """
    Simulate seizure with fractional-order dynamics
    
    Args:
        n_electrodes: Number of electrodes
        ez_indices: Indices of epileptogenic zone
        connectivity: Electrode connectivity matrix
        beta: Fractional time order (0 < beta <= 1)
        duration: Simulation duration (ms)
        dt: Time step (ms)
    
    Returns:
        x: State trajectory [n_steps, n_electrodes]
        times: Time vector
    """
    n_steps = int(duration / dt)
    
    # State history (need memory for fractional derivative)
    x_history = []
    
    # Initialize
    x_init = np.ones(n_electrodes) * -2.0  # Rest state
    x_init[ez_indices] = -1.6  # EZ excited
    x_history.append(x_init)
    
    # Storage for full trajectory
    x_trajectory = np.zeros((n_steps, n_electrodes))
    x_trajectory[0, :] = x_init
    
    # Simulate with fractional dynamics
    for t in range(1, n_steps):
        x_curr = x_history[-1]
        
        # Network coupling
        coupling = connectivity @ (x_curr - x_curr.mean())
        
        # Right-hand side: f(x) = dynamics
        f_x = 0.01 * (x_curr + 2.0) * (1.0 - x_curr) + 0.05 * coupling
        f_x[x_curr > -1.0] += 0.1  # Threshold crossing
        
        # Fractional integration: x(t+dt) computed from fractional derivative
        if len(x_history) > 1:
            # Use fractional derivative with memory
            x_new = np.zeros(n_electrodes)
            for i in range(n_electrodes):
                # Get history for this electrode
                electrode_history = np.array([x_history[-(j+1)][i] for j in range(len(x_history))])
                
                # Fractional derivative
                frac_deriv = fractional_derivative(electrode_history, beta, dt)
                
                # Update: D^beta x = f(x) --> x_new = x_curr + dt^beta * f(x) - correction
                x_new[i] = x_curr[i] + dt**beta * f_x[i] - dt**beta * frac_deriv
        else:
            # First step: standard Euler
            x_new = x_curr + f_x * dt
        
        # Keep bounded
        x_new = np.clip(x_new, -3, 2)
        
        # Store
        x_history.append(x_new)
        x_trajectory[t, :] = x_new
        
        # Keep memory limited (last 100 steps)
        if len(x_history) > 100:
            x_history.pop(0)
    
    times = np.arange(n_steps) * dt
    
    return x_trajectory, times