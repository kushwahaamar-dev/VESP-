"""
Simple Fractional-Order Seizure Simulation
Uses existing electrode CSV files
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from fractional_dynamics import simulate_fractional_seizure

def load_patient_electrodes(patient_id):
    """Load electrode data from CSV"""
    filename = f'{patient_id}_electrode_mapping.csv'
    df = pd.read_csv(filename)
    
    coords = df[['X', 'Y', 'Z']].values
    names = df['Electrode'].values
    regions = df['Brain_Region'].values
    
    return coords, names

def simulate_patient(patient_id, beta=0.8):
    """Run fractional seizure simulation"""
    print(f"\nSimulating {patient_id} (beta={beta})...")
    
    # Load electrodes
    coords, names = load_patient_electrodes(patient_id)
    n_electrodes = len(names)
    print(f"  - {n_electrodes} electrodes loaded")
    
    # Create connectivity (distance-based)
    connectivity = np.zeros((n_electrodes, n_electrodes))
    for i in range(n_electrodes):
        for j in range(i+1, n_electrodes):
            dist = np.linalg.norm(coords[i] - coords[j])
            connectivity[i, j] = np.exp(-dist / 30.0) * np.random.rand() * 0.1
            connectivity[j, i] = connectivity[i, j]
    
    # Define EZ (first 3 electrodes)
    ez_indices = [0, 1, 2]
    print(f"  - EZ: {[names[i] for i in ez_indices]}")
    
    # Run fractional simulation
    print(f"  - Running fractional simulation...")
    x, times = simulate_fractional_seizure(
        n_electrodes=n_electrodes,
        ez_indices=ez_indices,
        connectivity=connectivity,
        beta=beta,
        duration=2000,
        dt=1
    )
    print(f"  - Complete: {len(times)} time steps")
    
    # Create visualization
    fig = make_subplots(
        rows=2, cols=1,
        specs=[[{'type': 'scatter3d'}], [{'type': 'scatter'}]],
        subplot_titles=(
            f'{patient_id} - Fractional Seizure (beta={beta})',
            'Neural Activity with Memory Effects'
        ),
        vertical_spacing=0.15,
        row_heights=[0.6, 0.4]
    )
    
    # 3D electrodes
    final_activity = np.abs(x[-1, :])
    activity_norm = (final_activity - final_activity.min()) / (final_activity.max() - final_activity.min() + 1e-6)
    
    fig.add_trace(
        go.Scatter3d(
            x=coords[:, 0], y=coords[:, 1], z=coords[:, 2],
            mode='markers',
            marker=dict(
                size=10,
                color=activity_norm,
                colorscale='Hot',
                showscale=True,
                colorbar=dict(title="Activity")
            ),
            name='Electrodes'
        ),
        row=1, col=1
    )
    
    # Time series
    for i in ez_indices:
        fig.add_trace(
            go.Scatter(x=times, y=x[:, i], name=f'{names[i]} (EZ)'),
            row=2, col=1
        )
    
    propagated = np.argsort(np.abs(x[-1, :]))[-6:-3]
    for i in propagated:
        if i not in ez_indices:
            fig.add_trace(
                go.Scatter(x=times, y=x[:, i], name=f'{names[i]}', 
                          line=dict(dash='dash')),
                row=2, col=1
            )
    
    fig.update_layout(height=900, showlegend=True)
    fig.update_xaxes(title_text="Time (ms)", row=2, col=1)
    fig.update_yaxes(title_text="Activity", row=2, col=1)
    
    filename = f'{patient_id}_FRACTIONAL_beta{beta}.html'
    fig.write_html(filename)
    print(f"  SAVED: {filename}")
    
    return n_electrodes

# Run simulations
patients = ['sub-R1001P', 'sub-R1002P', 'sub-R1003P']
beta_values = [1.0, 0.8, 0.6]

print("="*70)
print("FRACTIONAL-ORDER SEIZURE SIMULATIONS")
print("="*70)

for patient in patients:
    print(f"\n{patient}:")
    for beta in beta_values:
        n_elec = simulate_patient(patient, beta=beta)

print("\n" + "="*70)
print("COMPLETE!")
print("="*70)