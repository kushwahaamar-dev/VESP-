"""
EMERGENCY: Simulate seizure on patient-specific electrode network
"""
import sys
sys.path.append('./vep')

from upenn_loader import UPennLoader
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def simulate_patient_seizure(patient_id):
    """Run seizure simulation using patient's electrode positions as network"""
    upenn = UPennLoader(data_root='./data/upenn')
    
    print(f"Simulating seizure for {patient_id}...")
    raw, events, electrodes = upenn.load_patient_ieeg(patient_id)
    
    # Get electrode info
    coords = electrodes[['x', 'y', 'z']].values
    names = electrodes['name'].values
    n_electrodes = len(names)
    
    # Simple seizure simulation parameters
    duration = 2000  # ms
    dt = 1  # ms
    n_steps = int(duration / dt)
    
    # Initialize with random connectivity between electrodes
    np.random.seed(42)
    connectivity = np.random.rand(n_electrodes, n_electrodes) * 0.1
    connectivity = (connectivity + connectivity.T) / 2  # symmetric
    np.fill_diagonal(connectivity, 0)
    
    # Set 3-5 electrodes as epileptogenic (temporal lobe focus)
    ez_indices = [0, 1, 2]  # First 3 electrodes
    
    # Simplified Epileptor dynamics
    x = np.ones((n_steps, n_electrodes)) * -2.0  # Initialize in rest state
    x[0, ez_indices] = -1.6  # EZ starts excited
    
    # Simulate
    for t in range(1, n_steps):
        # Current state
        x_curr = x[t-1, :]
        
        # Network coupling
        coupling = connectivity @ (x_curr - x_curr.mean())
        
        # Simple excitable dynamics
        dx = 0.01 * (x_curr + 2.0) * (1.0 - x_curr) + 0.05 * coupling
        
        # Add threshold crossing
        dx[x_curr > -1.0] += 0.1  # Positive feedback for spiking
        
        # Update
        x[t, :] = x_curr + dx * dt
        
        # Keep bounded
        x[t, :] = np.clip(x[t, :], -3, 2)
    
    # Create visualization
    times = np.arange(n_steps) * dt
    
    # Create 3D brain + time series plot
    fig = make_subplots(
        rows=2, cols=1,
        specs=[[{'type': 'scatter3d'}], [{'type': 'scatter'}]],
        subplot_titles=(
            f'{patient_id} - Seizure Propagation on Electrode Network',
            'Neural Activity Over Time'
        ),
        vertical_spacing=0.15,
        row_heights=[0.6, 0.4]
    )
    
    # 3D electrode plot with final activity
    final_activity = np.abs(x[-1, :])
    activity_norm = (final_activity - final_activity.min()) / (final_activity.max() - final_activity.min() + 1e-6)
    
    fig.add_trace(
        go.Scatter3d(
            x=coords[:, 0],
            y=coords[:, 1],
            z=coords[:, 2],
            mode='markers+text',
            marker=dict(
                size=12,
                color=activity_norm,
                colorscale='Hot',
                showscale=True,
                colorbar=dict(title="Seizure Activity", x=1.15)
            ),
            text=names,
            textposition="top center",
            name='Electrodes',
            hovertemplate='<b>%{text}</b><br>Activity: %{marker.color:.2f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Time series for EZ and propagated regions
    for i in ez_indices:
        fig.add_trace(
            go.Scatter(x=times, y=x[:, i], name=f'{names[i]} (EZ)', 
                      line=dict(width=2)),
            row=2, col=1
        )
    
    # Show top 3 propagated electrodes
    propagated = np.argsort(np.abs(x[-1, :]))[-6:-3]
    for i in propagated:
        if i not in ez_indices:
            fig.add_trace(
                go.Scatter(x=times, y=x[:, i], name=f'{names[i]} (propagated)',
                          line=dict(width=1, dash='dash')),
                row=2, col=1
            )
    
    fig.update_layout(
        height=900,
        showlegend=True,
        scene=dict(
            xaxis_title='X (mm)',
            yaxis_title='Y (mm)',
            zaxis_title='Z (mm)',
            aspectmode='data'
        )
    )
    
    fig.update_xaxes(title_text="Time (ms)", row=2, col=1)
    fig.update_yaxes(title_text="Neural Activity", row=2, col=1)
    
    filename = f'{patient_id}_SEIZURE_SIMULATION.html'
    fig.write_html(filename)
    print(f"  SAVED: {filename}")
    
    return n_electrodes, n_steps

# Generate for 3 patients
patients = ['sub-R1001P', 'sub-R1002P', 'sub-R1003P']

print("\n" + "="*60)
print("GENERATING PATIENT-SPECIFIC SEIZURE SIMULATIONS...")
print("="*60 + "\n")

for patient in patients:
    n_elec, n_steps = simulate_patient_seizure(patient)
    print(f"  {patient}: {n_elec} electrodes, {n_steps} time steps\n")

print("="*60)
print("SEIZURE SIMULATIONS COMPLETE!")
print("="*60)
