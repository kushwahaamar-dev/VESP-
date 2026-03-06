"""
IMPROVED: Better visualization with colors and clarity
"""
import sys
sys.path.append('./vep')

from upenn_loader import UPennLoader
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def create_improved_viz(patient_id):
    """Better visualization with clearer colors"""
    upenn = UPennLoader(data_root='./data/upenn')
    
    print(f"Creating improved visualization for {patient_id}...")
    raw, events, electrodes = upenn.load_patient_ieeg(patient_id)
    
    coords = electrodes[['x', 'y', 'z']].values
    names = electrodes['name'].values
    n_electrodes = len(names)
    
    # Seizure simulation (same as before)
    duration = 2000
    dt = 1
    n_steps = int(duration / dt)
    
    np.random.seed(42)
    connectivity = np.random.rand(n_electrodes, n_electrodes) * 0.1
    connectivity = (connectivity + connectivity.T) / 2
    np.fill_diagonal(connectivity, 0)
    
    ez_indices = [0, 1, 2]
    
    x = np.ones((n_steps, n_electrodes)) * -2.0
    x[0, ez_indices] = -1.6
    
    for t in range(1, n_steps):
        x_curr = x[t-1, :]
        coupling = connectivity @ (x_curr - x_curr.mean())
        dx = 0.01 * (x_curr + 2.0) * (1.0 - x_curr) + 0.05 * coupling
        dx[x_curr > -1.0] += 0.1
        x[t, :] = x_curr + dx * dt
        x[t, :] = np.clip(x[t, :], -3, 2)
    
    times = np.arange(n_steps) * dt
    
    # IMPROVED: Better activity calculation
    peak_activity = np.max(np.abs(x), axis=0)  # Max activity per electrode
    activity_norm = (peak_activity - peak_activity.min()) / (peak_activity.max() - peak_activity.min() + 1e-6)
    
    # IMPROVED: Force color variation
    activity_norm = activity_norm ** 0.5  # Make colors more spread out
    
    # IMPROVED: Separate EZ from propagated
    ez_mask = np.zeros(n_electrodes, dtype=bool)
    ez_mask[ez_indices] = True
    
    fig = make_subplots(
        rows=2, cols=1,
        specs=[[{'type': 'scatter3d'}], [{'type': 'scatter'}]],
        subplot_titles=(
            f'{patient_id} - Seizure Activity Map (88 electrodes)',
            'Seizure Propagation Time Series (2 seconds)'
        ),
        vertical_spacing=0.12,
        row_heights=[0.55, 0.45]
    )
    
    # IMPROVED: EZ electrodes (RED)
    fig.add_trace(
        go.Scatter3d(
            x=coords[ez_mask, 0],
            y=coords[ez_mask, 1],
            z=coords[ez_mask, 2],
            mode='markers+text',
            marker=dict(
                size=14,
                color='red',
                symbol='diamond',
                line=dict(width=2, color='darkred')
            ),
            text=[names[i] for i in np.where(ez_mask)[0]],
            textposition="top center",
            textfont=dict(size=10, color='red'),
            name='Epileptogenic Zone (EZ)',
            hovertemplate='<b>%{text}</b><br>SEIZURE ORIGIN<extra></extra>'
        ),
        row=1, col=1
    )
    
    # IMPROVED: Other electrodes (colored by activity)
    fig.add_trace(
        go.Scatter3d(
            x=coords[~ez_mask, 0],
            y=coords[~ez_mask, 1],
            z=coords[~ez_mask, 2],
            mode='markers+text',
            marker=dict(
                size=10,
                color=activity_norm[~ez_mask],
                colorscale=[
                    [0, 'rgb(50, 50, 150)'],      # Blue (low)
                    [0.3, 'rgb(100, 200, 100)'],  # Green
                    [0.6, 'rgb(255, 200, 0)'],    # Yellow
                    [1, 'rgb(255, 50, 50)']       # Red (high)
                ],
                showscale=True,
                colorbar=dict(
                    title="Propagation<br>Level",
                    x=1.02,
                    len=0.5,
                    y=0.75
                ),
                cmin=0,
                cmax=1
            ),
            text=[names[i] for i in np.where(~ez_mask)[0]],
            textposition="top center",
            textfont=dict(size=8),
            name='Propagated Regions',
            hovertemplate='<b>%{text}</b><br>Activity: %{marker.color:.2f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # IMPROVED: Time series with better colors
    # EZ electrodes
    colors_ez = ['red', 'darkred', 'crimson']
    for i, idx in enumerate(ez_indices):
        fig.add_trace(
            go.Scatter(
                x=times, 
                y=x[:, idx], 
                name=f'{names[idx]} (EZ)', 
                line=dict(width=3, color=colors_ez[i]),
                showlegend=True
            ),
            row=2, col=1
        )
    
    # Top 5 propagated electrodes
    propagated_indices = np.argsort(peak_activity[~ez_mask])[-5:]
    actual_indices = np.where(~ez_mask)[0][propagated_indices]
    colors_prop = ['orange', 'gold', 'yellow', 'lightgreen', 'cyan']
    
    for i, idx in enumerate(actual_indices):
        fig.add_trace(
            go.Scatter(
                x=times, 
                y=x[:, idx], 
                name=f'{names[idx]} (propagated)', 
                line=dict(width=2, dash='dash', color=colors_prop[i]),
                showlegend=True
            ),
            row=2, col=1
        )
    
    fig.update_layout(
        height=1000,
        showlegend=True,
        title_text=f"<b>{patient_id}</b> - Patient-Specific Seizure Simulation",
        title_font_size=20,
        scene=dict(
            xaxis_title='Left ← X (mm) → Right',
            yaxis_title='Posterior ← Y (mm) → Anterior',
            zaxis_title='Inferior ← Z (mm) → Superior',
            aspectmode='data',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.2)
            )
        ),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.5,
            xanchor="left",
            x=1.15
        )
    )
    
    fig.update_xaxes(title_text="<b>Time (milliseconds)</b>", row=2, col=1)
    fig.update_yaxes(title_text="<b>Neural Activity Level</b>", row=2, col=1)
    
    # Add annotations
    fig.add_annotation(
        text="Red diamonds = Seizure starts here",
        xref="paper", yref="paper",
        x=0.5, y=0.98, showarrow=False,
        font=dict(size=12, color="red")
    )
    
    filename = f'{patient_id}_IMPROVED_SIMULATION.html'
    fig.write_html(filename)
    print(f"  SAVED: {filename}")
    
    return n_electrodes

# Generate improved visualizations
patients = ['sub-R1001P', 'sub-R1002P', 'sub-R1003P']

print("\n" + "="*60)
print("GENERATING IMPROVED VISUALIZATIONS...")
print("="*60 + "\n")

for patient in patients:
    n_elec = create_improved_viz(patient)
    print(f"  {patient}: {n_elec} electrodes\n")

print("="*60)
print("IMPROVED VISUALIZATIONS COMPLETE!")
print("="*60)
