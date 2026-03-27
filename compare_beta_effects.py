"""
Beta Comparison: Show how fractional order affects seizure dynamics
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from fractional_dynamics import simulate_fractional_seizure

# Load one patient
patient_id = 'sub-R1001P'
df = pd.read_csv(f'{patient_id}_electrode_mapping.csv')
coords = df[['X', 'Y', 'Z']].values
names = df['Electrode'].values
n_electrodes = len(names)

# Simple connectivity
connectivity = np.zeros((n_electrodes, n_electrodes))
for i in range(n_electrodes):
    for j in range(i+1, n_electrodes):
        dist = np.linalg.norm(coords[i] - coords[j])
        connectivity[i, j] = np.exp(-dist / 30.0) * np.random.rand() * 0.1
        connectivity[j, i] = connectivity[i, j]

ez_indices = [0, 1, 2]

# Run 3 simulations with different beta
print("Generating comparison...")
beta_values = [1.0, 0.8, 0.6]
results = {}

for beta in beta_values:
    print(f"  Beta = {beta}...")
    x, times = simulate_fractional_seizure(
        n_electrodes=n_electrodes,
        ez_indices=ez_indices,
        connectivity=connectivity,
        beta=beta,
        duration=2000,
        dt=1
    )
    results[beta] = x

# Create comparison figure
fig = make_subplots(
    rows=3, cols=1,
    subplot_titles=(
        'Beta = 1.0 (Standard Dynamics - No Memory)',
        'Beta = 0.8 (Moderate Memory Effects)',
        'Beta = 0.6 (Strong Memory Effects)'
    ),
    vertical_spacing=0.12
)

colors = ['red', 'blue', 'green', 'orange', 'purple']

for row, beta in enumerate(beta_values, 1):
    x = results[beta]
    
    # Plot EZ electrodes
    for i, ez_idx in enumerate(ez_indices):
        fig.add_trace(
            go.Scatter(
                x=times, 
                y=x[:, ez_idx],
                name=f'{names[ez_idx]}' if row == 1 else None,
                line=dict(color=colors[i], width=2),
                showlegend=(row == 1)
            ),
            row=row, col=1
        )
    
    # Plot top 2 propagated
    propagated = np.argsort(np.abs(x[-1, :]))[-5:-3]
    for i, prop_idx in enumerate(propagated):
        if prop_idx not in ez_indices:
            fig.add_trace(
                go.Scatter(
                    x=times,
                    y=x[:, prop_idx],
                    name=f'{names[prop_idx]}' if row == 1 else None,
                    line=dict(color=colors[3+i], width=1, dash='dash'),
                    showlegend=(row == 1)
                ),
                row=row, col=1
            )

# Update axes
for row in range(1, 4):
    fig.update_xaxes(title_text="Time (ms)", row=row, col=1)
    fig.update_yaxes(title_text="Neural Activity", row=row, col=1)

fig.update_layout(
    height=1000,
    title=dict(
        text=f"{patient_id}: Effect of Fractional Order (Beta) on Seizure Dynamics",
        x=0.5,
        xanchor='center',
        font=dict(size=18)
    ),
    showlegend=True,
    legend=dict(x=1.05, y=0.5)
)

# Add annotations explaining differences
fig.add_annotation(
    text="<b>Key Observations:</b><br>" +
         "• Beta = 1.0: Fast rise, quick decay (memoryless)<br>" +
         "• Beta = 0.8: Slower dynamics, moderate persistence<br>" +
         "• Beta = 0.6: Very slow, sustained activity (strong memory)",
    xref="paper", yref="paper",
    x=0.02, y=0.98,
    showarrow=False,
    bgcolor="lightyellow",
    bordercolor="black",
    borderwidth=1,
    font=dict(size=11),
    align="left",
    xanchor="left",
    yanchor="top"
)

filename = f'{patient_id}_BETA_COMPARISON.html'
fig.write_html(filename)
print(f"\nSAVED: {filename}")
print("\nThis figure shows how memory effects change seizure propagation!")
