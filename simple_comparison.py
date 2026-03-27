"""
Simple 2D comparison - no WebGL needed
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from fractional_dynamics import simulate_fractional_seizure

# Load patient
patient_id = 'sub-R1001P'
df = pd.read_csv(f'{patient_id}_electrode_mapping.csv')
coords = df[['X', 'Y', 'Z']].values
names = df['Electrode'].values
n_electrodes = len(names)

# Connectivity
connectivity = np.zeros((n_electrodes, n_electrodes))
for i in range(n_electrodes):
    for j in range(i+1, n_electrodes):
        dist = np.linalg.norm(coords[i] - coords[j])
        connectivity[i, j] = np.exp(-dist / 30.0) * np.random.rand() * 0.1
        connectivity[j, i] = connectivity[i, j]

ez_indices = [0, 1, 2]

# Run simulations
print("Generating 2D comparison...")
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

# Create figure (2D only - no WebGL)
fig = go.Figure()

# Beta = 1.0 (solid lines)
x = results[1.0]
for i in ez_indices:
    fig.add_trace(go.Scatter(
        x=times, y=x[:, i],
        name=f'Beta=1.0: {names[i]}',
        line=dict(width=2)
    ))

# Beta = 0.8 (dashed)
x = results[0.8]
for i in ez_indices:
    fig.add_trace(go.Scatter(
        x=times, y=x[:, i],
        name=f'Beta=0.8: {names[i]}',
        line=dict(width=2, dash='dash')
    ))

# Beta = 0.6 (dotted)
x = results[0.6]
for i in ez_indices:
    fig.add_trace(go.Scatter(
        x=times, y=x[:, i],
        name=f'Beta=0.6: {names[i]}',
        line=dict(width=2, dash='dot')
    ))

fig.update_layout(
    title=f"{patient_id}: Fractional Order Comparison (EZ Electrodes Only)",
    xaxis_title="Time (ms)",
    yaxis_title="Neural Activity",
    height=600,
    showlegend=True,
    legend=dict(x=1.05, y=1)
)

fig.add_annotation(
    text="<b>Solid</b> = Beta 1.0 (standard)<br>" +
         "<b>Dashed</b> = Beta 0.8 (moderate memory)<br>" +
         "<b>Dotted</b> = Beta 0.6 (strong memory)",
    xref="paper", yref="paper",
    x=0.02, y=0.98,
    showarrow=False,
    bgcolor="lightyellow",
    bordercolor="black",
    borderwidth=1,
    align="left",
    xanchor="left",
    yanchor="top"
)

filename = f'{patient_id}_SIMPLE_COMPARISON.html'
fig.write_html(filename)
print(f"\nSAVED: {filename}")
print("This should work in any browser!")
