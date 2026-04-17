"""Quick visualization for poster - shows workflow"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Load patient
patient_id = 'sub-R1001P'
df = pd.read_csv(f'{patient_id}_electrode_mapping.csv')
coords = df[['X', 'Y', 'Z']].values
names = df['Electrode'].values
n = len(names)

# Simulate seizure activity (starts high in EZ, spreads outward)
ez_indices = [0, 1, 2]  # First 3 electrodes
activity = np.zeros(n)
activity[ez_indices] = 1.0  # High activity in EZ
for i in range(n):
    if i not in ez_indices:
        min_dist = min([np.linalg.norm(coords[i] - coords[j]) for j in ez_indices])
        activity[i] = np.exp(-min_dist / 50.0)

# FIGURE 1: Side-by-side comparison
fig1 = make_subplots(
    rows=1, cols=2,
    subplot_titles=('STEP 1: Patient Electrode Network', 
                    'STEP 2: Predicted Seizure Spread'),
    specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]],
    horizontal_spacing=0.05
)

# Left: All electrodes
fig1.add_trace(
    go.Scatter3d(
        x=coords[:, 0], y=coords[:, 1], z=coords[:, 2],
        mode='markers',
        marker=dict(size=8, color='steelblue', opacity=0.8),
        text=names,
        hovertemplate='<b>%{text}</b><extra></extra>',
        showlegend=False
    ),
    row=1, col=1
)

# Right: Activity colored
fig1.add_trace(
    go.Scatter3d(
        x=coords[:, 0], y=coords[:, 1], z=coords[:, 2],
        mode='markers',
        marker=dict(
            size=12,
            color=activity,
            colorscale='Hot',
            showscale=True,
            cmin=0,
            cmax=1,
            colorbar=dict(
                title=dict(text="Seizure<br>Activity", side="right"),
                x=1.15,
                len=0.7
            )
        ),
        text=[f"{names[i]}<br>Activity: {activity[i]:.2f}" for i in range(n)],
        hovertemplate='%{text}<extra></extra>',
        showlegend=False
    ),
    row=1, col=2
)

fig1.update_layout(
    height=700,
    width=1400,
    title=dict(
        text="CortexCompass: Patient-Specific Seizure Prediction",
        x=0.5,
        xanchor='center',
        font=dict(size=28, color='#CC0000')
    ),
    font=dict(size=14),
    scene=dict(
        camera=dict(eye=dict(x=1.3, y=1.3, z=1.2)),
        xaxis=dict(showticklabels=False, title=''),
        yaxis=dict(showticklabels=False, title=''),
        zaxis=dict(showticklabels=False, title='')
    ),
    scene2=dict(
        camera=dict(eye=dict(x=1.3, y=1.3, z=1.2)),
        xaxis=dict(showticklabels=False, title=''),
        yaxis=dict(showticklabels=False, title=''),
        zaxis=dict(showticklabels=False, title='')
    )
)

fig1.write_html('POSTER_IMAGE2_WORKFLOW.html')
print("SAVED: POSTER_IMAGE2_WORKFLOW.html")

# FIGURE 2: Single detailed view
fig2 = go.Figure()

hover_text = [f"<b>{names[i]}</b><br>Activity: {activity[i]:.2f}" for i in range(n)]

fig2.add_trace(
    go.Scatter3d(
        x=coords[:, 0], y=coords[:, 1], z=coords[:, 2],
        mode='markers',
        marker=dict(
            size=14,
            color=activity,
            colorscale='Hot',
            showscale=True,
            cmin=0,
            cmax=1,
            colorbar=dict(
                title=dict(text="Predicted Seizure Activity", side="right"),
                x=1.02
            )
        ),
        text=hover_text,
        hovertemplate='%{text}<extra></extra>',
        showlegend=False
    )
)

# Highlight EZ
fig2.add_trace(
    go.Scatter3d(
        x=coords[ez_indices, 0],
        y=coords[ez_indices, 1],
        z=coords[ez_indices, 2],
        mode='markers',
        marker=dict(
            size=18,
            color='red',
            symbol='diamond',
            line=dict(color='white', width=2)
        ),
        name='Seizure Origin',
        hovertemplate='<b>SEIZURE ORIGIN</b><extra></extra>'
    )
)

fig2.update_layout(
    height=800,
    width=1000,
    title=dict(
        text="Interactive 3D Brain Visualization",
        x=0.5,
        xanchor='center',
        font=dict(size=26, color='#CC0000')
    ),
    scene=dict(
        camera=dict(eye=dict(x=1.5, y=1.5, z=1.3)),
        xaxis=dict(showticklabels=False, title=''),
        yaxis=dict(showticklabels=False, title=''),
        zaxis=dict(showticklabels=False, title='')
    ),
    showlegend=True,
    legend=dict(x=0.7, y=0.95, font=dict(size=14))
)

fig2.write_html('POSTER_IMAGE3_DETAILED.html')
print("SAVED: POSTER_IMAGE3_DETAILED.html")

print("\nDONE! Open files and screenshot!")
