"""
Emergency: 3 Patient Brain Models with Electrodes
"""
import sys
sys.path.append('./vep')

from upenn_loader import UPennLoader
import numpy as np
import plotly.graph_objects as go
import pandas as pd

def create_patient_brain_3d(patient_id):
    """Create 3D brain with electrodes for one patient"""
    upenn = UPennLoader(data_root='./data/upenn')
    
    print(f"Loading {patient_id}...")
    raw, events, electrodes = upenn.load_patient_ieeg(patient_id)
    
    # Get electrode positions
    coords = electrodes[['x', 'y', 'z']].values
    names = electrodes['name'].values
    
    # Check if has region info
    if 'ind.region' in electrodes.columns:
        regions = electrodes['ind.region'].values
    else:
        regions = ['unknown'] * len(names)
    
    # Create 3D scatter plot
    fig = go.Figure()
    
    # Add electrodes
    fig.add_trace(go.Scatter3d(
        x=coords[:, 0],
        y=coords[:, 1],
        z=coords[:, 2],
        mode='markers+text',
        marker=dict(size=8, color='red', opacity=0.8),
        text=names,
        textposition="top center",
        name='Electrodes',
        hovertemplate='<b>%{text}</b><br>X: %{x:.1f}<br>Y: %{y:.1f}<br>Z: %{z:.1f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=f'{patient_id} - Brain Electrode Positions ({len(names)} electrodes)',
        scene=dict(
            xaxis_title='X (mm)',
            yaxis_title='Y (mm)',
            zaxis_title='Z (mm)',
            aspectmode='data'
        ),
        height=700
    )
    
    filename = f'{patient_id}_brain_3d.html'
    fig.write_html(filename)
    print(f"✓ Saved: {filename}")
    
    return coords, names, regions, len(events)

# Generate for 3 patients
patients = ['sub-R1001P', 'sub-R1002P', 'sub-R1003P']
summary = []

for patient_id in patients:
    coords, names, regions, n_events = create_patient_brain_3d(patient_id)
    summary.append({
        'Patient': patient_id,
        'Electrodes': len(names),
        'Events': n_events,
        'X_range': f'{coords[:,0].min():.1f} to {coords[:,0].max():.1f}',
        'Y_range': f'{coords[:,1].min():.1f} to {coords[:,1].max():.1f}',
        'Z_range': f'{coords[:,2].min():.1f} to {coords[:,2].max():.1f}'
    })

# Print summary
print("\n" + "="*60)
print("PATIENT SUMMARY:")
print("="*60)
for s in summary:
    print(f"\n{s['Patient']}:")
    print(f"  Electrodes: {s['Electrodes']}")
    print(f"  Events: {s['Events']}")
    print(f"  Brain span: X={s['X_range']}, Y={s['Y_range']}, Z={s['Z_range']}")

print("\n✓ Generated 3 patient brain models!")
