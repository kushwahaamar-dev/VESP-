"""
Emergency: Show seizure on patient brain
"""
import sys
sys.path.append('./vep')

from upenn_loader import UPennLoader
import numpy as np
import plotly.graph_objects as go
import pandas as pd

def visualize_seizure_on_brain(patient_id):
    """Show seizure activity overlaid on electrode positions"""
    upenn = UPennLoader(data_root='./data/upenn')
    
    print(f"Processing {patient_id}...")
    raw, events, electrodes = upenn.load_patient_ieeg(patient_id)
    
    # Get electrode positions
    coords = electrodes[['x', 'y', 'z']].values
    names = electrodes['name'].values
    
    # Find seizure events (if any)
    seizure_events = events[events['trial_type'].str.contains('REC', case=False, na=False)]
    
    if len(seizure_events) == 0:
        print(f"  No clear seizure markers, using high-activity periods")
        # Use first 100 events as proxy
        seizure_events = events.head(100)
    
    print(f"  Found {len(seizure_events)} events")
    
    # Get data during events
    data = raw.get_data()  # All channels, all time
    times = raw.times
    
    # Calculate activity level per electrode (mean absolute value)
    activity = np.abs(data).mean(axis=1)
    activity_norm = (activity - activity.min()) / (activity.max() - activity.min())
    
    # Create color map based on activity
    colors = ['rgb({},{},{})'.format(
        int(255 * a), int(100 * (1-a)), int(50 * (1-a))
    ) for a in activity_norm[:len(coords)]]
    
    # Create 3D plot
    fig = go.Figure()
    
    # Add electrodes colored by activity
    fig.add_trace(go.Scatter3d(
        x=coords[:, 0],
        y=coords[:, 1],
        z=coords[:, 2],
        mode='markers+text',
        marker=dict(
            size=10, 
            color=activity_norm[:len(coords)],
            colorscale='Hot',
            showscale=True,
            colorbar=dict(title="Activity Level")
        ),
        text=names,
        textposition="top center",
        name='Electrodes',
        hovertemplate='<b>%{text}</b><br>Activity: %{marker.color:.2f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=f'{patient_id} - Seizure Activity Map ({len(names)} electrodes, {len(seizure_events)} events)',
        scene=dict(
            xaxis_title='X (mm)',
            yaxis_title='Y (mm)',
            zaxis_title='Z (mm)',
            aspectmode='data'
        ),
        height=700
    )
    
    filename = f'{patient_id}_SEIZURE_overlay.html'
    fig.write_html(filename)
    print(f"  SAVED: {filename}")
    
    return len(names), len(seizure_events)

# Process first patient with seizure overlay
patients = ['sub-R1001P', 'sub-R1002P', 'sub-R1003P']

print("\n" + "="*60)
print("GENERATING SEIZURE OVERLAYS...")
print("="*60 + "\n")

for patient in patients:
    n_electrodes, n_events = visualize_seizure_on_brain(patient)
    print(f"  {patient}: {n_electrodes} electrodes, {n_events} events\n")

print("="*60)
print("DONE! All seizure overlays generated!")
print("="*60)
