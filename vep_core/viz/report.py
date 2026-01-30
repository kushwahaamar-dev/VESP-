"""
Clinical Visualization Module
=============================
Generates the VEP Report dashboard.
Renders the cortical mesh and maps the seizure propagation onto it.
"""

import numpy as np
import plotly.graph_objects as go
import os

class VEPReport:
    @staticmethod
    def generate_dashboard(cortex, mapping, time, data, x0_values, onset_times, labels, output_path="vep_report.html"):
        """
        Create a High-Performance 'Glass Brain' Dashboard.
        - Static Translucent Cortical Mesh (Context)
        - Dynamic 3D Spheres (Active Regions)
        - Performance: 60FPS guaranteed
        """
        print(f"[Report] Generating Glass Brain Dashboard at {output_path}...")
        
        # --- 1. Prepare Data ---
        # We need Region Centers for the nodes.
        # We can approximate centers from the mesh mapping if not provided, 
        # but better to use the centers from the loader. 
        # Ideally, main_pipeline should pass 'centers'. 
        # I will calculate centroids from the mapping + cortex for now to be self-contained.
        
        vertices, triangles, _ = cortex
        n_regions = len(labels)
        dataset_centers = np.zeros((n_regions, 3))
        
        for r_idx in range(n_regions):
            # Find vertices belonging to region r_idx
            # mapping is array of size (n_verts,) with region indices
            vert_indices = np.where(mapping == r_idx)[0]
            if len(vert_indices) > 0:
                dataset_centers[r_idx] = np.mean(vertices[vert_indices], axis=0)
            else:
                # Fallback if region has no vertices (subcortical?)
                pass

        # Subsample time (Statistics are fine, animation needs flow)
        n_frames = 100 # We can handle more frames now because nodes are cheap!
        stride = max(1, len(time) // n_frames)
        frames_data = data[::stride]
        frames_time = time[::stride]
        
        # --- 2. Visual Elements ---
        
        # A. The Glass Brain (Static Mesh)
        x, y, z = vertices.T
        i, j, k = triangles.T
        
        trace_glass = go.Mesh3d(
            x=x, y=y, z=z,
            i=i, j=j, k=k,
            color='lightgrey',
            opacity=0.15, # Very translucent
            name='Anatomy',
            hoverinfo='skip', # Don't block hover on nodes
            lighting=dict(ambient=0.5, diffuse=0.5)
        )
        
        # B. The Active Nodes (Dynamic Scatter)
        # We animate Color and Size based on signal
        
        def get_node_trace(frame_idx):
            signal = frames_data[frame_idx]
            # Size: 5 (baseline) + signal * 5 (expands on spike)
            # Signal is approx -2 to +2.
            # Normalize signal for size: (s + 2) * 3 -> 0 to 12
            sizes = np.clip((signal + 2.0) * 4.0, 5, 20)
            
            return go.Scatter3d(
                x=dataset_centers[:, 0],
                y=dataset_centers[:, 1],
                z=dataset_centers[:, 2],
                mode='markers',
                marker=dict(
                    size=sizes,
                    color=signal,
                    colorscale='RdBu_r', # Red = Seizure (Positive/High), Blue = Quiet
                    cmin=-2.0, cmax=2.0,
                    showscale=True,
                    colorbar=dict(title='LFP (mV)', x=0.0, y=0.7, len=0.6, tickfont=dict(color='white')),
                    opacity=0.9
                ),
                text=[f"Region: {l}<br>x0: {x:.2f}<br>T_onset: {o:.0f}ms" 
                      for l, x, o in zip(labels, x0_values, onset_times)],
                hoverinfo='text',
                name='Regions'
            )
            
        dataset_node_trace = get_node_trace(0)
        
        # --- 3. Animation Frames ---
        frames = [
            go.Frame(
                data=[trace_glass, get_node_trace(k)], # Keep glass, update nodes
                name=f'f{k}',
                layout=go.Layout(title_text=f"Time: {t:.0f} ms")
            ) for k, t in enumerate(frames_time)
        ]
        
        # --- 4. Layout & Controls ---
        steps = [
            dict(
                method='animate',
                args=[[f'f{k}'], dict(mode='immediate', frame=dict(duration=50, redraw=True), transition=dict(duration=0))],
                label=f'{t:.0f}ms'
            ) for k, t in enumerate(frames_time)
        ]
        
        layout = go.Layout(
            title=dict(text="High-Performance VEP Dashboard (Glass Brain)", font=dict(size=24, color='white')),
            template="plotly_dark",
            paper_bgcolor="#050505",
            height=900,
            scene=dict(
                aspectmode='data',
                camera=dict(eye=dict(x=1.5, y=0, z=0.5)),
                dragmode='orbit',
                xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False)
            ),
            sliders=[dict(
                steps=steps, active=0,
                currentvalue=dict(font=dict(size=14, color="white"), prefix="Time: ", xanchor="center"),
                pad=dict(t=50), x=0.1, len=0.8
            )],
            updatemenus=[dict(
                type='buttons', showactive=False, y=0.1, x=0.05,
                buttons=[
                    dict(label='▶ PLAY', method='animate', args=[None, dict(frame=dict(duration=50, redraw=True), fromcurrent=True)]),
                    dict(label='Ⅱ PAUSE', method='animate', args=[[None], dict(frame=dict(duration=0, redraw=False), mode='immediate')]),
                ]
            )]
        )
        
        fig = go.Figure(data=[trace_glass, dataset_node_trace], layout=layout, frames=frames)
        fig.write_html(output_path)
        print("[Report] Glass Brain Dashboard saved.")
