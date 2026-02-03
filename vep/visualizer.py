"""
Brain Visualizer
================
Production-quality 3D brain visualization with:
- Solid translucent brain mesh (not sparse points)
- Color-coded region nodes (EZ=red, PZ=orange, healthy=blue)
- Time-series subplot with proper legend
- Smooth animation
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging

from .config import viz_config, physics_config

logger = logging.getLogger(__name__)


class BrainVisualizer:
    """High-quality clinical brain visualization."""
    
    def __init__(self, anatomy, config=None):
        self.anatomy = anatomy
        self.config = config or viz_config
        
        # Precompute region centroids
        self.region_centers = self._compute_centroids()
        
    def _compute_centroids(self):
        """Compute centroid for each brain region."""
        n_regions = self.anatomy.n_regions
        centroids = np.zeros((n_regions, 3))
        
        for r in range(n_regions):
            mask = self.anatomy.region_mapping == r
            if np.sum(mask) > 0:
                centroids[r] = np.mean(self.anatomy.vertices[mask], axis=0)
            else:
                # Fallback to connectivity centers
                centroids[r] = self.anatomy.centers[r] if self.anatomy.centers is not None else [0, 0, 0]
        
        return centroids
    
    def create_report(self, time, data, onset_times, x0_values, output_path="vep_report.html"):
        """
        Generate interactive HTML report.
        
        Args:
            time: (T,) time vector in ms
            data: (T, N) x1 activity per region
            onset_times: (N,) onset time per region (-1 if no seizure)
            x0_values: (N,) excitability parameter per region
            output_path: Output HTML file
        """
        logger.info(f"Generating report: {output_path}")
        
        n_regions = self.anatomy.n_regions
        labels = self.anatomy.labels
        full_labels = self.anatomy.get_full_labels()
        
        # Determine region status
        is_ez = x0_values > physics_config.x0_healthy
        is_pz = (onset_times > 0) & (~is_ez)
        is_healthy = ~is_ez & ~is_pz
        
        # Create figure with subplots
        fig = make_subplots(
            rows=2, cols=1,
            row_heights=[0.65, 0.35],
            specs=[[{'type': 'scene'}], [{'type': 'xy'}]],
            vertical_spacing=0.08,
            subplot_titles=["3D Brain Model", "Neural Activity Time Series"]
        )
        
        # ========== 1. BRAIN MESH (Solid, translucent) ==========
        # Use actual mesh triangles for proper brain shape
        fig.add_trace(
            go.Mesh3d(
                x=self.anatomy.vertices[:, 0],
                y=self.anatomy.vertices[:, 1],
                z=self.anatomy.vertices[:, 2],
                i=self.anatomy.triangles[:, 0],
                j=self.anatomy.triangles[:, 1],
                k=self.anatomy.triangles[:, 2],
                color=self.config.mesh_color,
                opacity=self.config.mesh_opacity,
                name="Brain Surface",
                hoverinfo='skip',
                lighting=dict(ambient=0.8, diffuse=0.5),
                lightposition=dict(x=100, y=200, z=0)
            ),
            row=1, col=1
        )
        
        # ========== 2. REGION NODES (colored spheres) ==========
        # Assign colors based on status
        node_colors = np.array([self.config.healthy_color] * n_regions)
        node_colors[is_ez] = self.config.ez_color
        node_colors[is_pz] = self.config.pz_color
        
        # Node sizes based on initial activity
        initial_activity = data[0] if len(data) > 0 else np.zeros(n_regions)
        norm = (initial_activity + 2) / 4  # Normalize from [-2, 2] to [0, 1]
        norm = np.clip(norm, 0, 1)
        node_sizes = self.config.node_size_min + norm * (self.config.node_size_max - self.config.node_size_min)
        
        # Hover text
        hover_texts = []
        for i in range(n_regions):
            status = "Epileptogenic Zone (EZ)" if is_ez[i] else ("Propagated (PZ)" if is_pz[i] else "Healthy")
            onset = f"{onset_times[i]:.0f} ms" if onset_times[i] > 0 else "None"
            hover_texts.append(
                f"<b>{full_labels[i]}</b><br>"
                f"Code: {labels[i]}<br>"
                f"Status: {status}<br>"
                f"Excitability (x0): {x0_values[i]:.3f}<br>"
                f"Seizure Onset: {onset}"
            )
        
        fig.add_trace(
            go.Scatter3d(
                x=self.region_centers[:, 0],
                y=self.region_centers[:, 1],
                z=self.region_centers[:, 2],
                mode='markers',
                marker=dict(
                    size=node_sizes,
                    color=node_colors,
                    opacity=0.9,
                    line=dict(width=1, color='white')
                ),
                text=hover_texts,
                hoverinfo='text',
                name="Brain Regions"
            ),
            row=1, col=1
        )
        
        # ========== 3. TIME SERIES (2D plot) ==========
        # Plot EZ and top propagated regions
        plot_indices = list(np.where(is_ez)[0])
        pz_indices = np.where(is_pz)[0]
        if len(pz_indices) > 0:
            # Sort by onset time and take first 3
            sorted_pz = pz_indices[np.argsort(onset_times[pz_indices])]
            plot_indices.extend(sorted_pz[:3].tolist())
        
        # Limit to 6 traces
        plot_indices = plot_indices[:6]
        
        colors = ['#ff3333', '#ff6666', '#ffaa00', '#ffcc00', '#44aaff', '#88ccff']
        for i, idx in enumerate(plot_indices):
            status = "EZ" if is_ez[idx] else "PZ"
            fig.add_trace(
                go.Scatter(
                    x=time,
                    y=data[:, idx],
                    mode='lines',
                    name=f"{labels[idx]} ({status})",
                    line=dict(width=2, color=colors[i % len(colors)]),
                    opacity=0.9
                ),
                row=2, col=1
            )
        
        # ========== 4. ANIMATION FRAMES ==========
        # Create frames for 3D animation
        n_frames = min(200, len(time))
        frame_stride = max(1, len(time) // n_frames)
        
        frames = []
        for k in range(0, len(time), frame_stride):
            activity = data[k]
            norm = (activity + 2) / 4
            norm = np.clip(norm, 0, 1)
            sizes = self.config.node_size_min + norm * (self.config.node_size_max - self.config.node_size_min)
            
            # Color by current activity
            colors_frame = []
            for i in range(n_regions):
                if is_ez[i]:
                    colors_frame.append(self.config.ez_color)
                elif activity[i] > -1.0:  # Spiking
                    colors_frame.append(self.config.pz_color)
                else:
                    colors_frame.append(self.config.healthy_color)
            
            frames.append(go.Frame(
                data=[go.Scatter3d(
                    x=self.region_centers[:, 0],
                    y=self.region_centers[:, 1],
                    z=self.region_centers[:, 2],
                    mode='markers',
                    marker=dict(
                        size=sizes,
                        color=colors_frame,
                        opacity=0.9,
                        line=dict(width=1, color='white')
                    ),
                    text=hover_texts,
                    hoverinfo='text'
                )],
                traces=[1],  # Update only the nodes trace
                name=f"{time[k]:.0f}"
            ))
        
        fig.frames = frames
        
        # ========== 5. LAYOUT ==========
        fig.update_layout(
            height=900,
            paper_bgcolor='#0a0a0a',
            plot_bgcolor='#0a0a0a',
            font=dict(color='white', size=12),
            margin=dict(l=20, r=20, t=60, b=20),
            
            # 3D scene
            scene=dict(
                aspectmode='data',
                camera=dict(eye=dict(x=1.8, y=0.5, z=0.5)),
                xaxis=dict(visible=False, showbackground=False),
                yaxis=dict(visible=False, showbackground=False),
                zaxis=dict(visible=False, showbackground=False),
                bgcolor='#0a0a0a'
            ),
            
            # 2D axes
            xaxis=dict(
                title="Time (ms)",
                color='white',
                gridcolor='#333',
                showgrid=True
            ),
            yaxis=dict(
                title="Neural Activity (x₁)",
                color='white',
                gridcolor='#333',
                showgrid=True,
                range=[-2.5, 2.0]
            ),
            
            # Legend - positioned to avoid overlap
            legend=dict(
                orientation='h',
                yanchor='top',
                y=0.32,
                xanchor='center',
                x=0.5,
                bgcolor='rgba(10,10,10,0.8)',
                bordercolor='#333',
                borderwidth=1,
                font=dict(size=11)
            ),
            
            # Animation controls
            updatemenus=[dict(
                type='buttons',
                showactive=False,
                y=0.02,
                x=0.02,
                xanchor='left',
                buttons=[
                    dict(label='▶ Play', method='animate',
                         args=[None, dict(frame=dict(duration=50, redraw=True), fromcurrent=True)]),
                    dict(label='⏸ Pause', method='animate',
                         args=[[None], dict(frame=dict(duration=0, redraw=False), mode='immediate')])
                ]
            )],
            
            sliders=[dict(
                active=0,
                yanchor='top',
                xanchor='left',
                currentvalue=dict(prefix='Time: ', suffix=' ms', font=dict(size=12, color='white')),
                pad=dict(b=10, t=50),
                len=0.9,
                x=0.05,
                y=0.35,
                steps=[dict(args=[[f.name], dict(mode='immediate', frame=dict(duration=0, redraw=True))],
                           method='animate', label=f.name) for f in frames]
            )]
        )
        
        # ========== 6. EXPORT ==========
        config = {'scrollZoom': True, 'displayModeBar': True, 'responsive': True}
        html = fig.to_html(config=config, include_plotlyjs='cdn', full_html=True)
        
        # Inject custom CSS for better styling
        custom_css = """
        <style>
            body { background: #0a0a0a; margin: 0; }
            .js-plotly-plot { border-radius: 8px; }
        </style>
        """
        html = html.replace('</head>', f'{custom_css}</head>')
        
        with open(output_path, 'w') as f:
            f.write(html)
        
        logger.info(f"Report saved: {output_path}")
        return output_path
