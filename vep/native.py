"""
Native High-Performance Visualizer
==================================
GPU-accelerated 3D brain viewer using PyVista (VTK).
Includes time series chart like the HTML report.
"""

import numpy as np
import pyvista as pv
import logging
from .config import viz_config, physics_config

logger = logging.getLogger(__name__)


class NativeVisualizer:
    """PyVista-based native desktop viewer with time series chart."""
    
    def __init__(self, anatomy, config=None):
        self.anatomy = anatomy
        self.config = config or viz_config
        self.pl = None
        
        # Data containers
        self.brain_mesh = None
        self.region_cloud = None
        self.current_frame = 0
        self.playing = False
        self.slider_widget = None
        
    def _create_brain_mesh(self):
        """Create VTK mesh from anatomy data."""
        n_faces = self.anatomy.triangles.shape[0]
        padding = np.full((n_faces, 1), 3, dtype=int)
        faces = np.hstack((padding, self.anatomy.triangles)).flatten()
        return pv.PolyData(self.anatomy.vertices, faces)

    def show(self, time, data, onset_times, x0_values):
        """Launch interactive 3D viewer with time series."""
        logger.info("Launching native high-performance viewer...")
        
        # Identify EZ, PZ, and healthy regions
        is_ez = x0_values > physics_config.x0_healthy
        ez_indices = np.where(is_ez)[0]
        pz_indices = np.where((onset_times > 0) & ~is_ez)[0]
        healthy_indices = np.where((onset_times <= 0) & ~is_ez)[0]
        
        # Select representative regions for time series
        n_plot = min(3, len(ez_indices))
        ez_plot = ez_indices[:n_plot] if len(ez_indices) > 0 else [0]
        pz_plot = pz_indices[:min(2, len(pz_indices))] if len(pz_indices) > 0 else []
        healthy_plot = healthy_indices[:1] if len(healthy_indices) > 0 else []
        
        # 1. Setup Plotter with subplots
        self.pl = pv.Plotter(
            title="VEP Native Viewer", 
            window_size=[1400, 900],
            shape=(1, 2),  # 2 columns
            border=True
        )
        
        # ========== LEFT: 3D Brain View ==========
        self.pl.subplot(0, 0)
        self.pl.set_background("#0a0a0a")
        
        # Brain Mesh
        self.brain_mesh = self._create_brain_mesh()
        self.pl.add_mesh(
            self.brain_mesh,
            color=self.config.mesh_color,
            opacity=self.config.mesh_opacity,
            smooth_shading=True,
            specular=0.5,
            name="brain"
        )
        
        # Region Nodes
        centroids = np.zeros((self.anatomy.n_regions, 3))
        for r in range(self.anatomy.n_regions):
            mask = self.anatomy.region_mapping == r
            if np.sum(mask) > 0:
                centroids[r] = np.mean(self.anatomy.vertices[mask], axis=0)
            else:
                centroids[r] = self.anatomy.centers[r] if self.anatomy.centers is not None else [0, 0, 0]
        
        self.region_cloud = pv.PolyData(centroids)
        sphere = pv.Sphere(theta_resolution=16, phi_resolution=16)
        
        # Initial colors
        base_colors = np.zeros((self.anatomy.n_regions, 3))
        for i in range(self.anatomy.n_regions):
            if is_ez[i]:
                base_colors[i] = [1.0, 0.2, 0.2]  # Red (EZ)
            elif onset_times[i] > 0:
                base_colors[i] = [1.0, 0.7, 0.0]  # Orange (PZ)
            else:
                base_colors[i] = [0.2, 0.5, 1.0]  # Blue (Healthy)
                
        self.region_cloud["colors"] = base_colors
        self.region_cloud["radius"] = np.ones(self.anatomy.n_regions) * self.config.node_size_min
        self.region_cloud.set_active_scalars("colors")
        
        glyphs = self.region_cloud.glyph(scale="radius", geom=sphere, orient=False)
        self.pl.add_mesh(glyphs, scalars="colors", rgb=True, show_scalar_bar=False, name="nodes")
        
        # Legend
        self.pl.add_text(
            "🔴 EZ (Epileptogenic)\n🟠 PZ (Propagated)\n🔵 Healthy",
            position='upper_left',
            font_size=11,
            color='white',
            name='legend'
        )
        
        # ========== RIGHT: Time Series Chart ==========
        self.pl.subplot(0, 1)
        self.pl.set_background("#0a0a0a")
        
        # Create chart
        chart = pv.Chart2D()
        chart.background_color = (0.05, 0.05, 0.05, 1.0)
        chart.x_axis.label = "Time (ms)"
        chart.y_axis.label = "Neural Activity (x₁)"
        chart.x_axis.range = [time[0], time[-1]]
        chart.y_axis.range = [-2.5, 2.0]
        
        # Downsample for performance
        step = max(1, len(time) // 500)
        t_plot = time[::step]
        
        # Plot EZ regions (red)
        for idx in ez_plot:
            y = data[::step, idx]
            chart.line(t_plot, y, color='red', width=2.0, label=f"EZ: {self.anatomy.labels[idx]}")
        
        # Plot PZ regions (orange)
        for idx in pz_plot:
            y = data[::step, idx]
            chart.line(t_plot, y, color='orange', width=2.0, label=f"PZ: {self.anatomy.labels[idx]}")
        
        # Plot healthy regions (blue)
        for idx in healthy_plot:
            y = data[::step, idx]
            chart.line(t_plot, y, color='dodgerblue', width=2.0, label=f"Healthy: {self.anatomy.labels[idx]}")
        
        # Add vertical time indicator line (will update during animation)
        self.time_line = chart.line([0, 0], [-2.5, 2.0], color='white', width=1.5)
        
        self.pl.add_chart(chart)
        
        # ========== Control Panel (Bottom) ==========
        self.pl.subplot(0, 0)  # Back to 3D view for widgets
        
        # Time Slider
        def update_time(value):
            idx = int(value)
            self.current_frame = idx
            
            # Update node colors based on current activity
            activity = data[idx]
            new_colors = base_colors.copy()
            
            # Dynamic coloring: active regions glow brighter
            for i in range(self.anatomy.n_regions):
                if activity[i] > -1.0:
                    if is_ez[i]:
                        new_colors[i] = [1.0, 0.4, 0.4]  # Bright red
                    else:
                        new_colors[i] = [1.0, 0.8, 0.2]  # Bright orange/yellow
                        
            # Update node sizes
            norm = (activity + 2.0) / 4.0
            sizes = self.config.node_size_min + np.clip(norm, 0, 1) * (self.config.node_size_max - self.config.node_size_min)
            
            self.region_cloud["radius"] = sizes
            self.region_cloud["colors"] = new_colors
            self.region_cloud.set_active_scalars("colors")
            
            new_glyphs = self.region_cloud.glyph(scale="radius", geom=sphere, orient=False)
            self.pl.add_mesh(new_glyphs, scalars="colors", rgb=True, show_scalar_bar=False, name="nodes")
            
            # Update time label
            self.pl.add_text(f"Time: {time[idx]:.1f} ms", position='upper_right', name='time_label', font_size=14, color='white')
            
            # Update time line on chart
            current_t = time[idx]
            self.time_line.update([current_t, current_t], [-2.5, 2.0])
        
        self.slider_widget = self.pl.add_slider_widget(
            update_time,
            [0, len(time)-1],
            value=0,
            title="Time Step",
            pointa=(0.1, 0.08),
            pointb=(0.9, 0.08),
            style='modern'
        )
        
        # Play/Pause Button
        def toggle_play(state):
            self.playing = state
            
        self.pl.add_checkbox_button_widget(
            toggle_play,
            position=(10, 100),
            size=35,
            border_size=2,
            color_on='lime',
            color_off='gray',
            background_color='white'
        )
        self.pl.add_text("▶ Play", position=(55, 112), font_size=11, color='white')
        
        # Animation Timer
        def animation_callback(obj, event):
            if self.playing:
                next_idx = (self.current_frame + 4) % len(time)
                rep = self.slider_widget.GetRepresentation()
                rep.SetValue(next_idx)
                update_time(next_idx)
        
        iren = self.pl.iren.interactor
        iren.AddObserver('TimerEvent', animation_callback)
        iren.CreateRepeatingTimer(50)
        
        # Instructions
        self.pl.add_text(
            "Controls: Click green box to Play/Pause | Rotate: LMB | Zoom: RMB | Pan: Shift+LMB | Quit: q",
            position=(10, 10),
            font_size=9,
            color='gray'
        )
        
        logger.info("Viewer ready. Opening window...")
        self.pl.show()
