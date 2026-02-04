"""
Native High-Performance Visualizer
==================================
GPU-accelerated 3D brain viewer using PyVista (VTK).
Capable of rendering 1M+ vertices smoothly.
"""

import numpy as np
import pyvista as pv
import logging
from .config import viz_config, physics_config

logger = logging.getLogger(__name__)

class NativeVisualizer:
    """PyVista-based native desktop viewer."""
    
    def __init__(self, anatomy, config=None):
        self.anatomy = anatomy
        self.config = config or viz_config
        self.pl = None  # Plotter instance
        
        # Data containers
        self.brain_mesh = None
        self.region_cloud = None
        self.time_series = None
        self.current_frame = 0
        
    def _create_brain_mesh(self):
        """Create VTK mesh from anatomy data."""
        # PyVista expects faces as [n_points, p0, p1, p2, ...]
        # We need to prepend 3 to each triangle
        n_faces = self.anatomy.triangles.shape[0]
        padding = np.full((n_faces, 1), 3, dtype=int)
        faces = np.hstack((padding, self.anatomy.triangles)).flatten()
        
        mesh = pv.PolyData(self.anatomy.vertices, faces)
        return mesh

    def show(self, time, data, onset_times, x0_values):
        """
        Launch interactive 3D viewer.
        
        Args:
            time: (T,) Time vector
            data: (T, N) Activity matrix
            onset_times: (N,) Onset times
            x0_values: (N,) Excitability
        """
        logger.info("Launching native high-performance viewer...")
        
        # 1. Setup Plotter
        self.pl = pv.Plotter(title="VEP Native Viewer (PyVista)", window_size=[1200, 900])
        self.pl.set_background("#0a0a0a")
        
        # 2. Add Brain Mesh
        self.brain_mesh = self._create_brain_mesh()
        self.pl.add_mesh(
            self.brain_mesh,
            color=self.config.mesh_color,
            opacity=self.config.mesh_opacity,
            smooth_shading=True,
            specular=0.5,
            name="brain"
        )
        
        # 3. Add Region Nodes
        # Compute centroids
        centroids = np.zeros((self.anatomy.n_regions, 3))
        for r in range(self.anatomy.n_regions):
            mask = self.anatomy.region_mapping == r
            if np.sum(mask) > 0:
                centroids[r] = np.mean(self.anatomy.vertices[mask], axis=0)
            else:
                centroids[r] = self.anatomy.centers[r] if self.anatomy.centers is not None else [0,0,0]
        
        # Create spheres
        self.region_cloud = pv.PolyData(centroids)
        
        # Initial scalars (Healthy/EZ colors)
        status_colors = np.zeros((self.anatomy.n_regions, 3))
        is_ez = x0_values > physics_config.x0_healthy
        
        # Base color (Blue for healthy, Red for EZ)
        for i in range(self.anatomy.n_regions):
            if is_ez[i]:
                status_colors[i] = [1.0, 0.2, 0.2]  # Red
            else:
                status_colors[i] = [0.2, 0.5, 1.0]  # Blue
                
        self.region_cloud["colors"] = status_colors
        self.region_cloud["radius"] = np.ones(self.anatomy.n_regions) * self.config.node_size_min
        
        # Use glyphs for spheres
        sphere = pv.Sphere(theta_resolution=20, phi_resolution=20)
        start_nodes = self.region_cloud.glyph(scale="radius", geom=sphere)
        
        node_actor = self.pl.add_mesh(
            start_nodes,
            scalars="colors",
            rgb=True,
            show_scalar_bar=False,
            name="nodes"
        )
        
        # 4. Interactive Time Slider
        def update_time(value):
            """Callback for time slider."""
            idx = int(value)
            self.current_frame = idx
            
            # Update node sizes/colors based on activity
            activity = data[idx]
            
            # Normalize size
            norm = (activity + 2.0) / 4.0
            sizes = self.config.node_size_min + np.clip(norm, 0, 1) * (self.config.node_size_max - self.config.node_size_min)
            
            # Update colors (flash orange if active)
            new_colors = status_colors.copy()
            active_mask = activity > -1.0
            for i in np.where(active_mask)[0]:
                if not is_ez[i]:
                    new_colors[i] = [1.0, 0.7, 0.0]  # Orange for PZ
            
            self.region_cloud["radius"] = sizes
            self.region_cloud["colors"] = new_colors
            
            # Re-glyph
            new_nodes = self.region_cloud.glyph(scale="radius", geom=sphere)
            
            # Update mesh (in-place replacement is faster in PyVista)
            self.pl.add_mesh(
                new_nodes,
                scalars="colors",
                rgb=True,
                show_scalar_bar=False,
                name="nodes"
            )
            
            # Update text
            self.pl.add_text(f"Time: {time[idx]:.1f} ms", position='upper_right', name='time_label', font_size=12)

        # Add slider
        self.pl.add_slider_widget(
            update_time,
            [0, len(time)-1],
            value=0,
            title="Time Step",
            pointa=(0.1, 0.1),
            pointb=(0.9, 0.1),
            style='modern'
        )
        
        # 5. Add Legend
        self.pl.add_text(
            "Controls:\n- Left Click: Rotate\n- Right Click: Zoom\n- Shift+Click: Pan\n- Press 'q' to quit",
            position='upper_left',
            font_size=10,
            color='white'
        )
        
        logger.info("Viewer ready. Opening window...")
        self.pl.show()

