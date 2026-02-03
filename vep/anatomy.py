"""
Brain Anatomy Loader
====================
Loads structural connectivity and cortical mesh from TVB data.
"""

import os
import numpy as np
import zipfile
import logging

logger = logging.getLogger(__name__)

class BrainAnatomy:
    """Container for brain structural data."""
    
    def __init__(self, data_root=None):
        self.data_root = data_root or self._find_tvb_data()
        logger.info(f"Using TVB data: {self.data_root}")
        
        # Loaded data
        self.weights = None
        self.distances = None
        self.labels = None
        self.centers = None
        self.n_regions = 0
        
        # Cortical surface
        self.vertices = None
        self.triangles = None
        self.region_mapping = None
        
    def _find_tvb_data(self):
        """Auto-discover TVB data location."""
        import tvb_data
        return os.path.dirname(tvb_data.__file__)
    
    def load_connectivity(self, n_regions=76):
        """Load structural connectivity matrices."""
        zip_path = os.path.join(self.data_root, 'connectivity', f'connectivity_{n_regions}.zip')
        
        with zipfile.ZipFile(zip_path, 'r') as zf:
            # Weights
            with zf.open('weights.txt') as f:
                self.weights = np.loadtxt(f)
            
            # Distances (tract lengths)
            try:
                with zf.open('tract_lengths.txt') as f:
                    self.distances = np.loadtxt(f)
            except KeyError:
                logger.warning("tract_lengths.txt missing, computing from centers")
                
            # Centers and labels
            with zf.open('centres.txt') as f:
                content = f.read().decode('utf-8').strip().split('\n')
                self.labels = [line.split()[0] for line in content]
                self.centers = np.array([[float(x) for x in line.split()[1:4]] for line in content])
        
        # Compute distances from centers if not loaded
        if self.distances is None:
            diff = self.centers[:, np.newaxis, :] - self.centers[np.newaxis, :, :]
            self.distances = np.sqrt(np.sum(diff**2, axis=2))
        
        # Normalize weights
        self.weights = self.weights / (np.max(self.weights) + 1e-10)
        self.n_regions = len(self.labels)
        
        logger.info(f"Loaded connectivity: {self.n_regions} regions")
        return self
    
    def load_cortex(self):
        """Load high-resolution cortical mesh."""
        mesh_zip = os.path.join(self.data_root, 'surfaceData', 'cortex_16384.zip')
        mapping_file = os.path.join(self.data_root, 'regionMapping', 'regionMapping_16k_76.txt')
        
        with zipfile.ZipFile(mesh_zip, 'r') as zf:
            with zf.open('vertices.txt') as f:
                self.vertices = np.loadtxt(f)
            with zf.open('triangles.txt') as f:
                self.triangles = np.loadtxt(f, dtype=int)
        
        self.region_mapping = np.loadtxt(mapping_file, dtype=int)
        
        logger.info(f"Loaded cortex: {len(self.vertices)} vertices, {len(self.triangles)} triangles")
        return self
    
    def get_region_name(self, code):
        """Expand region code to full anatomical name."""
        # Hemisphere
        if code.startswith('r'):
            hemi, body = "Right", code[1:]
        elif code.startswith('l'):
            hemi, body = "Left", code[1:]
        else:
            hemi, body = "", code
        
        # Region lookup
        names = {
            'AMYG': 'Amygdala', 'HC': 'Hippocampus', 'PHC': 'Parahippocampal',
            'V1': 'Primary Visual', 'V2': 'Secondary Visual',
            'M1': 'Primary Motor', 'S1': 'Primary Sensory',
            'PFC': 'Prefrontal', 'OFC': 'Orbitofrontal',
            'STC': 'Superior Temporal', 'ITC': 'Inferior Temporal',
            'CC': 'Cingulate', 'INS': 'Insula',
            'Thal': 'Thalamus', 'Put': 'Putamen', 'Caud': 'Caudate'
        }
        
        region = names.get(body, body)
        return f"{hemi} {region}".strip()
    
    def get_full_labels(self):
        """Get expanded labels for all regions."""
        return [self.get_region_name(l) for l in self.labels]
