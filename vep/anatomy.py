"""
Brain Anatomy Loader
====================
Loads structural connectivity and cortical mesh from local TVB data.
Supports multiple atlas resolutions and species.
"""

import os
import numpy as np
import zipfile
import logging

logger = logging.getLogger(__name__)

# Default data root (local extended dataset)
DEFAULT_DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'tvb_data', 'tvb_data')

# Available atlas configurations
ATLAS_OPTIONS = {
    # Human atlases
    66:  {'conn': 'connectivity_66.zip',  'mesh': 'cortex_16384.zip', 'mapping': 'regionMapping_16k_76.txt'},
    68:  {'conn': 'connectivity_68.zip',  'mesh': 'cortex_16384.zip', 'mapping': 'regionMapping_16k_76.txt'},
    76:  {'conn': 'connectivity_76.zip',  'mesh': 'cortex_16384.zip', 'mapping': 'regionMapping_16k_76.txt'},
    80:  {'conn': 'connectivity_80.zip',  'mesh': 'cortex_80k.zip',   'mapping': 'regionMapping_80k_80.txt'},
    96:  {'conn': 'connectivity_96.zip',  'mesh': 'cortex_16384.zip', 'mapping': 'regionMapping_16k_76.txt'},
    192: {'conn': 'connectivity_192.zip', 'mesh': 'cortex_16384.zip', 'mapping': 'regionMapping_16k_192.txt'},
    998: {'conn': 'connectivity_998.zip', 'mesh': 'cortex_80k.zip',   'mapping': 'regionMapping_80k_80.txt'},
    # Macaque
    84:  {'conn': 'macaque_v3/connectivity_84.zip', 'mesh': 'macaque_v3/surface_147k.zip', 'mapping': 'macaque_v3/regionMapping_147k_84.txt'},
}


class BrainAnatomy:
    """Container for brain structural data."""
    
    def __init__(self, data_root=None):
        self.data_root = data_root or self._find_data_root()
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
        
        # Metadata
        self.atlas_regions = None
        self.species = "human"
        
    def _find_data_root(self):
        """Find local TVB data folder."""
        # Check local extended dataset first
        if os.path.exists(DEFAULT_DATA_ROOT):
            return DEFAULT_DATA_ROOT
        
        # Fallback to pip-installed tvb_data
        try:
            import tvb_data
            return os.path.dirname(tvb_data.__file__)
        except ImportError:
            raise FileNotFoundError("TVB data not found. Please install tvb-data or provide data_root.")
    
    def load_connectivity(self, n_regions=76):
        """
        Load structural connectivity matrices.
        
        Args:
            n_regions: Atlas resolution (66, 68, 76, 80, 96, 192, 998, or 84 for macaque)
        """
        self.atlas_regions = n_regions
        
        if n_regions == 84:
            self.species = "macaque"
            zip_path = os.path.join(self.data_root, 'macaque_v3', 'connectivity_84.zip')
        else:
            self.species = "human"
            zip_path = os.path.join(self.data_root, 'connectivity', f'connectivity_{n_regions}.zip')
        
        logger.info(f"Loading {self.species} connectivity: {n_regions} regions")
        
        with zipfile.ZipFile(zip_path, 'r') as zf:
            # Get list of files
            files = zf.namelist()
            
            # Weights
            weight_file = next((f for f in files if 'weight' in f.lower()), None)
            if weight_file:
                with zf.open(weight_file) as f:
                    self.weights = np.loadtxt(f)
            
            # Distances (tract lengths)
            tract_file = next((f for f in files if 'tract' in f.lower() or 'length' in f.lower()), None)
            if tract_file:
                with zf.open(tract_file) as f:
                    self.distances = np.loadtxt(f)
            
            # Centers and labels
            center_file = next((f for f in files if 'centre' in f.lower() or 'center' in f.lower()), None)
            if center_file:
                with zf.open(center_file) as f:
                    content = f.read().decode('utf-8').strip().split('\n')
                    self.labels = [line.split()[0] for line in content]
                    self.centers = np.array([[float(x) for x in line.split()[1:4]] for line in content])
        
        # Compute distances from centers if not loaded
        if self.distances is None and self.centers is not None:
            logger.warning("tract_lengths missing, computing from centers")
            diff = self.centers[:, np.newaxis, :] - self.centers[np.newaxis, :, :]
            self.distances = np.sqrt(np.sum(diff**2, axis=2))
        
        # Normalize weights
        if self.weights is not None:
            self.weights = self.weights / (np.max(self.weights) + 1e-10)
            self.n_regions = self.weights.shape[0]
        
        logger.info(f"Loaded connectivity: {self.n_regions} regions")
        return self
    
    def load_cortex(self, resolution='16k'):
        """
        Load cortical mesh.
        
        Args:
            resolution: '16k', '80k', or '120k' (high-res)
        """
        mesh_files = {
            '16k': 'cortex_16384.zip',
            '80k': 'cortex_80k.zip',
            '120k': 'cortex_2x120k.zip',
        }
        
        mesh_file = mesh_files.get(resolution, mesh_files['16k'])
        
        if self.species == "macaque":
            mesh_zip = os.path.join(self.data_root, 'macaque_v3', 'surface_147k.zip')
        else:
            mesh_zip = os.path.join(self.data_root, 'surfaceData', mesh_file)
        
        logger.info(f"Loading cortex mesh: {resolution}")
        
        with zipfile.ZipFile(mesh_zip, 'r') as zf:
            files = zf.namelist()
            
            vert_file = next((f for f in files if 'vert' in f.lower()), None)
            tri_file = next((f for f in files if 'tri' in f.lower()), None)
            
            if vert_file:
                with zf.open(vert_file) as f:
                    self.vertices = np.loadtxt(f)
            if tri_file:
                with zf.open(tri_file) as f:
                    self.triangles = np.loadtxt(f, dtype=int)
        
        # Load region mapping
        self._load_region_mapping(resolution)
        
        logger.info(f"Loaded cortex: {len(self.vertices)} vertices, {len(self.triangles)} triangles")
        return self
    
    def _load_region_mapping(self, resolution):
        """Load vertex-to-region mapping."""
        mapping_files = {
            '16k': {
                76: 'regionMapping_16k_76.txt',
                192: 'regionMapping_16k_192.txt',
            },
            '80k': {
                80: 'regionMapping_80k_80.txt',
            }
        }
        
        if self.species == "macaque":
            mapping_file = os.path.join(self.data_root, 'macaque_v3', 'regionMapping_147k_84.txt')
        else:
            # Find best matching mapping
            res_mappings = mapping_files.get(resolution, mapping_files['16k'])
            if self.atlas_regions in res_mappings:
                fname = res_mappings[self.atlas_regions]
            else:
                fname = 'regionMapping_16k_76.txt'
            mapping_file = os.path.join(self.data_root, 'regionMapping', fname)
        
        if os.path.exists(mapping_file):
            self.region_mapping = np.loadtxt(mapping_file, dtype=int)
        else:
            # Create dummy mapping if file missing
            logger.warning(f"Region mapping not found: {mapping_file}")
            self.region_mapping = np.zeros(len(self.vertices), dtype=int)
    
    def get_region_name(self, code):
        """Expand region code to full anatomical name."""
        if self.species == "macaque":
            return code  # Macaque labels are already descriptive
        
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
    
    def list_available_atlases(self):
        """List available connectivity atlases."""
        conn_dir = os.path.join(self.data_root, 'connectivity')
        available = []
        
        if os.path.exists(conn_dir):
            for f in os.listdir(conn_dir):
                if f.startswith('connectivity_') and f.endswith('.zip'):
                    n = f.replace('connectivity_', '').replace('.zip', '')
                    try:
                        available.append(int(n))
                    except ValueError:
                        pass
        
        # Check for macaque
        macaque_path = os.path.join(self.data_root, 'macaque_v3', 'connectivity_84.zip')
        if os.path.exists(macaque_path):
            available.append(84)
        
        return sorted(available)
