"""
Data Ingestion Module
=====================
Handles loading of Structural Connectivity, Cortical Mesh, and Region Mappings.
Abstraction layer over the file-system paths.
"""

import os
import numpy as np
import zipfile
import re

class VEPLoader:
    def __init__(self, data_root=None):
        """
        Initialize loader with path to tvb-data.
        If None, attempts to auto-discover in site-packages.
        """
        self.data_root = data_root or self._find_data_root()
        print(f"[VEPLoader] Using data root: {self.data_root}")
        
    def _find_data_root(self):
        # Common locations for tvb-data
        import tvb_data
        return os.path.dirname(tvb_data.__file__)

    def load_connectivity(self, n_regions=76):
        """
        Load weights and tract lengths.
        Returns: weights (NxN), centers (Nx3), region_labels (N)
        """
        file_path = os.path.join(self.data_root, 'connectivity', f'connectivity_{n_regions}.zip')
        
        with zipfile.ZipFile(file_path, 'r') as zf:
            # Load Weights
            with zf.open('weights.txt') as f:
                weights = np.loadtxt(f)
                
            # Load Centers
            with zf.open('centres.txt') as f:
                centers = np.loadtxt(f, usecols=(1, 2, 3))
                
            # Load Labels
            with zf.open('centres.txt') as f:
                # Labels are typically the first column (string)
                content = f.read().decode('utf-8')
                labels = [line.split()[0] for line in content.strip().split('\n')]
            
            # Load Tract Lengths
            try:
                with zf.open('tract_lengths.txt') as f:
                    lengths = np.loadtxt(f)
            except KeyError:
                print("[VEPLoader] tract_lengths.txt not found. Computing Euclidean distances.")
                # Compute from centers
                pos = centers
                lengths = np.sqrt(np.sum((pos[:, np.newaxis, :] - pos[np.newaxis, :, :]) ** 2, axis=2))

        # Normalize weights
        weights = weights / np.max(weights)
        
        # Expand Labels to Full Anatomical Names
        full_labels = [self._expand_label(l) for l in labels]
                
        return weights, lengths, labels, full_labels

    def _expand_label(self, code):
        """
        Convert abbreviated region code (e.g. rAMYG) to full name (Right Amygdala).
        Based on standard TVB/CoCoMac nomenclature.
        """
        # 1. Hemisphere
        if code.startswith('r'):
            hemi = "Right"
            body = code[1:]
        elif code.startswith('l'):
            hemi = "Left"
            body = code[1:]
        else:
            hemi = ""
            body = code
            
        # 2. Region Lookup (Common TVB/VEP Abbreviations)
        # Note: This is a robust heuristics map
        mapping = {
            'AMYG': 'Amygdala', 'HC': 'Hippocampus', 'PHC': 'Parahippocampal Cortex',
            'T': 'Temporal Cortex', 'F': 'Frontal Cortex', 'P': 'Parietal Cortex', 'O': 'Occipital Cortex',
            'V1': 'Primary Visual Cortex', 'V2': 'Secondary Visual Cortex',
            'CC': 'Cingulate Cortex', 'PFC': 'Prefrontal Cortex',
            'M1': 'Primary Motor Cortex', 'S1': 'Primary Somatosensory Cortex',
            'Thal': 'Thalamus', 'Put': 'Putamen', 'Caud': 'Caudate',
            'GPe': 'Globus Pallidus Ext', 'GPi': 'Globus Pallidus Int',
            'STC': 'Superior Temporal Cortex', 'ITC': 'Inferior Temporal Cortex',
            'OFC': 'Orbital Frontal Cortex'
        }
        
        # Try direct match
        name = mapping.get(body)
        
        # If not found, try robust expansion
        if not name:
            if 'C' in body and not 'Cortex' in body:
                body = body.replace('C', ' Cortex')
            name = body
            
        return f"{hemi} {name}".strip()


    def load_cortex(self):
        """
        Load the high-resolution cortical mesh.
        Returns: vertices (Vx3), triangles (Tx3), mapping (V -> RegionIdx)
        """
        mesh_zip = os.path.join(self.data_root, 'surfaceData', 'cortex_16384.zip')
        mapping_file = os.path.join(self.data_root, 'regionMapping', 'regionMapping_16k_76.txt')
        
        # Load Mesh
        with zipfile.ZipFile(mesh_zip, 'r') as zf:
            with zf.open('vertices.txt') as f:
                vertices = np.loadtxt(f)
            with zf.open('triangles.txt') as f:
                triangles = np.loadtxt(f, dtype=int)
                
        # Load Mapping
        mapping = np.loadtxt(mapping_file, dtype=int)
        
        return vertices, triangles, mapping

if __name__ == "__main__":
    loader = VEPLoader()
    w, c, l = loader.load_connectivity()
    print(f"Loaded Connectivity: shape {w.shape}")
