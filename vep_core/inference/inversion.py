"""
Bayesian Inference Module
=========================
Simulates the parameter estimation phase of the VEP workflow.
In a clinical setting, this module fits the Epileptor model to SEEG power envelopes.
Here, we implement the logic for mapping clinical hypothesis to model parameters.
"""

import numpy as np
from .. import config

class VEPInference:
    def __init__(self, n_regions, region_labels):
        self.n_regions = n_regions
        self.labels = region_labels
        
    def generate_hypothesis(self, target_region_str="Temporal"):
        """
        Generate a synthetic posterior distribution (Epileptogenicity Values)
        based on a clinical hypothesis (e.g., 'Right Temporal Lobe Epilepsy').
        
        This simulates the output of the HMC/NUTS sampler.
        """
        # EV = Epileptogenicity Value [0, 1]
        # 0 = Healthy, 1 = Highly Epileptogenic
        ev_distribution = np.random.beta(0.5, 10.0, self.n_regions) # Mostly low values
        
        target_indices = [i for i, label in enumerate(self.labels) 
                         if target_region_str.lower() in label.lower() or 
                            "Hippocampus" in label or "Amygdala" in label]
                            
        if not target_indices:
            print("[Inference] Text matching failed for labels. Defaulting to Right Hemisphere regions (indices 40-42).")
            # In 76-region atlas, indices 38-75 are Right Hemisphere usually
            target_indices = [35, 36, 40] 

        print(f"[Inference] Clinical Prior targets regions: {[self.labels[i] for i in target_indices]}")
        
        # Set high EV for target regions (Epileptogenic Zone)
        for idx in target_indices:
            # Force high EV for the demo
            ev_distribution[idx] = 0.95 # Max epileptogenicity
                
        return ev_distribution

    def map_ev_to_x0(self, ev_values):
        """
        Map Epileptogenicity Values (0-1) to Model Parameter x0 (-2.2 to -1.6).
        
        Mapping rule (approximate from Makhalova 2022):
        EV < 0.3  -> x0 ≈ -2.2 (Healthy)
        EV > 0.6  -> x0 > -2.0 (Epileptogenic)
        """
        # Linear map: 0 -> -2.2, 1 -> -1.6
        # Range span = 0.6
        x0 = -2.2 + (ev_values * 0.6)
        
        # Ensure bounds
        x0 = np.clip(x0, -2.4, -1.2)
        return x0
