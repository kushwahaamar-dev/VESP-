"""
VEP Pipeline Configuration & Physics Constants
==============================================
References:
1. Jirsa et al., "The Virtual Epileptic Patient", NeuroImage 2017.
2. Proix et al., "Permittivity Coupling", 2017.
3. Makhalova et al., "VEP Modeling", Epilepsia 2022.

Global physics constants and clinical thresholds for the VEP pipeline.
"""

import numpy as np

# --- Time Integration Settings ---
DT = 0.05               # Integration step (ms) - High fidelity
SAMPLING_RATE = 1000.0  # Hz
SIMULATION_LENGTH = 10000.0 # 10 seconds

# --- Connectome Physics ---
CONDUCTION_VELOCITY = 3.0   # m/ms (or mm/ms? TVB uses mm/ms usually, wait. 3 m/s = 0.003 mm/ms? No. 3 m/s = 3 mm/ms?
# 3 m/s = 3000 mm / 1000 ms = 3 mm/ms. Yes.
# Standard white matter velocity is 2-10 m/s.
GLOBAL_COUPLING = 0.05      # Scaling factor 'a'

# --- Epileptor Model Constants (Jirsa 2014) ---
# Population 1 (Fast Discharges)
I_EXT_1 = 3.1
TAU_0 = 2857.0          # Time constant separation

# Population 2 (Spike-Wave Events)
I_EXT_2 = 0.45

# Permittivity Coupling (Slow variable z)
R_TIMESCALE = 0.00035   # Very slow timescale of permittivity
X0_CRITICAL = -2.0      # Critical bifurcation point

# Clinical Thresholds
EPILEPTOGENICITY_THRESHOLD = 0.5  # Makhalova 2022
RESECTION_THRESHOLD = 0.4         # Indices >= 0.4 considered for resection
