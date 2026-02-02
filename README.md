# VESP: Virtual Epileptic Patient Surgical Pipeline

**High-Fidelity Computational Modeling for Epilepsy Surgery Planning**

[![License](https://img.shields.io/badge/License-GPL%203.0-blue.svg)](https://opensource.org/licenses/GPL-3.0)
[![Python](https://img.shields.io/badge/Python-3.10%2B-green.svg)](https://www.python.org/)
[![Status](https://img.shields.io/badge/Status-Clinical%20R%26D-red.svg)](https://github.com/kushwahaamar-dev/VESP-)

## Overview

VESP is a state-of-the-art computational framework designed to assist neurosurgeons in identifying the Epileptogenic Zone (EZ) in drug-resistant epilepsy patients. 

It implements the Virtual Epileptic Patient (VEP) workflow (Jirsa et al., 2016; Makhalova et al., 2022) with a focus on high-performance computing and real-time clinical visualization.

## 🧠 Key Features

- **"Glass Brain" Visualization**: A lightweight, high-performance 3D dashboard (60 FPS) that combines anatomical context with dynamic network activity.
- **HPCI Physics Engine**: Solves 200,000+ coupled differential equations (Epileptor Model) using **Numba JIT compilation** for microsecond-scale execution.
- **Spatiotemporal Delays**: Full implementation of white-matter transmission delays using a Ring-Buffer memory architecture ($v = 3.0 m/s$).
- **Clinical Analytics**: Automatic computation of Seizure Onset Times and Recruitment Latencies for surgical decision support.

## ⚙️ Physical Model Configuration

The physics engine is governed by rigorous biophysical constants defined in `vep_core/config.py`. These parameters ensure the model adheres to realistic brain dynamics (Jirsa et al., 2014).

| Parameter | Value | Description |
|-----------|-------|-------------|
| `conduction_velocity` | 3.0 m/s | Speed of signal transmission along white matter tracts |
| `global_coupling` | 0.05 | Scaling factor $G$ for long-range network inputs |
| `x0_critical` | -2.0 | Bifurcation parameter threshold for seizure onset |
| `r` | 0.00035 | Slow permittivity variable timescale ($1/\tau_0$) |

Simulation Time Integration:
- **Heun Stochastic Integration** with $dt = 0.05$ ms
- **Ring-Buffer Delay Compensation** for transmission delays

## 🚀 Quick Start

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/kushwahaamar-dev/VESP-.git
cd VESP-

# 2. Create and activate a virtual environment
python -m venv vep_env
source vep_env/bin/activate

# 3. Install dependencies
pip install numpy scipy numba plotly tqdm tvb-library
```

### Running the Pipeline

The full pipeline (Data -> Inference -> Simulation -> Report) can be executed with a single command:

```bash
python main_pipeline.py
```

### Advanced Usage

The pipeline supports several command-line arguments for research workflows:

| Argument | Default | Description |
|----------|---------|-------------|
| `--patient` | `PAT001` | Patient ID for metadata/logging |
| `--duration` | `4000` | Simulation duration in milliseconds |
| `--resume` | `False` | Resume simulation from last checkpoint |
| `--checkpoint` | `simulation_checkpoint.npz` | Path to save/load state |

Example running a longer simulation for a specific patient:

```bash
python main_pipeline.py --patient PAT002 --duration 8000 --output report_pat002.html
```

### Running Tests

To verify the integrity of the physics engine and pipeline components:

```bash
pytest tests/
```

### Viewing the Surgical Report

The pipeline generates an interactive HTML dashboard:

```bash
open VEP_Clinical_Report.html
```

## 🏗️ Architecture (`vep_core`)

The codebase is structured as a modular R&D package:

```
vep_core/
├── config.py             # Rigorous physics constants (Jirsa 2014)
├── models/
│   └── epileptor.py      # JIT-compiled 6D Epileptor Kernels (Numba)
│                         # Implements the phenomenological model of seizure genesis
├── data/
│   └── loader.py         # Robust Data Ingestion (TVB Connectivity)
│                         # Handles MRI, fMRI, and SEEG multimodal datasets
├── inference/
│   └── inversion.py      # Bayesian Parameter Estimation Logic
│                         # Uses Hamiltonian Monte Carlo / Variational Inference
├── simulation/
│   └── forward.py        # Ring-Buffer Time Integration Engine
│                         # Handles spatiotemporal delays efficiently
└── viz/
    └── report.py         # "Glass Brain" Visualization Engine
                          # Generates interactive HTML dashboards using Plotly
```

## 📚 References

1. **Jirsa et al. (2017)**. The Virtual Epileptic Patient: Individualized whole-brain models of epilepsy spread. *NeuroImage*.
2. **Makhalova et al. (2022)**. Virtual epileptic patient brain modeling. *Epilepsia*.
3. **Proix et al. (2017)**. Permittivity coupling across brain regions determines seizure recruitment in partial epilepsy. *Journal of Neuroscience*.

## License

Copyright © 2026 Amar Kushwaha.
Licensed under the GPL-3.0 License.
