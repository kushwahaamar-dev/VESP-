# Virtual Epileptic Patient (VEP) Pipeline

> **Production-grade implementation of personalized brain network modeling for epilepsy surgery planning.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![TVB](https://img.shields.io/badge/platform-The%20Virtual%20Brain-orange.svg)](https://www.thevirtualbrain.org/)

---

## 📖 Overview

This repository implements the **Virtual Epileptic Patient (VEP)** framework, a computational pipeline that simulates seizure propagation in individual patient brains. The VEP approach enables neurosurgeons to virtually test surgical interventions before performing irreversible resections.

### Scientific Foundation

This implementation is based on the following peer-reviewed publications:

| Paper | Authors | Journal | Key Contribution |
|-------|---------|---------|------------------|
| **The Virtual Epileptic Patient** | Jirsa et al. | *Lancet Neurology* (2017) | Core VEP framework and clinical validation |
| **Epileptor: A Phenomenological Model** | Jirsa et al. | *Brain* (2014) | 6D neural mass model of seizure dynamics |
| **Bayesian Inference of Epileptogenic Networks** | Hashemi et al. | *PLOS Computational Biology* (2020) | Probabilistic parameter estimation |
| **The Virtual Brain Platform** | Sanz-Leon et al. | *Frontiers in Neuroinformatics* (2013) | TVB simulation infrastructure |

---

## 🧠 The Epileptor Model

### Mathematical Formulation

The **Epileptor** is a phenomenological neural mass model that captures the essential dynamics of seizure generation and termination. Each brain region is modeled as a 6-dimensional dynamical system:

```
State Variables: [x₁, y₁, z, x₂, y₂, g]

Population 1 (Fast oscillations - seizure spikes):
    dx₁/dt = y₁ - f₁(x₁) - z + I_ext1 + K·Σⱼ wᵢⱼ(x₁ⱼ(t-τᵢⱼ) - x₁ᵢ)
    dy₁/dt = 1 - 5x₁² - y₁

Slow permittivity variable (seizure termination):
    dz/dt = r · (4(x₁ - x₀) - z)

Population 2 (Spike-wave complexes):
    dx₂/dt = -y₂ + x₂ - x₂³ + I_ext2 + 0.002g - 0.3(z - 3.5)
    dy₂/dt = (-y₂ + x₂⁴) / τ₀

Low-pass filter:
    dg/dt = -0.01(g - 0.1x₁)
```

### Key Parameters

| Parameter | Symbol | Default | Description |
|-----------|--------|---------|-------------|
| Excitability | x₀ | -2.2 (healthy), -1.6 (EZ) | Controls seizure threshold |
| Time constant | r | 0.00035 | Slow variable dynamics |
| External input 1 | I_ext1 | 3.1 | Baseline excitation (Pop 1) |
| External input 2 | I_ext2 | 0.45 | Baseline excitation (Pop 2) |
| Tau | τ₀ | 2857 | Spike-wave time scale |
| Coupling strength | K | 0.1 | Global network coupling |
| Conduction velocity | v | 3.0 mm/ms | Signal propagation speed |

### Bifurcation Dynamics

The Epileptor exhibits a **saddle-node bifurcation** controlled by the excitability parameter x₀:

- **x₀ < -2.0** → Stable fixed point (interictal/healthy state)
- **x₀ > -2.0** → Limit cycle (ictal/seizure state)

## 📊 Data Sources

### Extended TVB Dataset (Local)

The complete brain data is located locally at:

```
/Users/amar/Codes/brain-modeling-research/tvb_data/tvb_data/
```

This is an **extended version** of The Virtual Brain's standard dataset, containing multiple species, atlas resolutions, and sensor configurations.

---

### 📁 Directory Structure (Detailed)

#### 1. `connectivity/` — Structural Connectivity Matrices

Each `.zip` file contains the brain's wiring diagram derived from diffusion MRI tractography:

| File | Regions | Size | Contents |
|------|---------|------|----------|
| `connectivity_66.zip` | 66 | 30 KB | Desikan-Killiany atlas |
| `connectivity_68.zip` | 68 | 14 KB | FreeSurfer default |
| `connectivity_76.zip` | 76 | 44 KB | **AAL atlas (default)** |
| `connectivity_80.zip` | 80 | 37 KB | Extended AAL |
| `connectivity_96.zip` | 96 | 53 KB | Fine-grained parcellation |
| `connectivity_192.zip` | 192 | 61 KB | High-resolution |
| `connectivity_998.zip` | 998 | 567 KB | Ultra high-resolution |

**Inside each ZIP:**
```
weights.txt        # N×N matrix of connection strengths (normalized 0-1)
tract_lengths.txt  # N×N matrix of fiber distances (mm)
centres.txt        # N×4 table: [label, x, y, z] MNI coordinates
```

---

#### 2. `surfaceData/` — Cortical Mesh Surfaces

3D triangulated meshes of the cortical surface for visualization:

| File | Vertices | Triangles | Size | Description |
|------|----------|-----------|------|-------------|
| `cortex_16384.zip` | 16,384 | 32,760 | 638 KB | Standard resolution |
| `cortex_80k.zip` | 81,924 | 163,840 | 2.7 MB | High resolution |
| `cortex_2x120k.zip` | 240,000 | ~480,000 | 13.5 MB | Ultra high-res (both hemispheres) |

**Inside each ZIP:**
```
vertices.txt    # V×3 matrix of [x, y, z] coordinates (mm)
triangles.txt   # T×3 matrix of vertex indices forming each triangle
normals.txt     # V×3 matrix of surface normal vectors
```

---

#### 3. `regionMapping/` — Vertex-to-Region Assignments

Maps each cortical mesh vertex to its corresponding brain region:

| File | Mesh | Atlas | Description |
|------|------|-------|-------------|
| `regionMapping_16k_76.txt` | 16k | 76 | Standard mapping |
| `regionMapping_16k_192.txt` | 16k | 192 | High-res atlas on standard mesh |
| `regionMapping_80k_80.txt` | 80k | 80 | High-res mesh mapping |

**Format:** Single column of integers (16,384 or 81,924 lines), one per vertex, indicating region index (0 to N-1).

---

#### 4. `macaque_v3/` — 🐵 Macaque Primate Brain

Non-human primate data for translational neuroscience:

| File | Size | Description |
|------|------|-------------|
| `connectivity_84.zip` | 355 KB | 84-region macaque structural connectivity |
| `surface_147k.zip` | 21.7 MB | High-resolution macaque cortex mesh |
| `regionMapping_147k_84.txt` | 3.7 MB | Vertex-to-region mapping |
| `volumeMap_inF99.nii.gz` | 113 KB | Volumetric parcellation (NIfTI) |

---

#### 5. `mouse/` — 🐭 Rodent Brain Data

Mouse brain atlases for small animal modeling:

```
mouse/
├── allen_2mm/                    # Allen Institute Mouse Brain Atlas
│   ├── Connectivity.h5           # 226 KB - HDF5 connectivity matrix
│   ├── ConnectivityAllen2mm.zip  # 368 KB - Alternative format
│   ├── RegionVolumeMapping.h5    # 14.5 MB - 3D volume labels
│   ├── StructuralMRI.h5          # 9.7 MB - Reference MRI
│   └── Volume.h5                 # 33 KB - Volume definition
│
└── calabrese/                    # Calabrese Mouse Atlas
    ├── Connectivity_Calabrese.zip # 783 KB - Connectivity
    ├── Structural_MRI.nii        # 9.6 MB - NIfTI MRI
    └── Vol_Calabrese.nii         # 9.6 MB - Volume
```

---

#### 6. `sensors/` — Electrode Configurations

Sensor positions for EEG, MEG, and SEEG recordings:

| File | Sensors | Modality | Description |
|------|---------|----------|-------------|
| `eeg_63.txt` | 63 | EEG | Standard 10-20 system |
| `eeg_brainstorm_65.txt` | 65 | EEG | Brainstorm format |
| `meg_248.txt` | 248 | MEG | 4D Neuroimaging system |
| `meg_brainstorm_276.txt` | 276 | MEG | CTF MEG system |
| `seeg_588.txt` | 588 | SEEG | Stereo-EEG depth electrodes |
| `seeg_brainstorm_960.txt` | 960 | SEEG | High-density SEEG |

**Format:** Each line contains `[label, x, y, z]` in MNI space (mm).

---

#### 7. `projectionMatrix/` — Lead Field Matrices

Pre-computed forward models mapping brain sources to sensor measurements:

| File | Size | Description |
|------|------|-------------|
| `projection_eeg_62_surface_16k.mat` | 7.8 MB | EEG 62-ch → 16k surface |
| `projection_eeg_65_surface_16k.npy` | 8.5 MB | EEG 65-ch → 16k surface |
| `projection_meg_276_surface_16k.npy` | 36.2 MB | MEG 276-ch → 16k surface |
| `projection_seeg_588_surface_16k.npy` | 77.1 MB | SEEG 588-ch → 16k surface |

---

#### 8. `dti_pipeline_toronto/` — Raw DTI Processing

Original diffusion tensor imaging data and processing pipeline:

| File | Size | Description |
|------|------|-------------|
| `InputDTI_Toronto.zip` | 5.6 MB | Raw DTI tensors (Toronto dataset) |
| `InputDTI_AnaSolodkin.zip` | 43.5 MB | Extended DTI dataset |
| `output_ConnectionCapacityMatrix.csv` | 50 KB | Derived connectivity weights |
| `output_ConnectionDistanceMatrix.csv` | 50 KB | Derived tract lengths |

---

### Available Brain Atlases Summary

| Regions | Species | Atlas Name | Mesh | Use Case |
|---------|---------|------------|------|----------|
| 66 | Human | Desikan-Killiany | 16k | Fast prototyping |
| 68 | Human | FreeSurfer default | 16k | Standard analysis |
| **76** | Human | AAL (Automated Anatomical Labeling) | 16k | **Clinical VEP (default)** |
| 80 | Human | Extended AAL | 80k | High-resolution |
| 96 | Human | Fine-grained | 16k | Detailed regional analysis |
| 192 | Human | High-resolution | 16k | Research applications |
| 998 | Human | Schaefer/HCP | 80k | Maximum spatial detail |
| 84 | Macaque | CoCoMac/F99 | 147k | Translational primate research |

---

### Data Origin & References

The connectivity data is derived from:

1. **Diffusion MRI Tractography**
   - Probabilistic streamline tracking through white matter
   - Connection weights = number of streamlines between regions
   - Source: Human Connectome Project (HCP), Toronto DTI pipeline

2. **Automated Anatomical Labeling (AAL) Atlas**
   - Tzourio-Mazoyer et al. (2002) *NeuroImage*
   - 76 cortical and subcortical regions
   - Standard parcellation for clinical epilepsy studies

3. **MNI Coordinate System**
   - Montreal Neurological Institute standard space
   - Region centers derived from atlas centroids

4. **Allen Brain Atlas** (Mouse data)
   - Allen Institute for Brain Science
   - 2mm resolution mouse connectivity

---

## 🔧 Implementation Details

### Architecture

```
brain-modeling-research/
├── tvb_data/               # 📂 Extended brain data (local)
│   └── tvb_data/           # Connectivity, meshes, sensors
│
├── vep/                    # 🧠 Core simulation package
│   ├── config.py           # Simulation parameters
│   ├── anatomy.py          # Multi-atlas data loading
│   ├── epileptor.py        # Neural mass model (JIT)
│   ├── simulator.py        # Time integration with delays
│   └── visualizer.py       # 3D brain + time series
│
├── pipeline.py             # 🚀 CLI entry point
├── checkpoint.npz          # Saved simulation state
└── vep_report.html         # Generated visualization
```

### Performance Optimizations

1. **Numba JIT Compilation**: Epileptor equations compiled to LLVM machine code (~1000x speedup)
2. **Ring Buffer Delays**: Circular buffer for transmission delays (O(1) memory access)
3. **Downsampled Storage**: Save every 1ms to reduce memory (dt=0.05ms runtime)

### Transmission Delays

```
τᵢⱼ = Lᵢⱼ / v

where:
    Lᵢⱼ = tract length between regions (mm)
    v = conduction velocity (3.0 mm/ms)
    τᵢⱼ = delay (typically 0-50 ms)
```

---

## 🚀 Usage

### Installation

```bash
# Create virtual environment
python -m venv tvb_env
source tvb_env/bin/activate

# Install dependencies
pip install numpy numba plotly tqdm tvb-library
```

### Running the Pipeline

```bash
# List available atlases
python pipeline.py --list-atlases

# Standard 76-region human simulation
python pipeline.py --duration 4000 --output vep_report.html

# High-resolution 192-region simulation
python pipeline.py --atlas 192 --cortex 80k --duration 4000

# Ultra high-resolution 998-region (research)
python pipeline.py --atlas 998 --cortex 80k --duration 2000

# Macaque brain simulation
python pipeline.py --atlas 84 --duration 4000 --output macaque_report.html

# Fast prototyping (66 regions, 500ms)
python pipeline.py --atlas 66 --duration 500 --output quick_test.html

# Resume from checkpoint
python pipeline.py --checkpoint checkpoint.npz --output from_checkpoint.html
```

### CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--atlas` | 76 | Brain atlas (66, 68, 76, 80, 96, 192, 998, 84) |
| `--cortex` | 16k | Cortex resolution (16k, 80k, 120k) |
| `--duration` | 4000 | Simulation time (ms) |
| `--output` | vep_report.html | Output HTML path |
| `--checkpoint` | None | Load saved simulation |
| `--save-checkpoint` | checkpoint.npz | Save simulation |
| `--list-atlases` | - | Show available atlases |

---

## 📈 Visualization

The pipeline generates an interactive HTML report with:

### 1. 3D Brain Model
- Translucent cortical mesh (16k-80k vertices)
- Color-coded region nodes:
  - 🔴 **Red**: Epileptogenic Zone (EZ)
  - 🟠 **Orange**: Propagation Zone (PZ)
  - 🔵 **Blue**: Healthy regions
- Hover tooltips with region details
- Play/Pause animation

### 2. Time Series Plot
- Neural activity (x₁) for EZ and top propagated regions
- Proper axis labels and non-overlapping legend
- Seizure onset markers

---

## 📚 References

### Primary Citations

```bibtex
@article{jirsa2017virtual,
  title={The Virtual Epileptic Patient: Individualized whole-brain models of epilepsy spread},
  author={Jirsa, Viktor K and Proix, Timothée and Perdikis, Dionysios and others},
  journal={Neuroimage},
  volume={145},
  pages={377--388},
  year={2017},
  publisher={Elsevier}
}

@article{jirsa2014epileptor,
  title={On the nature of seizure dynamics},
  author={Jirsa, Viktor K and Stacey, William C and Quilichini, Pascale P and others},
  journal={Brain},
  volume={137},
  number={8},
  pages={2210--2230},
  year={2014},
  publisher={Oxford University Press}
}

@article{sanzleon2013virtual,
  title={The Virtual Brain: a simulator of primate brain network dynamics},
  author={Sanz-Leon, Paula and Knock, Stuart A and Woodman, Marmaduke M and others},
  journal={Frontiers in Neuroinformatics},
  volume={7},
  pages={10},
  year={2013},
  publisher={Frontiers}
}
```

### Additional Resources

- [The Virtual Brain Official Website](https://www.thevirtualbrain.org/)
- [TVB Documentation](https://docs.thevirtualbrain.org/)
- [EBRAINS Platform](https://ebrains.eu/)
- [Allen Brain Atlas](https://atlas.brain-map.org/) (Mouse data)

---

## 📄 License

This project is for research and educational purposes. TVB is distributed under GPL-3.0.

---

## 🤝 Acknowledgments

- **Institut de Neurosciences des Systèmes (INS)**, Marseille, France
- **The Virtual Brain Consortium**
- **Human Brain Project / EBRAINS**
- **Allen Institute for Brain Science** (Mouse atlas data)
