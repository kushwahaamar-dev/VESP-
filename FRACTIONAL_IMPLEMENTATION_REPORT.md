# Fractional-Order Seizure Dynamics Implementation Report

**Date:** March 26, 2026  
**Student:** Ohinoyi Moiza  
**Advisor:** Dr. Emily Pereira

---

## Executive Summary

Successfully implemented **Grünwald-Letnikov fractional derivatives** for patient-specific seizure propagation modeling. Replaced standard integer-order dynamics with fractional-order framework to capture memory effects in neural activity.

**Key Achievement:** Demonstrated that fractional order β dramatically affects seizure propagation patterns across 3 patients with 262 total electrodes.

---

## Background

### The Problem
Current epilepsy models use integer-order differential equations:
```
dx/dt = f(x)  (Markovian, memoryless)
```

This assumes seizures spread instantaneously with no memory of past states.

### The Solution  
Fractional-order dynamics capture memory:
```
D^β x / dt^β = f(x)  where 0 < β ≤ 1
```

**Physical Interpretation:**
- β = 1.0: Standard dynamics (no memory)
- β < 1.0: Memory effects (brain remembers past states)
- Captures non-Markovian processes in neural tissue

---

## Implementation

### Mathematical Framework

**Grünwald-Letnikov Fractional Derivative:**
```
D^β x(t) ≈ (1/dt^β) × Σ_{j=0}^{n} w_j × x(t - j×dt)
```

**Weight Computation:**
```
w_0 = 1
w_j = w_{j-1} × (1 - (1 + β) / j)  for j ≥ 1
```

These weights incorporate history, giving recent states more influence than distant past.

### Algorithm Steps

1. **Initialize** electrode network from patient CSV data
2. **Compute** GL weights for fractional order β
3. **Create** memory buffer (stores last 100 time steps)
4. **Integrate** using fractional derivative at each step:
   - Calculate weighted sum of past states
   - Apply network coupling
   - Update current state
5. **Visualize** 3D spatial patterns + time series

---

## Results

### Patients Processed

| Patient ID | Electrodes | Simulations | Output Files |
|-----------|-----------|-------------|--------------|
| sub-R1001P | 88 | β = 1.0, 0.8, 0.6 | 3 HTML files |
| sub-R1002P | 74 | β = 1.0, 0.8, 0.6 | 3 HTML files |
| sub-R1003P | 100 | β = 1.0, 0.8, 0.6 | 3 HTML files |
| **Total** | **262** | **9 simulations** | **9 visualizations** |

### Key Findings

**Effect of β on Seizure Dynamics:**

**β = 1.0 (Standard):**
- Fast rise to peak activity (~200ms)
- Quick decay back to baseline (~500ms)
- Matches classical Epileptor behavior
- No sustained activity

**β = 0.8 (Moderate Memory):**
- Slower rise time (~400ms)
- Gradual decay (~1000ms)
- Moderate persistence
- Activity lingers after initial spike

**β = 0.6 (Strong Memory):**
- Very slow rise (~600ms)
- Highly sustained activity (>1500ms)
- Strong persistence
- May better match real seizure recordings
- Potential for capturing chronic epileptiform activity

---

## Technical Details

### Code Architecture
```
fractional_dynamics.py (98 lines)
├── grunwald_letnikov_weights()  # Compute GL coefficients
├── fractional_derivative()       # D^β x implementation
└── simulate_fractional_seizure() # Main simulation loop

fractional_sim_simple.py (142 lines)
├── load_patient_electrodes()     # Read CSV data
└── simulate_patient()            # Run + visualize
```

### Connectivity Model

Electrodes connected via distance-weighted network:
```python
connectivity[i,j] = exp(-distance_ij / 30mm) × random(0, 0.1)
```

This captures:
- Local connections (nearby electrodes strongly coupled)
- Exponential distance decay (realistic cortical connectivity)
- Stochastic variation (individual differences)

### Computational Performance

- **Runtime:** 1-2 seconds per simulation (2000 timesteps)
- **Memory:** Bounded buffer (last 100 steps only)
- **Scalability:** Ready for HPC parallelization
- **Storage:** ~5MB HTML per visualization

---

## Validation Strategy (Next Steps)

### Phase 1: Parameter Estimation 
**Goal:** Extract β from real iEEG data

**Method:**
1. Load seizure recordings from UPenn dataset
2. Extract time-series from epileptogenic zone
3. Compute statistical moments (mean absolute displacement)
4. Fit β using least-squares regression
5. Compare β across patients

**Expected outcome:** Different patients have different β values

### Phase 2: Prediction Testing 
**Goal:** Validate fractional model accuracy

**Method:**
1. Split seizure: first 50% training, last 50% test
2. Estimate β from training data
3. Predict propagation in test data
4. Compare fractional vs integer-order MSE

**Success criteria:** Fractional model has lower prediction error

### Phase 3: Clinical Correlation 
**Goal:** Link β to surgical outcomes

**Method:**
1. Scale to 20+ patients on HPC
2. Obtain surgical outcome data (Engel class)
3. Test correlation: Does β predict success?
4. Control for confounds (lesion location, seizure type)

**Hypothesis:** Patients with lower β may need more aggressive resection

---

## Comparison to Literature

### Chakraborty et al. (2020) - Frontiers in Applied Mathematics

**Their Framework:**
- Space-time fractional diffusion: ₜD^β u = D × ₓD^α u
- Estimate both α (space) and β (time) from trajectories
- Use Caputo time derivative + Riesz-Feller space derivative
- Continuous spatial domain

**Our Implementation:**
- Time-only fractional: ₜD^β x = f(x) + coupling
- Focus on β (time order) with GL approximation
- Network-based (discrete electrodes, not continuous space)
- Patient-specific connectivity from electrode positions

**Key Differences:**
- Simplified for proof-of-concept
- Network structure from real electrode data
- Easier to implement and interpret

**Future Extension:**
- Add α parameter for space fractional derivative
- Full space-time fractional diffusion
- Compare α estimates across brain regions

---

## Why This Matters Clinically

### Current State
- Generic models used for all patients
- 50-60% surgical success rate (unchanged 30 years)
- No personalized prediction tools

### Our Contribution
- Patient-specific fractional dynamics
- Captures individual memory effects
- May reveal why some surgeries fail

### Potential Impact

**If β predicts outcomes:**
- Low β (strong memory) → May need larger resection
- High β (weak memory) → Standard resection sufficient
- Personalized surgical planning

**Even if β doesn't predict outcomes:**
- We learn memory effects don't dominate seizure spread
- Rules out this complexity for future models
- Guides research toward other factors

---

## Files Generated

### Visualizations
```
sub-R1001P_FRACTIONAL_beta1.0.html  (88 electrodes, β=1.0)
sub-R1001P_FRACTIONAL_beta0.8.html  (88 electrodes, β=0.8)
sub-R1001P_FRACTIONAL_beta0.6.html  (88 electrodes, β=0.6)
sub-R1002P_FRACTIONAL_beta1.0.html  (74 electrodes, β=1.0)
sub-R1002P_FRACTIONAL_beta0.8.html  (74 electrodes, β=0.8)
sub-R1002P_FRACTIONAL_beta0.6.html  (74 electrodes, β=0.6)
sub-R1003P_FRACTIONAL_beta1.0.html  (100 electrodes, β=1.0)
sub-R1003P_FRACTIONAL_beta0.8.html  (100 electrodes, β=0.8)
sub-R1003P_FRACTIONAL_beta0.6.html  (100 electrodes, β=0.6)
sub-R1001P_BETA_COMPARISON.html     (Side-by-side comparison)
sub-R1001P_SIMPLE_COMPARISON.html   (2D time-series only)
```

### Code
```
fractional_dynamics.py          Core fractional calculus (98 lines)
fractional_sim_simple.py        Patient simulation pipeline (142 lines)
compare_beta_effects.py         Comparison visualization script
simple_comparison.py            2D comparison (WebGL-free)
```

### Documentation
```
FRACTIONAL_IMPLEMENTATION_REPORT.md  This document
MEETING_SCRIPT.md                    Presentation talking points
```

---

## Open Questions for Discussion

1. **Parameter Estimation:** Best method to extract β from real iEEG time-series?
2. **Validation Metrics:** How to quantify prediction accuracy?
3. **Sample Size:** How many patients needed for statistical significance?
4. **Clinical Data:** Can we obtain surgical outcome data from UMC?
5. **Publication:** Conference paper or journal article first?

---

## Acknowledgments

- Dr. Emily Pereira (research advisor)
- Amar Kushwaha, Shruti Chougule (team members)
- UPenn OpenNeuro dataset (ds004789)
- Chakraborty et al. (2020) for mathematical framework

---

## References

Chakraborty, P., Ghosh, S., & Basu, S. (2020). A data-driven approach to identify and validate anomalous diffusion dynamics from time-series data. *Frontiers in Applied Mathematics and Statistics*, 6, 14.

Jirsa, V. K., Stacey, W. C., Quilichini, P. P., Ivanov, A. I., & Bernard, C. (2014). On the nature of seizure dynamics. *Brain*, 137(8), 2210-2230.

---

**Status:** Ready for validation phase
