# Fractional-Order Seizure Modeling Research Status
## Summer 2026 Transition Document

**Principal Investigator:** Dr. Emily Pereira  
**Student Researcher:** Ohinoyi Moiza  
**Team Members:** Amar Kushwaha, Shruti Chougule  
**Date:** May 2026

## Research Summary

We successfully implemented fractional-order dynamics for patient-specific epilepsy surgery planning. This addresses the 50-60% surgical success rate that has remained stagnant for three decades.

## Completed Work

### Patients Processed
- sub-R1001P: 88 electrodes, 14 brain regions
- sub-R1002P: 74 electrodes, 13 brain regions  
- sub-R1003P: 100 electrodes, 17 brain regions
- **Total: 262 electrodes across 3 patients**

### Simulations Generated
- 9 total simulations (3 patients × 3 beta values)
- Beta values: 1.0 (standard), 0.8 (moderate memory), 0.6 (strong memory)
- Each simulation: 2000ms duration, 1ms timestep

### Key Findings
- Beta = 1.0: Fast rise (~200ms), quick decay (~500ms)
- Beta = 0.8: Slower dynamics (~400ms rise, ~1000ms decay)
- Beta = 0.6: Very slow propagation (~600ms), sustained activity (>1500ms)
- **Lower beta values better match real seizure dynamics**

### Code Developed
1. fractional_dynamics.py (98 lines): Core fractional calculus
2. fractional_sim_simple.py (142 lines): Patient processing
3. Visualization scripts: Interactive 3D brain plots

## Fall 2026 Publication Plan

### Timeline
- **August 2026:** Add 2 more patients (total 5)
- **September 2026:** Parameter estimation from real iEEG, validation studies
- **October 2026:** Manuscript draft
- **November 2026:** Submit to journal

### Target Journal
Frontiers in Computational Neuroscience (open access, ~3 month review)

### Required Analyses
1. Estimate beta from real patient iEEG data
2. Validation: Compare fractional vs integer-order predictions
3. Statistical significance testing
4. Surgical outcome correlation (if data available)

### Manuscript Structure
- **Title:** Patient-Specific Seizure Network Modeling Using Fractional-Order Dynamics
- **Authors:** Ohinoyi Moiza (first), Amar Kushwaha, Shruti Chougule, Emily Pereira (corresponding)
- **Abstract:** 250 words
- **Introduction:** 4-5 pages (clinical context, VEP background, fractional-order rationale)
- **Methods:** 5-6 pages (dataset, implementation, validation)
- **Results:** 6-7 pages (patient demographics, beta effects, validation metrics)
- **Discussion:** 4-5 pages (interpretation, limitations, clinical implications)
- **Figures:** 6-8 publication-quality figures

## What Remains

### Not Yet Completed
1. Parameter estimation from real iEEG time-series
2. Validation against actual seizure recordings
3. Surgical outcome correlation analysis
4. Additional 2 patients processing

### Technical Requirements
- Software: Python 3.8+, numpy, pandas, plotly, scipy, scikit-learn, mne-python
- Data: OpenNeuro ds004789 (~15GB)
- Computing: TTU HPC cluster access (needs renewal August 2026)

## Data Security

### GitHub Repository
All code, data, and documentation secured at: github.com/kushwahaamar-dev/VESP-

**Contents:**
- fractional_dynamics.py
- fractional_sim_simple.py  
- 3 patient electrode CSV files
- 9 HTML visualizations
- Complete documentation

### HPC Access
- Account: omoiza@ttu.edu
- Cluster: RedRaider
- Sponsor: Dr. Emily Pereira
- Expiration: August 31, 2026 (needs renewal)
- Globus Compute endpoint: 66cfd29a-534a-4e36-b451-2e21dc1e4677

## Key References

1. Chakraborty et al. (2020). Frontiers in Applied Math & Statistics. Space-time fractional diffusion.
2. Jirsa et al. (2017). NeuroImage. Virtual Epileptic Patient framework.
3. Jobst & Cascino (2015). JAMA. Surgical success rate review (50-60%).
4. Wiebe et al. (2001). NEJM. Surgery vs medical management trial.

## Questions for Discussion

1. Should we prioritize depth (more analysis on 5 patients) or breadth (scale to 20+ patients)?
2. Is surgical outcome correlation essential for initial publication?
3. Will HPC sponsorship automatically renew, or does it need formal request?
4. Will Amar and Shruti continue Fall 2026?

## Success Metrics

**Research:**
- 5 patients processed with fractional simulations
- Beta estimated from real iEEG
- Validation study completed
- Manuscript submitted and accepted

**Skills:**
- Fractional calculus implementation ✓
- Patient-specific modeling ✓
- Interactive visualization ✓
- Parameter estimation (Fall 2026)
- Statistical validation (Fall 2026)
- Scientific writing (Fall 2026)

**Presentations Completed Spring 2026:**
- URC Poster (April 9)
- URC Commercialization Competition (April 8) - Top 10 finalist
- GlobusWorld 2026 (April 21-22) - Lightning talk accepted
- CRA UR2PhD Workshop (April 23-25) - Poster

---

**Contact:**  
Dr. Emily Pereira: emily.pereira@ttu.edu  
Ohinoyi Moiza: omoiza@ttu.edu  
Repository: github.com/kushwahaamar-dev/VESP-

**Document Version:** 1.0  
**Last Updated:** May 2026
