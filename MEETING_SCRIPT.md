# Research Meeting Script - March 27, 2026
## Fractional-Order Seizure Dynamics Implementation

---

## OPENING (30 seconds)

"Good morning Dr. Pereira. Since our last meeting two weeks ago, I've implemented the fractional-order dynamics framework you mentioned. I have results to show you from 3 patients with different memory parameters."

---

## WHAT I DID (2 minutes)

### 1. Implemented Fractional Calculus
"I implemented the Grünwald-Letnikov fractional derivative to add memory effects to our seizure simulations."

**Show: fractional_dynamics.py**

"This replaces standard derivatives with fractional-order ones:
- Standard: dx/dt = f(x) - no memory
- Fractional: D^beta x = f(x) - includes memory from past states"

### 2. Processed 3 Patients
"I ran simulations on 3 patients from the UPenn dataset:
- sub-R1001P: 88 electrodes
- sub-R1002P: 74 electrodes  
- sub-R1003P: 100 electrodes
- Total: 262 electrodes mapped to brain regions"

### 3. Tested Different Beta Values
"I compared three beta values:
- Beta = 1.0: Standard dynamics (no memory)
- Beta = 0.8: Moderate memory effects
- Beta = 0.6: Strong memory effects"

**Show: sub-R1001P_SIMPLE_COMPARISON.html**

---

## KEY FINDINGS (1 minute)

"The beta parameter dramatically affects seizure propagation:"

**Point to the visualization:**

**Beta = 1.0 (solid lines):**
- Fast rise to peak activity
- Quick decay back to baseline
- Like standard textbook models

**Beta = 0.8 (dashed lines):**
- Slower dynamics
- Activity persists longer
- More gradual return to baseline

**Beta = 0.6 (dotted lines):**
- Very slow propagation
- Highly sustained activity
- Strong memory effects
- May better match real seizure recordings

---

## WHY THIS MATTERS (1 minute)

"This connects to your suggestion about anomalous diffusion from the Chakraborty paper."

**The big question:**
"Do real patients have different beta values? If so, patients with low beta (strong memory) might need different surgical strategies than patients with beta close to 1.0 (standard dynamics)."

**Clinical relevance:**
"If we can estimate beta from pre-surgical recordings, it could help predict:
- Which patients need more aggressive resection
- Which might respond to different interventions
- Why some surgeries succeed and others fail"

---

## TECHNICAL DETAILS (if she asks)

### Implementation:
- Used Grünwald-Letnikov approximation (simplest to implement)
- Memory buffer stores last 100 time steps
- Distance-based connectivity between electrodes
- Runtime: 1-2 seconds per simulation

### Mathematical Framework:
```
Fractional derivative: D^β x(t) ≈ Σ w_j * x(t - j*dt)
Weights: w_j = w_{j-1} * (1 - (1 + β) / j)
```

### Code Structure:
- fractional_dynamics.py: Core fractional calculus (98 lines)
- fractional_sim_simple.py: Patient pipeline (142 lines)
- Generated 9 visualizations + comparison figures

---

## NEXT STEPS (2 minutes)

### Immediate (This Week):
**1. Parameter Estimation**
"Extract real seizure time-series from iEEG recordings and estimate beta for each patient using statistical moments approach from the Chakraborty paper."

**Question for you:**
"Should I use the absolute moments method they describe, or is there a simpler approach you'd recommend?"

**2. Validation Metrics**
"Define how to measure prediction accuracy - maybe mean squared error between predicted and actual propagation?"

### Medium-Term (Next 2-3 Weeks):
**3. Scale to 20+ Patients**
"Run on HPC using Globus Compute to process multiple patients in parallel. Statistical validation requires larger sample size."

**4. Compare Integer vs Fractional**
"Quantitative comparison: Does fractional-order model predict seizure spread better than integer-order? Need held-out test data."

### Long-Term:
**5. Space-Time Fractional**
"Extend to full space-time fractional diffusion with both alpha (space) and beta (time) parameters."

**6. Surgical Outcome Correlation**
"Test if beta values correlate with surgical success rates."

---

## QUESTIONS FOR DR. PEREIRA

1. **Parameter estimation:** "What's the best way to extract beta from real iEEG time-series? Statistical moments or another method?"

2. **Validation approach:** "How should we validate predictions? Split seizure in half and predict second half?"

3. **Publication timeline:** "When should we start drafting? After 20 patients or can we publish proof-of-concept with current 3?"

4. **Collaboration:** "Should we reach out to clinicians at UMC for surgical outcome data?"

5. **Next meeting:** "Would you like weekly updates as I scale this up, or wait until I have 20 patients done?"

---

## CLOSING

"In summary: I've successfully implemented fractional-order dynamics for patient-specific seizure modeling. The beta parameter clearly affects propagation patterns. Next steps are parameter estimation from real data and validation across 20+ patients on HPC."

"I have all the code documented and the visualizations ready to share. What questions do you have?"

---

## FILES TO HAVE READY

1. ✅ sub-R1001P_SIMPLE_COMPARISON.html (SHOW FIRST)
2. ✅ sub-R1001P_FRACTIONAL_beta0.6.html (if she wants detail)
3. ✅ FRACTIONAL_IMPLEMENTATION_REPORT.md (technical details)
4. ✅ fractional_dynamics.py (show code if asked)
5. ✅ fractional_sim_simple.py (show code if asked)

---

## IF SHE ASKS TOUGH QUESTIONS

**Q: "How do you know this is better than standard models?"**
A: "I don't yet - that's the validation step. These simulations show fractional-order CAN capture different dynamics. Next is testing if they SHOULD be used by comparing predictions to real data."

**Q: "What if beta doesn't matter clinically?"**
A: "That's a valid outcome too. We'd learn that memory effects don't significantly impact seizure propagation, which would guide us away from this complexity."

**Q: "Why not use Caputo derivative like the paper?"**
A: "Grünwald-Letnikov is computationally simpler and gives same results for our discrete time steps. Can switch to Caputo if needed for theoretical reasons."

**Q: "How long to finish validation?"**
A: "With HPC: 2-3 weeks for 20 patients. Each patient takes ~2 seconds to simulate, but parameter estimation from real iEEG will take longer."

---

## CONFIDENCE BOOSTERS

✅ "I have working code that runs successfully"
✅ "I understand the math - can explain GL weights"
✅ "I have concrete next steps planned"
✅ "I can answer questions about implementation"
✅ "I have 9 visualizations showing results"
✅ "I know what I don't know yet (parameter estimation, validation)"

---

**YOU'VE GOT THIS!** 💪
