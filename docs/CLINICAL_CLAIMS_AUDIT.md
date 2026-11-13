# Clinical Claims & Biomedical Interpretation Audit

**Report Date:** 2026-11-13  
**Auditor:** Biomedical Signal Processing & Clinical Safety Specialist  

---

## 1. Dissection of Clinical Claims

| System Claim | Scientific / Physiological Basis | Clinical Reality & Safety Boundary |
|---|---|---|
| **Apnea Event Forecasting (5s Lead Time)** | Sympathetic tone surges prior to airway obstruction, altering arterial compliance. | **Plausible research hypothesis.** However, a 5-second lead time is very short for therapeutic positional adjustment. It is primarily applicable to auto-titrating CPAP pressure pre-empting. |
| **Early Autonomic Perturbations** | Mayer wave blood pressure oscillations (~0.1 Hz) appear as low-frequency pulse amplitude modulation. | **Scientifically sound.** Extensively documented in cardiovascular literature. |
| **HRV Spectral LF/HF Shift** | Sympathetic activation increases Low Frequency (0.04-0.15 Hz) power while vagal breathing modulation (HF) diminishes. | **Valid physiological observation.** Reflects acute sympathovagal imbalance. |
| **Confirmed Medical Diagnosis** | **UNSUPPORTED.** | A 1D-CNN analyzing a 30s finger-probe PPG window cannot replace 16-channel overnight Polysomnography (EEG, EOG, EMG, ECG, nasal cannula pressure, thorax/abdominal belts, SpO₂). |

---

## 2. Regulatory & Software Language Adjustments

To ensure compliance with clinical decision-support safety standards:
1. **Removed all claims of "confirmed medical diagnosis"** from UI and PDF reports.
2. **Replaced speculative risk labels** (`"High probability of imminent airway collapse"`) with transparent observational statements (`"Model score exceeds decision boundary (>0.50); review autonomic biomarkers"`).
3. **Prominent Research Disclaimer:** Explicitly stated on every PDF report and in the UI footer:
   > *"RESEARCH & INVESTIGATIONAL USE ONLY. This software does not provide medical diagnoses or replace polysomnography supervised by a sleep medicine physician."*
