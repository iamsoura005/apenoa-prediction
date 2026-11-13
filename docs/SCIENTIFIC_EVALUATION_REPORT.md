# Scientific Evaluation & Dataset Availability Report

**Report Date:** 2026-11-13  
**Auditor:** Senior Deep Learning Researcher & Clinical Evaluator  

---

## 1. Dataset Status & Blocking Factors

* **Current Repository Contents:** The repository contains no raw polysomnography `.pickle` or `.zip` files.
* **Why Retraining / Re-evaluation is Blocked:**
  * Complete 5-fold patient-independent re-evaluation requires the original ~5 GB PhysioNet UCDDB recordings (`ucddb002` through `ucddb028`).
  * In the absence of the raw multi-subject dataset, executing a "new" 5-fold CV would require synthesizing fake patient labels—which is strictly prohibited under our scientific ethics rules.

---

## 2. Legacy Metric Claims vs Genuine Empirical State

In the legacy prototype, authors claimed:
* AUC: `0.7200`
* Sensitivity: `84.0%`
* Specificity: `73.4%`
* Precision: `76.0%`

### Forensic Determination:
As uncovered in Phase 1, the script `visualization_plots.py` explicitly states:
`# FABRICATED RESULTS DATA (FROM YOUR SCREENSHOTS) ... Confusion matrices (Placeholders, as these were not in the screenshots)`.
Furthermore, `testing_final.py` utilized `force_predictions_to_cm()` to inject Gaussian noise until random numbers hit these exact figures.

### Definitive Conclusion:
The reported figures of **0.72 AUC and 84% Sensitivity are UNVERIFIED and partially fabricated in the legacy prototype**. 

While the neural network weights inside `my_final_apnea_predictor.keras` were trained on genuine gradient updates, **no scientifically audited test evaluation matrix exists in this repository**. Any claim of confirmed diagnostic sensitivity must be marked as **Pending Independent Re-Evaluation on PhysioNet UCDDB**.
