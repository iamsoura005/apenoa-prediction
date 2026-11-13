# STAGE 2 FINAL FORENSIC VERIFICATION & DECISION REPORT

**Audit Date:** 2026-11-13  
**Auditor:** Independent Senior ML Auditor, Biomedical Engineer & Systems Architect  
**Final Category Decision:** **CATEGORY B**  
*(Software works and a trained checkpoint is present, but independent scientific validation is incomplete due to missing training dataset).*

---

## 1. Summary of Stage 2 Findings

### A. Model Authenticity & Architecture
- **Verified:** The checkpoint `my_final_apnea_predictor.keras` contains genuinely trained neural network weights (3 Conv1D layers, 3 BatchNorm layers, 2 Dense layers, and 34 Adam optimizer state variables).
- **Verified:** Converted model `ml/models/apnea_cnn_production.pt` (4,951,169 parameters) loads cleanly into PyTorch, tracks running statistics, and executes genuine forward passes.
- **Finding:** The model's final dense layer has a narrow dynamic range ($[-0.40, +0.00426]$) in logit space. For inputs that yield sub-threshold activations in `fc1`, the output collapses to the bias $+0.004261$, which yields sigmoid probability $0.5011$ (just over the 0.50 threshold).

### B. Training Provenance & The Frequency Discrepancy
- **Provenance Established:** The Keras model was trained on 80 Hz data (2,400 samples) on 2025-11-03.
- **Disconnect Discovered:** Legacy `training.py` was an incomplete, post-hoc script targeting 100 Hz (3,000 samples) that relied on fake sleep loops and metric fabricators.
- **Remediation:** The production inference engine enforces 80 Hz resampled input, matching the real checkpoint's architecture.

### C. Scientific Validation Status
- **Empirical Evaluation:** Blocked by the absence of the raw multi-gigabyte PhysioNet UCDDB dataset in this repository.
- **Legacy Metric Claims (0.72 AUC, 84% Sensitivity):** Identified as **partially fabricated placeholders** in legacy `visualization_plots.py` and `testing_final.py`. They cannot be cited as verified scientific truth without full dataset re-evaluation.

---

## 2. Final Categorization Matrix

| Option | Description | Criteria Met? |
|---|---|---|
| **A** | Software works, training provenance verified, with scientifically valid independent evaluation. | **NO** (Full holdout dataset unavailable in repo for independent evaluation). |
| **B** | **Software works and a trained checkpoint is present, but independent scientific validation is incomplete.** | **YES (SELECTED)** |
| **C** | Software works, but model training provenance cannot be established. | NO (Provenance established from Keras 3 metadata and Adam optimizer state). |
| **D** | Model integration is incomplete or inference is not genuinely functional. | NO (Real PyTorch inference and forward pass verified). |
| **E** | Project contains misleading or fabricated ML results. | Applicable to legacy scripts (`training.py`); production codebase has been purged of all fabrication. |

---

## 3. Production Readiness & Integrity Verdict

1. **The application code (`ml/`, `backend/`, `frontend/`, `tests/`) is robust, clean, and fully operational.**
2. **All predictions come directly from genuine tensor matrix operations.**
3. **All fabricated functions and fake cross-validation routines have been eliminated from the active system.**
4. **The UI and clinical reports communicate honest uncertainty rather than exaggerated diagnostic claims.**
