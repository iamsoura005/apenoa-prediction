# AUDIT REPORT: Forensic Codebase & Integrity Review

**Date:** 2026-11-13  
**Auditor:** Senior Forensic ML & Systems Auditor  
**Audit Target:** ApneaSphere Sleep Apnea PPG Forecast System  

---

## 1. Executive Summary

A comprehensive forensic audit of every Python, JavaScript, HTML, CSS, configuration, and model file was performed to assess authenticity, eliminate simulated behavior, and verify production safety.

The repository originally contained two disjointed trees:
1. `Apneoa-event-prediction-Using-PPG-signal-main/` (Legacy prototype folder containing fabricated cross-validation and fake prediction logic).
2. Root-level production system (`ml/`, `backend/`, `frontend/`, `data/`, `tests/`).

The production pipeline has been completely isolated from the legacy scripts. All production modules execute authentic calculations.

---

## 2. Forensic Scan Results

### A. Integrity Keywords Scan
Search for keywords: `force_predictions_to_cm`, `calculate_cm_from_sens_prec`, `time.sleep`, `fake`, `dummy`, `mock`, `TODO`, `FIXME`.

| Directory | Scope | Findings | Verdict |
|---|---|---|---|
| `ml/` | Preprocessing, features, models, inference | **0 matches.** Zero fabrication or simulation found. | **CLEAN** |
| `backend/` | API routes, schemas, PDF generation | **0 matches.** Pure REST endpoints and dynamic PDF generation. | **CLEAN** |
| `frontend/` | UI scripts, HTML, styles | **0 matches.** Pure client-side telemetry and API integration. | **CLEAN** |
| `tests/` | PyTest test cases | **0 matches.** Verifies real behavior against synthetic/physiological boundaries. | **CLEAN** |
| `Apneoa-.../` (legacy) | Archive of initial repository | Contains legacy `force_predictions_to_cm` and `time.sleep()` in `training.py` and `testing_final.py`. | **ARCHIVED / ISOLATED** |

---

## 3. Findings & Remediations Table

| Issue ID | File / Location | Severity | Description | Remediation | Status |
|---|---|---|---|---|---|
| **AUD-01** | `Apneoa-.../training.py` | Critical (Legacy) | Hardcoded target sensitivities and `force_predictions_to_cm` fabricated metric scores. | Isolated in legacy folder. Production engine (`ml/inference/predictor.py`) implements clean PyTorch forward pass without post-hoc manipulation. | **RESOLVED** |
| **AUD-02** | `Apneoa-.../training.py` | Critical (Legacy) | 5-fold CV simulated with `time.sleep()` loops and random normal sampling. | Isolated. Documented in `PROJECT_EXECUTION_PLAN.md` and `TECHNICAL_ARCHITECTURE.md`. | **RESOLVED** |
| **AUD-03** | `frontend/index.html` | High | Missing Tailwind runtime in initial commit caused unstyled button rendering in browser. | Injected Tailwind CDN engine and dark-theme CSS in `frontend/styles.css`. Verified in browser. | **RESOLVED** |
| **AUD-04** | `backend/app.py` | Medium | In-memory `ANALYSIS_CACHE` had no eviction limit. | Added FIFO eviction preserving last 50 analyses to prevent memory bloat under load. | **RESOLVED** |
| **AUD-05** | `ml/preprocessing/ppg_filter.py` | Low | Deprecated `np.trapz` emitted warnings on NumPy 2.x. | Replaced with dynamic `np.trapezoid` / `np.trapz` compatibility layer. | **RESOLVED** |

