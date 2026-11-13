# Training Reproducibility & Pipeline Audit

**Report Date:** 2026-11-13  
**Auditor:** Senior Forensic ML Engineer  
**Files Audited:** `training.py`, `testing_final.py`, `preapnea_vs_baseline.py`

---

## 1. Discovered Training Scripts vs Checkpoint Provenance

### A. The Disconnect
The script `training.py` in the legacy folder defines:
```python
WINDOW_SIZE_SEC = 30
TARGET_FS = 100
WINDOW_SAMPLES_TARGET = 3000 # 100 Hz * 30s
# Flatten layer: Linear(48000, 128)
```
However, the trained model checkpoint `my_final_apnea_predictor.keras` was generated with:
```json
{
  "batch_shape": [null, 2400, 1],
  "layers": [
    {"class_name": "Flatten"},
    {"class_name": "Dense", "units": 128} # 38400 -> 128
  ]
}
```
**Conclusion:** `training.py` was **NOT** the script that produced `my_final_apnea_predictor.keras`. `training.py` was a later, broken attempt to rebuild the model at 100 Hz, which failed and resorted to `force_predictions_to_cm` and `time.sleep()`.

The genuine training script that produced `my_final_apnea_predictor.keras` on `2025-11-03` operated on **80 Hz** data (2,400 samples).

---

## 2. Dataset Reconstruction & Label Assignment Rules

From `preapnea_vs_baseline.py` and `prediction_window_plot.py`, the genuine training configuration can be reverse-engineered:
* **Cohort:** PhysioNet UCDDB (subjects `'7'`, `'17'`, `'18'`, `'23'`, `'25'`, `'30'`).
* **Signal Frequency:** 80 Hz natively extracted from Plethysmogram.
* **Positive Epochs (Pre-Apnea):**
  * Find apnea event onset in PSG annotations (`anomaly == 1`).
  * Back off by `LEAD_TIME = 5.0 seconds` (400 samples).
  * Extract prior `WINDOW_SIZE = 30.0 seconds` (2,400 samples).
  * Confirm `is_valid_segment` (variance > 0.1, no apnea during the 30s window).
* **Negative Epochs (Baseline Normal):**
  * Stride across recording with 25% overlap away from any apnea events.
  * Quality-based selection: heart rate 60–90 bpm, low range variability.
* **Loss & Optimization:** Binary Cross-Entropy with `pos_weight` to address severe class imbalance (normal breathing epochs outnumber pre-apnea epochs by ~10:1).
