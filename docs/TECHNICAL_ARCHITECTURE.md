# Technical Architecture: ApneaSphere PPG Sleep Apnea Prediction

## 1. System Overview

ApneaSphere is a machine-learning-driven clinical decision-support pipeline designed to forecast impending Obstructive Sleep Apnea (OSA) events from Photoplethysmography (PPG) waveforms prior to onset.

```
[ Patient PPG Signal ] 
         │ (30s window @ 80 Hz = 2,400 samples)
         ▼
[ Scientific Preprocessing ]
   ├── Linear NaN / Artifact Interpolation
   ├── 4th-Order Butterworth Bandpass (0.5 - 8.0 Hz)
   ├── Resampling & Length Alignment (80 Hz, 2400 pts)
   └── Z-Score Standardization [(x - μ) / (σ + 1e-6)]
         │
         ├───► [ Physiological Telemetry Engine ]
         │        ├── Systolic Peak Detection (scipy.signal)
         │        ├── Time-Domain HRV (Mean HR, SDNN, RMSSD, pNN50, CV)
         │        └── Spectral Analysis (Welch PSD: LF, HF, LF/HF Ratio)
         │
         ▼
[ Deep 1D-CNN Inference Head ]
   ├── Block 1: Conv1D(32, k=7, p=3) -> BN -> ReLU -> MaxPool(2) -> Drop(0.3)
   ├── Block 2: Conv1D(64, k=5, p=2) -> BN -> ReLU -> MaxPool(2) -> Drop(0.3)
   ├── Block 3: Conv1D(128, k=3, p=1) -> BN -> ReLU -> MaxPool(2) -> Drop(0.3)
   ├── Flatten: 128 x 300 = 38,400 features
   ├── Dense 1: Linear(38400 -> 128) -> ReLU -> Dropout(0.5)
   └── Dense 2: Linear(128 -> 1) -> Sigmoid Logit
         │
         ▼
[ Prediction & Risk Stratification ]
   ├── Probability Score (0.0 - 1.0)
   ├── Classification (APNEA_EVENT vs NORMAL_BREATHING)
   ├── Risk Tiering (LOW, MODERATE, HIGH)
   └── Clinical Interpretation Guidance
         │
         ▼
[ Serving & Presentation ]
   ├── FastAPI High-Performance REST API
   ├── Interactive Clinical Web Dashboard (Chart.js + Tailwind)
   └── Automated ReportLab PDF Generation
```

---

## 2. Timing Paradigm & Prediction Window

- **Observation Window:** 30.0 seconds (2,400 samples at 80 Hz).
- **Lead Time Buffer:** 5.0 seconds. No event during this interval; allows proactive auto-titrating CPAP activation or patient notification before oxygen desaturation.
- **Forecast Horizon:** 10.0 seconds following the lead-time buffer.

---

## 3. Technology Stack

- **Deep Learning Framework:** PyTorch 2.x
- **Signal Processing & Feature Extraction:** SciPy, NumPy
- **Backend Application Server:** FastAPI, Uvicorn, Pydantic v2
- **Clinical Reporting:** ReportLab PDF Engine
- **Frontend Dashboard:** Vanilla JavaScript, Chart.js, Lucide Icons, Modern CSS
- **Test Suite:** PyTest, TestClient

