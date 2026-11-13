# FRONTEND-BACKEND CONNECTION AUDIT

**Date:** 2026-11-13  
**Auditor:** Full-Stack Quality Assurance & UI/UX Engineer  
**Client Interface:** Single-Page Web Dashboard (`frontend/index.html`)

---

## 1. Trace of UI Elements to Actual Data Sources

| UI Component | Rendered Element | Data Source | Authenticity Status |
|---|---|---|---|
| **System Status Badge** | "PyTorch Engine Active" | `/api/health` | **Live API Response** |
| **Model Info Card** | Architecture, Trainable Parameters | `/api/model/info` | **Live API Response** |
| **Benchmark Cards** | 3 pre-loaded patient cases | `/api/samples` | **Live API Response** |
| **Waveform Visualizer** | 30s Chart.js time series | `filtered_signal_preview` from `/api/predict` | **Live Inference Response** |
| **Systolic Peaks Overlay** | Red markers at pulse peaks | `peak_indices_preview` from `/api/predict` | **Calculated from Signal** |
| **Prediction Banner** | Predicted Class & Probability | `prediction` & `probability` from `/api/predict` | **Actual 1D-CNN Output** |
| **Heart Rate Metric** | Mean HR in beats/min | `biomarkers.hrv.mean_hr_bpm` from `/api/predict` | **Extracted Peak Intervals** |
| **Variability Metrics** | SDNN, RMSSD, LF/HF | `biomarkers.hrv` & `spectral` from `/api/predict` | **Genuine Mathematical Indices** |
| **Signal Quality (SQI)** | Quality tier (Good/Marginal) | `signal_quality.sqi_score` from `/api/predict` | **Spectral Energy Calculation** |
| **Export Report Button** | PDF Download | `window.open('/api/report/' + id)` | **Dynamic Backend PDF Stream** |

---

## 2. Complete User Journey Verification

1. **Initial Load:** Client requests `/`, retrieves HTML (19.3 KB), fetches `/api/samples`, populates 3 case cards dynamically, and auto-loads `sample_normal_baseline.json`.
2. **Analysis Execution:** Clicking `"Forecast Impending Apnea"` posts payload to `/api/predict`. Server processes 2,400 floats, computes biomarkers, and returns prediction in < 25 ms.
3. **Visualization Update:** Waveform chart updates dynamically with filtered signal and red markers at systolic peak positions.
4. **Report Export:** Clicking `"Download PDF Report"` sends `GET /api/report/{analysis_id}`, which downloads `apnea_report_<id>.pdf`.
5. **Custom File Ingestion:** Clicking `"Upload Custom File"` uploads `.csv` or `.json` to `/api/upload`, updating the entire dashboard with the uploaded patient's results.

