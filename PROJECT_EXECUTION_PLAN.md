# Complete Implementation & Execution Plan: PPG Sleep Apnea Prediction System

## 1. Executive Summary & Audit Findings

### Core Problems Uncovered in Legacy Repository
1. **Fabricated Evaluation Metrics:** `training.py` and `testing_final.py` utilized synthetic functions (`force_predictions_to_cm`, `calculate_cm_from_sens_prec`) which artificially altered model outputs with random noise to match hardcoded targets (0.71-0.94 AUC).
2. **Simulated Cross-Validation:** 5-fold cross-validation in `training.py` consisted of `time.sleep()` loops coupled with Gaussian random sampling around `CV_TARGET_*` parameters.
3. **Framework & Architecture Inconsistency:**
   - Saved Keras model (`my_final_apnea_predictor.keras`) was trained on **80 Hz signals (2400 samples for 30s window)** and contained genuine, converged trained weights (3 Conv1D layers + 3 BatchNorm + MaxPool + Dropout + Dense).
   - Saved PyTorch checkpoint (`my_final_apnea_predictor.pth`) was a fresh/dummy initialization with **100 Hz / 3000 sample** dimension mismatch, where BatchNorm tracked zero batches.
4. **Missing Dataset Files:** Pickle files (`data/CNN_input_data.pickle`, `CNN_input_data.zip`) were not committed.
5. **Absence of User Interface & Production Serving:** The system was restricted to standalone terminal scripts.

---

## 2. Target Architecture

```
Apneoa-event-prediction-Using-PPG-signal-main/
├── ml/
│   ├── models/           # Unified PyTorch 1D-CNN (recovering authentic Keras weights)
│   ├── preprocessing/    # Butterworth bandpass (0.5-8.0 Hz), outlier handling, z-score norm
│   ├── features/         # Peak detection, HRV (RMSSD, SDNN, LF/HF, sample entropy)
│   ├── training/         # Genuine train/val loop with PyTorch, EarlyStopping, BCEWithLogits
│   ├── evaluation/       # Genuine scikit-learn metrics, patient-wise split
│   └── inference/        # Standalone Predictor class for production scoring
├── backend/
│   ├── app.py            # FastAPI production server with CORS, file upload, PDF generator
│   └── schemas.py        # Pydantic v2 schemas for requests & responses
├── frontend/             # Single-Page Healthcare Analytics Web App
│   ├── index.html        # Clean, modern responsive biomedical dashboard
│   ├── app.js            # Signal rendering (Chart.js), interactive segment inspection, PDF trigger
│   └── styles.css        # Professional medical UI styling (Tailwind-inspired CSS)
├── data/
│   ├── sample_signals/   # Benchmark PPG signals (pre-apnea, normal, raw CSV/JSON)
│   └── ucddb_loader.py   # Automated downloader & extractor for PhysioNet UCDDB
├── tests/
│   ├── test_preprocessing.py
│   ├── test_model_inference.py
│   └── test_api_endpoints.py
├── docs/
│   ├── TECHNICAL_ARCHITECTURE.md
│   ├── MODEL_CARD.md
│   ├── DATASET_DOCUMENTATION.md
│   ├── API_DOCUMENTATION.md
│   └── TESTING_REPORT.md
├── PROJECT_EXECUTION_PLAN.md
├── README.md
└── requirements.txt
```

---

## 3. Engineering Phases

### Phase 1: Model Recovery & Framework Unification
- Extract authentic trained weights from `my_final_apnea_predictor.keras` (Keras 3 zip format).
- Port weights into unified PyTorch architecture (`Apnea1DCNN`, 80 Hz, 2400 samples).
- Export production PyTorch model: `ml/models/apnea_cnn_production.pt`.
- Verify zero discrepancy in tensor outputs.

### Phase 2: Scientific Preprocessing & Feature Extraction
- Reusable Butterworth bandpass filter (0.5 - 8.0 Hz, 4th order filtfilt).
- Baseline wander removal and linear interpolation for missing/NaN segments.
- Peak detection (scipy.signal find_peaks with prominence & physiological constraints).
- Real-time HRV metrics: Mean HR, SDNN, RMSSD, CV, Welch PSD band powers (LF: 0.04-0.15 Hz, HF: 0.15-0.4 Hz, LF/HF ratio).

### Phase 3: Dataset Ingestion & Benchmark Samples
- Script `data/ucddb_loader.py` capable of pulling and parsing PhysioNet UCDDB data.
- Generate verified test benchmark samples (Normal Baseline vs Pre-Apnea onset) for instant evaluation and demo upload.

### Phase 4: Production FastAPI Backend
- `POST /api/predict`: Runs inference on 30s PPG signal array.
- `POST /api/analyze`: Comprehensive signal processing, peak detection, HRV extraction, and apnea prediction.
- `POST /api/upload`: Multi-format file ingest (CSV, JSON, TXT).
- `GET /api/model/info`: Model metadata, architecture, weights verification status.
- `GET /api/samples`: Access pre-loaded clinical benchmark signals.
- `POST /api/report/generate`: Generates downloadable PDF clinical report using ReportLab.

### Phase 5: High-Performance Biomedical Frontend
- Responsive dashboard with tabbed workflows:
  - **Overview & Research Problem** (Pathophysiology of OSA, autonomic PPG markers).
  - **Live Signal Analysis Workspace** (Interactive Chart.js waveform, zoom, peak highlights, probability gauge).
  - **Sample Library** (One-click loading of verified benchmark cases).
  - **Model Architecture & Transparency** (Honest metrics, layers, parameters).
  - **Clinical Report Export** (One-click PDF download).

### Phase 6: Automated Testing & Verification
- Unit tests for preprocessing, model conversion, and FastAPI endpoints.
- End-to-end integration test of signal upload -> analysis -> report generation.

