# ApneaSphere Enterprise: Final System Architecture

## 1. System Overview

ApneaSphere is an enterprise-grade biomedical software platform designed to ingest raw Photoplethysmography (PPG) waveforms, perform digital filtering and autonomic feature extraction, execute 1D Convolutional Neural Network (1D-CNN) predictive inference, persist clinical history, and generate PDF diagnostic summary reports.

```
                                  USER BROWSER / CLIENT
                                            │
                     ┌──────────────────────┴──────────────────────┐
                     │ (Static HTML / JS / Tailwind CSS via CDN)    │
                     │  - Interactive Waveform Viewer (Chart.js)   │
                     │  - Persistent History Log                   │
                     │  - Live Telemetry & Biomarkers Dashboard    │
                     └──────────────────────┬──────────────────────┘
                                            │  HTTP REST (JSON / Multipart)
                                            ▼
                                     FASTAPI GATEWAY
                                  (backend/app.py: Uvicorn)
                                            │
        ┌───────────────────────────────────┼───────────────────────────────────┐
        │                                   │                                   │
        ▼                                   ▼                                   ▼
[ Preprocessing & Telemetry ]     [ ML Inference Head ]                [ Data Persistence ]
  - Butterworth 0.5-8.0 Hz          - Model Checkpoint                   - SQLite Engine
  - Artifact Interpolation            (apnea_cnn_production.pt)            (backend/database.py)
  - Z-Score Normalization           - PyTorch 1D-CNN (4.95M params)      - Analysis Records
  - Peak Detection & HRV            - Raw Logits & Probabilities         - Real Historical Logs
  - Spectral PSD (LF / HF)          - Calibrated Risk Stratification
        │                                   │                                   │
        └───────────────────────────────────┼───────────────────────────────────┘
                                            │
                                            ▼
                               [ ReportLab PDF Engine ]
                                  (backend/pdf_report.py)
                               - Dynamic Vector PDF Output
```

---

## 2. Component Directory Architecture

```
Apneoa-event-prediction-Using-PPG-signal-main/
├── backend/
│   ├── app.py                # FastAPI gateway, REST routing, file ingest
│   ├── database.py           # SQLAlchemy SQLite persistence engine
│   ├── schemas.py            # Pydantic v2 data transfer schemas
│   └── pdf_report.py         # ReportLab clinical document compiler
├── ml/
│   ├── models/
│   │   ├── apnea_cnn.py      # PyTorch 1D-CNN topology (4,951,169 parameters)
│   │   ├── apnea_cnn_production.pt # Checkpoint with weights & metadata
│   │   └── convert_weights.py# Direct porting script from Keras H5
│   ├── preprocessing/
│   │   └── ppg_filter.py     # 4th-order Butterworth, interpolation, SQI
│   ├── features/
│   │   └── hrv_features.py   # Peak detection, SDNN, RMSSD, Welch LF/HF
│   └── inference/
│       └── predictor.py      # Thread-safe ApneaPredictor singleton
├── frontend/
│   ├── index.html            # Single-page enterprise biomedical dashboard
│   ├── app.js                # Chart.js dynamic visualizer & API controller
│   └── styles.css            # Medical UI styling & dark-theme overrides
├── data/
│   ├── apneasphere.db        # SQLite persistent clinical database
│   ├── sample_signals/       # Benchmark cases (Normal, Pre-Apnea, Borderline)
│   └── generate_samples.py   # Benchmark generation script
├── tests/
│   ├── test_preprocessing.py # Digital signal processing unit tests
│   └── test_inference_and_api.py # End-to-end API, DB, and inference tests
├── docs/                     # Technical, model, API, and audit documentation
└── Dockerfile                # Production container specification
```

---

## 3. Security, Privacy & Integrity Rules
- Zero Protected Health Information (PHI) is persisted without explicit user ingestion.
- Multi-part file uploads are parsed in-memory and strictly validated against numerical length limits ($N \ge 80$).
- All predictions originate strictly from the neural network's genuine mathematical forward pass.
