# ApneaSphere: Predictive Sleep Apnea AI Platform

[![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0%2Bcu124-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-NVIDIA%20RTX%204050-76B900.svg?style=flat&logo=nvidia)](https://developer.nvidia.com/cuda-zone)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg?style=flat&logo=fastapi)](https://fastapi.tiangolo.com/)
[![SQLite](https://img.shields.io/badge/SQLite-SQLAlchemy%202.0-003B57.svg?style=flat&logo=sqlite)](https://sqlite.org/)
[![Docker](https://img.shields.io/badge/Docker-Hugging%20Face%20Ready-2496ED.svg?style=flat&logo=docker)](https://www.docker.com/)
[![Tests](https://img.shields.io/badge/Tests-11%20Passing-brightgreen.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**ApneaSphere** is a biomedical AI platform that forecasts impending **Obstructive Sleep Apnea (OSA)** episodes from continuous single-channel Photoplethysmography (PPG) waveforms with a **5-second lead time** prior to airway collapse.

---

## 🌟 Key Capabilities

- **Deep 1D-CNN Network**: 3 convolutional blocks (32, 64, 128 filters) with batch normalization, max pooling, and dropout, operating on 30s windows at 80 Hz (4,951,169 parameters).
- **GPU Accelerated**: Native CUDA acceleration verified on NVIDIA GeForce RTX 4050 Laptop GPU with PyTorch 2.6.0+cu124 and Automatic Mixed Precision (AMP).
- **Zero Synthetic Results**: Complete elimination of fake metrics and simulated cross-validation. All predictions originate strictly from genuine PyTorch tensor operations.
- **Biomedical Signal Processing**: 4th-order zero-phase Butterworth bandpass filtering (0.5 – 8.0 Hz), linear NaN interpolation, and Z-score standardization.
- **Real-Time Autonomic Biomarkers**: Automatic systolic peak detection, time-domain HRV (Mean HR, SDNN, RMSSD, pNN50, CV), and Welch spectral powers (LF, HF, LF/HF ratio).
- **Persistent Clinical Database**: Integrated SQLite database powered by SQLAlchemy 2.x for persistent historical records and trend analysis.
- **Interactive Medical Dashboard**: Single-page clinical UI featuring Chart.js waveform rendering, systolic peak indicators, and an auditable model transparency panel.
- **Automated PDF Reports**: Dynamic ReportLab compiler producing downloadable clinical summary reports.
- **Enterprise Deployment**: Production-ready `Dockerfile` for Hugging Face Spaces (Port 7860) and `vercel.json` for Vercel static hosting.

---

## 🚀 Quickstart

### 1. Local Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Launch the platform
python -m uvicorn backend.app:app --host 0.0.0.0 --port 8000
```
Open **`http://localhost:8000`** in your browser.

### 2. Local GPU Training
```bash
python ml/train.py --device cuda --mixed-precision --epochs 15 --batch-size 32
```

### 3. Docker Deployment (Hugging Face Spaces)
```bash
docker build -t apneasphere-backend .
docker run -p 7860:7860 apneasphere-backend
```

---

## 🧪 Automated Testing

Execute the test suite verifying preprocessing, model inference, database persistence, and PDF compilation:
```bash
python -m pytest tests -v
```

---

## 📚 Technical Documentation

- [Project Audit](PROJECT_AUDIT.md): Detailed forensic audit of legacy code, model weights, and signal compatibility.
- [Dataset Validation](DATASET_VALIDATION.md): PPG vs. ECG modalities, PhysioNet UCDDB analysis, and annotation integrity.
- [Model Evaluation](MODEL_EVALUATION.md): Independent test set metrics (ROC-AUC: 0.944, F1: 0.806), confusion matrix, and threshold calibration.
- [Testing Report](TESTING_REPORT.md): Pytest automated test results and coverage.
- [Deployment Guide](DEPLOYMENT.md): Hugging Face Spaces backend and Vercel frontend deployment.
- [Final Delivery Report](FINAL_PROJECT_DELIVERY.md): Complete summary of issues resolved, architecture, and live links.
