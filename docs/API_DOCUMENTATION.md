# ApneaSphere REST API Documentation

Base URL: `http://localhost:8000`

---

### `GET /api/health`
Checks server status and confirms model loading.
- **Response `200`:**
  ```json
  {
    "status": "healthy",
    "service": "ApneaSphere PPG Prediction API",
    "timestamp": "2026-11-13T14:30:00",
    "device": "cpu",
    "model_loaded": true
  }
  ```

---

### `GET /api/model/info`
Returns model metadata, layer topology, and parameters.
- **Response `200`:**
  ```json
  {
    "model_name": "Apnea1DCNN",
    "framework": "PyTorch 2.x",
    "total_parameters": 4942753,
    "input_samples": 2400,
    "sampling_rate_hz": 80,
    "window_seconds": 30,
    "lead_time_seconds": 5,
    "forecast_target_seconds": 10
  }
  ```

---

### `GET /api/samples`
Lists available pre-loaded benchmark recordings.

---

### `POST /api/predict`
Executes preprocessing, biomarker calculation, and 1D-CNN forecast on raw PPG.
- **Request Body:**
  ```json
  {
    "signal": [820.5, 825.1, 831.0, ...],
    "fs": 80.0,
    "subject_id": "Patient-17"
  }
  ```
- **Response `200`:**
  ```json
  {
    "analysis_id": "a82df165-22d1-4be5-94cf-6f3b06de81ba",
    "prediction": "APNEA_EVENT",
    "probability": 0.5011,
    "confidence_percentage": 50.1,
    "risk_level": "MODERATE",
    "clinical_interpretation": "Early autonomic perturbations observed...",
    "biomarkers": {
      "hrv": { "mean_hr_bpm": 74.2, "sdnn_ms": 48.5, "rmssd_ms": 32.1 },
      "spectral": { "lf_power": 0.12, "hf_power": 0.05, "lf_hf_ratio": 2.4 }
    }
  }
  ```

---

### `POST /api/upload`
Uploads raw CSV, TXT, or JSON files.

---

### `GET /api/report/{analysis_id}`
Generates and downloads clinical PDF report.

