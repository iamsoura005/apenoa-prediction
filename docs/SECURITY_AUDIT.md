# SECURITY & RELIABILITY AUDIT

**Date:** 2026-11-13  
**Auditor:** Application Security & Reliability Engineer  
**Scope:** FastAPI Server, File Upload Handler, ML Serving Engine

---

## 1. Security Analysis Findings

### A. File Upload Safety (`POST /api/upload`)
- **File Type Restrictions:** Upload accepts `.csv`, `.json`, and `.txt`.
- **Parsing Robustness:** Files are parsed line-by-line or via `json.loads` within try-except blocks. Malformed inputs throw `400 Bad Request` rather than crashing the process or leaking stack traces.
- **In-Memory Buffer:** Upload contents are read into memory streams and decoded safely without writing untrusted raw files to the host filesystem, preventing arbitrary file overwrite or path traversal attacks.

### B. Input Validation (`POST /api/predict`)
- **Pydantic Validation:** All request payloads are strictly validated against `PredictRequest`.
- **Length Checking:** Signals shorter than 80 samples (1 second at 80 Hz) are rejected with a clear `400 Bad Request`.
- **NaN / Inf Shielding:** `ml/preprocessing/ppg_filter.py` checks `np.isfinite()`. Non-finite numbers are linearly interpolated from neighboring valid values. If all values are non-finite, a controlled `ValueError` is raised.

### C. CORS Configuration
- **State:** Configured with `allow_origins=["*"]`, suitable for local research workstations and multi-port frontend dev. For hospital production intranet deployment, recommend restricting origins to the specific clinical portal domain.

### D. Memory & Resource Exhaustion (DoS Protection)
- **Signal Truncation/Padding:** Preprocessing enforces fixed array size (2,400 samples) before feeding the neural network. Excessive signal length is safely truncated to 2,400 samples, preventing GPU/CPU memory spikes.
- **Analysis Cache Bound:** In-memory PDF report cache is limited to a maximum of 50 concurrent items with FIFO eviction, preventing memory leaks over extended uptime.

### E. Privacy & Sensitive Patient Data
- No Protected Health Information (PHI) is persisted to local disk.
- Log outputs contain only telemetry (batch size, elapsed inference time, status codes) and no raw patient waveform arrays or personal identifiers.

