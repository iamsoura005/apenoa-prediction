# Security, Privacy & Compliance Documentation

**Platform:** ApneaSphere Enterprise Biomedical Platform  
**Version:** 2.0.0  
**Classification:** Research & Clinical Decision-Support System  

---

## 1. Data Ingestion & Memory Isolation
* **In-Memory Streaming:** Uploaded files (`.csv`, `.json`, `.txt`) are decoded directly into memory buffers. No untrusted raw files are written to the host operating system, preventing path traversal (`../`) and arbitrary file execution.
* **Strict Numerical Validation:** Files are verified to contain at least 80 numerical samples (1 second at 80 Hz). Malformed or non-numerical rows are skipped gracefully.
* **Bounded Allocation:** Input arrays are truncated or zero-padded strictly to 2,400 samples before passing into the PyTorch tensor memory space, preventing Out-Of-Memory (OOM) denial-of-service vulnerabilities.

---

## 2. Database Persistence & SQL Injection Shielding
* **ORM Parameterization:** SQLite persistence is handled via **SQLAlchemy 2.x ORM**. All queries and inserts use bound parameters, eliminating raw string interpolation and SQL injection risks.
* **Sanitized Record Metadata:** Analysis identifiers are generated via cryptographically secure UUID4 (`uuid.uuid4()`).

---

## 3. Privacy & Protected Health Information (PHI)
* **Zero PII Required:** The system requires no patient names, government IDs, or demographic data. Inputs are referenced solely by a user-supplied identifier (defaulting to `"Anonymous"`).
* **De-Identified Waveforms:** The SQLite database stores only the downsampled 600-point preview array for visual inspection, not the full raw clinical continuous recording.

---

## 4. Regulatory & Diagnostic Safety Boundaries
* **Investigational Use Only:** Explicitly disclaimed on all exported PDF reports and the dashboard footer.
* **Calibrated Output Bands:** Output scores near the decision boundary ($0.48 \le P \le 0.50$) are flagged as `EQUIVOCAL / BORDERLINE`, preventing false clinical certainty.
