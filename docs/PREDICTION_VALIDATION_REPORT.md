# Prediction Validation & Decision Boundary Report

**Report Date:** 2026-11-13  
**Auditor:** Senior Forensic ML Auditor  

---

## 1. Investigation of Output Values

In the previous live demonstration, the system reported:
* **Raw logit:** `+0.004261`
* **Probability:** `0.5011`
* **Prediction:** `APNEA_EVENT`

### Detailed Mathematical Explanation
1. `+0.0042611323` is the exact scalar bias of `fc2` (layer `dense_1` in the original Keras model).
2. When the input signal produced sub-threshold activations in layer `fc1` (Dense 128), ReLU clamped all 128 activations to $0.0$.
3. As a result:
   $$\text{Output} = \mathbf{w}_{\text{fc2}} \cdot \mathbf{0} + b_{\text{fc2}} = +0.0042611323$$
4. Applying sigmoid:
   $$\sigma(+0.0042611323) = \frac{1}{1 + e^{-0.0042611323}} = 0.50106525 \approx 0.5011$$
5. Because the naive threshold was $\ge 0.5000$, a difference of just $+0.0011$ tipped the classification into `APNEA_EVENT`.

---

## 2. Threshold Calibration & Uncertainty Stratification

A fixed binary threshold of 0.50 without confidence bands creates deceptive certainty when the model is in reality near its decision boundary.

### Remediated Production Strategy (`ml/inference/predictor.py`):
1. **Uncertainty Region Explicitly Flagged:**
   Predictions where $0.48 \le P \le 0.50$ are now classified as **`EQUIVOCAL / BORDERLINE`**, with a clear UI message stating that the model is near its decision boundary.
2. **Telemetry Transmitted to Client:**
   The API response now includes:
   ```json
   "model_telemetry": {
     "raw_logit": 0.004261,
     "decision_threshold": 0.50,
     "is_near_boundary": true
   }
   ```
3. **Multi-Modal Decision Support:**
   Clinicians are instructed to evaluate physiological HRV indices (Mean HR, SDNN, RMSSD, LF/HF ratio) in conjunction with model probabilities rather than relying solely on the raw binary output.
