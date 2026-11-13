# Signal Compatibility & Sampling Frequency Report

**Report Date:** 2026-11-13  
**Auditor:** Biomedical Signal Processing Engineer  

---

## 1. The 80 Hz vs 100 Hz Discrepancy

### Root Cause Analysis
* Early documentation and the unverified `training.py` script assumed a universal 100 Hz sampling frequency (3,000 samples for a 30s window).
* However, medical PSG monitors (such as those in the UCDDB dataset) record photoplethysmography at varying clock rates (commonly downsampled to 80 Hz or 128 Hz).
* In `prediction_window_plot.py`, line 35 confirms `current_fs = 80` was the true baseline frequency of the valid PPG entries in the original dataset.
* The Keras checkpoint `my_final_apnea_predictor.keras` strictly defines its input layer as `batch_shape: [null, 2400, 1]`. Since the window duration is 30 seconds:
  $$\text{Frequency} = \frac{2,400 \text{ samples}}{30 \text{ seconds}} = 80 \text{ Hz}$$

---

## 2. Ingestion & Resampling Guarantees

In the production inference engine (`ml/preprocessing/ppg_filter.py`):
1. **Arbitrary Frequency Handling:** The API accepts signals recorded at any frequency (`fs: float`), defaulting to 80 Hz.
2. **Scientifically Valid Resampling:**
   If $f_s \neq 80.0 \text{ Hz}$, the pipeline executes Fourier-based anti-aliasing resampling:
   $$N_{\text{target}} = \text{round}\left(N_{\text{orig}} \times \frac{80.0}{f_s}\right)$$
   using `scipy.signal.resample` *prior* to feeding the neural network.
3. **No Blind Reshaping:** Raw arrays are never reshaped or padded blindly; they are cleaned of NaNs, bandpass filtered in their native frequency domain, resampled, and then normalized to zero mean and unit variance.
