import numpy as np
from scipy import signal as scipy_signal
from typing import Dict, Any, List

def detect_systolic_peaks(filtered_ppg: np.ndarray, fs: float = 80.0) -> np.ndarray:
    """
    Detects systolic peaks in the filtered PPG waveform.
    Constraints:
    - Minimum distance: 0.35s (corresponds to max HR ~170 bpm)
    - Prominence: proportional to signal standard deviation
    """
    if len(filtered_ppg) < int(0.5 * fs):
        return np.array([], dtype=int)
        
    std_val = np.std(filtered_ppg)
    distance = max(1, int(0.35 * fs))
    prominence = max(0.05, 0.2 * std_val)
    
    peaks, _ = scipy_signal.find_peaks(
        filtered_ppg, 
        distance=distance, 
        prominence=prominence
    )
    return peaks

def compute_hrv_features(peaks: np.ndarray, fs: float = 80.0) -> Dict[str, float]:
    """
    Computes genuine Heart Rate & HRV indices from systolic peak locations:
    - Mean HR (bpm)
    - SDNN: Standard deviation of NN (peak-to-peak) intervals (ms)
    - RMSSD: Root mean square of successive differences (ms) - parasympathetic index
    - PNN50: Percentage of successive intervals differing by > 50ms
    - CV: Coefficient of variation of NN intervals
    """
    if len(peaks) < 2:
        return {
            "mean_hr_bpm": 0.0,
            "sdnn_ms": 0.0,
            "rmssd_ms": 0.0,
            "pnn50_pct": 0.0,
            "cv": 0.0,
            "peak_count": len(peaks)
        }
        
    # Peak-to-peak intervals in milliseconds
    rr_intervals_ms = np.diff(peaks) * (1000.0 / fs)
    
    # Filter physiologically implausible intervals (< 300ms or > 1800ms)
    valid_rr = rr_intervals_ms[(rr_intervals_ms >= 300) & (rr_intervals_ms <= 1800)]
    if len(valid_rr) == 0:
        valid_rr = rr_intervals_ms
        
    mean_rr_ms = float(np.mean(valid_rr))
    mean_hr_bpm = float(60000.0 / mean_rr_ms) if mean_rr_ms > 0 else 0.0
    sdnn_ms = float(np.std(valid_rr))
    
    if len(valid_rr) > 1:
        diff_rr = np.diff(valid_rr)
        rmssd_ms = float(np.sqrt(np.mean(diff_rr ** 2)))
        pnn50_pct = float(np.sum(np.abs(diff_rr) > 50) / len(diff_rr) * 100.0)
    else:
        rmssd_ms = 0.0
        pnn50_pct = 0.0
        
    cv = float(sdnn_ms / mean_rr_ms) if mean_rr_ms > 0 else 0.0
    
    return {
        "mean_hr_bpm": round(mean_hr_bpm, 1),
        "sdnn_ms": round(sdnn_ms, 2),
        "rmssd_ms": round(rmssd_ms, 2),
        "pnn50_pct": round(pnn50_pct, 1),
        "cv": round(cv, 4),
        "peak_count": int(len(peaks))
    }

def compute_spectral_features(filtered_ppg: np.ndarray, fs: float = 80.0) -> Dict[str, float]:
    """
    Computes Welch Power Spectral Density bands:
    - VLF: Very Low Frequency (0.003 - 0.04 Hz)
    - LF: Low Frequency (0.04 - 0.15 Hz) -> sympathetic/baroreflex
    - HF: High Frequency (0.15 - 0.4 Hz) -> parasympathetic/vagal respiratory
    - LF/HF ratio: Sympathovagal balance index (often elevated pre-apnea)
    """
    freqs, psd = scipy_signal.welch(filtered_ppg, fs=fs, nperseg=min(256, len(filtered_ppg)))
    
    lf_band = (freqs >= 0.04) & (freqs < 0.15)
    hf_band = (freqs >= 0.15) & (freqs < 0.40)
    
    integrate_fn = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
    lf_power = float(integrate_fn(psd[lf_band], freqs[lf_band])) if np.any(lf_band) else 0.0
    hf_power = float(integrate_fn(psd[hf_band], freqs[hf_band])) if np.any(hf_band) else 0.0
    
    lf_hf_ratio = float(lf_power / hf_power) if hf_power > 1e-8 else 0.0
    
    return {
        "lf_power": round(lf_power, 4),
        "hf_power": round(hf_power, 4),
        "lf_hf_ratio": round(lf_hf_ratio, 2)
    }

def extract_all_biomarkers(filtered_ppg: np.ndarray, fs: float = 80.0) -> Dict[str, Any]:
    """
    Extracts complete set of physiological biomarkers from a 30-second window.
    """
    peaks = detect_systolic_peaks(filtered_ppg, fs=fs)
    hrv = compute_hrv_features(peaks, fs=fs)
    spectral = compute_spectral_features(filtered_ppg, fs=fs)
    
    return {
        "peaks": peaks.tolist(),
        "hrv": hrv,
        "spectral": spectral
    }
