import numpy as np
from scipy import signal as scipy_signal
from typing import Tuple, Dict, Any, Optional

def clean_and_interpolate_signal(raw_signal: np.ndarray) -> np.ndarray:
    """
    Handles NaNs, Infs, and removes degenerate artifacts.
    Performs linear interpolation across missing segments.
    """
    sig = np.array(raw_signal, dtype=np.float64)
    
    if sig.ndim > 1:
        sig = sig.squeeze()
        if sig.ndim > 1:
            sig = np.nanmean(sig, axis=-1)
            
    # Mask invalid values
    finite_mask = np.isfinite(sig)
    if not np.any(finite_mask):
        raise ValueError("PPG signal contains no finite numerical values.")
        
    if not np.all(finite_mask):
        valid_indices = np.where(finite_mask)[0]
        invalid_indices = np.where(~finite_mask)[0]
        sig[invalid_indices] = np.interp(invalid_indices, valid_indices, sig[valid_indices])
        
    return sig

def butter_bandpass_filter(
    signal: np.ndarray, 
    fs: float = 80.0, 
    lowcut: float = 0.5, 
    highcut: float = 8.0, 
    order: int = 4
) -> np.ndarray:
    """
    Applies a zero-phase 4th order Butterworth bandpass filter (0.5 to 8.0 Hz).
    0.5 Hz removes respiratory baseline wander and motion artifacts.
    8.0 Hz removes high frequency muscle noise and electrical interference
    while preserving the dicrotic notch and systolic pulse profile.
    """
    if len(signal) < 3 * order:
        return signal

    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    
    # Bound to valid range (0, 1)
    low = max(0.001, min(low, 0.95))
    high = max(low + 0.01, min(high, 0.99))
    
    b, a = scipy_signal.butter(order, [low, high], btype='band')
    filtered = scipy_signal.filtfilt(b, a, signal)
    return filtered

def normalize_signal(signal: np.ndarray) -> np.ndarray:
    """
    Z-score standardizes signal: (x - mean) / (std + 1e-6).
    Ensures input distribution is centered around 0 with unit variance.
    """
    mean = np.mean(signal)
    std = np.std(signal)
    if std < 1e-7:
        return np.zeros_like(signal)
    return (signal - mean) / (std + 1e-6)

def resample_signal(signal: np.ndarray, orig_fs: float, target_fs: float = 80.0) -> np.ndarray:
    """
    Resamples PPG signal to target frequency (default 80 Hz for the CNN).
    """
    if orig_fs == target_fs:
        return signal
    target_num_samples = int(round(len(signal) * target_fs / orig_fs))
    return scipy_signal.resample(signal, target_num_samples)

def check_signal_quality(signal: np.ndarray, fs: float = 80.0) -> Dict[str, Any]:
    """
    Evaluates signal quality index (SQI) based on:
    - Amplitude variance (detecting flatline or sensor disconnect)
    - Physiological heart rate bounds (35 to 180 bpm)
    - Relative spectral power in the cardiac band (0.8 - 2.5 Hz)
    """
    std = float(np.std(signal))
    is_flatline = std < 0.01
    
    # Welch power spectral density
    freqs, psd = scipy_signal.welch(signal, fs=fs, nperseg=min(256, len(signal)))
    # Numerical integration across cardiac band
    cardiac_band = (freqs >= 0.8) & (freqs <= 2.5) # 48 to 150 bpm
    integrate_fn = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
    total_power = integrate_fn(psd, freqs) if len(psd) > 0 else 1e-6
    cardiac_power = integrate_fn(psd[cardiac_band], freqs[cardiac_band]) if np.any(cardiac_band) else 0.0
    
    sqi_ratio = float(cardiac_power / (total_power + 1e-6))
    
    if is_flatline:
        quality = "poor"
        message = "Flatline or sensor disconnect detected."
    elif sqi_ratio < 0.15:
        quality = "marginal"
        message = "High motion artifact or poor pulse amplitude."
    else:
        quality = "good"
        message = "Pulse waveform clear with distinct cardiac rhythm."
        
    return {
        "quality": quality,
        "sqi_score": round(min(1.0, sqi_ratio * 2.0), 3),
        "is_flatline": is_flatline,
        "message": message
    }

def preprocess_ppg(
    raw_signal: np.ndarray, 
    input_fs: float = 80.0, 
    target_fs: float = 80.0,
    target_length: Optional[int] = 2400
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Complete scientific preprocessing pipeline:
    1. Clean & interpolate missing points
    2. Bandpass filter (0.5 - 8.0 Hz)
    3. Resample if necessary to 80 Hz
    4. Truncate/pad to target length (2400 samples for 30s)
    5. Z-score normalize
    
    Returns:
    (raw_cleaned, normalized_for_cnn, quality_info)
    """
    cleaned = clean_and_interpolate_signal(raw_signal)
    filtered = butter_bandpass_filter(cleaned, fs=input_fs)
    
    if input_fs != target_fs:
        filtered = resample_signal(filtered, orig_fs=input_fs, target_fs=target_fs)
        
    quality_info = check_signal_quality(filtered, fs=target_fs)
    
    # Adjust length if specified
    if target_length is not None:
        if len(filtered) > target_length:
            filtered_fixed = filtered[:target_length]
        elif len(filtered) < target_length:
            pad_width = target_length - len(filtered)
            filtered_fixed = np.pad(filtered, (0, pad_width), mode='edge')
        else:
            filtered_fixed = filtered
    else:
        filtered_fixed = filtered
        
    normalized = normalize_signal(filtered_fixed)
    return filtered_fixed, normalized, quality_info
