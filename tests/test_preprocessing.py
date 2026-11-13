"""
Unit tests for scientific signal preprocessing and biomarker extraction.
"""
import pytest
import numpy as np
from ml.preprocessing.ppg_filter import (
    clean_and_interpolate_signal,
    butter_bandpass_filter,
    normalize_signal,
    check_signal_quality,
    preprocess_ppg
)
from ml.features.hrv_features import (
    detect_systolic_peaks,
    compute_hrv_features,
    compute_spectral_features
)

def test_clean_and_interpolate_nan():
    # Signal with NaNs and Infs
    arr = np.array([1.0, np.nan, 3.0, np.inf, 5.0])
    cleaned = clean_and_interpolate_signal(arr)
    assert not np.any(np.isnan(cleaned))
    assert not np.any(np.isinf(cleaned))
    assert len(cleaned) == 5

def test_butter_bandpass_filtering():
    fs = 80.0
    t = np.linspace(0, 10, int(10 * fs), endpoint=False)
    # 0.1 Hz DC drift + 1.2 Hz pulse + 25 Hz noise
    drift = 5.0 * np.sin(2 * np.pi * 0.1 * t)
    pulse = 1.0 * np.sin(2 * np.pi * 1.2 * t)
    noise = 0.5 * np.sin(2 * np.pi * 25.0 * t)
    raw = drift + pulse + noise
    
    filtered = butter_bandpass_filter(raw, fs=fs, lowcut=0.5, highcut=8.0)
    assert len(filtered) == len(raw)
    # Variance of filtered should be significantly lower than raw due to drift/noise attenuation
    assert np.std(filtered) < np.std(raw)

def test_z_score_normalization():
    arr = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    norm = normalize_signal(arr)
    assert np.isclose(np.mean(norm), 0.0, atol=1e-5)
    assert np.isclose(np.std(norm), 1.0, atol=1e-5)

def test_peak_detection_and_hrv():
    fs = 80.0
    t = np.linspace(0, 30, int(30 * fs), endpoint=False)
    # Regular 60 bpm pulse (1 Hz -> 30 cycles)
    sig = np.sin(2 * np.pi * 1.0 * t)
    peaks = detect_systolic_peaks(sig, fs=fs)
    assert len(peaks) >= 28 and len(peaks) <= 32
    
    hrv = compute_hrv_features(peaks, fs=fs)
    assert np.isclose(hrv["mean_hr_bpm"], 60.0, atol=2.0)
    assert hrv["sdnn_ms"] >= 0.0

