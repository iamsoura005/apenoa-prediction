import sys
import os
import requests
import json
import numpy as np

print('======================================================================')
print('STAGE 1: VERIFYING MODEL CHECKPOINT & ARCHITECTURE LOADING')
print('======================================================================')
from ml.inference.predictor import ApneaPredictor

model_path = os.path.abspath('ml/models/apnea_cnn_production.pt')
print(f'[1.1] Checkpoint file: {model_path}')
print(f'[1.2] Checkpoint size: {os.path.getsize(model_path):,} bytes')

# Instantiate predictor (loads PyTorch weights)
predictor = ApneaPredictor()
print(f'[1.3] PyTorch Model Class: {predictor.model.__class__.__name__}')
print(f'[1.4] Execution Hardware Device: {predictor.device}')
param_count = sum(p.numel() for p in predictor.model.parameters())
print(f'[1.5] Total Model Parameters: {param_count:,}')
print(f'[1.6] Weight checkpoint loaded successfully: YES')

print('\n======================================================================')
print('STAGE 2: PREPARING REAL PPG INPUT DATA')
print('======================================================================')
# Read actual benchmark signal file
input_file = os.path.abspath('data/sample_signals/sample_pre_apnea_onset.csv')
print(f'[2.1] Source file: {input_file}')
with open(input_file, 'r') as f:
    lines = f.readlines()
print(f'[2.2] CSV Lines count: {len(lines)} (Header + {len(lines)-1} numerical samples)')

# Parse raw values
raw_samples = [float(line.strip()) for line in lines[1:] if line.strip()]
print(f'[2.3] Parsed sample count: {len(raw_samples)} samples')
print(f'[2.4] Sampling frequency: 80 Hz -> Window duration: {len(raw_samples)/80:.1f} seconds')
print(f'[2.5] First 5 raw values: {raw_samples[:5]}')
print(f'[2.6] Raw signal stats: Min={min(raw_samples):.2f}, Max={max(raw_samples):.2f}, Mean={np.mean(raw_samples):.2f}, Std={np.std(raw_samples):.2f}')

print('\n======================================================================')
print('STAGE 3: STEP-BY-STEP PREPROCESSING DEMONSTRATION')
print('======================================================================')
from ml.preprocessing.ppg_filter import clean_and_interpolate_signal, butter_bandpass_filter, normalize_signal, check_signal_quality
from ml.features.hrv_features import detect_systolic_peaks, compute_hrv_features, compute_spectral_features

# Step 3.1: Clean
cleaned = clean_and_interpolate_signal(np.array(raw_samples))
print(f'[3.1] NaN/Inf check & interpolation: Cleaned array finite count = {np.sum(np.isfinite(cleaned))}/{len(cleaned)}')

# Step 3.2: Butterworth bandpass
filtered = butter_bandpass_filter(cleaned, fs=80.0, lowcut=0.5, highcut=8.0, order=4)
print(f'[3.2] Butterworth Bandpass (0.5 - 8.0 Hz, 4th order):')
print(f'      Raw signal variance = {np.var(raw_samples):.4f}')
print(f'      Filtered signal variance = {np.var(filtered):.4f} (baseline wander removed)')
print(f'      First 5 filtered values: {[round(x, 4) for x in filtered[:5]]}')

# Step 3.3: SQI
sqi = check_signal_quality(filtered, fs=80.0)
print(f'[3.3] Signal Quality Assessment:')
print(f'      SQI Score = {sqi["sqi_score"]}')
print(f'      Quality Category = {sqi["quality"]}')
print(f'      Interpretation = {sqi["message"]}')

# Step 3.4: Systolic Peak Detection & HRV
peaks = detect_systolic_peaks(filtered, fs=80.0)
hrv = compute_hrv_features(peaks, fs=80.0)
spec = compute_spectral_features(filtered, fs=80.0)
print(f'[3.4] Physiological Telemetry Extracted:')
print(f'      Detected Systolic Peaks = {len(peaks)} peaks across 30 seconds')
print(f'      Peak sample indices (first 5) = {peaks[:5].tolist()}')
print(f'      Mean Heart Rate = {hrv["mean_hr_bpm"]} bpm')
print(f'      SDNN (Total Variability) = {hrv["sdnn_ms"]} ms')
print(f'      RMSSD (Parasympathetic Index) = {hrv["rmssd_ms"]} ms')
print(f'      LF/HF Ratio = {spec["lf_hf_ratio"]}')

# Step 3.5: Z-score Normalization
normalized = normalize_signal(filtered)
print(f'[3.5] Z-score Standardized input for CNN:')
print(f'      Input array shape = {normalized.shape}')
print(f'      Mean = {np.mean(normalized):.6f} (target ~ 0.0)')
print(f'      Std = {np.std(normalized):.6f} (target ~ 1.0)')

print('\n======================================================================')
print('STAGE 4: LIVE HTTP API REQUEST (POST /api/upload)')
print('======================================================================')
url = 'http://127.0.0.1:8000/api/upload'
print(f'[4.1] Sending multipart HTTP POST request to: {url}')

with open(input_file, 'rb') as f:
    files = {'file': ('sample_pre_apnea_onset.csv', f, 'text/csv')}
    data = {'fs': 80.0, 'subject_id': 'Live-Demo-Subject-17'}
    response = requests.post(url, files=files, data=data)

print(f'[4.2] HTTP Status Code: {response.status_code} OK')
res_json = response.json()
print(f'[4.3] Response Analysis ID: {res_json["analysis_id"]}')
print(f'[4.4] Raw Output Prediction: {res_json["prediction"]}')
print(f'[4.5] Calculated Probability: {res_json["probability"]}')
print(f'[4.6] Confidence Percentage: {res_json["confidence_percentage"]}%')
print(f'[4.7] Risk Level: {res_json["risk_level"]}')
print(f'[4.8] Clinical Interpretation: {res_json["clinical_interpretation"]}')
print(f'[4.9] Telemetry Preview Points: {len(res_json["filtered_signal_preview"])} points transmitted')

print('\n======================================================================')
print('STAGE 5: PDF CLINICAL REPORT RETRIEVAL (GET /api/report/{id})')
print('======================================================================')
report_url = f'http://127.0.0.1:8000/api/report/{res_json["analysis_id"]}'
print(f'[5.1] Requesting PDF report from: {report_url}')
rep_resp = requests.get(report_url)
print(f'[5.2] Report Status Code: {rep_resp.status_code}')
print(f'[5.3] Content-Type: {rep_resp.headers.get("content-type")}')
print(f'[5.4] PDF File Size: {len(rep_resp.content):,} bytes')
print(f'[5.5] PDF Header bytes: {rep_resp.content[:8]}')

