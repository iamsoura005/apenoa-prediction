import os
import json
import numpy as np

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'sample_signals')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_sample_cases():
    fs = 80
    duration = 30
    n_samples = fs * duration
    t = np.linspace(0, duration, n_samples, endpoint=False)
    
    # -------------------------------------------------------------
    # Case 1: Healthy Baseline Normal (Normal Breathing, stable pulse)
    # -------------------------------------------------------------
    hr_normal = 70.0 / 60.0 # 1.167 Hz
    resp_rate = 14.0 / 60.0 # 0.233 Hz
    
    # Fundamental pulse + dicrotic notch harmonics
    pulse_normal = (
        np.sin(2 * np.pi * hr_normal * t) +
        0.50 * np.sin(4 * np.pi * hr_normal * t + 0.4) +
        0.25 * np.sin(6 * np.pi * hr_normal * t + 0.8) +
        0.10 * np.sin(8 * np.pi * hr_normal * t + 1.2)
    )
    # Natural respiratory-induced amplitude modulation (RIIV / RIAV)
    modulation_normal = 1.0 + 0.12 * np.sin(2 * np.pi * resp_rate * t)
    # Minor pink noise
    noise_normal = np.random.normal(0, 0.05, n_samples)
    sig_normal = (pulse_normal * modulation_normal + noise_normal) * 100.0 + 800.0
    
    # -------------------------------------------------------------
    # Case 2: Pre-Apnea Segment (Pre-apnea sympathetic surge & vasoconstriction)
    # Characteristics: Heart rate acceleration followed by deceleration, 
    # progressive amplitude reduction (peripheral vasoconstriction), increased low-frequency oscillation
    # -------------------------------------------------------------
    # Dynamic HR varying from 68 to 86 bpm (autonomic fluctuation)
    hr_inst = (75.0 + 10.0 * np.sin(2 * np.pi * 0.05 * t)) / 60.0
    phase_apnea = 2 * np.pi * np.cumsum(hr_inst) / fs
    
    pulse_apnea = (
        np.sin(phase_apnea) +
        0.40 * np.sin(2 * phase_apnea + 0.3) +
        0.15 * np.sin(3 * phase_apnea + 0.7)
    )
    # Vasoconstriction: amplitude drop towards the end of the window (prior to lead time)
    vasoconstriction_envelope = 1.0 - 0.40 * (t / duration)
    # Low frequency Mayer waves (0.1 Hz sympathetic oscillation)
    mayer_wave = 0.25 * np.sin(2 * np.pi * 0.09 * t)
    noise_apnea = np.random.normal(0, 0.08, n_samples)
    
    sig_apnea = (pulse_apnea * vasoconstriction_envelope + mayer_wave + noise_apnea) * 100.0 + 780.0
    
    # -------------------------------------------------------------
    # Case 3: Borderline / High Variability Case
    # -------------------------------------------------------------
    hr_borderline = 82.0 / 60.0
    pulse_borderline = (
        np.sin(2 * np.pi * hr_borderline * t) +
        0.45 * np.sin(4 * np.pi * hr_borderline * t + 0.5)
    )
    mod_border = 1.0 + 0.22 * np.sin(2 * np.pi * 0.18 * t)
    sig_borderline = (pulse_borderline * mod_border + np.random.normal(0, 0.06, n_samples)) * 95.0 + 810.0

    samples = [
        {
            "id": "sample_normal_baseline",
            "name": "Subject 01 - Normal Stable Baseline",
            "condition": "Normal",
            "expected_outcome": "NORMAL_BREATHING",
            "description": "30-second resting PPG with regular heart rate (~70 bpm), robust pulse amplitude, and healthy respiratory sinus arrhythmia (RSA).",
            "fs": 80,
            "duration_sec": 30,
            "signal": [round(float(v), 2) for v in sig_normal]
        },
        {
            "id": "sample_pre_apnea_onset",
            "name": "Subject 17 - Impending Obstructive Apnea Onset",
            "condition": "Pre-Apnea",
            "expected_outcome": "APNEA_EVENT",
            "description": "30-second pre-apnea epoch (5s lead time prior to desaturation). Shows autonomic instability, peripheral vasoconstrictive amplitude attenuation, and 0.1 Hz sympathetic modulation.",
            "fs": 80,
            "duration_sec": 30,
            "signal": [round(float(v), 2) for v in sig_apnea]
        },
        {
            "id": "sample_borderline_variability",
            "name": "Subject 25 - Borderline / Elevated Variability",
            "condition": "Borderline",
            "expected_outcome": "MODERATE_RISK",
            "description": "30-second epoch displaying elevated heart rate and pronounced respiratory modulation during light sleep.",
            "fs": 80,
            "duration_sec": 30,
            "signal": [round(float(v), 2) for v in sig_borderline]
        }
    ]

    for s in samples:
        filename = os.path.join(OUTPUT_DIR, f"{s['id']}.json")
        with open(filename, 'w') as f:
            json.dump(s, f, indent=2)
        print(f"Generated sample benchmark: {filename}")
        
        # Also export CSV for testing file upload functionality
        csv_filename = os.path.join(OUTPUT_DIR, f"{s['id']}.csv")
        np.savetxt(csv_filename, s['signal'], delimiter=",", fmt="%.2f", header="ppg_amplitude", comments="")
        print(f"Generated sample CSV: {csv_filename}")

if __name__ == '__main__':
    generate_sample_cases()

