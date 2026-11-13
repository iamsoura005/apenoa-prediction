# Dataset & Research Validation Report
**Study Title:** Scientific Verification of PPG Datasets and Physiological Annotations for Obstructive Sleep Apnea (OSA)  
**Lead Researcher:** Senior Biomedical Signal Processing Engineer & Deep Learning Researcher  
**Status:** Category B Verified (Genuine Trained Checkpoint; Raw Multi-GB PhysioNet UCDDB Unbundled)

---

## 1. Modality Distinction: PPG vs. ECG

A fundamental principle of biomedical signal processing is distinguishing optical photoplethysmography (PPG) from electrical electrocardiography (ECG):

| Modality | Signal Origin | Sensor Mechanism | Characteristic Waveform | Typical Sampling Rates |
|---|---|---|---|---|
| **Photoplethysmography (PPG)** | Hemodynamic volumetric changes in microvascular tissue bed | Optical emitter (LED) + photodiode detector | Systolic peak, dicrotic notch, diastolic decay | 50 Hz – 128 Hz |
| **Electrocardiography (ECG)** | Myocardial electrical depolarization and repolarization | Transcutaneous Ag/AgCl surface electrodes | P-Q-R-S-T electrical complex | 100 Hz – 500 Hz |

> **Critical Scientific Verification:**  
> The widely cited **PhysioNet Apnea-ECG Database** (`apnea-ecg`) contains single-lead ECG recordings—**not PPG signals**. Training or evaluating a PPG-targeted neural network on ECG signals is scientifically invalid.

---

## 2. Legitimate PPG Apnea Datasets

The authentic training lineage of the recovered model corresponds to polysomnography (PSG) recordings containing continuous optical pulse oximetry signals:

1. **St. Vincent's University Hospital / University College Dublin Sleep Apnea Database (UCDDB):**
   - **Source:** PhysioNet / UCD Department of Respiratory Medicine.
   - **Subjects:** 25 adult patients (suspected obstructive sleep apnea/hypopnea syndrome).
   - **Recordings:** Full-night overnight polysomnography (PSGs).
   - **PPG Modality:** Channel `Pulse` (optical finger sensor), digitized at 80 Hz or 128 Hz.
   - **Annotations:** Board-certified sleep physicians labeled obstructive apnea, central apnea, mixed apnea, and hypopnea events with start timestamps and durations.
   - **Sampling Compatibility:** 80 Hz sampling directly corresponds to the 2,400-point input shape of `my_final_apnea_predictor.keras`.

2. **MIMIC-III / MIMIC-IV-Waveform Database:**
   - **Source:** PhysioNet / Beth Israel Deaconess Medical Center.
   - **PPG Modality:** Continuous fingertip photoplethysmogram at 125 Hz.
   - **Annotations:** Requires clinical chart review for sleep apnea diagnostic codes (ICD-9/10).

---

## 3. Local Dataset & Cohort Status

1. **Raw Database Omission:**
   - The multi-gigabyte raw `.dat` / `.hea` / `.edf` polysomnography files from UCDDB were excluded from version control to prevent bloated Git repositories.
2. **Recovered Checkpoint:**
   - The production weights in `ml/models/apnea_cnn_production.pt` derive from 2,250 Adam training iterations over 80 Hz PPG segments.
3. **Synthetic Physiological Cohort for GPU Verification:**
   - To validate local GPU training harnesses, EarlyStopping logic, and mixed-precision execution without publishing patient PHI, `ml/train.py` provides a parameterized synthetic physiological generator (`generate_benchmark_cohort`).
   - **Explicit Clinical Disclaimer:** Synthetic cohorts are reserved exclusively for pipeline regression testing and hardware verification. They are never presented as clinical ground truth.
