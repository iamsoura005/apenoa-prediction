# Dataset Documentation & Provenance

## 1. Primary Dataset Target: UCDDB (PhysioNet)
- **Database:** University College Dublin Sleep Apnea Database (UCDDB)
- **Host:** PhysioNet / PhysioNet Challenge
- **Subjects:** 25 adult subjects (21 male, 4 female) with suspected Obstructive Sleep Apnea.
- **Signals:** Overnight polysomnography recordings including PPG (Plethysmogram), EEG, EOG, EMG, ECG, nasal airflow, and pulse oximetry.
- **PPG Sampling Rate:** Recorded at 128 Hz or downsampled to 80 Hz across standard cohorts.

## 2. Legacy Pickle Dataset Status
- The original repository referenced `data/CNN_input_data.pickle` and `CNN_input_data.zip`, which were not tracked in git due to storage constraints.
- To ensure full reproducibility without external dependencies, this system includes:
  1. Automated generator for validated physiological benchmark epochs (`data/generate_samples.py`).
  2. Direct CSV, JSON, and TXT upload handlers capable of receiving raw clinical recordings from PhysioNet, hospital monitors, or wearables.

## 3. Data Integrity & Scientific Ethics
- Zero synthetic clinical labels are ever passed off as real patient data.
- Benchmark cases are clearly labeled by their origin and physiological profiles.

