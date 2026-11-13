# Scientific Model Evaluation & Benchmark Metrics
**Model:** Apnea1DCNN (3-Stage 1D Convolutional Neural Network)  
**Trained Weights:** `ml/models/apnea_cnn_production.pt` & `experiments/gpu_run_01/best_model.pt`  
**Evaluation Harness:** Stratified Patient-Separated Partitions (Zero Window Overlap Data Leakage)  
**Hardware:** NVIDIA GeForce RTX 4050 6GB Laptop GPU (CUDA 12.4 + AMP)

---

## 1. Model Architecture Specification

The network processes continuous 30-second single-channel optical pulse segments ($N = 2,400$ samples @ 80 Hz):

```
Input: (Batch, 1, 2400)
 │
 ├── Stage 1: Conv1D(1 -> 32, kernel=7, stride=1, pad=3) + BatchNorm1d(32) + ReLU + MaxPool1d(2) + Dropout(0.2)
 │    └── Output: (Batch, 32, 1200)
 │
 ├── Stage 2: Conv1D(32 -> 64, kernel=5, stride=1, pad=2) + BatchNorm1d(64) + ReLU + MaxPool1d(2) + Dropout(0.2)
 │    └── Output: (Batch, 64, 600)
 │
 ├── Stage 3: Conv1D(64 -> 128, kernel=3, stride=1, pad=1) + BatchNorm1d(128) + ReLU + MaxPool1d(2) + Dropout(0.2)
 │    └── Output: (Batch, 128, 300)
 │
 ├── Flatten: 128 * 300 = 38,400 features
 │
 ├── Fully Connected 1: Linear(38400 -> 128) + ReLU + Dropout(0.5)
 │
 └── Fully Connected 2 (Logit Head): Linear(128 -> 1)
      └── Sigmoid: Probability of Impending Apnea Event
```

Total trainable parameters: **4,951,169**.

---

## 2. Independent Test Set Performance (GPU Training Run 01)

Evaluated on an independent patient cohort (144 windows across 6 completely unseen subjects):

| Metric | Score | Clinical Relevance |
|---|---|---|
| **Accuracy** | **83.33%** | Overall classification correctness across all windows |
| **Sensitivity (Recall)** | **83.33%** | Detection rate of true impending apnea events (critical to prevent false negatives) |
| **Specificity** | **83.33%** | Correct suppression of false alarms during normal rhythmic breathing |
| **Precision** | **78.12%** | Positive predictive value of triggered alerts |
| **F1-Score** | **0.8065** | Harmonic mean balancing recall and precision |
| **ROC-AUC** | **0.9444** | Area under the Receiver Operating Characteristic curve |
| **PR-AUC** | **0.9271** | Area under the Precision-Recall curve under class imbalance |

### Confusion Matrix
- **True Negatives (TN):** 70
- **False Positives (FP):** 14
- **False Negatives (FN):** 10
- **True Positives (TP):** 50

---

## 3. Threshold Calibration & Uncertainty Management

To maintain scientific rigor and prevent misleading binary classification on ambiguous recordings:

- **$P < 0.40$:** Confident Normal Breathing (Low Risk).
- **$0.40 \le P < 0.48$:** Mild Perturbation / Normal Rhythmic (Low-to-Moderate Risk).
- **$0.48 \le P \le 0.50$:** Equivocal / Borderline (Collapses to sub-threshold model bias; marked as unconfirmed).
- **$0.50 < P < 0.65$:** Elevated Sympathovagal Activity / Probable Disturbance (Moderate Risk).
- **$P \ge 0.65$:** Impending Obstructive Sleep Apnea Onset (High Risk).
