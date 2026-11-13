# Model Card: Apnea1DCNN

## Model Overview
- **Model Name:** Apnea1DCNN
- **Version:** 1.0.0 (PyTorch Production)
- **Model Type:** 1D Convolutional Neural Network
- **Input Dimension:** `(Batch, 1, 2400)` [30s window @ 80 Hz]
- **Output:** Scalar logit / Probability `[0.0, 1.0]`

## Architecture Summary
| Layer | Operation | Output Shape | Parameters |
|---|---|---|---|
| Input | 30s PPG signal | `(B, 1, 2400)` | 0 |
| Conv Block 1 | Conv1d(32, k=7, p=3) + BN + ReLU + MaxPool(2) + Dropout(0.3) | `(B, 32, 1200)` | 288 |
| Conv Block 2 | Conv1d(64, k=5, p=2) + BN + ReLU + MaxPool(2) + Dropout(0.3) | `(B, 64, 600)` | 10,432 |
| Conv Block 3 | Conv1d(128, k=3, p=1) + BN + ReLU + MaxPool(2) + Dropout(0.3) | `(B, 128, 300)` | 24,960 |
| Flatten | Reshape | `(B, 38400)` | 0 |
| Fully Connected 1 | Linear(38400, 128) + ReLU + Dropout(0.5) | `(B, 128)` | 4,915,328 |
| Fully Connected 2 | Linear(128, 1) | `(B, 1)` | 129 |
| **Total Parameters** | | | **4,942,753** |

## Weights Provenance & Verification
The model weights were recovered directly from `my_final_apnea_predictor.keras` (which contained genuine weights trained on 80 Hz PPG signals) and ported tensor-by-tensor into PyTorch format (`ml/models/apnea_cnn_production.pt`). 

## Ethical Considerations & Limitations
- **Investigational Use Only:** This model is designed for clinical research and decision support. It does not replace nocturnal Polysomnography (PSG).
- **Signal Quality Sensitivity:** Motion artifacts or sensor detachment degrade performance; always review the Signal Quality Index (SQI) before interpreting risk scores.

