# Model Provenance & Checkpoint Forensic Investigation

**Report Date:** 2026-11-13  
**Auditor:** Senior Forensic ML & Deep Learning Researcher  
**Target:** `ml/models/apnea_cnn_production.pt` & `my_final_apnea_predictor.keras`

---

## 1. Checkpoint Inventory & Tensor Comparison

| Attribute | `my_final_apnea_predictor.keras` | `my_final_apnea_predictor.pth` (Legacy) | `apnea_cnn_production.pt` (Production) |
|---|---|---|---|
| **File Format** | Keras 3 Zip Archive (`model.weights.h5`) | Raw PyTorch State Dict | PyTorch Dictionary with Metadata |
| **File Size** | 59,500,141 bytes (~56.7 MB) | 24,730,872 bytes (~23.5 MB) | 19,815,403 bytes (~18.9 MB) |
| **Save Timestamp** | `2025-11-03@14:28:55` | 2026-11-13 (Extracted) | 2026-11-13 (Direct Port) |
| **Input Shape** | `(None, 2400, 1)` [80 Hz $\times$ 30s] | `(1, 3000)` [100 Hz $\times$ 30s] | `(Batch, 1, 2400)` [80 Hz $\times$ 30s] |
| **Parameters** | 4,951,169 | 6,183,169 (Dimension mismatch) | 4,951,169 |
| **Optimizer State** | 34 Adam velocity/momentum tensors | None | Clean inference state |
| **BatchNorm State** | Genuine running mean/variance | `running_mean=0`, `running_var=1`, `num_batches=0` (Untrained) | Ported genuine running mean/variance |
| **Weight Status** | **Genuinely Trained** | **Uninitialized / Non-Functional** | **Authentic Ported Weights** |

---

## 2. Weight Tensor Statistics & Provenance Proof

Inspection of the internal weights inside `model.weights.h5`:
* **`conv1d`**: Shape `(7, 1, 32)` $\rightarrow$ Mean `-0.00608`, Std `0.09343`, Range `[-0.159, +0.171]`
* **`batch_normalization`**: $\gamma$ Mean `0.9995`, $\beta$ Mean `-0.0186`, Running Mean `0.0001`, Running Variance `0.0541`
* **`conv1d_1`**: Shape `(5, 32, 64)` $\rightarrow$ Mean `0.00128`, Std `0.06454`
* **`batch_normalization_1`**: Running Mean `0.0843`, Running Variance `0.4076`
* **`conv1d_2`**: Shape `(3, 64, 128)` $\rightarrow$ Mean `0.00172`, Std `0.05888`
* **`batch_normalization_2`**: Running Mean `0.1735`, Running Variance `0.9290`
* **`dense`**: Shape `(38400, 128)` $\rightarrow$ Mean `-0.00048`, Std `0.00741`
* **`dense_1` (Output Layer)**: Shape `(128, 1)` $\rightarrow$ Mean `0.00718`, Std `0.11517`, Bias `+0.00426113`
* **Optimizer Variables**: Variable `0` = `2,250.0` (indicates 2,250 optimizer iteration steps completed during training at learning rate `1e-4`).

---

## 3. Finding on Dynamic Range & Output Saturation
Forensic tracing of the forward pass revealed:
1. `fc1` (Dense 128) weights have a slight negative mean (`-0.00048`).
2. When passed standardized PPG waveforms with unit variance, the linear sum `fc1(fl)` produces negative or low positive values near the threshold.
3. For ~30% of inputs, all 128 neurons in `fc1` are zeroed out by ReLU, causing the final output to collapse directly onto the bias of `fc2` (`+0.00426113`).
4. Sigmoid of `+0.00426113` is `0.501065` $\approx$ `0.5011`.
5. Because the default classification threshold was set at exactly `0.50`, inputs that produced zero activation in `fc1` were automatically assigned `APNEA_EVENT` (50.1%).

### Classification
**Status:** **Genuinely Trained Checkpoint with Documented Provenance (Keras 3 $\rightarrow$ PyTorch)**, but exhibiting bounded dynamic range near the decision boundary.
