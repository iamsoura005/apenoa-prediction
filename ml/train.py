import os
import sys
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Ensure ml package is on path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ml.models.apnea_cnn import Apnea1DCNN
from ml.preprocessing.ppg_filter import preprocess_ppg

class PPGDataset(Dataset):
    def __init__(self, signals: np.ndarray, labels: np.ndarray):
        """
        signals: shape (N, 2400)
        labels: shape (N,)
        """
        self.signals = torch.tensor(signals, dtype=torch.float32).unsqueeze(1) # (N, 1, 2400)
        self.labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)  # (N, 1)

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        return self.signals[idx], self.labels[idx]

def generate_benchmark_cohort(n_subjects=30, windows_per_subject=20, fs=80, duration=30, seed=42):
    """
    Generates a structured, multi-subject benchmark physiological cohort
    with subject-level separation to guarantee zero data leakage between splits.
    
    NOTE: Explicitly flagged as synthetic physiological cohort for pipeline verification,
    architecture benchmarking, and GPU training harness validation.
    """
    np.random.seed(seed)
    n_samples = fs * duration
    t = np.linspace(0, duration, n_samples, endpoint=False)
    
    signals = []
    labels = []
    subject_ids = []
    
    for sub in range(n_subjects):
        # Patient specific baseline physiological parameters
        base_hr = np.random.uniform(60.0, 85.0) / 60.0
        base_resp = np.random.uniform(12.0, 18.0) / 60.0
        patient_vessel_stiffness = np.random.uniform(0.3, 0.6)
        
        for w in range(windows_per_subject):
            is_apnea = 1 if np.random.rand() > 0.65 else 0 # 35% positive class
            
            if is_apnea:
                # Pre-apnea sympathetic acceleration and peripheral vasoconstriction
                hr_mod = base_hr * (1.0 + 0.18 * np.sin(2 * np.pi * 0.04 * t))
                phase = 2 * np.pi * np.cumsum(hr_mod) / fs
                pulse = (
                    np.sin(phase) +
                    patient_vessel_stiffness * np.sin(2 * phase + 0.4) +
                    0.15 * np.sin(3 * phase + 0.8)
                )
                vasoconstriction = 1.0 - 0.35 * (t / duration)
                mayer_wave = 0.20 * np.sin(2 * np.pi * 0.10 * t)
                noise = np.random.normal(0, 0.06, n_samples)
                raw_sig = (pulse * vasoconstriction + mayer_wave + noise) * 100.0 + 800.0
            else:
                # Normal stable breathing with respiratory sinus arrhythmia
                pulse = (
                    np.sin(2 * np.pi * base_hr * t) +
                    patient_vessel_stiffness * np.sin(4 * np.pi * base_hr * t + 0.3) +
                    0.20 * np.sin(6 * np.pi * base_hr * t + 0.7)
                )
                rsa = 1.0 + 0.10 * np.sin(2 * np.pi * base_resp * t)
                noise = np.random.normal(0, 0.04, n_samples)
                raw_sig = (pulse * rsa + noise) * 100.0 + 800.0
                
            _, normalized, _ = preprocess_ppg(raw_sig, input_fs=fs, target_fs=fs, target_length=2400)
            signals.append(normalized)
            labels.append(is_apnea)
            subject_ids.append(sub)
            
    return np.array(signals, dtype=np.float32), np.array(labels, dtype=np.float32), np.array(subject_ids)

def train_epoch(model, dataloader, criterion, optimizer, scaler, device, use_amp):
    model.train()
    total_loss = 0.0
    for x_batch, y_batch in dataloader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        optimizer.zero_grad()
        
        if use_amp and device.type == 'cuda':
            with torch.amp.autocast('cuda'):
                logits = model(x_batch)
                loss = criterion(logits, y_batch)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            
        total_loss += loss.item() * len(y_batch)
    return total_loss / len(dataloader.dataset)

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            total_loss += loss.item() * len(y_batch)
            
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            preds = (probs >= 0.50).astype(int)
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_targets.extend(y_batch.cpu().numpy().flatten())
            
    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    acc = accuracy_score(all_targets, all_preds)
    prec = precision_score(all_targets, all_preds, zero_division=0)
    rec = recall_score(all_targets, all_preds, zero_division=0)
    f1 = f1_score(all_targets, all_preds, zero_division=0)
    
    try:
        roc_auc = roc_auc_score(all_targets, all_probs)
    except Exception:
        roc_auc = 0.5
        
    try:
        pr_auc = average_precision_score(all_targets, all_probs)
    except Exception:
        pr_auc = 0.5
        
    cm = confusion_matrix(all_targets, all_preds).tolist()
    
    # Specificity = TN / (TN + FP)
    tn = cm[0][0] if len(cm) > 0 and len(cm[0]) > 0 else 0
    fp = cm[0][1] if len(cm) > 0 and len(cm[0]) > 1 else 0
    fn = cm[1][0] if len(cm) > 1 and len(cm[1]) > 0 else 0
    tp = cm[1][1] if len(cm) > 1 and len(cm[1]) > 1 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    return {
        "loss": total_loss / len(dataloader.dataset),
        "accuracy": float(acc),
        "precision": float(prec),
        "recall_sensitivity": float(rec),
        "specificity": float(spec),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "confusion_matrix": cm,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "probs": all_probs.tolist(),
        "targets": all_targets.tolist()
    }

def run_training(args):
    print("=" * 80)
    print(" ApneaSphere Production 1D-CNN Training & GPU Acceleration Engine")
    print("=" * 80)
    
    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        
    # Device configuration
    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda:0")
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f" [HARDWARE] CUDA Accelerated Engine ACTIVE")
        print(f" [HARDWARE] GPU: {gpu_name}")
        print(f" [HARDWARE] VRAM: {vram_gb:.2f} GB")
        use_amp = args.mixed_precision
    else:
        device = torch.device("cpu")
        print(f" [HARDWARE] Running on CPU (CUDA unavailable or disabled)")
        use_amp = False
        
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Data Preparation: Patient-level split
    print(f"\n [DATA] Generating patient-separated benchmark physiological cohort...")
    signals, labels, subjects = generate_benchmark_cohort(
        n_subjects=args.n_subjects, 
        windows_per_subject=args.windows_per_subject, 
        seed=args.seed
    )
    
    unique_subs = np.unique(subjects)
    n_subs = len(unique_subs)
    train_subs = unique_subs[:int(0.70 * n_subs)]
    val_subs = unique_subs[int(0.70 * n_subs):int(0.85 * n_subs)]
    test_subs = unique_subs[int(0.85 * n_subs):]
    
    train_mask = np.isin(subjects, train_subs)
    val_mask = np.isin(subjects, val_subs)
    test_mask = np.isin(subjects, test_subs)
    
    X_train, y_train = signals[train_mask], labels[train_mask]
    X_val, y_val = signals[val_mask], labels[val_mask]
    X_test, y_test = signals[test_mask], labels[test_mask]
    
    print(f" [DATA] Train set: {len(X_train)} windows across {len(train_subs)} patients (Positives: {int(y_train.sum())})")
    print(f" [DATA] Val set:   {len(X_val)} windows across {len(val_subs)} patients (Positives: {int(y_val.sum())})")
    print(f" [DATA] Test set:  {len(X_test)} windows across {len(test_subs)} patients (Positives: {int(y_test.sum())})")
    print(f" [INTEGRITY] Patient-separated partitions: ZERO window-overlap data leakage guaranteed.")
    
    train_loader = DataLoader(PPGDataset(X_train, y_train), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(PPGDataset(X_val, y_val), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(PPGDataset(X_test, y_test), batch_size=args.batch_size, shuffle=False)
    
    # Model Initialization
    model = Apnea1DCNN(input_samples=2400, in_channels=1)
    model.to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n [MODEL] Initialized Apnea1DCNN with {total_params:,} trainable parameters.")
    
    # Class imbalance weighting
    pos_weight = torch.tensor([(len(y_train) - y_train.sum()) / max(y_train.sum(), 1.0)]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp and device.type == 'cuda')
    
    best_val_loss = float('inf')
    best_checkpoint_path = os.path.join(args.output_dir, "best_model.pt")
    history = {"train_loss": [], "val_loss": [], "val_f1": [], "val_roc_auc": [], "epoch_times": []}
    
    patience_counter = 0
    print("\n [TRAINING] Beginning Training Loop with Early Stopping & AMP...")
    start_train_time = time.time()
    
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, criterion, optimizer, scaler, device, use_amp)
        val_metrics = evaluate(model, val_loader, criterion, device)
        elapsed = time.time() - t0
        
        scheduler.step(val_metrics["loss"])
        
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["val_f1"].append(val_metrics["f1"])
        history["val_roc_auc"].append(val_metrics["roc_auc"])
        history["epoch_times"].append(elapsed)
        
        # GPU Memory telemetry
        mem_info = ""
        if device.type == 'cuda':
            alloc = torch.cuda.memory_allocated(0) / (1024**2)
            max_alloc = torch.cuda.max_memory_allocated(0) / (1024**2)
            mem_info = f" | VRAM: {alloc:.1f}MB (Peak: {max_alloc:.1f}MB)"
            
        print(f" Epoch [{epoch:02d}/{args.epochs:02d}] - Train Loss: {train_loss:.4f} | Val Loss: {val_metrics['loss']:.4f} | Val F1: {val_metrics['f1']:.3f} | Val AUC: {val_metrics['roc_auc']:.3f} | {elapsed:.2f}s{mem_info}")
        
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_metrics["loss"],
                "val_metrics": val_metrics,
                "model_name": "Apnea1DCNN",
                "sampling_rate_hz": 80,
                "window_seconds": 30,
                "lead_time_seconds": 5,
                "total_params": total_params,
                "training_device": str(device)
            }, best_checkpoint_path)
            print(f"   -> [CHECKPOINT] Saved new best validation checkpoint to {best_checkpoint_path}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\n [EARLY STOPPING] Validation loss stopped improving for {args.patience} epochs. Terminating early at epoch {epoch}.")
                break
                
    total_time = time.time() - start_train_time
    print(f"\n [COMPLETE] Training finished in {total_time:.2f}s.")
    
    # Independent Test Evaluation
    print(f"\n [EVALUATION] Loading best model checkpoint and evaluating on independent test set...")
    ckpt = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    test_metrics = evaluate(model, test_loader, criterion, device)
    
    print("\n" + "=" * 80)
    print(" INDEPENDENT TEST SET EVALUATION REPORT")
    print("=" * 80)
    print(f" Test Accuracy:          {test_metrics['accuracy'] * 100:.2f}%")
    print(f" Test Sensitivity (Rec): {test_metrics['recall_sensitivity'] * 100:.2f}%")
    print(f" Test Specificity:       {test_metrics['specificity'] * 100:.2f}%")
    print(f" Test Precision:         {test_metrics['precision'] * 100:.2f}%")
    print(f" Test F1 Score:          {test_metrics['f1']:.4f}")
    print(f" Test ROC-AUC:           {test_metrics['roc_auc']:.4f}")
    print(f" Test PR-AUC:            {test_metrics['pr_auc']:.4f}")
    print(f" Confusion Matrix:       TN={test_metrics['tn']}, FP={test_metrics['fp']}, FN={test_metrics['fn']}, TP={test_metrics['tp']}")
    print("=" * 80)
    
    # Save artifacts
    # 1. Training history JSON
    history_path = os.path.join(args.output_dir, "training_history.json")
    with open(history_path, "w") as fp:
        json.dump({
            "history": history,
            "test_metrics": {k: v for k, v in test_metrics.items() if k not in ["probs", "targets"]},
            "hyperparameters": vars(args),
            "device": str(device),
            "vram_gb": vram_gb if device.type == 'cuda' else None,
            "total_time_seconds": total_time
        }, fp, indent=2)
        
    # 2. Plot curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(history["train_loss"], label="Train Loss", color="#38bdf8", lw=2)
    axes[0].plot(history["val_loss"], label="Val Loss", color="#f43f5e", lw=2)
    axes[0].set_title("Training & Validation Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Binary Cross Entropy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(history["val_f1"], label="Val F1 Score", color="#10b981", lw=2)
    axes[1].plot(history["val_roc_auc"], label="Val ROC-AUC", color="#a855f7", lw=2)
    axes[1].set_title("Validation F1 & ROC-AUC Progression")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    loss_curve_path = os.path.join(args.output_dir, "loss_curve.png")
    plt.savefig(loss_curve_path, dpi=200)
    plt.close()
    
    print(f" [ARTIFACT] Saved metrics to {history_path}")
    print(f" [ARTIFACT] Saved loss curves to {loss_curve_path}")
    return test_metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ApneaSphere Production GPU Training Harness")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (optimized for 6GB VRAM)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--patience", type=int, default=4, help="Early stopping patience")
    parser.add_argument("--n-subjects", type=int, default=35, help="Number of cohort subjects")
    parser.add_argument("--windows-per-subject", type=int, default=24, help="Windows per subject")
    parser.add_argument("--seed", type=int, default=42, help="Reproducibility seed")
    parser.add_argument("--mixed-precision", action="store_true", default=True, help="Enable AMP mixed precision")
    parser.add_argument("--device", type=str, default="cuda", help="Target device (cuda or cpu)")
    parser.add_argument("--output-dir", type=str, default="experiments/gpu_run_01", help="Output directory")
    
    args = parser.parse_args()
    run_training(args)
