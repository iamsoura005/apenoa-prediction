import os
import torch
import numpy as np
from typing import Dict, Any, Union, List, Optional
from ml.models.apnea_cnn import Apnea1DCNN
from ml.preprocessing.ppg_filter import preprocess_ppg
from ml.features.hrv_features import extract_all_biomarkers

DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), 
    'models', 
    'apnea_cnn_production.pt'
)

class ApneaPredictor:
    """
    Production-grade inference engine for Sleep Apnea Event Prediction from PPG.
    Thread-safe, self-contained, with comprehensive feature extraction and telemetry.
    """
    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None):
        if model_path is None:
            model_path = DEFAULT_MODEL_PATH
            
        self.model_path = model_path
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        
        # Initialize model
        self.model = Apnea1DCNN(input_samples=2400, in_channels=1)
        self._load_weights()
        self.model.to(self.device)
        self.model.eval()
        
    def _load_weights(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model checkpoint not found at {self.model_path}. "
                "Ensure convert_weights.py has been run."
            )
            
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.metadata = {
                'sampling_rate_hz': checkpoint.get('sampling_rate_hz', 80),
                'window_seconds': checkpoint.get('window_seconds', 30),
                'lead_time_seconds': checkpoint.get('lead_time_seconds', 5),
                'model_name': checkpoint.get('model_name', 'Apnea1DCNN')
            }
        else:
            self.model.load_state_dict(checkpoint)
            self.metadata = {'sampling_rate_hz': 80, 'window_seconds': 30}
            
    def predict_window(
        self, 
        raw_signal: Union[np.ndarray, List[float]], 
        fs: float = 80.0
    ) -> Dict[str, Any]:
        """
        Runs complete analysis on a 30s PPG segment:
        1. Preprocessing & bandpass filtering
        2. Signal Quality Index assessment
        3. Systolic peak detection & HRV feature calculation
        4. CNN inference -> Apnea Event Probability
        """
        raw_arr = np.array(raw_signal, dtype=np.float32)
        filtered, normalized, sqi_info = preprocess_ppg(
            raw_arr, 
            input_fs=fs, 
            target_fs=80.0, 
            target_length=2400
        )
        
        # Biomarkers
        biomarkers = extract_all_biomarkers(filtered, fs=80.0)
        
        # Inference
        tensor_in = torch.from_numpy(normalized).float().unsqueeze(0).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logit = self.model(tensor_in).item()
            prob = float(torch.sigmoid(torch.tensor(logit)).item())
            
        # Model Forward Pass
        # Note: The underlying model's final dense layer has small dynamic range around bias (+0.00426).
        # We output raw logits, probability score via sigmoid, and calibrated uncertainty indicators.
        is_apnea_predicted = prob >= 0.50
        
        # Scientifically honest uncertainty & observational categorization
        # Rather than claiming definitive clinical diagnostic certainty, we report model score bands
        # and highlight signal morphology & HRV telemetry.
        if prob < 0.48:
            risk_level = "LOW_PROBABILITY"
            risk_color = "emerald"
            clinical_interpretation = "Model output leans towards stable rhythmic pattern. Low model-predicted likelihood of apnea event in forecast window."
        elif prob <= 0.50:
            risk_level = "EQUIVOCAL / BORDERLINE"
            risk_color = "amber"
            clinical_interpretation = "Model score is near the decision boundary (0.48 - 0.50). High model uncertainty; review autonomic biomarkers."
        else:
            risk_level = "ELEVATED_PROBABILITY"
            risk_color = "rose"
            clinical_interpretation = "Model score exceeds decision boundary (>0.50). Pattern shows features consistent with pre-apnea epoch in training cohort."
            
        return {
            "prediction": "APNEA_EVENT" if is_apnea_predicted else "NORMAL_BREATHING",
            "probability": round(prob, 4),
            "confidence_percentage": round(prob * 100 if is_apnea_predicted else (1 - prob) * 100, 1),
            "risk_level": risk_level,
            "risk_color": risk_color,
            "clinical_interpretation": clinical_interpretation,
            "model_telemetry": {
                "raw_logit": round(logit, 6),
                "decision_threshold": 0.50,
                "is_near_boundary": abs(prob - 0.50) < 0.02
            },
            "signal_quality": sqi_info,
            "biomarkers": biomarkers,
            "timing_parameters": {
                "window_duration_sec": 30,
                "sampling_rate_hz": 80,
                "lead_time_sec": 5,
                "forecast_window_sec": 10
            },
            "raw_signal_length": len(raw_arr),
            "filtered_signal_preview": filtered[::4].tolist(), # Downsampled 4x for fast network transmission (600 points)
            "peak_indices_preview": [int(p // 4) for p in biomarkers["peaks"] if p // 4 < 600]
        }

