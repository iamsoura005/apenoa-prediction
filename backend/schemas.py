from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class PredictRequest(BaseModel):
    signal: List[float] = Field(..., description="Array of PPG amplitude values (recommended 30s window)")
    fs: float = Field(default=80.0, description="Sampling rate of the signal in Hz")
    subject_id: Optional[str] = Field(default="Anonymous", description="Identifier for patient/recording")

class PreprocessRequest(BaseModel):
    signal: List[float] = Field(..., description="Raw PPG array")
    fs: float = Field(default=80.0, description="Sampling rate in Hz")

class PreprocessResponse(BaseModel):
    cleaned_signal: List[float]
    filtered_signal: List[float]
    normalized_signal: List[float]
    detected_peaks: List[int]
    sqi_info: Dict[str, Any]
    biomarkers: Dict[str, Any]
    samples_count: int
    duration_seconds: float

class BiomarkerSummary(BaseModel):
    mean_hr_bpm: float
    sdnn_ms: float
    rmssd_ms: float
    pnn50_pct: float
    cv: float
    peak_count: int

class SpectralSummary(BaseModel):
    lf_power: float
    hf_power: float
    lf_hf_ratio: float

class BiomarkersResponse(BaseModel):
    peaks: List[int]
    hrv: BiomarkerSummary
    spectral: SpectralSummary

class SignalQualityResponse(BaseModel):
    quality: str
    sqi_score: float
    is_flatline: bool
    message: str

class TimingParameters(BaseModel):
    window_duration_sec: int
    sampling_rate_hz: int
    lead_time_sec: int
    forecast_window_sec: int

class ModelTelemetryResponse(BaseModel):
    raw_logit: float
    decision_threshold: float
    is_near_boundary: bool

class PredictionResponse(BaseModel):
    analysis_id: str
    subject_id: str
    timestamp: str
    prediction: str
    probability: float
    confidence_percentage: float
    risk_level: str
    risk_color: str
    clinical_interpretation: str
    model_telemetry: Optional[ModelTelemetryResponse] = None
    signal_quality: SignalQualityResponse
    biomarkers: BiomarkersResponse
    timing_parameters: TimingParameters
    raw_signal_length: int
    filtered_signal_preview: List[float]
    peak_indices_preview: List[int]

class AnalysisHistoryItem(BaseModel):
    id: str
    subject_id: str
    timestamp: str
    prediction: str
    probability: float
    confidence_percentage: float
    risk_level: str
    risk_color: str
    sqi_quality: str
    sqi_score: float
    mean_hr_bpm: float
    sdnn_ms: float
    rmssd_ms: float
    has_report: bool

class ModelInfoResponse(BaseModel):
    model_name: str
    version: str
    framework: str
    architecture: str
    input_samples: int
    sampling_rate_hz: int
    window_seconds: int
    lead_time_seconds: int
    forecast_target_seconds: int
    weights_provenance: str
    training_steps: int
    total_parameters: int
    checkpoint_size_bytes: int
    evaluation_status: str
    checksum_sha256: str
