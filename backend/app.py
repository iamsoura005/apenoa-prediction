import os
import json
import uuid
import hashlib
import numpy as np
from datetime import datetime
from typing import List, Optional
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Response, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session

from ml.inference.predictor import ApneaPredictor
from ml.preprocessing.ppg_filter import preprocess_ppg, clean_and_interpolate_signal, butter_bandpass_filter, normalize_signal, check_signal_quality
from ml.features.hrv_features import extract_all_biomarkers, detect_systolic_peaks, compute_hrv_features, compute_spectral_features
from backend.database import get_db, AnalysisRecord
from backend.schemas import (
    PredictRequest, PredictionResponse, ModelInfoResponse,
    PreprocessRequest, PreprocessResponse, AnalysisHistoryItem
)
from backend.pdf_report import build_pdf_report

app = FastAPI(
    title="ApneaSphere Enterprise Biomedical Platform",
    description="Production-grade AI Platform for Predictive PPG Sleep Apnea Analysis and Telemetry",
    version="2.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize predictor singleton
predictor = ApneaPredictor()

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
FRONTEND_DIR = os.path.join(PROJECT_ROOT, "frontend")
SAMPLES_DIR = os.path.join(PROJECT_ROOT, "data", "sample_signals")

@app.get("/health")
@app.get("/api/health")
def health_check(db: Session = Depends(get_db)):
    # Verify DB connectivity
    db_ok = False
    try:
        db.execute(AnalysisRecord.__table__.select().limit(1))
        db_ok = True
    except Exception:
        db_ok = False
        
    return {
        "status": "healthy" if db_ok and predictor.model is not None else "degraded",
        "service": "ApneaSphere Enterprise Biomedical API",
        "timestamp": datetime.utcnow().isoformat(),
        "device": str(predictor.device),
        "model_loaded": predictor.model is not None,
        "database_connected": db_ok,
        "version": "2.0.0"
    }

@app.get("/api/model/status")
def get_model_status():
    total_params = sum(p.numel() for p in predictor.model.parameters()) if predictor.model else 0
    ckpt_path = predictor.model_path
    
    # Calculate sha256 checksum
    h = hashlib.sha256()
    if os.path.exists(ckpt_path):
        with open(ckpt_path, "rb") as f:
            while chunk := f.read(8192):
                h.update(chunk)
        checksum = h.hexdigest()
    else:
        checksum = "unknown"

    return {
        "model_loaded": predictor.model is not None,
        "model_version": "2.0.0",
        "architecture": "3-Stage Conv1D (32, 64, 128) + BatchNorm + MaxPool + Dropout + Dense(38400->128->1)",
        "device": str(predictor.device),
        "sampling_frequency_hz": 80,
        "window_duration_seconds": 30,
        "lead_time_seconds": 5,
        "prediction_threshold": 0.50,
        "validation_status": "Verified PyTorch Checkpoint (4,951,169 parameters)",
        "parameters": total_params,
        "checksum_sha256": checksum
    }

@app.get("/api/model/info", response_model=ModelInfoResponse)
def get_model_info():
    total_params = sum(p.numel() for p in predictor.model.parameters())
    ckpt_path = predictor.model_path
    ckpt_size = os.path.getsize(ckpt_path) if os.path.exists(ckpt_path) else 0
    
    # Calculate sha256 checksum
    h = hashlib.sha256()
    if os.path.exists(ckpt_path):
        with open(ckpt_path, "rb") as f:
            while chunk := f.read(8192):
                h.update(chunk)
        checksum = h.hexdigest()
    else:
        checksum = "unknown"
        
    return {
        "model_name": "Apnea1DCNN",
        "version": "2.0.0 (Production Checkpoint)",
        "framework": "PyTorch 2.x",
        "architecture": "3-Stage Conv1D (32, 64, 128) + BatchNorm + MaxPool + Dropout + Dense(38400->128->1)",
        "input_samples": 2400,
        "sampling_rate_hz": 80,
        "window_seconds": 30,
        "lead_time_seconds": 5,
        "forecast_target_seconds": 10,
        "weights_provenance": "Authentic weights recovered from trained Keras model (2,250 Adam training steps)",
        "training_steps": 2250,
        "total_parameters": total_params,
        "checkpoint_size_bytes": ckpt_size,
        "evaluation_status": "Category B: Trained checkpoint verified; independent external re-evaluation pending raw PhysioNet cohort access",
        "checksum_sha256": checksum
    }

@app.post("/api/signal/preprocess", response_model=PreprocessResponse)
def preprocess_signal_endpoint(req: PreprocessRequest):
    """
    Dedicated endpoint to test/inspect pure scientific signal processing
    without executing neural network inference.
    """
    if len(req.signal) < 80:
        raise HTTPException(status_code=400, detail="Signal too short. Minimum 1 second required.")
        
    raw_arr = np.array(req.signal, dtype=np.float32)
    cleaned = clean_and_interpolate_signal(raw_arr)
    filtered = butter_bandpass_filter(cleaned, fs=req.fs, lowcut=0.5, highcut=8.0, order=4)
    norm = normalize_signal(filtered)
    sqi = check_signal_quality(filtered, fs=req.fs)
    biomarkers = extract_all_biomarkers(filtered, fs=req.fs)
    
    return {
        "cleaned_signal": cleaned[::4].tolist(),
        "filtered_signal": filtered[::4].tolist(),
        "normalized_signal": norm[::4].tolist(),
        "detected_peaks": [int(p // 4) for p in biomarkers["peaks"] if p // 4 < len(filtered) // 4],
        "sqi_info": sqi,
        "biomarkers": biomarkers,
        "samples_count": len(raw_arr),
        "duration_seconds": round(len(raw_arr) / req.fs, 2)
    }

@app.get("/api/samples")
def list_sample_signals():
    samples = []
    if os.path.exists(SAMPLES_DIR):
        for f in os.listdir(SAMPLES_DIR):
            if f.endswith(".json"):
                try:
                    with open(os.path.join(SAMPLES_DIR, f), 'r') as fp:
                        data = json.load(fp)
                        samples.append({
                            "id": data.get("id"),
                            "name": data.get("name"),
                            "condition": data.get("condition"),
                            "expected_outcome": data.get("expected_outcome"),
                            "description": data.get("description"),
                            "fs": data.get("fs", 80),
                            "duration_sec": data.get("duration_sec", 30)
                        })
                except Exception:
                    continue
    return samples

@app.get("/api/samples/{sample_id}")
def get_sample_signal(sample_id: str):
    file_path = os.path.join(SAMPLES_DIR, f"{sample_id}.json")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Sample not found")
    with open(file_path, 'r') as fp:
        return json.load(fp)

@app.post("/api/predict", response_model=PredictionResponse)
def predict_signal(req: PredictRequest, db: Session = Depends(get_db)):
    if len(req.signal) < 80:
        raise HTTPException(status_code=400, detail="Signal too short. Minimum 1 second (80 samples) required.")
        
    result = predictor.predict_window(req.signal, fs=req.fs)
    analysis_id = str(uuid.uuid4())
    now = datetime.utcnow()
    
    # Save persistent record into database
    bm = result.get("biomarkers", {})
    hrv = bm.get("hrv", {})
    spec = bm.get("spectral", {})
    sqi = result.get("signal_quality", {})
    telem = result.get("model_telemetry", {})
    
    db_record = AnalysisRecord(
        id=analysis_id,
        subject_id=req.subject_id or "Anonymous",
        timestamp=now,
        prediction=result["prediction"],
        probability=result["probability"],
        confidence_percentage=result["confidence_percentage"],
        risk_level=result["risk_level"],
        risk_color=result["risk_color"],
        clinical_interpretation=result["clinical_interpretation"],
        raw_logit=telem.get("raw_logit"),
        decision_threshold=telem.get("decision_threshold", 0.50),
        is_near_boundary=telem.get("is_near_boundary", False),
        sqi_quality=sqi.get("quality", "good"),
        sqi_score=sqi.get("sqi_score", 1.0),
        mean_hr_bpm=hrv.get("mean_hr_bpm", 0.0),
        sdnn_ms=hrv.get("sdnn_ms", 0.0),
        rmssd_ms=hrv.get("rmssd_ms", 0.0),
        pnn50_pct=hrv.get("pnn50_pct", 0.0),
        cv=hrv.get("cv", 0.0),
        peak_count=hrv.get("peak_count", 0),
        lf_power=spec.get("lf_power", 0.0),
        hf_power=spec.get("hf_power", 0.0),
        lf_hf_ratio=spec.get("lf_hf_ratio", 0.0),
        raw_signal_length=len(req.signal),
        sampling_rate_hz=int(req.fs),
        signal_preview_json=json.dumps(result["filtered_signal_preview"]),
        peak_indices_json=json.dumps(result["peak_indices_preview"])
    )
    db.add(db_record)
    db.commit()
    
    response_payload = {
        "analysis_id": analysis_id,
        "subject_id": req.subject_id or "Anonymous",
        "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
        **result
    }
    return response_payload

@app.post("/api/upload")
async def upload_signal_file(
    file: UploadFile = File(...),
    fs: float = Form(80.0),
    subject_id: Optional[str] = Form("Uploaded-Patient"),
    db: Session = Depends(get_db)
):
    content = await file.read()
    filename = file.filename.lower()
    
    try:
        if filename.endswith(".json"):
            data = json.loads(content.decode("utf-8"))
            if isinstance(data, dict) and "signal" in data:
                signal_vals = data["signal"]
                fs = data.get("fs", fs)
                subject_id = data.get("name", subject_id)
            elif isinstance(data, list):
                signal_vals = data
            else:
                raise ValueError("JSON must contain an array or an object with 'signal' key.")
        else:
            text = content.decode("utf-8")
            lines = [l.strip() for l in text.splitlines() if l.strip()]
            signal_vals = []
            for line in lines:
                parts = line.split(",")
                for p in parts:
                    try:
                        signal_vals.append(float(p))
                    except ValueError:
                        continue
                        
        if len(signal_vals) < 80:
            raise HTTPException(status_code=400, detail="File did not contain enough valid numerical samples (>80).")
            
        req = PredictRequest(signal=signal_vals, fs=fs, subject_id=subject_id)
        return predict_signal(req, db=db)
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse signal file: {str(e)}")

@app.get("/api/history", response_model=List[AnalysisHistoryItem])
def get_analysis_history(
    limit: int = Query(default=20, le=100),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db)
):
    """
    Returns authentic persistent historical records from SQLite database.
    """
    records = db.query(AnalysisRecord).order_by(AnalysisRecord.timestamp.desc()).offset(offset).limit(limit).all()
    results = []
    for r in records:
        results.append({
            "id": r.id,
            "subject_id": r.subject_id,
            "timestamp": r.timestamp.strftime("%Y-%m-%d %H:%M:%S") if r.timestamp else "",
            "prediction": r.prediction,
            "probability": r.probability,
            "confidence_percentage": r.confidence_percentage,
            "risk_level": r.risk_level,
            "risk_color": r.risk_color,
            "sqi_quality": r.sqi_quality,
            "sqi_score": r.sqi_score,
            "mean_hr_bpm": r.mean_hr_bpm,
            "sdnn_ms": r.sdnn_ms,
            "rmssd_ms": r.rmssd_ms,
            "has_report": True
        })
    return results

@app.get("/api/report/{analysis_id}")
def download_pdf_report(analysis_id: str, db: Session = Depends(get_db)):
    """
    Fetches the persistent analysis record from SQLite and compiles a clinical PDF.
    """
    record = db.query(AnalysisRecord).filter(AnalysisRecord.id == analysis_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="Analysis record not found in database.")
        
    # Reconstruct data dict for PDF builder
    analysis_data = {
        "analysis_id": record.id,
        "subject_id": record.subject_id,
        "timestamp": record.timestamp.strftime("%Y-%m-%d %H:%M:%S") if record.timestamp else "",
        "prediction": record.prediction,
        "probability": record.probability,
        "confidence_percentage": record.confidence_percentage,
        "risk_level": record.risk_level,
        "risk_color": record.risk_color,
        "clinical_interpretation": record.clinical_interpretation,
        "signal_quality": {
            "quality": record.sqi_quality,
            "sqi_score": record.sqi_score,
            "message": f"Signal quality evaluated as {record.sqi_quality.upper()}"
        },
        "biomarkers": {
            "hrv": {
                "mean_hr_bpm": record.mean_hr_bpm,
                "sdnn_ms": record.sdnn_ms,
                "rmssd_ms": record.rmssd_ms,
                "pnn50_pct": record.pnn50_pct,
                "cv": record.cv,
                "peak_count": record.peak_count
            },
            "spectral": {
                "lf_power": record.lf_power,
                "hf_power": record.hf_power,
                "lf_hf_ratio": record.lf_hf_ratio
            }
        }
    }
    
    pdf_bytes = build_pdf_report(analysis_data)
    
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename=apneasphere_report_{analysis_id[:8]}.pdf"}
    )

if os.path.exists(FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
