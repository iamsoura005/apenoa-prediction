"""
Unit and integration tests for ApneaPredictor, Database, and FastAPI endpoints.
"""
import pytest
import numpy as np
from fastapi.testclient import TestClient
from ml.inference.predictor import ApneaPredictor
from backend.app import app
from backend.database import SessionLocal, AnalysisRecord

client = TestClient(app)

def test_predictor_initialization_and_forward():
    predictor = ApneaPredictor()
    assert predictor.model is not None
    
    # 30-second signal at 80 Hz
    test_sig = np.sin(np.linspace(0, 30 * 2 * np.pi, 2400))
    res = predictor.predict_window(test_sig, fs=80.0)
    
    assert "prediction" in res
    assert res["prediction"] in ["APNEA_EVENT", "NORMAL_BREATHING"]
    assert 0.0 <= res["probability"] <= 1.0
    assert "biomarkers" in res
    assert "hrv" in res["biomarkers"]
    assert "model_telemetry" in res

def test_api_health_endpoint():
    resp = client.get("/api/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["database_connected"] is True
    
    # Also test /health alias
    resp_root = client.get("/health")
    assert resp_root.status_code == 200
    assert resp_root.json()["status"] == "healthy"

def test_api_model_status():
    resp = client.get("/api/model/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["model_loaded"] is True
    assert data["sampling_frequency_hz"] == 80
    assert data["window_duration_seconds"] == 30
    assert data["parameters"] > 1_000_000
    assert "checksum_sha256" in data

def test_api_model_info():
    resp = client.get("/api/model/info")
    assert resp.status_code == 200
    data = resp.json()
    assert data["model_name"] == "Apnea1DCNN"
    assert data["sampling_rate_hz"] == 80
    assert data["total_parameters"] > 1_000_000
    assert "checksum_sha256" in data

def test_api_samples_list():
    resp = client.get("/api/samples")
    assert resp.status_code == 200
    samples = resp.json()
    assert len(samples) >= 3
    sample_ids = [s["id"] for s in samples]
    assert "sample_normal_baseline" in sample_ids
    assert "sample_pre_apnea_onset" in sample_ids

def test_standalone_preprocessing_endpoint():
    test_sig = np.sin(np.linspace(0, 10 * 2 * np.pi, 800)).tolist()
    resp = client.post("/api/signal/preprocess", json={"signal": test_sig, "fs": 80.0})
    assert resp.status_code == 200
    data = resp.json()
    assert "cleaned_signal" in data
    assert "filtered_signal" in data
    assert "biomarkers" in data

def test_api_predict_flow_db_and_pdf_generation():
    # 1. Fetch benchmark sample
    sample_resp = client.get("/api/samples/sample_normal_baseline")
    assert sample_resp.status_code == 200
    sample_data = sample_resp.json()
    
    # 2. Post prediction
    pred_resp = client.post("/api/predict", json={
        "signal": sample_data["signal"],
        "fs": sample_data["fs"],
        "subject_id": "Integration-Test-Subject"
    })
    assert pred_resp.status_code == 200
    pred_data = pred_resp.json()
    assert "analysis_id" in pred_data
    analysis_id = pred_data["analysis_id"]
    
    # 3. Verify record was stored in SQLite database
    db = SessionLocal()
    rec = db.query(AnalysisRecord).filter(AnalysisRecord.id == analysis_id).first()
    assert rec is not None
    assert rec.subject_id == "Integration-Test-Subject"
    assert rec.prediction == pred_data["prediction"]
    db.close()
    
    # 4. Verify History endpoint returns the record
    hist_resp = client.get("/api/history")
    assert hist_resp.status_code == 200
    history = hist_resp.json()
    assert len(history) > 0
    assert any(h["id"] == analysis_id for h in history)
    
    # 5. Request PDF Report
    report_resp = client.get(f"/api/report/{analysis_id}")
    assert report_resp.status_code == 200
    assert report_resp.headers["content-type"] == "application/pdf"
    assert len(report_resp.content) > 1000 # Valid non-empty PDF
