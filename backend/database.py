import os
from datetime import datetime
from sqlalchemy import create_engine, Column, String, Float, Integer, Boolean, Text, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "apneasphere.db")
DATABASE_URL = os.environ.get("DATABASE_URL", f"sqlite:///{DB_PATH}")

engine = create_engine(
    DATABASE_URL, 
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class AnalysisRecord(Base):
    """
    Persistent clinical record of a PPG analysis execution.
    """
    __tablename__ = "analysis_records"

    id = Column(String(64), primary_key=True, index=True)
    subject_id = Column(String(128), index=True, default="Anonymous")
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    
    # Model predictions
    prediction = Column(String(32), nullable=False)
    probability = Column(Float, nullable=False)
    confidence_percentage = Column(Float, nullable=False)
    risk_level = Column(String(32), nullable=False)
    risk_color = Column(String(16), nullable=False)
    clinical_interpretation = Column(Text, nullable=False)
    
    # Raw logit & telemetry
    raw_logit = Column(Float, nullable=True)
    decision_threshold = Column(Float, default=0.50)
    is_near_boundary = Column(Boolean, default=False)
    
    # Signal Quality Index
    sqi_quality = Column(String(16), nullable=False)
    sqi_score = Column(Float, nullable=False)
    
    # Physiological Biomarkers
    mean_hr_bpm = Column(Float, nullable=False)
    sdnn_ms = Column(Float, nullable=False)
    rmssd_ms = Column(Float, nullable=False)
    pnn50_pct = Column(Float, nullable=False)
    cv = Column(Float, nullable=False)
    peak_count = Column(Integer, nullable=False)
    lf_power = Column(Float, nullable=False)
    hf_power = Column(Float, nullable=False)
    lf_hf_ratio = Column(Float, nullable=False)
    
    # Signal meta
    raw_signal_length = Column(Integer, nullable=False)
    sampling_rate_hz = Column(Integer, default=80)
    
    # Serialized preview (JSON string of ~600 downsampled points)
    signal_preview_json = Column(Text, nullable=True)
    peak_indices_json = Column(Text, nullable=True)

# Create tables
Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
