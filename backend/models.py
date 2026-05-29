from sqlalchemy import Column, Integer, String, Float, DateTime
from datetime import datetime
from database import Base

class IncidentReport(Base):
    __tablename__ = "incident_reports"

    id = Column(Integer, primary_key=True, index=True)
    case_id = Column(String, unique=True, index=True)
    name = Column(String)
    email = Column(String)
    gender = Column(String)
    age = Column(String)
    location = Column(String)
    contact = Column(String)
    description = Column(String)
    status = Column(String, default="Filed")
    submitted_at = Column(DateTime, default=datetime.utcnow)

class AnalysisCase(Base):
    __tablename__ = "analysis_cases"

    id = Column(Integer, primary_key=True, index=True)
    case_id = Column(String, unique=True, index=True)
    image_status = Column(String)  # Safe, Suspicious, Manipulated
    confidence_score = Column(Float)
    forensic_score = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow)
    risk_level = Column(String)  # green, yellow, red
    face_manipulation = Column(Float)
    splice_detection = Column(Float)
    metadata_anomaly = Column(Float)
    noise_analysis = Column(Float)
    ela_image_data = Column(String, nullable=True)  # Store base64 ELA difference image for visual display
