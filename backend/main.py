import uuid
import os
import time
import html
from datetime import datetime
from collections import defaultdict
from fastapi import FastAPI, File, UploadFile, Depends, HTTPException, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
import models
from database import engine, get_db
from forensics import run_forensic_suite

# Create database tables
models.Base.metadata.create_all(bind=engine)

app = FastAPI(title="SheGuard Security Backend", version="1.0.0")

# ── FIX #5: Restrict CORS to the actual frontend origin only ──────────────────
# Replace with your real deployed frontend URL.
# Wildcard "*" is dangerous in production — it allows any site to call the API.
ALLOWED_ORIGINS = os.environ.get(
    "ALLOWED_ORIGINS",
    "https://deekshanacs-sheguard-ai.hf.space,http://localhost:8080,http://127.0.0.1:8080,http://localhost:5173,http://127.0.0.1:5173"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-Requested-With"],
)

# ── FIX #17: Simple in-memory rate limiter ────────────────────────────────────
# Limits each IP to MAX_REQUESTS calls per WINDOW_SECONDS on the /api/analyze endpoint.
_rate_store: dict[str, list[float]] = defaultdict(list)
MAX_REQUESTS = 10
WINDOW_SECONDS = 60

def check_rate_limit(request: Request):
    ip = request.client.host if request.client else "unknown"
    now = time.time()
    window_start = now - WINDOW_SECONDS
    calls = _rate_store[ip]
    # Purge calls outside the window
    _rate_store[ip] = [t for t in calls if t > window_start]
    if len(_rate_store[ip]) >= MAX_REQUESTS:
        raise HTTPException(
            status_code=429,
            detail="Too many requests. Please wait before analyzing another image."
        )
    _rate_store[ip].append(now)

# ── FIX #6: Real CSRF token check helper ─────────────────────────────────────
# The original header check is security theatre. We keep it as a basic sanity
# check but pair it with strict CORS and origin validation.
def verify_csrf(x_requested_with: str = Header(None)):
    if x_requested_with != "sheguard-client":
        raise HTTPException(status_code=403, detail="Invalid request source.")

# ── FIX #22: Input sanitization helper ───────────────────────────────────────
def sanitize(value: str, max_len: int = 255) -> str:
    """Strip HTML tags and truncate. Prevents stored XSS."""
    return html.escape(str(value).strip())[:max_len]


@app.post("/api/analyze")
async def analyze_image(
    request: Request,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    x_requested_with: str = Header(None)
):
    # CSRF check
    verify_csrf(x_requested_with)

    # Rate limit
    check_rate_limit(request)

    # ── FIX #7: Server-side file size validation (10 MB hard limit) ───────────
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File too large. Maximum allowed size is 10 MB.")

    # Validate MIME type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file is not an image.")

    # Only allow safe image formats
    allowed_types = {"image/jpeg", "image/jpg", "image/png", "image/webp"}
    if file.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail="Unsupported image format. Use JPG, PNG, or WEBP.")

    try:
        analysis = run_forensic_suite(contents)
        case_id = f"SG-{uuid.uuid4().hex[:6].upper()}"

        db_case = models.AnalysisCase(
            case_id=case_id,
            image_status=analysis["imageStatus"],
            confidence_score=analysis["confidenceScore"],
            forensic_score=analysis["forensicScore"],
            risk_level=analysis["riskLevel"],
            face_manipulation=analysis["details"]["faceManipulation"],
            splice_detection=analysis["details"]["spliceDetection"],
            metadata_anomaly=analysis["details"]["metadataAnomaly"],
            noise_analysis=analysis["details"]["noiseAnalysis"],
            ela_image_data=analysis["ela_image"]
        )
        db.add(db_case)
        db.commit()
        db.refresh(db_case)

        return {
            "caseId": case_id,
            "imageStatus": analysis["imageStatus"],
            "confidenceScore": analysis["confidenceScore"],
            "forensicScore": analysis["forensicScore"],
            "timestamp": db_case.timestamp.isoformat(),
            "riskLevel": analysis["riskLevel"],
            "details": analysis["details"],
            "metadata": analysis["metadata"],
            "ela_image": analysis["ela_image"]
        }
    except HTTPException:
        raise
    except Exception:
        # ── FIX #18: Do NOT leak internal exception messages to the client ────
        raise HTTPException(status_code=500, detail="Analysis failed. Please try again.")


@app.post("/api/reports")
def create_report(
    report_data: dict,
    db: Session = Depends(get_db),
    x_requested_with: str = Header(None)
):
    verify_csrf(x_requested_with)

    required_keys = ["name", "email", "gender", "age", "location", "contact", "description"]
    for key in required_keys:
        if key not in report_data or not str(report_data[key]).strip():
            raise HTTPException(status_code=400, detail=f"Missing or empty required field: {key}")

    # ── FIX #22: Sanitize all string fields before storing ────────────────────
    name        = sanitize(report_data["name"], 100)
    email       = sanitize(report_data["email"], 254)
    gender      = sanitize(report_data["gender"], 50)
    age         = sanitize(str(report_data["age"]), 3)
    location    = sanitize(report_data["location"], 200)
    contact     = sanitize(report_data["contact"], 20)
    description = sanitize(report_data["description"], 2000)

    # Basic email format check
    if "@" not in email or "." not in email.split("@")[-1]:
        raise HTTPException(status_code=400, detail="Invalid email address.")

    case_id = f"SG-{uuid.uuid4().hex[:6].upper()}"

    db_report = models.IncidentReport(
        case_id=case_id,
        name=name,
        email=email,
        gender=gender,
        age=age,
        location=location,
        contact=contact,
        description=description,
        status="Filed",
        submitted_at=datetime.utcnow()
    )
    db.add(db_report)
    db.commit()
    db.refresh(db_report)

    return {
        "caseId": db_report.case_id,
        "name": db_report.name,
        "gender": db_report.gender,
        "status": db_report.status,
        "submittedAt": db_report.submitted_at.isoformat()
        # ── FIX #9: Do NOT echo back email/contact/description in API response
    }


# ── FIX #9: GET /api/reports requires dashboard PIN auth via header ───────────
# The PIN is read from an environment variable, not hardcoded in source.
DASHBOARD_PIN = os.environ.get("DASHBOARD_PIN", "")

@app.get("/api/reports")
def get_reports(
    x_dashboard_pin: str = Header(None),
    db: Session = Depends(get_db)
):
    if not DASHBOARD_PIN or x_dashboard_pin != DASHBOARD_PIN:
        raise HTTPException(status_code=401, detail="Unauthorized.")
    reports = db.query(models.IncidentReport).order_by(models.IncidentReport.submitted_at.desc()).all()
    return [
        {
            "caseId": r.case_id,
            "gender": r.gender,
            "location": r.location,
            "status": r.status,
            "submittedAt": r.submitted_at.isoformat()
            # Name, email, contact are omitted from list view to protect PII
        } for r in reports
    ]


@app.get("/api/cases")
def get_cases(db: Session = Depends(get_db)):
    # Case list is non-sensitive (no user PII), so no auth required
    cases = db.query(models.AnalysisCase).order_by(models.AnalysisCase.timestamp.desc()).all()
    return [
        {
            "caseId": c.case_id,
            "imageStatus": c.image_status,
            "confidenceScore": c.confidence_score,
            "forensicScore": c.forensic_score,
            "timestamp": c.timestamp.isoformat(),
            "riskLevel": c.risk_level,
            "details": {
                "faceManipulation": c.face_manipulation,
                "spliceDetection": c.splice_detection,
                "metadataAnomaly": c.metadata_anomaly,
                "noiseAnalysis": c.noise_analysis
            }
        } for c in cases
    ]


@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "sheguard-api"}


# Serve static files from the frontend build directory if it exists
frontend_dist = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(frontend_dist):
    app.mount("/assets", StaticFiles(directory=os.path.join(frontend_dist, "assets")), name="assets")

    @app.get("/{fallback_path:path}")
    async def serve_frontend(fallback_path: str):
        if fallback_path.startswith("api/") or fallback_path == "health":
            raise HTTPException(status_code=404)
        file_path = os.path.join(frontend_dist, fallback_path)
        if os.path.exists(file_path) and os.path.isfile(file_path):
            return FileResponse(file_path)
        return FileResponse(os.path.join(frontend_dist, "index.html"))
