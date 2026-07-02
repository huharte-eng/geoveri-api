"""
GeoVeri / Saint Lucia Slope Stability — Prediction API (hardened, pilot-ready)

Adds to the base prototype:
  - API key authentication (header: X-API-Key)
  - Structured JSON request logging (audit trail for any eventual early-warning use)
  - Request ID + latency tracking
  - Basic rate limiting (per-key, in-memory — fine for a pilot, swap for Redis at scale)

Run locally with:
    export GEOVERI_API_KEYS="key1,key2"   # comma-separated, at least one required
    uvicorn main:app --reload --port 8000

Then:
    curl -X POST http://localhost:8000/predict \
      -H "Content-Type: application/json" -H "X-API-Key: key1" -d @sample_request.json
"""
from fastapi import FastAPI, HTTPException, Request, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Literal, Optional
import joblib
import pandas as pd
import json
import os
import time
import uuid
import logging
from logging.handlers import RotatingFileHandler
from collections import defaultdict, deque

MODEL_PATH = os.path.join(os.path.dirname(__file__), "best_model.pkl")
SUMMARY_PATH = os.path.join(os.path.dirname(__file__), "model_summary.json")
LOG_PATH = os.path.join(os.path.dirname(__file__), "requests.log")

API_VERSION = "1.1.0-pilot"

# --- auth config ---
VALID_API_KEYS = set(
    k.strip() for k in os.environ.get("GEOVERI_API_KEYS", "").split(",") if k.strip()
)
AUTH_ENABLED = len(VALID_API_KEYS) > 0

# --- rate limiting (simple sliding window, per key, in-memory) ---
RATE_LIMIT_PER_MIN = int(os.environ.get("GEOVERI_RATE_LIMIT_PER_MIN", "60"))
_request_log = defaultdict(deque)

# --- structured request logging ---
logger = logging.getLogger("geoveri")
logger.setLevel(logging.INFO)
handler = RotatingFileHandler(LOG_PATH, maxBytes=5_000_000, backupCount=3)
handler.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(handler)


def log_event(event: dict):
    event["ts"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    logger.info(json.dumps(event))


app = FastAPI(
    title="Saint Lucia Slope Stability Prediction API",
    description="Random Forest landslide-susceptibility classifier trained on consolidated "
                "Ministry of Infrastructure / NIPP / NEMO / Met Office data for Saint Lucia. "
                "Pilot-hardened build: API key auth, request logging, rate limiting.",
    version=API_VERSION,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
model_summary = None


@app.on_event("startup")
def load_model():
    global model, model_summary
    model = joblib.load(MODEL_PATH)
    with open(SUMMARY_PATH) as f:
        model_summary = json.load(f)
    log_event({"event": "startup", "auth_enabled": AUTH_ENABLED, "version": API_VERSION})


def require_api_key(x_api_key: Optional[str] = Header(default=None)):
    if not AUTH_ENABLED:
        return "anonymous"  # no keys configured — open mode, e.g. local dev
    if not x_api_key or x_api_key not in VALID_API_KEYS:
        raise HTTPException(status_code=401, detail="Missing or invalid X-API-Key header")
    now = time.time()
    window = _request_log[x_api_key]
    while window and now - window[0] > 60:
        window.popleft()
    if len(window) >= RATE_LIMIT_PER_MIN:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    window.append(now)
    return x_api_key


class SlopeCase(BaseModel):
    """
    One slope location / date to be scored. Feature definitions and units match
    Section 3.3.4 of the revised Methodology.
    """
    Rainfall_mm: float = Field(..., ge=0, description="Rainfall on the day of assessment (mm)")
    Rain_Lag3_mm: float = Field(..., ge=0, description="Cumulative rainfall, preceding 3 days (mm)")
    Rain_Lag7_mm: float = Field(..., ge=0, description="Cumulative rainfall, preceding 7 days (mm)")
    Rain_Lag14_mm: float = Field(..., ge=0, description="Cumulative rainfall, preceding 14 days (mm)")
    Rain_Lag30_mm: float = Field(..., ge=0, description="Cumulative rainfall, preceding 30 days (mm)")
    Distance_to_Road_m: float = Field(..., ge=0, description="Distance to nearest road (m)")
    Distance_to_River_m: float = Field(..., ge=0, description="Distance to nearest river (m)")
    Elevation_m: float = Field(..., description="Elevation (m)")
    Slope_deg: Literal[10, 20, 30, 40] = Field(..., description="NIPP slope class (categorical: 10/20/30/40 degrees)")
    Aspect_deg: float = Field(..., ge=0, le=360, description="Slope aspect (degrees)")
    Curvature: float = Field(..., description="Terrain curvature")
    Landcover: Literal["Forest", "Scrub", "Agriculture", "Bare"] = Field(..., description="Land cover class")
    Soil_Type: Literal["Clay", "Loam", "Sandy Loam", "Silty Clay"] = Field(..., description="Soil type")

    class Config:
        json_schema_extra = {
            "example": {
                "Rainfall_mm": 45.0,
                "Rain_Lag3_mm": 80.0,
                "Rain_Lag7_mm": 150.0,
                "Rain_Lag14_mm": 240.0,
                "Rain_Lag30_mm": 310.0,
                "Distance_to_Road_m": 60.0,
                "Distance_to_River_m": 90.0,
                "Elevation_m": 300.0,
                "Slope_deg": 30,
                "Aspect_deg": 210.0,
                "Curvature": 0.8,
                "Landcover": "Bare",
                "Soil_Type": "Clay",
            }
        }


class PredictionResponse(BaseModel):
    prediction: Literal["Stable", "Unstable"]
    probability_unstable: float
    probability_stable: float
    model_used: str
    request_id: str


class BatchRequest(BaseModel):
    cases: list[SlopeCase]


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None, "version": API_VERSION,
            "auth_enabled": AUTH_ENABLED}


@app.get("/model-info")
def model_info(api_key: str = Depends(require_api_key)):
    if model_summary is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return model_summary


@app.post("/predict", response_model=PredictionResponse)
def predict(case: SlopeCase, request: Request, api_key: str = Depends(require_api_key)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    request_id = str(uuid.uuid4())
    t0 = time.time()

    row = pd.DataFrame([case.dict()])
    proba = model.predict_proba(row)[0]
    pred = int(model.predict(row)[0])
    result = PredictionResponse(
        prediction="Unstable" if pred == 1 else "Stable",
        probability_unstable=round(float(proba[1]), 4),
        probability_stable=round(float(proba[0]), 4),
        model_used=model_summary["best_model"] if model_summary else "unknown",
        request_id=request_id,
    )

    log_event({
        "event": "predict", "request_id": request_id, "api_key": api_key,
        "input": case.dict(), "output": result.dict(),
        "latency_ms": round((time.time() - t0) * 1000, 1),
        "client_ip": request.client.host if request.client else None,
    })
    return result


@app.post("/predict-batch", response_model=list[PredictionResponse])
def predict_batch(req: BatchRequest, request: Request, api_key: str = Depends(require_api_key)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if len(req.cases) == 0:
        raise HTTPException(status_code=400, detail="No cases supplied")
    if len(req.cases) > 500:
        raise HTTPException(status_code=400, detail="Batch limited to 500 cases per request")

    request_id = str(uuid.uuid4())
    t0 = time.time()
    rows = pd.DataFrame([c.dict() for c in req.cases])
    probas = model.predict_proba(rows)
    preds = model.predict(rows)
    results = [
        PredictionResponse(
            prediction="Unstable" if p == 1 else "Stable",
            probability_unstable=round(float(pr[1]), 4),
            probability_stable=round(float(pr[0]), 4),
            model_used=model_summary["best_model"] if model_summary else "unknown",
            request_id=f"{request_id}-{i}",
        )
        for i, (p, pr) in enumerate(zip(preds, probas))
    ]

    log_event({
        "event": "predict_batch", "request_id": request_id, "api_key": api_key,
        "n_cases": len(req.cases),
        "latency_ms": round((time.time() - t0) * 1000, 1),
        "client_ip": request.client.host if request.client else None,
    })
    return results

