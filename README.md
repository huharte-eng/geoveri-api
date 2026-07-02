# Saint Lucia Slope Stability — Prediction API (Pilot-Hardened)

Wraps the trained Random Forest landslide-susceptibility model (`best_model.pkl`, Section 3.5
of the revised Methodology) in a FastAPI service, with API key auth, rate limiting, and
structured request logging added for pilot/early-warning use.

## What's new in this hardened build

- **API key authentication** — every endpoint except `/health` requires an `X-API-Key` header
- **Rate limiting** — per-key sliding window, default 60 requests/minute (configurable)
- **Structured JSON request logging** — every prediction logged with request ID, full
  input/output, latency, and client IP, rotated at 5MB (`requests.log`) — an audit trail,
  useful if this is ever used to inform NEMO early-warning decisions
- **Containerized** — `Dockerfile` + `docker-compose.yml` for consistent deployment
- **Batch size cap** (500 cases/request) to prevent accidental overload

## Run locally (no container)

```bash
pip install -r requirements.txt
export GEOVERI_API_KEYS="your-key-here"        # comma-separated for multiple keys
export GEOVERI_RATE_LIMIT_PER_MIN=60            # optional, defaults to 60
uvicorn main:app --reload --port 8000
```

Leave `GEOVERI_API_KEYS` unset and auth is disabled (open mode) — fine for local dev, **not**
for anything reachable outside your machine.

## Run with Docker

```bash
export GEOVERI_API_KEYS="your-key-here"
docker compose up --build
```

This wasn't build-tested in the sandbox this was developed in (no Docker available there) —
run a real `docker build` / `docker compose up` on your end before relying on it for a pilot.

## Endpoints

- `GET /health` — liveness check, no auth required
- `GET /model-info` — model metadata (best model, feature list, cross-val/test metrics)
- `POST /predict` — single slope case → `{prediction, probability_unstable, probability_stable, model_used, request_id}`
- `POST /predict-batch` — `{"cases": [ ... ]}` (max 500) → list of predictions

## Example

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-key-here" \
  -d @sample_request.json
```

## Input fields

| Field | Type | Notes |
|---|---|---|
| Rainfall_mm | float ≥ 0 | Rainfall on assessment day |
| Rain_Lag3_mm / Lag7 / Lag14 / Lag30 | float ≥ 0 | Antecedent rainfall, days *before* the assessment day (excludes the day itself) |
| Distance_to_Road_m | float ≥ 0 | |
| Distance_to_River_m | float ≥ 0 | |
| Elevation_m | float | |
| Slope_deg | 10 / 20 / 30 / 40 | NIPP categorical slope class — not a continuous angle |
| Aspect_deg | 0–360 | |
| Curvature | float | |
| Landcover | Forest / Scrub / Agriculture / Bare | |
| Soil_Type | Clay / Loam / Sandy Loam / Silty Clay | |

NDVI is deliberately not an input — it was excluded from the model as outcome leakage
(Section 3.3.3 of the revised Methodology).

## Still not production-grade — before a real pilot rollout, also add

- **Real key management**: this uses a static env-var list; move to a proper secrets store
  or API gateway (e.g. AWS API Gateway, Kong) before issuing keys to external partners like NEMO
- **HTTPS termination** — put this behind a reverse proxy (nginx, Caddy) or load balancer;
  the app itself serves plain HTTP
- **Persistent, queryable logging** — `requests.log` is file-based and rotates locally; for a
  real audit trail, ship logs to something queryable (CloudWatch, ELK, even just a database
  table)
- **A retraining/refresh pipeline** — this model is a static snapshot; an early-warning tool
  needs a defined process for retraining as new labelled events come in
- **Automated rainfall ingestion** — currently every request needs the caller to supply
  rainfall/antecedent values manually; a live pilot would pull current Met Office data
  automatically rather than relying on manual input
- **Load testing** — the in-memory rate limiter and single-process model won't necessarily
  hold up under real concurrent load; test before depending on it operationally

