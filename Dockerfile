FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py best_model.pkl model_summary.json ./

# GEOVERI_API_KEYS must be set at runtime (docker run -e GEOVERI_API_KEYS=... or via compose)
ENV GEOVERI_RATE_LIMIT_PER_MIN=60

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
  CMD python -c "import os,urllib.request; urllib.request.urlopen('http://localhost:'+os.environ.get('PORT','8000')+'/health')" || exit 1

# Render (and similar platforms) inject a PORT env var and expect the app to bind to it.
# Falls back to 8000 for local/docker-compose use where PORT isn't set.
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
