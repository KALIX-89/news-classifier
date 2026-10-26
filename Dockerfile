# ------------------------------------------------------------------
# Image for the news-classifier API (inference only, CPU).
#
# The model is NOT baked into the image. It comes from either
#   - a mounted folder:  -v ./models:/app/models
#   - or S3:             -e MODEL_S3_URI=s3://bucket/prefix
# -> same image can serve any model version.
# ------------------------------------------------------------------
FROM python:3.11-slim

# Don't write .pyc files, print logs immediately (no buffering)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# 1) CPU-only PyTorch: ~200 MB instead of ~2.5 GB for the CUDA build.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

# 2) Remaining dependencies. Copied before the code so Docker can cache
#    this layer: changing the code does not re-install everything.
COPY requirements-api.txt .
RUN pip install -r requirements-api.txt

# 3) Our code
COPY src/ src/
COPY api/ api/

# Run as a normal user, not root (security best practice)
RUN useradd --create-home appuser && mkdir -p /app/models && chown -R appuser /app
USER appuser

EXPOSE 8000

# Docker marks the container "unhealthy" if /health stops answering
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
