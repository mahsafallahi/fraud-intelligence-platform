# Fraud scoring API (Phase 6): model F behind FastAPI.
#
# Build from the project root (BuildKit uses docker/api/Dockerfile.dockerignore):
#   docker build -f docker/api/Dockerfile -t fraud-api:dev .
# Run with the serving snapshot mounted read-only (never baked into the image):
#   docker run --rm -p 8000:8000 --read-only \
#     -v "$PWD/data/serving/serving.duckdb:/data/serving.duckdb:ro" fraud-api:dev
#
# Python 3.14.6 = the .venv the model, its bundle and the parity tests were made
# with; pinned by digest so a rebuild gets the same base. Single stage: every
# package is a binary wheel (nothing compiles), so a build stage would only
# drop pip.
FROM python:3.14.6-slim-trixie@sha256:7bec7ddcddeff7975d6ba9b4be7dd6f6b2f55e7491539145e2978f7f97ce9144

# libgomp1: OpenMP runtime the LightGBM wheel links to (not in the slim image)
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# Every package pinned with its wheel hash; no dependency resolution at build time
COPY requirements-api.lock /tmp/requirements-api.lock
RUN pip install --no-cache-dir --only-binary=:all: --no-deps --require-hashes \
      -r /tmp/requirements-api.lock \
 && rm /tmp/requirements-api.lock

WORKDIR /app
# Only the code the API imports, and the model bundle
COPY src/__init__.py src/
COPY src/models/__init__.py src/models/features.py src/models/calibration.py src/models/
COPY src/api/ src/api/
COPY models/F/ models/F/

RUN useradd --system --uid 10001 --no-create-home --shell /usr/sbin/nologin app
USER 10001

# OMP_NUM_THREADS=1: one-row predictions are slower with a thread pool
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    BUNDLE_DIR=/app/models/F \
    SNAPSHOT_PATH=/data/serving.duckdb

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD ["python", "-c", "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"]

CMD ["uvicorn", "src.api.main:create_app_from_env", "--factory", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--no-access-log"]
