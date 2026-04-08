# ─────────────────────────────────────────────────────────────────────────────
# CodeDebug-RL — Production Dockerfile
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

FROM base AS deps
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

FROM deps AS app
WORKDIR /app

# Copy project files
COPY pyproject.toml .
COPY codedebug_env/ codedebug_env/
COPY configs/ configs/
COPY data/ data/
COPY inference.py .

# Install the package itself
RUN pip install --no-cache-dir -e .

# Default environment variables for HF Spaces
ENV PORT=7860 \
    CODEDEBUG_MAX_STEPS=10 \
    CODEDEBUG_TIMEOUT=30 \
    ENABLE_WEB_INTERFACE=true \
    CODEDEBUG_TASK_DIR=""

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:7860/health').raise_for_status()"

CMD ["sh", "-c", "uvicorn codedebug_env.server.app:app --host 0.0.0.0 --port ${PORT} --workers 1"]
