# ─────────────────────────────────────────────────────────────────────────────
# CodeDebug-RL — Production Dockerfile
# ─────────────────────────────────────────────────────────────────────────────
# Multi-stage build for minimal image size.
# Runs the FastAPI/Uvicorn server on port 8000.
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim AS base

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies for subprocess-based test execution
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# ─── Dependencies ────────────────────────────────────────────────────────────

FROM base AS deps

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ─── Application ─────────────────────────────────────────────────────────────

FROM deps AS app

WORKDIR /app

# Copy project files
COPY pyproject.toml .
COPY codedebug_env/ codedebug_env/
COPY configs/ configs/
COPY data/ data/

# Install the package itself
RUN pip install --no-cache-dir -e .

# ─── Runtime ─────────────────────────────────────────────────────────────────

# Default environment variables
ENV PORT=8000 \
    CODEDEBUG_MAX_STEPS=10 \
    CODEDEBUG_TIMEOUT=30 \
    ENABLE_WEB_INTERFACE=true \
    CODEDEBUG_TASK_DIR=""

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/health').raise_for_status()"

# Run with Uvicorn
CMD ["uvicorn", "codedebug_env.server.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
