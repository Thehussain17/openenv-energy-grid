# Multi-stage build for the Energy Grid Environment
# Compatible with HF Spaces (sdk: docker, app_port: 8000, tags: openenv)
#
# IMPORTANT: The openenv-base image uses Python 3.11 (python:3.11-slim).
# Our pyproject.toml requires python >=3.10, so 3.11 is fully compatible.
# We use --no-frozen so uv resolves deps against Python 3.11 inside the
# container, avoiding conflicts with the local py3.10-generated uv.lock.

ARG BASE_IMAGE=ghcr.io/meta-pytorch/openenv-base:latest
FROM ${BASE_IMAGE} AS builder

WORKDIR /app

# Ensure git + curl are available (curl is needed for uv fallback install)
RUN apt-get update && \
    apt-get install -y --no-install-recommends git curl && \
    rm -rf /var/lib/apt/lists/*

# Ensure uv is available (base image ships with it; this is a safety fallback)
RUN if ! command -v uv >/dev/null 2>&1; then \
        curl -LsSf https://astral.sh/uv/install.sh | sh && \
        mv /root/.local/bin/uv /usr/local/bin/uv && \
        mv /root/.local/bin/uvx /usr/local/bin/uvx; \
    fi

# Copy project files into the builder
COPY . /app/env
WORKDIR /app/env

# Install all dependencies with Python 3.11.
# --no-frozen: resolve fresh against py3.11 (our local uv.lock was made on py3.10)
# --no-editable: install the project itself as a wheel for clean imports
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --python 3.11 --no-frozen --no-editable

# ── Runtime stage ─────────────────────────────────────────────────────────────
FROM ${BASE_IMAGE}

WORKDIR /app

# Copy the resolved virtualenv and project code from the builder stage
COPY --from=builder /app/env/.venv /app/.venv
COPY --from=builder /app/env /app/env

# Use the venv's Python + binaries; add project root to PYTHONPATH
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app/env:$PYTHONPATH"
ENV PYTHONUNBUFFERED=1

# Expose the server port
EXPOSE 8000

# Health check — server must respond within 15 s of startup
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Launch the FastAPI / uvicorn server
CMD ["sh", "-c", "cd /app/env && uvicorn server.app:app --host 0.0.0.0 --port 8000"]
