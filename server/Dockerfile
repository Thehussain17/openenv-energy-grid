# Compatible with HF Spaces (sdk: docker, app_port: 7860, tags: openenv)

# Stage 1: Builder
FROM python:3.11-slim AS builder

WORKDIR /app

# Install git, curl
RUN apt-get update && \
    apt-get install -y --no-install-recommends git curl && \
    rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    mv /root/.cargo/bin/uv /usr/local/bin/uv && \
    mv /root/.cargo/bin/uvx /usr/local/bin/uvx || \
    (mv /root/.local/bin/uv /usr/local/bin/uv && \
    mv /root/.local/bin/uvx /usr/local/bin/uvx)

# Copy project files
COPY . /app/env
WORKDIR /app/env

# Install dependencies using uv
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --python 3.11 --no-frozen --no-editable

# Stage 2: Runtime
FROM python:3.11-slim

WORKDIR /app

# Install curl for HEALTHCHECK
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Copy virtualenv and project code from builder
COPY --from=builder /app/env/.venv /app/.venv
COPY --from=builder /app/env /app/env

# Set environment paths
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app/env:$PYTHONPATH"
ENV PYTHONUNBUFFERED=1

WORKDIR /app/env

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Launch the FastAPI server
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
