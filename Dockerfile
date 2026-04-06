# Compatible with HF Spaces (sdk: docker, app_port: 7860, tags: openenv)

FROM python:3.11-slim

WORKDIR /app/env

# Install curl for HEALTHCHECK
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . .

# Install dependencies using standard built-in pip
# This reads pyproject.toml automagically securely across any architecture
RUN pip install --no-cache-dir .

# Set environment
ENV PYTHONUNBUFFERED=1

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Launch the FastAPI server
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
