# Production Dockerfile for phomem-cosim
FROM python:3.11-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Create wheel directory
RUN mkdir /wheels

# Copy requirements and build wheels
COPY requirements.txt /
RUN pip wheel --wheel-dir=/wheels -r requirements.txt

# Production stage
FROM python:3.11-slim

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy wheels and install
COPY --from=builder /wheels /wheels
RUN pip install --no-index --find-links=/wheels /wheels/* \
    && rm -rf /wheels

# Create directories
RUN mkdir -p /app /app/data /app/logs \
    && chown -R appuser:appuser /app

# Copy application
COPY --chown=appuser:appuser . /app/
WORKDIR /app

# Security: Remove shell access
RUN rm /bin/sh /bin/bash || true

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD python -c "import requests; requests.get('http://localhost:8080/health')"

# Expose port
EXPOSE 8080

# Run application
CMD ["python", "-m", "phomem.server", "--production"]
