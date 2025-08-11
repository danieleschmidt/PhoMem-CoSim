FROM python:3.11-slim as base

# System dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r phomem && useradd --no-log-init -r -g phomem phomem

# Set work directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY phomem/ phomem/
COPY *.py ./

# Set ownership
RUN chown -R phomem:phomem /app

# Switch to non-root user
USER phomem

# Production stage
FROM base as production

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Expose ports
EXPOSE 8000 9090

# Start application
CMD ["python", "-m", "phomem.cli", "--production"]
