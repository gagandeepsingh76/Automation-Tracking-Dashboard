# Use Python 3.9 slim image for smaller footprint
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libpq-dev \
    libssl-dev \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash trendscope
USER trendscope

# Copy requirements and install Python dependencies
COPY --chown=trendscope:trendscope requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Copy application code
COPY --chown=trendscope:trendscope . .

# Create necessary directories
RUN mkdir -p logs data/raw data/processed dashboards/exports models/trained

# Set proper permissions
RUN chmod +x schedule/run_workflow.py

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Expose port for monitoring
EXPOSE 8000

# Default command
CMD ["python", "schedule/run_workflow.py", "--scheduler"]
