# Multi-stage Dockerfile for Industrial Audio RAG
# Optimized for production with security hardening

# Build stage - includes development tools
FROM python:3.13-slim as builder

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy package configuration
COPY pyproject.toml ./
COPY README.md ./

# Install build dependencies and build wheel
RUN pip install --no-cache-dir build && \
    python -m build

# Runtime stage - minimal image for production
FROM python:3.13-slim as runtime

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Install runtime system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set working directory
WORKDIR /app

# Copy built wheel from builder stage
COPY --from=builder /app/dist/*.whl ./

# Install the application
RUN pip install --no-cache-dir *.whl && \
    rm *.whl

# Copy source code
COPY src/ ./src/

# Create necessary directories and set permissions
RUN mkdir -p /app/logs /app/data && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Add health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 --start-period=40s \
    CMD curl -f http://localhost:8000/ || exit 1

# Expose port
EXPOSE 8000

# Set environment variables
ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Default command
CMD ["uvicorn", "rag_audio.api:app", "--host", "0.0.0.0", "--port", "8000"]