# Backend Dockerfile for Car-Psycho Microservices
# Supports all three backend services: Manager, Inference, Data

FROM python:3.11-slim

# Build argument for service directory
ARG SERVICE_DIR

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy backend code
# From context backend/services/{service_name}, go up 2 levels to get backend/
COPY ../../ /app/backend
# Go up 3 levels to get project root requirements.txt
COPY ../../../requirements.txt /app/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

# Install service-specific requirements if they exist
RUN if [ -f "/app/backend/${SERVICE_DIR}/requirements.txt" ]; then \
        pip install --no-cache-dir -r "/app/backend/${SERVICE_DIR}/requirements.txt"; \
    fi

# Set Python path
ENV PYTHONPATH=/app/backend:$PYTHONPATH

# Create necessary directories
RUN mkdir -p /app/data /app/models /app/logs

# Expose port
EXPOSE 8000

# Set working directory to service directory
WORKDIR /app/backend/${SERVICE_DIR}

# Default command (can be overridden in docker-compose)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
