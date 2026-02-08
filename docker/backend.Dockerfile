# Backend Dockerfile for Car-Psycho Microservices
# Supports all three backend services: Manager, Inference, Data
# Includes PyTorch CPU for ML training in Docker

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

# Copy requirements first (for better caching)
COPY requirements-docker.txt /app/requirements.txt

# Install Python dependencies
# 1. Install PyTorch CPU from the PyTorch wheel index
# 2. Install remaining requirements
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        torch==2.1.2 \
        --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r /app/requirements.txt

# Verify torch installed correctly
RUN python -c "import torch; print(f'PyTorch {torch.__version__} installed successfully')"

# Copy backend code
COPY backend /app/backend

# Install service-specific requirements if they exist
RUN if [ -f "/app/backend/${SERVICE_DIR}/requirements.txt" ]; then \
        pip install --no-cache-dir -r "/app/backend/${SERVICE_DIR}/requirements.txt"; \
    fi

# Set Python path to include backend and ML modules
ENV PYTHONPATH=/app/backend:/app/backend/ml:/app:$PYTHONPATH

# Create necessary directories
RUN mkdir -p /app/data/raw /app/data/processed /app/data/car_questionnaire /app/data/synthetic \
    /app/models/checkpoints /app/models/saved /app/logs

# Expose port
EXPOSE 8000

# Set working directory to service directory
WORKDIR /app/backend/${SERVICE_DIR}

# Default command (can be overridden in docker-compose)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
