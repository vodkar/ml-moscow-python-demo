FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY pyproject.toml .

# Install Python dependencies
RUN pip install --no-cache-dir -e .

# Copy application code
COPY titanic_ml/ ./titanic_ml/
COPY data/ ./data/

# Create artifacts directory
RUN mkdir -p artifacts

# Expose port for API
EXPOSE 8000

# Default command to start API server
CMD ["python", "-m", "titanic_ml.cli", "serve", "--host", "0.0.0.0", "--port", "8000"]