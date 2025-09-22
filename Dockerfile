FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY pyproject.toml ./
COPY titanic_ml/ ./titanic_ml/
COPY README.md ./

# Install the package
RUN pip install --no-cache-dir -e .

# Create directories for data and models
RUN mkdir -p /app/data /app/models /app/output

# Copy data files if they exist
COPY data/ /app/data/

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Expose port (if running web service)
EXPOSE 8000

# Default command
CMD ["titanic-ml", "--help"]