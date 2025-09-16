# Deployment Guide

This document provides comprehensive deployment strategies for the Titanic ML application.

## 🐳 Docker Deployment

### Build Docker Image

```bash
# Build the image
docker build -t titanic-ml .

# Run the container
docker run -it titanic-ml

# Run training
docker run -v $(pwd)/data:/app/data -v $(pwd)/models:/app/models titanic-ml \
  titanic-ml train --data-path /app/data --output-path /app/models
```

### Docker Compose

Create a `docker-compose.yml`:

```yaml
version: '3.8'

services:
  titanic-ml:
    build: .
    volumes:
      - ./data:/app/data
      - ./models:/app/models
      - ./output:/app/output
    environment:
      - PYTHONUNBUFFERED=1
    command: titanic-ml train --data-path /app/data --output-path /app/models

  prediction-service:
    build: .
    volumes:
      - ./models:/app/models
      - ./input:/app/input
      - ./output:/app/output
    ports:
      - "8000:8000"
    command: python -m titanic_ml.api  # If you implement web API
```

## ☁️ Cloud Deployment

### AWS Deployment

#### Using AWS Lambda

1. **Create Lambda Package**:
   ```bash
   # Install dependencies in Lambda-compatible way
   pip install -e . -t lambda_package/
   cp -r titanic_ml/ lambda_package/
   cd lambda_package && zip -r ../titanic-ml-lambda.zip .
   ```

2. **Deploy Lambda Function**:
   ```python
   # lambda_function.py
   import json
   from titanic_ml.pipeline.pipeline import TitanicMLPipeline
   from titanic_ml.core.models import PipelineConfig
   
   def lambda_handler(event, context):
       # Load pre-trained model from S3
       # Make prediction
       # Return result
       pass
   ```

#### Using AWS SageMaker

```python
# sagemaker_deploy.py
import sagemaker
from sagemaker.sklearn.estimator import SKLearn

# Create SageMaker estimator
sklearn_estimator = SKLearn(
    entry_point='train_script.py',
    framework_version='1.0-1',
    instance_type='ml.m5.large',
    role=sagemaker.get_execution_role()
)

# Deploy model
predictor = sklearn_estimator.deploy(
    instance_type='ml.m5.large',
    initial_instance_count=1
)
```

### Google Cloud Platform

#### Using Cloud Run

1. **Build and Push Image**:
   ```bash
   # Build for GCP
   docker build -t gcr.io/your-project/titanic-ml .
   docker push gcr.io/your-project/titanic-ml
   ```

2. **Deploy to Cloud Run**:
   ```bash
   gcloud run deploy titanic-ml \
     --image gcr.io/your-project/titanic-ml \
     --platform managed \
     --region us-central1 \
     --allow-unauthenticated
   ```

### Azure Deployment

#### Using Azure Container Instances

```bash
az container create \
  --resource-group myResourceGroup \
  --name titanic-ml \
  --image titanic-ml:latest \
  --cpu 2 --memory 4 \
  --environment-variables PYTHONUNBUFFERED=1
```

## 🖥️ On-Premise Deployment

### Production Server Setup

1. **Install Dependencies**:
   ```bash
   # Create production environment
   python -m venv /opt/titanic-ml/venv
   source /opt/titanic-ml/venv/bin/activate
   
   # Install application
   pip install -e .
   ```

2. **Create Service File** (`/etc/systemd/system/titanic-ml.service`):
   ```ini
   [Unit]
   Description=Titanic ML Service
   After=network.target
   
   [Service]
   Type=simple
   User=titanic-ml
   WorkingDirectory=/opt/titanic-ml
   Environment=PATH=/opt/titanic-ml/venv/bin
   ExecStart=/opt/titanic-ml/venv/bin/titanic-ml train --data-path /data --output-path /models
   Restart=always
   
   [Install]
   WantedBy=multi-user.target
   ```

3. **Enable and Start Service**:
   ```bash
   sudo systemctl enable titanic-ml
   sudo systemctl start titanic-ml
   ```

## 📊 Batch Processing Deployment

### Cron Jobs for Regular Training

```bash
# Add to crontab (crontab -e)
# Train model daily at 2 AM
0 2 * * * /opt/titanic-ml/venv/bin/titanic-ml train --data-path /data/daily --output-path /models/$(date +\%Y\%m\%d)

# Generate predictions hourly
0 * * * * /opt/titanic-ml/venv/bin/titanic-ml predict --model-path /models/latest --input /data/new --output /predictions/$(date +\%Y\%m\%d\%H).csv
```

### Apache Airflow DAG

```python
# titanic_ml_dag.py
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash_operator import BashOperator

default_args = {
    'owner': 'data-team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

dag = DAG(
    'titanic_ml_pipeline',
    default_args=default_args,
    description='Titanic ML training and prediction pipeline',
    schedule_interval='@daily',
    catchup=False
)

train_task = BashOperator(
    task_id='train_model',
    bash_command='titanic-ml train --data-path /data --output-path /models/{{ ds }}',
    dag=dag
)

predict_task = BashOperator(
    task_id='make_predictions',
    bash_command='titanic-ml predict --model-path /models/{{ ds }} --input /data/test.csv --output /predictions/{{ ds }}.csv',
    dag=dag
)

train_task >> predict_task
```

## 🔒 Security Considerations

### Authentication and Authorization

```python
# For API deployment, add authentication
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    # Implement token verification
    if credentials.credentials != "your-secret-token":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)
    return credentials
```

### Environment Variables

```bash
# .env file for production
DATABASE_URL=postgresql://user:password@localhost/titanic_ml
MODEL_PATH=/secure/models
SECRET_KEY=your-secret-key
LOG_LEVEL=INFO
```

## 📈 Monitoring and Logging

### Application Monitoring

```python
# Add to your application
import logging
from prometheus_client import Counter, Histogram, generate_latest

# Metrics
prediction_counter = Counter('predictions_total', 'Total predictions made')
training_histogram = Histogram('training_duration_seconds', 'Training duration')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/var/log/titanic-ml.log'),
        logging.StreamHandler()
    ]
)
```

### Health Checks

```python
# health_check.py
from titanic_ml.pipeline.pipeline import TitanicMLPipeline
from pathlib import Path

def health_check():
    try:
        # Check if models exist
        model_path = Path("/models")
        if not model_path.exists():
            return {"status": "unhealthy", "reason": "Models not found"}
        
        # Try loading pipeline
        pipeline = TitanicMLPipeline.load_trained_pipeline(model_path)
        return {"status": "healthy", "models": str(model_path)}
    except Exception as e:
        return {"status": "unhealthy", "reason": str(e)}
```

## 🚀 Performance Optimization

### Resource Allocation

- **CPU**: 2-4 cores for training, 1-2 for prediction
- **Memory**: 4-8GB for training, 2-4GB for prediction
- **Storage**: SSD recommended, 10-50GB depending on data size

### Scaling Strategies

1. **Horizontal Scaling**: Multiple instances behind load balancer
2. **Vertical Scaling**: Increase CPU/memory for single instance
3. **GPU Acceleration**: For XGBoost/LightGBM if available

### Caching Strategy

```python
# Redis caching for predictions
import redis
import pickle

redis_client = redis.Redis(host='localhost', port=6379, db=0)

def cached_prediction(features_hash, features):
    # Check cache
    cached = redis_client.get(features_hash)
    if cached:
        return pickle.loads(cached)
    
    # Make prediction
    result = model.predict(features)
    
    # Cache result (expire in 1 hour)
    redis_client.setex(features_hash, 3600, pickle.dumps(result))
    return result
```

## 🔧 Configuration Management

### Environment-specific Configurations

```yaml
# config/production.yaml
database:
  host: prod-db.example.com
  port: 5432
  name: titanic_ml_prod

models:
  path: /secure/models
  cache_size: 100

logging:
  level: INFO
  file: /var/log/titanic-ml.log
```

```yaml
# config/staging.yaml
database:
  host: staging-db.example.com
  port: 5432
  name: titanic_ml_staging

models:
  path: /tmp/models
  cache_size: 10

logging:
  level: DEBUG
  file: /tmp/titanic-ml.log
```

This deployment guide covers various scenarios from simple Docker deployments to enterprise-scale cloud deployments. Choose the approach that best fits your infrastructure and requirements.