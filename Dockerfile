FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml README.md /app/
COPY titanic_ml /app/titanic_ml
COPY data /app/data

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir .

ENTRYPOINT ["titanic-ml"]
