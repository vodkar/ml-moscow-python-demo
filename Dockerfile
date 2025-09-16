FROM python:3.12-slim

WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir -e .

ENTRYPOINT ["titanic-ml"]
CMD ["train", "--train-csv", "data/train.csv", "--artifacts-dir", "artifacts"]
