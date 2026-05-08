FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y awscli && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY train.py .
COPY retrain_if_needed.py .
COPY evaluate.py .
COPY drift_detector.py .
COPY db/ ./db/

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
