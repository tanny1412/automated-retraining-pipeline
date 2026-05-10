# CI/CD test
import os

MODEL_NAME = os.getenv("MODEL_NAME", "distilbert-base-uncased")
NUM_LABELS = int(os.getenv("NUM_LABELS", "2"))
import json
LABEL_MAP = json.loads(os.getenv("LABEL_MAP", '{"0": "negative", "1": "positive"}'))
MAX_LENGTH = int(os.getenv("MAX_LENGTH", "128"))
MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", "32"))
import mlflow
from mlflow.tracking import MlflowClient
import torch
from fastapi import FastAPI, Response, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from db.database import engine
from db.models import Base, Prediction
from db.dependencies import get_db

PREDICT_COUNT = Counter("predict_requests_total", "Total prediction requests", ["prediction"])
PREDICT_LATENCY = Histogram("predict_latency_seconds", "Model inference latency")
REQUEST_LATENCY = Histogram("request_latency_seconds", "Full API request latency")
CONFIDENCE = Gauge("prediction_confidence", "Confidence of predictions")

Base.metadata.create_all(bind=engine) 

app = FastAPI()

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

def load_model_from_registry():
    try:
        artifact_path = mlflow.artifacts.download_artifacts("models:/sentiment-classifier@Production")
        weights_path = f"{artifact_path}/best_model.pt"
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"No weights at {weights_path}")
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"Registry load failed ({e}), falling back to disk")
        artifact_path = "models"
    loaded_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)
    loaded_model.load_state_dict(torch.load(f"{artifact_path}/best_model.pt", map_location=device))
    loaded_model.to(device)
    loaded_model.eval()
    return loaded_model

tokenizer = AutoTokenizer.from_pretrained("models/tokenizer")
model = load_model_from_registry()

class PredictRequest(BaseModel):
    text: str

class PredictResponse(BaseModel):
    prediction: str
    confidence: float

@app.get("/")
def root():
    return {"message": "ML API running"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/reload")
def reload_model():
    global model
    model = load_model_from_registry()
    return {"status": "model reloaded"}

class RollbackRequest(BaseModel):
    version: str

@app.post("/rollback")
def rollback(request: RollbackRequest):
    global model
    client = MlflowClient()
    client.set_registered_model_alias("sentiment-classifier", "Production", request.version)
    model = load_model_from_registry()
    return {"status": f"rolled back to version {request.version}"}

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest, db: Session = Depends(get_db)):
    with REQUEST_LATENCY.time():
        with PREDICT_LATENCY.time():
            inputs = tokenizer(request.text, return_tensors="pt", truncation=True, padding="max_length", max_length=MAX_LENGTH)
            inputs = {k: v.to(device) for k, v in inputs.items() if k != "token_type_ids"}

            with torch.no_grad():
                outputs = model(**inputs)

            probs = torch.softmax(outputs.logits, dim=-1)
            confidence, predicted_class = probs.max(dim=-1)

            label = LABEL_MAP[str(predicted_class.item())]
            conf = round(confidence.item(), 4)

        PREDICT_COUNT.labels(prediction=label).inc()
        CONFIDENCE.set(conf)

        record = Prediction(text=request.text, prediction=label, confidence=conf)

        db.add(record)
        db.commit()

        return PredictResponse(prediction=label, confidence=conf)

class PredictBatchRequest(BaseModel):
    texts: list[str]

class PredictBatchResponse(BaseModel):
    predictions: list[PredictResponse]

@app.post("/predict_batch", response_model=PredictBatchResponse)
def predict_batch(request: PredictBatchRequest, db: Session = Depends(get_db)):
    if len(request.texts) > MAX_BATCH_SIZE:
        raise HTTPException(status_code=400, detail=f"Batch size {len(request.texts)} exceeds maximum of {MAX_BATCH_SIZE}")

    inputs = tokenizer(request.texts, return_tensors="pt", truncation=True, padding="max_length", max_length=MAX_LENGTH)
    inputs = {k: v.to(device) for k, v in inputs.items() if k != "token_type_ids"}

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=-1)
    results = []
    records = []
    for i in range(len(request.texts)):
        confidence, predicted_class = probs[i].max(dim=-1)
        label = LABEL_MAP[str(predicted_class.item())]
        conf = round(confidence.item(), 4)
        PREDICT_COUNT.labels(prediction=label).inc()
        results.append(PredictResponse(prediction=label, confidence=conf))
        records.append(Prediction(text=request.texts[i], prediction=label, confidence=conf))

    db.bulk_save_objects(records)
    db.commit()

    return PredictBatchResponse(predictions=results)

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
