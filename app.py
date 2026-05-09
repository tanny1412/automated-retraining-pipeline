# CI/CD test
import os
import mlflow
from mlflow.tracking import MlflowClient
import torch
from fastapi import FastAPI, Response, Depends
from sqlalchemy.orm import Session
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from db.database import engine
from db.models import Base, Prediction
from db.dependencies import get_db

PREDICT_COUNT = Counter("predict_requests_total", "Total prediction requests", ["prediction"])
PREDICT_LATENCY = Histogram("predict_latency_seconds", "Model inference latency")
REQUEST_LATENCY = Histogram("request_latency_seconds", "Full API request latency")

Base.metadata.create_all(bind=engine) 

app = FastAPI()

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

def load_model_from_registry():
    artifact_path = mlflow.artifacts.download_artifacts("models:/sentiment-classifier@Production")
    loaded_model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
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
            inputs = tokenizer(request.text, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
            inputs = {k: v.to(device) for k, v in inputs.items() if k != "token_type_ids"}

            with torch.no_grad():
                outputs = model(**inputs)

            probs = torch.softmax(outputs.logits, dim=-1)
            confidence, predicted_class = probs.max(dim=-1)

            label = "positive" if predicted_class.item() == 1 else "negative"
            conf = round(confidence.item(), 4)

        PREDICT_COUNT.labels(prediction=label).inc()

        record = Prediction(text=request.text, prediction=label, confidence=conf)

        db.add(record)
        db.commit()

        return PredictResponse(prediction=label, confidence=conf)

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
