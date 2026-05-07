import csv
import os
import torch
from datetime import datetime
from fastapi import FastAPI, Response
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

PREDICT_COUNT = Counter("predict_requests_total", "Total prediction requests", ["prediction"])
PREDICT_LATENCY = Histogram("predict_latency_seconds", "Model inference latency")
REQUEST_LATENCY = Histogram("request_latency_seconds", "Full API request latency")

PREDICTIONS_LOG = "predictions.csv"

def init_log():
    if not os.path.exists(PREDICTIONS_LOG):
        with open(PREDICTIONS_LOG, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "text", "prediction", "confidence"])

init_log()

app = FastAPI()

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained("models/tokenizer")
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
model.load_state_dict(torch.load("models/best_model.pt", map_location=device))
model.to(device)
model.eval()

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
    model.load_state_dict(torch.load("models/best_model.pt", map_location=device))
    model.to(device)
    model.eval()
    return {"status": "model reloaded"}

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
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

        with open(PREDICTIONS_LOG, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([datetime.now().isoformat(), request.text, label, conf])

        return PredictResponse(prediction=label, confidence=conf)

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
