import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification

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
    inputs = tokenizer(request.text, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items() if k != "token_type_ids"}

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=-1)
    confidence, predicted_class = probs.max(dim=-1)

    label = "positive" if predicted_class.item() == 1 else "negative"
    return PredictResponse(prediction=label, confidence=round(confidence.item(), 4))
