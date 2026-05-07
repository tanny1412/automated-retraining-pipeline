# Automated ML Retraining Pipeline

An end-to-end MLOps project demonstrating automated model retraining, experiment tracking, drift detection, and serving.

## What This Builds

A production-style closed-loop pipeline where a sentiment classification model is trained, versioned, served via API, monitored for drift, and automatically retrained when performance degrades — with zero manual intervention.

## Stack

- **Model:** DistilBERT fine-tuned on SST-2 (binary sentiment classification)
- **Training:** Manual PyTorch loop
- **Experiment Tracking:** MLflow
- **Serving:** FastAPI + uvicorn
- **Drift Detection:** Confidence + prediction distribution monitoring
- **Device:** CUDA / Apple MPS / CPU (auto-detected)

## Project Phases

| Step | Focus | Status |
|------|-------|--------|
| 1 | Core train → save → serve loop | Done |
| 2 | MLflow experiment tracking | Done |
| 3 | Automated retraining trigger | Done |
| 4 | Drift detection + monitoring | In progress |
| 5 | Docker + CI/CD | Not started |

## How It Works

```
train.py          → fine-tunes DistilBERT, logs to MLflow, saves model
app.py            → serves predictions via FastAPI, logs every request
evaluate.py       → measures current model accuracy on validation set
drift_detector.py → checks confidence + label distribution from request logs
retrain_if_needed.py → orchestrates: evaluate → retrain → hot-reload API
```

## Quickstart

```bash
pip install -r requirements.txt

# Train initial model
python train.py --epochs 3 --batch-size 16

# Serve
uvicorn app:app --reload

# Check if retraining is needed
python retrain_if_needed.py
```

## API

```
GET  /health   → {"status": "ok"}
POST /predict  → {"text": "this movie was amazing"}
               ← {"prediction": "positive", "confidence": 0.98}
POST /reload   → hot-swaps model weights without server restart
```
