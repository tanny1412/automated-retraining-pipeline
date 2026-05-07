# Automated ML Retraining Pipeline

An end-to-end MLOps project demonstrating automated model retraining, experiment tracking, and serving.

## What This Builds

A production-style pipeline where a sentiment classification model is trained, versioned, served via API, and automatically retrained when performance degrades.

## Stack

- **Model:** DistilBERT fine-tuned on SST-2 (binary sentiment classification)
- **Serving:** FastAPI
- **Experiment Tracking:** MLflow *(coming in Step 2)*
- **Orchestration:** *(coming in Step 3)*
- **Monitoring + Drift Detection:** *(coming in Step 4)*

## Project Phases

| Step | Focus | Status |
|------|-------|--------|
| 1 | Core train → save → serve loop | In progress |
| 2 | MLflow experiment tracking | Not started |
| 3 | Automated retraining trigger | Not started |
| 4 | Drift detection + monitoring | Not started |
| 5 | Docker + CI/CD | Not started |

## Quickstart

```bash
pip install -r requirements.txt

# Train
python train.py --epochs 3 --batch-size 16

# Serve
uvicorn app:app --reload
```

## API

```
POST /predict
{"text": "this movie was amazing"}
→ {"prediction": "positive", "confidence": 0.98}

GET /health
```
