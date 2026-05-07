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
- **Containerization:** Docker + docker-compose
- **CI/CD:** GitHub Actions — tests + Docker build on every push
- **Monitoring:** Prometheus (metrics scraping) + Grafana (dashboards)
- **Device:** CUDA / Apple MPS / CPU (auto-detected)

## Project Phases

| Step | Focus | Status |
|------|-------|--------|
| 1 | Core train → save → serve loop | Done |
| 2 | MLflow experiment tracking | Done |
| 3 | Automated retraining trigger | Done |
| 4 | Drift detection + monitoring | Done |
| 5 | Docker + CI/CD | Done |

## How It Works

```
train.py             → fine-tunes DistilBERT, logs to MLflow, saves model
app.py               → serves predictions via FastAPI, logs every request to CSV
evaluate.py          → measures current model accuracy on validation set
drift_detector.py    → checks confidence drop + prediction distribution shift
retrain_if_needed.py → orchestrates full loop:
                         check drift → check accuracy → retrain → hot-reload API
```

## Quickstart

### Local
```bash
pip install -r requirements.txt

# Train initial model
python train.py --epochs 3 --batch-size 16

# Serve
uvicorn app:app --reload

# Check if retraining is needed
python retrain_if_needed.py
```

### Docker
```bash
# Start all services
docker compose up

# API:        http://localhost:8000
# MLflow UI:  http://localhost:5001
# Prometheus: http://localhost:9090
# Grafana:    http://localhost:3000  (admin/admin)
```

## API

```
GET  /health   → {"status": "ok"}
POST /predict  → {"text": "this movie was amazing"}
               ← {"prediction": "positive", "confidence": 0.98}
POST /reload   → hot-swaps model weights without server restart
GET  /metrics  → Prometheus metrics (scraped every 15s)
```

## Monitoring

Prometheus scrapes `/metrics` every 15 seconds. Key metrics:

| Metric | Description |
|---|---|
| `predict_requests_total` | Total predictions by label (positive/negative) |
| `predict_latency_seconds` | Model inference latency histogram |
| `request_latency_seconds` | Full API request latency histogram |

Grafana dashboards at `http://localhost:3000` visualize all metrics in real time.
