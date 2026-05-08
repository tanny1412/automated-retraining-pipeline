# Automated ML Retraining Pipeline

An end-to-end MLOps platform demonstrating automated model retraining, experiment tracking, drift detection, and production serving — built to mirror how ML lifecycle management operates at scale.

## What This Builds

A self-healing ML system running on real AWS infrastructure:

- DistilBERT served via a public URL on AWS (not localhost)
- Model weights persisted on EBS — survives pod restarts
- Automated hourly retraining via Kubernetes CronJob
- Drift detection monitoring every prediction
- MLflow tracking all experiments
- Prometheus + Grafana dashboards live on AWS
- CI/CD pipeline that builds, tests, and pushes to ECR on every commit

**What makes this rare for a portfolio project:**
Most candidates have notebooks. Some have Docker. Very few have Kubernetes on real cloud infrastructure with persistent storage, IAM roles, CSI drivers, and a CI/CD pipeline attached.

## Stack

- **Model:** DistilBERT fine-tuned on SST-2 (binary sentiment classification)
- **Training:** Manual PyTorch loop with gradient optimization and per-epoch validation
- **Experiment Tracking:** MLflow (params, metrics, artifacts per run)
- **Serving:** FastAPI + uvicorn
- **Drift Detection:** Confidence degradation + prediction distribution shift monitoring
- **Containerization:** Docker + docker-compose
- **CI/CD:** GitHub Actions — automated testing + Docker build on every push
- **Monitoring:** Prometheus (metrics scraping) + Grafana (real-time dashboards)
- **Orchestration:** Kubernetes — Deployments, Services, ConfigMaps, CronJob
- **Device:** CUDA / Apple MPS / CPU (auto-detected)

## Project Phases

| Step | Focus | Status |
|------|-------|--------|
| 1 | Core train → save → serve loop | Done |
| 2 | MLflow experiment tracking | Done |
| 3 | Automated retraining trigger | Done |
| 4 | Drift detection + monitoring | Done |
| 5 | Docker + CI/CD | Done |
| 6 | Prometheus + Grafana | Done |
| 7 | Kubernetes orchestration | Done |

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

### Kubernetes (minikube)
```bash
minikube start
eval $(minikube docker-env)
docker build -t automated-retraining-pipeline-inference:latest .

# Mount model weights and data into minikube
minikube mount ./models:/mnt/models &
minikube mount ./mlruns:/mnt/mlruns &
minikube mount .:/mnt/project &

# Deploy all services
kubectl apply -f k8s/

# Get service URLs
minikube service inference-service --url
minikube service mlflow-service --url
minikube service prometheus-service --url
minikube service grafana-service --url

# Trigger retraining manually
kubectl create job retrain-test --from=cronjob/retrain-cronjob
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

## Kubernetes Architecture

```
k8s/
├── inference-deployment.yaml   → FastAPI server (Deployment + NodePort Service)
├── mlflow-deployment.yaml      → MLflow UI (Deployment + NodePort Service)
├── prometheus-deployment.yaml  → Prometheus (Deployment + ConfigMap + NodePort Service)
├── grafana-deployment.yaml     → Grafana (Deployment + NodePort Service)
└── retrain-cronjob.yaml        → Automated retraining (CronJob, runs hourly)
```
