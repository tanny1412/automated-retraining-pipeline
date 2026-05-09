# Automated ML Retraining Pipeline

An end-to-end MLOps platform demonstrating automated model retraining, experiment tracking, drift detection, and production serving — built to mirror how ML lifecycle management operates at scale.

## What This Builds

A self-healing ML system running on real AWS infrastructure:

- DistilBERT served via a public URL on AWS (not localhost)
- Model weights persisted on EBS — survives pod restarts
- Model weights stored on S3 — init container downloads latest on every pod start
- Automated hourly retraining via Kubernetes CronJob — uploads new weights to S3 after each run
- Drift detection monitoring every prediction via PostgreSQL
- MLflow tracking all experiments + Model Registry managing production lifecycle (versioning, promotion, rollback)
- Prometheus + Grafana dashboards live on AWS
- CI/CD pipeline that builds, tests, and pushes to ECR on every commit
- PostgreSQL for prediction logging — replaces CSV, supports querying and scale

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
- **Prediction Storage:** PostgreSQL — stores every prediction with timestamp, text, label, confidence
- **Monitoring:** Prometheus (metrics scraping) + Grafana (real-time dashboards)
- **Orchestration:** Kubernetes — Deployments, StatefulSets, Services, ConfigMaps, Secrets, CronJob
- **Model Storage:** S3 — single source of truth for best model weights, init container downloads on pod start
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
| 8 | PostgreSQL prediction logging | Done |
| 9 | S3 model storage + init container | Done |
| 10 | Full EKS deployment with PostgreSQL + S3 | Done |
| 11 | End-to-end pipeline test + CI/CD fully wired | Done |
| 12 | MLflow Model Registry — versioning + automated Production promotion | Done |
| 13 | Quality gate — promote only if new model beats Production accuracy | Done |

## How It Works

```
train.py             → fine-tunes DistilBERT, logs to MLflow, saves model, registers new version in MLflow Registry
app.py               → serves predictions via FastAPI, logs every request to PostgreSQL
evaluate.py          → measures current model accuracy on validation set
drift_detector.py    → checks confidence drop + prediction distribution shift
retrain_if_needed.py → orchestrates full loop:
                         check drift → check accuracy → retrain → quality gate (new vs production) → promote if better → upload to S3 → hot-reload API
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

## CI/CD + Deploy Flow

```
git push to main
      ↓
GitHub Actions — tests run (pytest)
      ↓
Docker builds image with --platform linux/amd64
      ↓
Image pushed to ECR with :latest tag
      ↓
kubectl rollout restart deployment/inference-deployment
      ↓
Kubernetes kills old pod, starts new pod
      ↓
imagePullPolicy: Always → pulls :latest image fresh from ECR
      ↓
Init container runs → downloads model weights from S3
      ↓
Inference container starts → serving predictions with new code + latest model
```

## Kubernetes Architecture

```
k8s/
├── inference-deployment.yaml   → FastAPI server (Deployment + LoadBalancer Service + S3 init container)
├── mlflow-deployment.yaml      → MLflow UI (Deployment + NodePort Service)
├── prometheus-deployment.yaml  → Prometheus (Deployment + ConfigMap + NodePort Service)
├── grafana-deployment.yaml     → Grafana (Deployment + NodePort Service)
├── postgres-statefulset.yaml   → PostgreSQL (StatefulSet + headless Service)
├── postgres-secret.yaml        → DB credentials (Kubernetes Secret)
├── volumes.yaml                → PVCs for models and MLflow runs (EBS gp2)
└── retrain-cronjob.yaml        → Automated retraining (CronJob, runs hourly, uploads to S3)
```
