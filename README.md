# Production MLOps Platform — Microservices, Self-Healing, Fully Automated

A production-grade MLOps platform built on AWS EKS — supports any HuggingFace encoder-based sequence classifier on any text classification dataset. Microservices architecture, automated retraining, drift detection, model registry, autoscaling, and real-time alerting. Built to mirror how ML lifecycle management operates at scale.

## What This Builds

A self-healing ML system running on real AWS infrastructure:

- **Microservices** — inference and training are independently deployable services, each with its own Docker image, resource profile, and scaling policy
- **Self-healing** — drift detected → Slack alert fired → human investigates → retrain if needed. Model registry quality gate ensures Production only gets better, never worse
- **Self-bootstrapping** — fresh cluster deploy → CronJob detects no model → trains → registers → serves. Zero manual steps
- **Autoscaling** — HPA scales inference pods 1→3 based on CPU load. GPU nodes for training, CPU nodes for inference
- **Full observability** — Prometheus metrics, Grafana dashboards, Slack alerting on confidence drop — all provisioned as code, zero UI clicks
- **GitOps-ready CI/CD** — every `git push` builds, tests, pushes to ECR, and rolls out new inference pods automatically

**Architecture highlights:**
Two Docker images, two node types, independently scalable. Inference runs on CPU nodes behind a LoadBalancer with HPA. Training runs on GPU nodes as a Kubernetes CronJob. MLflow Model Registry manages versioning, promotion, and rollback. S3 as single source of truth for model weights — init container downloads latest on every pod start. Everything reproducible on a fresh cluster with two manual steps.

## Stack

- **Model:** Any HuggingFace encoder-based sequence classifier (default: DistilBERT on SST-2)
- **Training:** Manual PyTorch loop — AdamW with weight decay, linear warmup + LR decay, gradient clipping, best checkpoint, weighted F1
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
| 14 | Two Dockerfiles — separate inference and training images, self-bootstrapping CronJob | Done |
| 15 | HPA — autoscale inference pods based on CPU utilization | Done |
| 16 | GPU node support — CUDA base image for training, nodeSelector in CronJob | Done |
| 17 | Grafana alerting — Slack notification on confidence drop via provisioning | Done |
| 18 | Configurable model — MODEL_NAME env var, supports any HuggingFace sequence classifier | Done |
| 19 | Full platform configurability — DATASET_NAME, NUM_LABELS, TEXT_COLUMN, VAL_SPLIT env vars | Done |
| 20 | Production-grade training loop — LR scheduler, gradient clipping, best checkpoint, F1, weight decay | Done |
| 21 | Early stopping + /predict_batch endpoint with bulk PostgreSQL logging and MAX_BATCH_SIZE guard | Done |

## How It Works

```
train.py             → fine-tunes any HuggingFace sequence classifier, logs to MLflow, saves best checkpoint, registers version in MLflow Registry
app.py               → serves predictions via FastAPI, maps class index → label via LABEL_MAP, logs every request to PostgreSQL
evaluate.py          → measures accuracy + weighted F1 on validation set
drift_detector.py    → checks confidence drop + prediction distribution shift
retrain_if_needed.py → orchestrates full loop:
                         check drift → check accuracy → retrain → quality gate (new vs production) → promote if better → upload to S3 → hot-reload API
```

## Supported Models

Any HuggingFace encoder-based model compatible with `AutoModelForSequenceClassification`:

| Model | Use case |
|-------|----------|
| `distilbert-base-uncased` | Fast sentiment / binary classification (default) |
| `bert-base-uncased` | General text classification |
| `roberta-base` | Higher accuracy classification |
| `xlm-roberta-base` | Multilingual classification |
| `biobert-base-cased-v1.2` | Biomedical text |
| `ProsusAI/finbert` | Financial sentiment |

**Not supported:** generative models (GPT, LLaMA), seq2seq (T5, BART), token classification (NER).

## Configuration

Every aspect of the platform is configurable via env vars — zero code changes required:

| Env Var | Default | Description |
|---------|---------|-------------|
| `MODEL_NAME` | `distilbert-base-uncased` | Any HuggingFace sequence classifier |
| `NUM_LABELS` | `2` | Number of output classes |
| `LABEL_MAP` | `{"0": "negative", "1": "positive"}` | Class index → label name |
| `DATASET_NAME` | `sst2` | Any HuggingFace dataset |
| `TEXT_COLUMN` | `sentence` | Column containing input text |
| `VAL_SPLIT` | `validation` | Validation split name |
| `MAX_LENGTH` | `128` | Tokenizer max sequence length |
| `EPOCHS` | `3` | Training epochs |
| `BATCH_SIZE` | `16` | Training batch size |
| `LEARNING_RATE` | `2e-5` | Peak learning rate |
| `WEIGHT_DECAY` | `0.01` | AdamW L2 regularization |
| `WARMUP_STEPS` | `100` | Linear warmup steps |
| `GRAD_CLIP` | `1.0` | Gradient clipping max norm |
| `ACCURACY_THRESHOLD` | `0.80` | Min accuracy to skip retraining |
| `PATIENCE` | `3` | Early stopping — epochs without improvement before stopping |
| `MAX_BATCH_SIZE` | `32` | Max texts per /predict_batch request |
| `MLFLOW_MODEL_NAME` | `sentiment-classifier` | MLflow registry model name |
| `MLFLOW_EXPERIMENT_NAME` | `sentiment-classifier` | MLflow experiment name |
| `S3_BUCKET` | `ml-pipeline-models-tanish` | S3 bucket for model weights |
| `BASELINE_CONFIDENCE` | `0.90` | Expected avg confidence for drift detection |
| `BASELINE_DOMINANT_RATIO` | `0.50` | Expected ratio of most common label |
| `CONFIDENCE_THRESHOLD` | `0.10` | Confidence drop that triggers drift alert |
| `DISTRIBUTION_THRESHOLD` | `0.20` | Label distribution shift that triggers drift alert |
| `MIN_PREDICTIONS` | `10` | Min predictions needed before drift detection runs |

**Example — AG News topic classification:**
```yaml
MODEL_NAME: roberta-base
NUM_LABELS: "4"
DATASET_NAME: ag_news
TEXT_COLUMN: text
VAL_SPLIT: test
LABEL_MAP: '{"0": "World", "1": "Sports", "2": "Business", "3": "Sci/Tech"}'
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
# Configure environment
cp .env.example .env
# Edit .env and set SLACK_WEBHOOK_URL to your Slack incoming webhook URL

# Set up Grafana Slack alerting
cp grafana/provisioning/alerting/contact-points.yml.example grafana/provisioning/alerting/contact-points.yml
# Edit contact-points.yml and replace the placeholder URL with your Slack webhook URL

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
GET  /health         → {"status": "ok"}
POST /predict        → {"text": "this movie was amazing"}
                     ← {"prediction": "positive", "confidence": 0.98}
POST /predict_batch  → {"texts": ["great movie", "terrible film"]}
                     ← {"predictions": [{"prediction": "positive", "confidence": 0.97}, ...]}
POST /reload         → hot-swaps model weights without server restart
GET  /metrics        → Prometheus metrics (scraped every 15s)
```

## Monitoring

Prometheus scrapes `/metrics` every 15 seconds. Key metrics:

| Metric | Description |
|---|---|
| `predict_requests_total` | Total predictions by label (positive/negative) |
| `predict_latency_seconds` | Model inference latency histogram |
| `request_latency_seconds` | Full API request latency histogram |

Grafana dashboards at `http://localhost:3000` visualize all metrics in real time.

## CI/CD Setup

Before pushing, configure these in your GitHub repository settings:

**Secrets** (`Settings → Secrets and variables → Actions → Secrets`):
| Secret | Description |
|--------|-------------|
| `AWS_ACCESS_KEY_ID` | AWS IAM access key |
| `AWS_SECRET_ACCESS_KEY` | AWS IAM secret key |
| `POSTGRES_PASSWORD` | PostgreSQL password (optional — defaults to `postgres`) |

**Variables** (`Settings → Secrets and variables → Actions → Variables`):
| Variable | Example | Description |
|----------|---------|-------------|
| `AWS_REGION` | `us-east-1` | AWS region for ECR and EKS |
| `ECR_INFERENCE_REPO` | `ml-pipeline-inference` | ECR repository for inference image |
| `ECR_TRAINING_REPO` | `ml-pipeline-training` | ECR repository for training image |
| `EKS_CLUSTER_NAME` | `ml-pipeline` | EKS cluster name |
| `POSTGRES_USER` | `postgres` | PostgreSQL user (optional) |
| `POSTGRES_DB` | `predictions` | PostgreSQL database name (optional) |

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
└── retrain-cronjob.yaml        → Automated retraining (CronJob, runs hourly, uses training image)
```

## Docker Images

Two separate images — inference and training are independently deployable:

| Image | Dockerfile | Contents | Used by |
|---|---|---|---|
| `ml-pipeline-inference` | `Dockerfile` | `app.py`, `db/` | Inference Deployment |
| `ml-pipeline-training` | `Dockerfile.training` | `train.py`, `evaluate.py`, `retrain_if_needed.py`, `drift_detector.py`, `db/`, `awscli` | Retrain CronJob |
