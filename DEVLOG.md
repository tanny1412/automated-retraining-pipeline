# Dev Log — Automated ML Retraining Pipeline

A running log of decisions, problems, and learnings as I built this project.
Written for interview prep and to document real engineering thinking — not just the final result.

---

## Step 1 — Core ML Loop (train → save → serve)

**Goal:** Prove the fundamental lifecycle works before adding any tooling around it.
The principle: MLflow, Airflow, Docker should wrap a working system — not replace understanding.

---

### Decision: DistilBERT on SST-2

Chose `distilbert-base-uncased` fine-tuned on SST-2 (binary sentiment classification).

**Why DistilBERT over BERT:**
- 40% smaller, 60% faster, retains ~97% of BERT's accuracy
- Fast enough to iterate locally on an M2 MacBook without needing cloud
- Well-understood architecture — easy to explain in interviews

**Why SST-2:**
- Clean, well-labeled dataset (67K examples)
- Binary labels (positive/negative) — simple enough to focus on the pipeline, not the problem
- Standard HuggingFace benchmark — reproducible results

---

### Decision: Manual PyTorch training loop over HuggingFace Trainer

Could have used `Trainer` from HuggingFace which abstracts the entire loop.
Chose to write it manually instead.

**Why:**
- The training loop is the thing you need to understand and explain
- `Trainer` hides: gradient accumulation, optimizer steps, eval logic — all things interviewers ask about
- Manual loop = I can explain every line

**The loop in plain English:**
1. Feed a batch of 16 examples to the model (forward pass)
2. Model outputs a loss — how wrong it was
3. `loss.backward()` — compute how much each weight contributed to being wrong
4. `optimizer.step()` — nudge every weight to be a little less wrong
5. Repeat 4,200 times per epoch (67K examples / batch size 16)

---

### Decision: Separate inference from training

`app.py` (the API server) only loads and serves the model. It never retrains.
`train.py` only trains. It never serves.

**Why this matters:**
- Separation of concerns — a core software engineering and MLOps principle
- In production, retraining is triggered by a scheduler or drift detector, not by an API request
- Makes each component independently deployable and testable

---

### Decision: Parameterize training via CLI args

```bash
python train.py --epochs 3 --batch-size 16 --lr 2e-5 --max-samples 500
```

**Why:**
- Later the retraining pipeline triggers `train.py` automatically with different params
- No hardcoded values = no code changes needed between runs
- `--max-samples` flag added for fast experimentation before full training runs

---

### Problem: No visibility into training progress

First run looked stuck — no output while training.

**Fix:** Added `tqdm` progress bars and replaced all `print()` with Python `logging`.

**Why logging over print:**
- Adds timestamps automatically to every line
- Has log levels (`INFO`, `WARNING`, `ERROR`) — in production you filter by level
- Can be routed to a file or monitoring system without changing the code
- `print` is for scripts; `logging` is for systems

---

### Decision: Virtual environment scoped to project

Used `.venv` inside the project directory instead of a global environment.

**Why:**
- VSCode auto-detects it
- Keeps dependencies isolated to this project
- `.venv/` added to `.gitignore` — environments are never committed, only `requirements.txt` is

---

### What the LOAD REPORT warnings mean

When loading `distilbert-base-uncased` for sequence classification, you see:

```
UNEXPECTED: vocab_layer_norm, vocab_transform, vocab_projector  ← MLM head, not needed
MISSING: classifier, pre_classifier                              ← classification head, newly initialized
```

This is expected. The base checkpoint was pretrained for Masked Language Modeling.
We're loading it for classification — the MLM head is discarded, a fresh classification head is added.
Fine-tuning trains both the transformer body and the new head.

---

### Model saving structure

```
models/
├── best_model.pt      ← learned weights (state_dict)
└── tokenizer/         ← tokenizer config + vocab
```

**Why save state_dict and not the whole model:**
- Smaller file
- Portable across PyTorch versions
- To load: reconstruct the architecture, then load weights into it
- Tokenizer saved alongside model because inference requires the exact same tokenizer used in training

---

---

## Step 2 — MLflow Experiment Tracking

Added MLflow directly into `train.py` rather than as a separate step.
The original plan said "add MLflow after Step 1" — but once the training loop worked, it made sense to integrate immediately while the code was fresh.

**What MLflow tracks per run:**
- Params: `epochs`, `batch_size`, `lr`, `max_samples`, `model`, `dataset` — logged once at start
- Metrics: `loss` and `val_accuracy` per epoch with `step` so charts show progression over time
- Artifacts: full `models/` directory (weights + tokenizer) — every run has its own saved model

**Why `step=epoch+1` in log_metrics:**
Without `step`, MLflow stores metrics as a flat value with no time axis.
With `step`, you get a chart showing loss going down epoch by epoch — that's what you show in interviews and LinkedIn posts.

**Why log artifacts:**
Params and metrics tell you how a run performed.
Artifacts let you go back and load the exact model from any run.
This is the foundation of model versioning — before you need a full model registry.

**Decision: MLflow runs locally for now**
MLflow stores everything in `./mlruns/` by default.
No server, no cloud, no setup — just `mlflow ui` to view.
In production this would point to a remote tracking server (MLflow on EC2, Databricks, etc).

---

## Step 3 — Automated Retraining Trigger

Built a closed-loop retraining pipeline across 3 scripts:

**`evaluate.py`** — loads current model, runs it on SST-2 validation set, exits with code 0 (healthy) or 1 (needs retraining) based on a configurable accuracy threshold. Runs fast — no weight updates, just a forward pass on 872 examples.

**`retrain_if_needed.py`** — orchestrates the loop. Calls `evaluate.py` via subprocess, checks exit code, triggers `train.py` if needed, then calls `POST /reload` on the running API.

**`/reload` endpoint in `app.py`** — hot-swaps model weights in memory without restarting the server. Uses `global model` + `load_state_dict()` to replace weights in-place.

**Why subprocess instead of importing?** Each script stays independently runnable from CLI or any orchestrator (cron, Airflow, GitHub Actions) without code changes. Exit codes are the universal signal.

**Why `global model` in /reload?** The model is loaded at module level on startup. To replace it from inside a function, Python needs `global` to know we're modifying the outer variable, not creating a local one.

### Problem: MLflow schema mismatch after version upgrade
After upgrading MLflow, the local `mlruns/` database schema was outdated.
**Fix for dev:** `rm -rf mlruns/` — MLflow recreates it with the correct schema.
**Fix for production:** `mlflow db upgrade sqlite:///mlruns/mlruns.db` before running any training after an upgrade.

---

---

## Step 4 — Drift Detection

Built `drift_detector.py` that reads `predictions.csv` (logged by `app.py` on every request) and checks two signals:

**Signal 1 — Confidence drift:** Tracks average prediction confidence across recent requests. If it drops more than 10% below baseline (0.90), the model is uncertain about incoming data — likely seeing text it wasn't trained on.

**Signal 2 — Prediction distribution drift:** Tracks ratio of positive vs negative predictions. SST-2 is ~50/50 so baseline positive ratio is 0.50. If it shifts more than 20%, the incoming data distribution has changed.

**Why two signals:**
- Confidence drift = model doesn't recognize new data style (leading indicator)
- Distribution drift = input data looks different from training data (structural shift)
- A model can be confident but wrong (distribution drift without confidence drift)
- Together they give a fuller picture than either alone

**Real example from testing:**
Sent Gen Z slang inputs ("this movie lowkey cooked", "mid ahh ending fr", "absolute cinema no cap") — model predicted negative for all of them with high confidence. It understood the words but not the meaning. Positive ratio shifted to 85% (from expected 50%) when we then stress-tested with positive inputs — distribution drift triggered at 0.35 shift vs 0.20 threshold.

**Why log to CSV instead of a database:**
At this scale (thousands of predictions), CSV is sufficient. No extra dependencies, human-readable, easy to inspect. In production you'd stream to BigQuery, S3, or a time-series database.

**How it plugs into the pipeline:**
`retrain_if_needed.py` now checks drift first, then accuracy. Either signal triggers retraining. The inner `if drift` / `if not healthy` blocks are only for logging *why* retraining triggered — `retrain()` sits at the `else` level and runs for both cases.

---

## Step 5 — Docker + CI/CD

**Dockerfile** — inference-only container. Key decision: model weights are NOT baked into the image. They mount at runtime via Docker volume. This means retraining updates the model without rebuilding the image — critical for automated pipelines.

**docker-compose.yml** — four services:
- `inference` — FastAPI server
- `mlflow` — experiment tracking UI
- `prometheus` — metrics scraping every 15s
- `grafana` — dashboards

Shared volume (`./models:/app/models`) connects inference and training — same folder both read/write. This is the local equivalent of S3/model registry in production.

**GitHub Actions CI** — two jobs on every push:
1. `test` — installs deps, runs pytest (7 tests)
2. `docker` — builds image, only runs if tests pass (`needs: test`)

**Key lesson from CI:** `models/` is gitignored so `COPY models/` in Dockerfile failed in CI. Fix was removing it — models come in via volume, not baked into image. This is actually MORE correct: image = immutable code, weights = mutable data.

---

## Step 6 — Prometheus + Grafana

Three metrics instrumented in `app.py`:
- `predict_requests_total` — Counter, labeled by prediction class. Tracks volume and positive/negative split.
- `predict_latency_seconds` — Histogram, measures model inference time only.
- `request_latency_seconds` — Histogram, measures full request including CSV logging + response.

**Why two latency metrics:** Inference latency isolates model performance. Request latency is what users actually experience. Comparing them shows overhead of non-ML work (CSV logging ≈ 0ms in practice).

**MLflow vs Prometheus distinction:**
- MLflow = offline, training-time. "Which run had best accuracy?"
- Prometheus = live, serving-time. "How many requests right now, how fast?"
- Both needed. A well-trained model can still be a poorly-operating service.

**Prometheus scraping:** `prometheus.yml` points at `inference:8000/metrics`. Container name `inference` resolves via Docker's internal DNS — not localhost, not IP. This is how services communicate inside Docker networks.

---

## Step 7 — Kubernetes

Migrated the full docker-compose stack to Kubernetes running on minikube locally.

**The core mapping:**

| Docker Compose | Kubernetes |
|---|---|
| `service:` block | `Deployment` + `Service` |
| `ports:` | `Service` (NodePort) |
| `volumes:` | `hostPath` volume + `minikube mount` |
| `environment:` | `env:` in container spec |

**4 files built in `k8s/`:**
- `inference-deployment.yaml` — FastAPI server, model weights mounted via hostPath
- `mlflow-deployment.yaml` — MLflow UI, mlruns mounted via hostPath
- `prometheus-deployment.yaml` — Prometheus with a ConfigMap holding the scrape config
- `grafana-deployment.yaml` — Grafana with admin password set via env var

**Why ConfigMap for Prometheus:**
In Compose you mount a single file (`./prometheus.yml`). Kubernetes can't mount individual files from your Mac directly. Instead you store the config in a ConfigMap — a Kubernetes object that holds key-value data — and mount that as a volume. Same result, different mechanism.

**Why `imagePullPolicy: Never` for inference:**
The inference image is built locally inside minikube's Docker daemon (via `eval $(minikube docker-env)`). Without this flag, Kubernetes tries to pull from DockerHub and fails with "image not found".

**Why `minikube mount`:**
minikube runs inside a Docker VM on Mac. `hostPath` volumes point to paths inside that VM, not your Mac. `minikube mount ./models:/mnt/models` bridges the gap — it makes your Mac's folder visible inside the VM.

**OOMKilled lesson:**
First deployment crashed with `OOMKilled` — memory limit was set to 1Gi but DistilBERT needs ~2-3Gi to load. Fixed by raising the limit to 3Gi. Always size memory limits for ML models generously.

**NodePort vs the tunnel:**
Set `nodePort: 30800` but `minikube service --url` returned a different port (e.g. 59997). On minikube with the Docker driver, you can't reach the VM's NodePort directly — Docker is in the way. `minikube service --url` creates a tunnel through Docker and assigns a random localhost port. On a real cloud cluster, NodePort is directly accessible on the node IP.

**Feature branch workflow:**
Built everything on `feature/kubernetes`, opened a PR on GitHub, merged into main — same flow as a real team.

---

### CronJob — Automated Retraining on Kubernetes

Added `k8s/retrain-cronjob.yaml` to close the retraining loop inside Kubernetes.

**What it does:** Runs `retrain_if_needed.py` on a schedule (`0 * * * *` = every hour). Kubernetes spins up a pod, runs the script, pod exits. If drift or low accuracy is detected, it retrains and hot-reloads the API.

**Why `kind: CronJob` and not `kind: Deployment`:**
- Deployments run forever — they restart if the pod exits
- CronJobs run on a schedule and exit — `restartPolicy: OnFailure` instead of `Always`
- CronJob → Job → Pod is the nesting. Job tracks completion, CronJob handles scheduling

**Why reuse the inference image:**
The inference image already has all dependencies installed. Adding the retraining scripts (`retrain_if_needed.py`, `evaluate.py`, `drift_detector.py`) to the same image with `COPY` avoids maintaining a second Dockerfile.

**CPU vs MPS lesson:**
Locally `evaluate.py` runs on Apple MPS (fast). Inside the container it runs on CPU with 0.5 cores (slow). In production you'd schedule this on a GPU node. The logic is correct — it's a resource constraint, not a bug.

**How to trigger manually (without waiting for the hour):**
```bash
kubectl create job retrain-test --from=cronjob/retrain-cronjob
kubectl logs <pod-name> -f
```

---

## Step 8 — EKS Deployment

Migrated the full Kubernetes stack from local minikube to AWS EKS.

**What changed from minikube → EKS:**

| minikube | EKS |
|---|---|
| Local Docker image | ECR (Elastic Container Registry) |
| `imagePullPolicy: Never` | `imagePullPolicy: Always` |
| `hostPath` volumes | `PersistentVolumeClaim` backed by EBS |
| `minikube mount` | EBS CSI driver provisions disks automatically |
| `NodePort` + tunnel | `LoadBalancer` → real public AWS URL |

**Why 3 PVCs:**
- `models-pvc` — model weights. Inference reads, CronJob writes after retraining
- `predictions-pvc` — predictions.csv. Inference writes every request, CronJob reads for drift detection
- `mlruns-pvc` — MLflow experiment runs

Each PVC has one clear purpose. Shared between pods that need the same data.

**EBS vs EFS tradeoff:**
Used EBS (`ReadWriteOnce`) — one node, one pod at a time. Sufficient for `replicas: 1`.
In production with multiple replicas, use EFS (`ReadWriteMany`) — shared across all nodes simultaneously.

**Problem 1: PVCs stuck in Pending**
PVCs were created without `storageClassName`. EKS had no default StorageClass set so PVCs didn't know how to provision disks.
Fix: added `storageClassName: gp2` explicitly to all PVCs in `volumes.yaml`. Deleted and reapplied PVCs.

**Problem 2: EBS CSI driver not installed**
Even with `storageClassName: gp2`, PVCs stayed Pending. `kubectl get pods -n kube-system | grep ebs` returned nothing — the EBS CSI driver wasn't installed.
Fix: `eksctl create addon --name aws-ebs-csi-driver --cluster ml-pipeline --region us-east-1`

**Problem 3: OIDC not enabled — driver had no IAM permissions**
The EBS CSI driver needs to call AWS APIs to create EBS volumes. Without OIDC, the driver pod has no AWS identity so AWS rejects the request.

The mental model:
```
OIDC     = the driver's ID card
IAM policy = the guest list (AmazonEBSCSIDriverPolicy)
AWS      = the bouncer
Driver shows ID → bouncer checks guest list → allowed to create EBS disk
```

Fix:
```bash
eksctl utils associate-iam-oidc-provider --cluster ml-pipeline --region us-east-1 --approve
eksctl create iamserviceaccount \
  --name ebs-csi-controller-sa \
  --namespace kube-system \
  --cluster ml-pipeline \
  --region us-east-1 \
  --attach-policy-arn arn:aws:iam::aws:policy/service-role/AmazonEBSCSIDriverPolicy \
  --approve \
  --override-existing-serviceaccounts
```

**Problem 5: models/tokenizer not found — EBS volume empty on first start**
Pod crashed with `models/tokenizer is not a local folder`. The EBS volume was provisioned empty — no model weights on it yet. `kubectl cp` is the manual fix for initial setup.

Used a `busybox` loader pod to copy files into the PVC without a race condition:
```bash
kubectl run model-loader --image=busybox --restart=Never \
  --overrides='{"spec":{"containers":[{"name":"model-loader","image":"busybox","command":["sleep","3600"],"volumeMounts":[{"name":"models","mountPath":"/app/models"}]}],"volumes":[{"name":"models","persistentVolumeClaim":{"claimName":"models-pvc"}}]}}'
kubectl cp models/ model-loader:/app/models/
kubectl exec model-loader -- sh -c "mv /app/models/models/* /app/models/ && rm -rf /app/models/models"
kubectl delete pod model-loader
```

Note: `kubectl cp` creates a nested directory (`models/models/`) — always verify with `kubectl exec -- ls` and move files up if needed.

Long-term fix: use an init container that pulls weights from S3 on pod startup. See "Future Improvement" section below.

**Problem 4: exec format error — ARM vs x86 architecture mismatch**
Inference pod crashed immediately with `exec format error`. The Docker image was built on an M2 Mac (ARM architecture) but EKS t3.medium nodes run x86 (AMD64). ARM binaries cannot execute on x86.

This is below Docker's abstraction layer — Docker guarantees same OS/dependencies/config, but cannot abstract CPU instruction sets.

Fix: rebuild with `--platform linux/amd64` to force x86 compilation:
```bash
docker build --platform linux/amd64 -t <ecr-uri>:latest .
docker push <ecr-uri>:latest
```

Also added `--platform linux/amd64` to GitHub Actions CI so every future build targets x86 automatically.

Production solution: multi-arch builds (`--platform linux/amd64,linux/arm64`) — one image containing both architectures, Docker picks the right one at runtime.

**CI/CD update:**
Added ECR push to GitHub Actions. On every merge to main:
1. Tests run
2. Image built and tagged with commit SHA + latest
3. Pushed to ECR automatically

Commit SHA tagging enables rollback — `kubectl set image` with a previous SHA to instantly revert.

---

## Step 9 — PostgreSQL Prediction Logging

Replaced CSV-based prediction logging with PostgreSQL. Every prediction is now stored in a relational database instead of appended to a flat file.

**Why PostgreSQL over CSV:**

| CSV | PostgreSQL |
|---|---|
| No concurrent writes | Handles concurrent writes safely |
| Full file scan to query | Indexed queries |
| Corrupts under load | ACID transactions |
| No schema enforcement | Typed columns |
| Hard to filter/aggregate | Full SQL |

**New `db/` package:**
```
db/
├── __init__.py
├── database.py     → SQLAlchemy engine + session factory
├── dependencies.py → get_db() for FastAPI dependency injection
└── models.py       → Prediction table definition
```

**Why dependency injection for DB sessions:**
FastAPI's `Depends(get_db)` automatically creates a session before each request and closes it after — even if an error occurs. No manual session management in every route.

**Why `Base.metadata.create_all(bind=engine)` on startup:**
SQLAlchemy reads all models inheriting from `Base` and creates tables if they don't exist. No manual `CREATE TABLE` SQL needed. Safe to call on every restart — uses `IF NOT EXISTS`.

**Why `lambda: datetime.now(timezone.utc)` instead of `datetime.utcnow`:**
`datetime.utcnow` is deprecated in Python 3.12+. The `lambda` is required so SQLAlchemy calls it at insert time — without it, all rows would get the same timestamp (when the class was defined).

**Docker Compose change:**
Added `postgres` service with a named volume (`postgres_data`) so data persists across restarts. Inference container gets `DATABASE_URL` env var pointing to `postgres:5432` — Docker's internal DNS resolves service names automatically.

**Environment-based config:**
```
Local Mac    → DATABASE_URL not set → uses localhost:5432 default
Docker       → DATABASE_URL=postgresql://postgres:postgres@postgres:5432/predictions
Kubernetes   → DATABASE_URL from Secret → postgres-service:5432
```
Same code, different config per environment. 12-factor app principle.

---

---

## Step 10 — PostgreSQL on Kubernetes + S3 Model Storage

### PostgreSQL on Kubernetes

Added PostgreSQL to the Kubernetes stack using a StatefulSet instead of a Deployment.

**Why StatefulSet over Deployment for PostgreSQL:**

| Deployment | StatefulSet |
|---|---|
| Pods are interchangeable | Each pod has a stable identity (`postgres-0`) |
| All replicas share one PVC | Each pod gets its own PVC via `volumeClaimTemplates` |
| Suitable for stateless apps | Required for databases |

**3 files added:**
- `postgres-secret.yaml` — stores `POSTGRES_PASSWORD` and `DATABASE_URL` as a Kubernetes Secret. Secrets are base64-encoded at rest. `stringData` lets you write plain text — Kubernetes encodes it automatically.
- `postgres-statefulset.yaml` — StatefulSet + headless Service (`clusterIP: None`). Headless service gives each pod a stable DNS name (`postgres-0.postgres-service`) instead of routing through a virtual IP.
- Updated `inference-deployment.yaml` and `retrain-cronjob.yaml` — `DATABASE_URL` now injected from the Secret via `secretKeyRef` instead of hardcoded. Removed `predictions-pvc` — no more CSV file.

**How the Secret flows:**
```
postgres-secret → POSTGRES_PASSWORD → PostgreSQL pod (sets DB password on init)
postgres-secret → DATABASE_URL      → inference pod + CronJob (os.environ.get override)
```

**`volumeClaimTemplates` vs referencing an existing PVC:**
The StatefulSet creates `postgres-data-postgres-0` automatically — you don't create a PVC manually in `volumes.yaml`. This is how StatefulSets bind each pod to dedicated storage permanently, even across restarts.

---

### S3 Model Storage + Init Container

Replaced the manual `kubectl cp` model loading process with a proper automated flow using S3 and a Kubernetes init container.

**The problem with manual copy:**
Every new cluster required running a busybox loader pod to copy model weights into the empty EBS PVC — not automated, not repeatable, not production-grade.

**The solution — S3 as single source of truth:**

```
First deploy:
  aws s3 cp models/ s3://ml-pipeline-models-tanish/models/ --recursive (once, manual)

Every retrain (CronJob):
  train.py saves new weights → /app/models/
  upload_model_to_s3() → pushes to S3

Every pod restart:
  init container → aws s3 cp s3://.../ /app/models/ --recursive
  inference container starts → weights already on disk
```

**Why the init container needs the same volumeMount as the inference container:**
Init container and inference container don't share a filesystem automatically. Both must mount the same PVC (`models-volume`) at `/app/models/`. The init container writes to EBS via the PVC; the inference container reads from EBS via the same PVC. Without the mount in the init container, downloaded files disappear when it exits.

**`retrain_if_needed.py` update:**
Added `upload_model_to_s3()` — calls `aws s3 cp models/ s3://ml-pipeline-models-tanish/models/ --recursive` after every successful retrain. S3 is always kept at the latest trained model, not the original hand-uploaded baseline.

**IAM permission:**
EKS nodes need `AmazonS3ReadOnlyAccess` attached to the node group's IAM role so the init container's `aws s3 cp` call is authorized. The CronJob needs write access (`AmazonS3FullAccess` or a scoped policy) to upload after retraining.

**Hot-reload vs init container — two different mechanisms:**
- Init container: runs on pod restart, downloads from S3 → handles cold start
- `/reload` endpoint: called by CronJob after retrain, swaps weights in memory → handles live update without restart
Both are needed. Init container is the safety net for restarts; hot-reload is the fast path during normal operation.

---

## Step 11 — EKS Redeployment with PostgreSQL + S3

Deployed the full updated stack (PostgreSQL + S3 init container) to a fresh EKS cluster.

**Problem 1: IAM service account conflict from old cluster**
Creating the new cluster with the same name `ml-pipeline` left behind the old IAM service account CloudFormation stack. eksctl saw it already existed and skipped creating a new one. But the old role had the old cluster's OIDC provider in its trust policy — AWS rejected all requests with `AccessDenied: Not authorized to perform sts:AssumeRoleWithWebIdentity`.

Fix: delete the leftover CloudFormation stack from the AWS console (`eksctl-ml-pipeline-addon-iamserviceaccount-kube-system-ebs-csi-controller-sa`), then recreate the IAM service account fresh for the new cluster.

**Lesson:** When deleting a cluster, always use `eksctl delete cluster` — not the console. eksctl cleans up all associated IAM resources automatically. Console delete only removes the cluster itself. Alternatively, use a new cluster name each time to avoid all conflicts.

**Problem 2: EBS CSI driver stuck in CREATE_FAILED**
The driver kept failing because it was picking up the old OIDC-linked role. Fixed by explicitly passing `--service-account-role-arn` with the newly created role ARN when recreating the addon.

**Problem 3: Postgres CrashLoopBackOff — `lost+found` directory**
PostgreSQL failed to initialize with:
```
initdb: error: directory "/var/lib/postgresql/data" exists but is not empty
initdb: detail: It contains a lost+found directory, perhaps due to it being a mount point.
```
EBS volumes have a `lost+found` directory at the root. PostgreSQL refuses to initialize in a non-empty directory.

Fix: added `PGDATA=/var/lib/postgresql/data/pgdata` env var — tells Postgres to use a subdirectory instead of the EBS root.

**Problem 4: Two `env:` blocks in StatefulSet YAML**
When adding `PGDATA`, accidentally created a second `env:` block instead of adding to the existing one. YAML doesn't error on duplicate keys — the second block silently overwrites the first, wiping out `POSTGRES_PASSWORD`. Postgres crashed with "superuser password is not specified."

Fix: merge both env vars into a single `env:` block.

**Problem 5: Inference pod DNS failure — `postgres-service` not found**
Inference pod started before Postgres DNS was registered. `app.py` calls `Base.metadata.create_all()` at startup which immediately tries to connect to `postgres-service` — failed with `could not translate host name "postgres-service"`.

Fix: delete the inference pod after Postgres is Running — Kubernetes restarts it and DNS resolves correctly.

**Final state:** All pods Running — inference, postgres, mlflow, prometheus, grafana, retrain CronJob. Inference serving predictions via public AWS LoadBalancer URL.

---

## Step 12 — End-to-End Pipeline Test + CI/CD Fixes

Triggered the full retraining loop manually and caught several bugs that only surface when running end to end.

**Problem 1: `train.py` missing from Docker image**
`retrain_if_needed.py` calls `python train.py` via subprocess — but `train.py` was never added to the Dockerfile. The inference image only had `app.py`, `evaluate.py`, `drift_detector.py`. Fix: added `COPY train.py .` to Dockerfile.

**Problem 2: `aws` CLI missing from Docker image**
`upload_model_to_s3()` calls `aws s3 cp` as a shell command, but `python:3.11-slim` doesn't include the AWS CLI. Fix: added `RUN apt-get install -y awscli` to Dockerfile.

**Problem 3: Tests failing in CI — no PostgreSQL**
`app.py` calls `Base.metadata.create_all(bind=engine)` at import time. When tests import `app`, this immediately tries to connect to PostgreSQL — which doesn't exist in GitHub Actions. Fix: added a `postgres:15` service to the GitHub Actions workflow so tests run against a real database.

**Problem 4: Drift tests using dicts instead of objects**
Tests were written when `drift_detector.py` read CSV rows as dicts. After switching to SQLAlchemy, `detect_drift` expects objects with `.confidence` and `.prediction` attributes. Tests were still passing dicts — `AttributeError: 'dict' object has no attribute 'confidence'`. Fix: replaced dicts with a `FakePrediction` class in `test_drift.py`.

**Problem 5: Full dataset too slow on CPU**
CronJob ran `train.py` on all 67K SST-2 examples — takes 1-2 hours on CPU t3.medium nodes. Fix: added `MAX_SAMPLES` env var support to `retrain_if_needed.py`. If set, passes `--max-samples` to `train.py`. Set to 500 in the CronJob for testing — reduces training time to ~5 minutes.

**CI/CD fully wired:**
Added `kubectl rollout restart deployment/inference-deployment` as final step in GitHub Actions. Now every `git push` to main automatically: runs tests → builds image → pushes to ECR → redeploys EKS. Full GitOps loop without ArgoCD.

---

## Step 13 — MLflow Model Registry

Added production model lifecycle management on top of the existing MLflow experiment tracking.

**Why Registry when we already had artifact logging:**
`mlflow.log_artifacts()` is just a file copy — it dumps weights into the run's storage with no concept of version history or production state. If retraining ran 10 times, there was no way to know which run's weights were currently serving production. The Registry adds: numbered versions (v1, v2, v3...), aliases (`Production`), and a full audit trail of promotions.

Without Registry: new weights silently overwrite `best_model.pt` — no history, no rollback.
With Registry: every retrain creates a new version, exactly one is tagged Production at any moment.

**What was built:**

`train.py` — after every training run, registers a new version in the Registry using `MlflowClient`:
```python
client.create_registered_model("sentiment-classifier")   # creates model name if not exists
client.create_model_version(name=..., source=mlflow.get_artifact_uri("model"), run_id=run.info.run_id)
```

`retrain_if_needed.py` — after every successful retrain, promotes the latest version to Production:
```python
versions = client.search_model_versions("name='sentiment-classifier'")
latest = max(versions, key=lambda v: int(v.version))
client.set_registered_model_alias("sentiment-classifier", "Production", latest.version)
```

`app.py` — loads the Production model by alias instead of hardcoded file path:
```python
artifact_path = mlflow.artifacts.download_artifacts("models:/sentiment-classifier@Production")
model.load_state_dict(torch.load(f"{artifact_path}/best_model.pt", map_location=device))
```

**Why `MlflowClient` instead of `mlflow.register_model()`:**
The fluent `mlflow.register_model()` validates that the artifact contains an `MLmodel` metadata file — only created by MLflow's own flavor-specific loggers (`mlflow.pytorch.log_model()`). We use `log_artifacts()` to keep our custom PyTorch state dict format, so we use the lower-level client API which skips that validation. The Registry does bookkeeping; PyTorch owns serialization.

**How the alias moves:**
`set_registered_model_alias("Production", v3)` automatically removes the alias from v2. An alias can only point to one version at a time — no manual cleanup needed. v1 and v2 stay in the Registry for history and rollback.

**`MLFLOW_TRACKING_URI` per environment:**
```
local testing   → MLFLOW_TRACKING_URI=http://localhost:5001
docker compose  → MLFLOW_TRACKING_URI=http://mlflow:5000  (Docker internal DNS)
kubernetes      → MLFLOW_TRACKING_URI=http://mlflow-service:5000
```
Same code, different env var value per environment. The service name `mlflow` resolves inside Docker because all services share a Docker network.

**Full automated loop after this step:**
```
drift detected / accuracy drops
        ↓
retrain_if_needed.py triggers train.py
        ↓
train.py trains → saves .pt → registers new version in Registry
        ↓
retrain_if_needed.py promotes new version to Production alias
        ↓
retrain_if_needed.py calls /reload on app.py
        ↓
app.py downloads Production version from Registry → serving new model
```
Every step automated. The Registry is the single source of truth for what's in production.

---

## Step 14 — Quality Gate Before Promotion

Added a comparison step between retraining and promotion — the new model only goes to Production if it actually beats the current Production model's accuracy.

**The problem with promoting immediately after retraining:**
Before this step, every retrain automatically promoted to Production. A model trained on 500 samples with bad hyperparameters could silently replace a better model. No validation, no comparison, no safety net.

**What was built:**

`evaluate.py` — added `model_path=None` parameter to `load_model()`:
- No `model_path` → downloads Production model from MLflow Registry
- `model_path="models/"` → loads newly retrained model from disk

`retrain_if_needed.py` — replaced `check_model_health()` subprocess with direct import of `load_model` and `evaluate`. Added `get_accuracy(model_path=None)` function. New flow in `__main__`:

```
1. get_accuracy()              → Production model accuracy from Registry
2. retrain()                   → new weights saved to models/best_model.pt
3. get_accuracy("models/")     → new model accuracy from disk
4. if new > production         → promote + upload to S3 + reload API
5. else                        → log and keep current Production
```

**Why import instead of subprocess for accuracy:**
`subprocess` only returns an exit code (0 or 1). For comparison we need the actual float. Importing `load_model` and `evaluate` directly gives us the float. `check_model_health()` (True/False) is replaced by `get_accuracy()` (float) — more information, same call.

**S3 upload moved inside quality gate:**
Previously `upload_model_to_s3()` ran after every retrain. Now it only runs if the new model is promoted. S3 should always match Production — no point uploading weights that didn't beat the current model.

**`/rollback` endpoint added to `app.py`:**
Accepts a version number, moves the Production alias to that version, and hot-reloads the model. One API call to roll back to any previous version without redeployment.

**Known limitations (future improvements):**
- Quality gate evaluates on static SST-2 validation set. In production this should be labeled incoming user traffic from a feature store (Feast/Tecton) — last 7 days of real traffic, not a held-out benchmark set.
- Two separate Dockerfiles (inference + training) would be cleaner than copying training scripts into the inference image.

**Future improvement — split inference and training into separate Docker images:**
Currently `train.py`, `evaluate.py`, and `retrain_if_needed.py` are all copied into the inference image. In production these should be two separate images:
- `inference-image` → `app.py` only. Small, fast startup, scales independently.
- `training-image` → `train.py`, `evaluate.py`, `retrain_if_needed.py`. Larger, runs periodically as a Kubernetes CronJob.

The CronJob YAML references the training image; the Deployment references the inference image. Clean separation of concerns, faster inference deployments.

**Future improvement — quality gate should evaluate on real production data:**
Currently `get_accuracy()` evaluates on the static SST-2 validation set (872 examples). In a real production system, the quality gate should evaluate on labeled incoming user data — actual traffic that has been labeled via a human feedback loop or labeling pipeline. That's the true measure of "how well is the model doing on real traffic?" The static validation set tells you the model works in general; production data tells you it works for your specific users.

**Future improvement — evaluate.py loads from disk, not Registry:**
`evaluate.py` currently loads from `models/best_model.pt` on disk. This works right after retraining (new weights just written there) but is fragile at any other time. Better fix: load from Registry by default, with an optional `--model-path` CLI arg override for evaluating a specific version.

---

## Step 16 — HPA (Horizontal Pod Autoscaler)

Added `k8s/hpa.yaml` to automatically scale inference pods based on CPU utilization.

**Why HPA only on inference, not training:**
HPA scales long-running services that handle ongoing requests. Training is a CronJob — one pod, runs once, exits. Scaling training horizontally would just create duplicate independent runs, not distributed training. Distributed training (PyTorch DDP) requires coordinated gradient sync across pods — a fundamentally different architecture.

**Configuration:**
- Min replicas: 1 — always at least one pod serving
- Max replicas: 3 — cost cap, sufficient for portfolio scale
- Target CPU: 70% of requests — scales before pods get overwhelmed

**How utilization is calculated:**
HPA measures `actual CPU / requested CPU` averaged across all pods. With `requests: 500m`, 70% threshold = 350m actual usage triggers a new pod. CPU limit raised to `2000m` (from 500m) so pods can burst under load without being throttled at the request ceiling.

**Scaling stack:**
```
Traffic spikes → HPA adds pods (up to 3)
Node full → Cluster Autoscaler adds EC2 node (not implemented — future)
```

**Training image doesn't need HPA** — one CronJob run at a time is correct. Parallel training requires PyTorch DDP code rewrite — planned as a separate project.

---

## Step 15 — Two Dockerfiles + Self-Bootstrapping CronJob

Split the single Docker image into two purpose-built images and made the retraining pipeline fully self-bootstrapping on a fresh cluster.

**Why two Dockerfiles:**
One image containing both serving and training code is wasteful and couples two unrelated concerns. The inference image only needs to serve predictions — it doesn't need `train.py`, `awscli`, or any training dependencies. Keeping them separate means:
- Inference image is smaller → faster pulls, faster pod starts
- Each image can be versioned and scaled independently
- Training image can run on GPU nodes; inference on CPU nodes

**What was built:**

`Dockerfile` (inference) — `app.py` + `db/` only. No awscli, no training scripts.

`Dockerfile.training` (training) — `train.py`, `evaluate.py`, `retrain_if_needed.py`, `drift_detector.py`, `db/`, `awscli`. CMD runs `retrain_if_needed.py` automatically.

`k8s/retrain-cronjob.yaml` — updated to use `ml-pipeline-training:latest` instead of the inference image.

**CI/CD updated:**
Builds and pushes both images on every commit. Also auto-creates ECR repositories if they don't exist — so a fresh AWS account works without any manual ECR setup.

**Self-bootstrapping CronJob:**
Added a check at the start of `retrain_if_needed.py` — if no Production model exists in the Registry, run initial training first before the normal loop. Fresh cluster flow:
```
kubectl apply -f k8s/
CronJob fires → no Production model found → trains → registers → promotes to Production
Inference pod loads Production model from Registry → serving
```
No manual `python train.py` needed on first deploy.

---

## Step 16 — GPU Node Support for Training

Updated the training image and CronJob to run on GPU-enabled nodes in the cluster.

**Why GPU for training, not inference:**
Training is computationally expensive — gradient computation, backprop, and weight updates across 67K examples are exactly what GPUs are built for. Inference is a single forward pass on one input at a time — CPU is fast enough and far cheaper.

**What was changed:**

`Dockerfile.training` — swapped `python:3.11-slim` for `nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04`. The CUDA base image ships with cuDNN and CUDA runtime so PyTorch can find and use the GPU. Python 3.11 is then installed on top via `apt`.

```
python:3.11-slim         → no CUDA runtime → GPU invisible to PyTorch
nvidia/cuda:12.1.1-...   → CUDA + cuDNN bundled → torch.cuda.is_available() = True
```

`k8s/retrain-cronjob.yaml` — replaced CPU requests/limits with GPU resources and added `nodeSelector`:

```yaml
resources:
  requests:
    memory: "6Gi"
    nvidia.com/gpu: "1"
  limits:
    memory: "6Gi"
    nvidia.com/gpu: "1"
nodeSelector:
  nvidia.com/gpu: "true"
```

**How Kubernetes GPU scheduling works:**
`nvidia.com/gpu: "1"` is a custom resource managed by the NVIDIA device plugin (runs as a DaemonSet on GPU nodes). Kubernetes uses this to schedule the pod only onto nodes that have a GPU available and reserves it exclusively for this pod — GPUs are not time-shared like CPU.

`nodeSelector: nvidia.com/gpu: "true"` is a label-based filter — ensures the pod only lands on nodes that have been explicitly labeled as GPU nodes. Without it, Kubernetes might schedule it on any node and the GPU request would be unschedulable.

**Why MPS (Apple Silicon) doesn't work here:**
MPS is Apple's Metal-based GPU framework — it only works on macOS, not inside Linux Docker containers. The CUDA base image runs on Linux, so the training image uses CUDA. Locally on a Mac, `train.py` still auto-detects MPS via `torch.backends.mps.is_available()` — the CUDA image only applies when running in the cluster.

**Architecture after this step:**
```
Training CronJob  → GPU node  (nvidia/cuda base, 6Gi RAM, 1 GPU)
Inference pods    → CPU nodes (python:3.11-slim, 500m–2000m CPU, no GPU)
HPA               → scales inference pods 1–3 based on CPU load
```
Training and inference now run on entirely different node types — independently scalable and independently priced.

---

## Step 17 — Grafana Alerting (Slack on Confidence Drop)

Added Grafana alerting to notify a Slack channel when prediction confidence drops below 0.70, replacing the previous auto-retraining trigger.

**Why alerting instead of auto-retraining:**
Auto-retraining is expensive and can make things worse — if confidence drops due to bad data quality or a data pipeline issue, retraining on bad data produces a worse model. A human should investigate first and decide whether retraining is the right fix. Alerting is the production-realistic pattern.

**Why confidence drop over drift flag:**
Confidence is a continuous metric — every prediction updates it, Prometheus scrapes it every 15 seconds, and Grafana can graph the trend over time. The drift flag only exists inside the CronJob once per hour. Confidence gives a live, continuous signal. Better for alerting.

**What was built:**

`app.py` — added a Prometheus Gauge for prediction confidence:
```python
CONFIDENCE = Gauge("prediction_confidence", "Confidence of predictions")
CONFIDENCE.set(conf)  # set on every prediction
```

`grafana/provisioning/datasources/datasources.yml` — auto-configures Prometheus as a Grafana datasource on startup. No manual UI clicks needed.

`grafana/provisioning/alerting/contact-points.yml` — configures the Slack webhook contact point. Stored outside git (gitignored) because it contains the webhook URL secret.

`grafana/provisioning/alerting/notification-policies.yml` — routes all alerts to the Slack contact point.

`grafana/provisioning/alerting/alert-rules.yml` — the alert rule:
- Query A: `avg_over_time(prediction_confidence[5m])` — average confidence over last 5 minutes
- Step B: reduce time series to single value (last)
- Step C: threshold < 0.70 → fire
- `for: 5m` — must be below threshold for 5 continuous minutes before firing (prevents noise from brief dips)

**Why Grafana alerting provisioning (not UI):**
Everything in code, version controlled, reproducible on any cluster. Fresh deploy → Grafana reads provisioning files → datasource, alert rule, contact point, notification policy all configured automatically. Zero UI clicks.

**The Slack message when it fires:**
```
[FIRING:1] Low Prediction Confidence
Value: B=0.33 (actual avg confidence), C=1 (threshold breached)
Labels: severity=warning, instance=inference:8000
Description: Average prediction confidence over the last 5 minutes is below 0.70
Source: link to alert in Grafana
Silence: one-click silence if already investigating
```

**Grafana alert pipeline — A → B → C:**
```
A → raw time series from Prometheus (avg_over_time over 5 min = one number per eval)
B → reduce to scalar (last value — required step, Grafana can't threshold a series directly)
C → compare scalar against 0.70 → true/false → fire or not
```

**Tested end to end:**
Sent predictions with a poorly-trained model (50 samples) — confidence ~0.50. Alert went Pending → Firing after 5 minutes. Slack message received in #alerts with correct labels, annotations, and source link.

**K8s wiring:**
Two ConfigMaps + one Secret mounted into the Grafana pod:

- `grafana-datasources` ConfigMap → mounted at `/etc/grafana/provisioning/datasources/`
- `grafana-alerting` ConfigMap → mounted at `/etc/grafana/provisioning/alerting/`
- `grafana-secret` Secret → `contact-points.yml` mounted into `/etc/grafana/provisioning/alerting/` (gitignored, applied manually like postgres-secret)

Two ConfigMaps instead of one because datasources and alerting files belong in different subfolders — mounting one ConfigMap at `/etc/grafana/provisioning/` would put all files in the root and Grafana wouldn't find them.

**Secrets approach:**
`grafana-secret.yaml` is gitignored — webhook URL never touches git. Applied manually on fresh cluster deploy alongside `postgres-secret.yaml`. In production this would be managed by External Secrets Operator pulling from AWS Secrets Manager.

---

## Step 18 — Configurable Model via MODEL_NAME Env Var

**What changed:**
Replaced every hardcoded `"distilbert-base-uncased"` string across `train.py`, `app.py`, and `evaluate.py` with `MODEL_NAME = os.getenv("MODEL_NAME", "distilbert-base-uncased")`. Also added `MODEL_NAME` as an env var in `k8s/retrain-cronjob.yaml` and `k8s/inference-deployment.yaml`.

**Why:**
The platform now supports any HuggingFace sequence classification model — not just DistilBERT. An engineer deploying a different model only changes one value (`MODEL_NAME`) in the K8s manifests (or `values.yaml` once Helm is added). Nothing else changes — the training loop, tokenizer loading, inference serving, evaluation, and registry promotion all adapt automatically.

**Files changed:**
- `train.py` — `AutoTokenizer.from_pretrained(MODEL_NAME)`, `AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, ...)`
- `app.py` — same two calls + `os.getenv` at top
- `evaluate.py` — same
- `k8s/retrain-cronjob.yaml` — added `MODEL_NAME: distilbert-base-uncased` env var
- `k8s/inference-deployment.yaml` — added `MODEL_NAME: distilbert-base-uncased` env var

**Interview angle:**
Parameterizing the model name is the difference between a one-model script and a reusable platform. Any sequence classifier on HuggingFace Hub (BERT, RoBERTa, ALBERT, domain-specific variants) can now be swapped in without touching Python code. This is the same pattern production MLOps platforms use — model choice is configuration, not code.

---

## Step 19 — Full Platform Configurability (Dataset, Labels, Text Column)

**What changed:**
Added four more env vars across `train.py`, `evaluate.py`, `app.py`, and both K8s manifests:

- `NUM_LABELS` — number of output classes. Default `2`. Set to `5` for star rating classification.
- `DATASET_NAME` — which HuggingFace dataset to load. Default `sst2`. Could be `imdb`, `ag_news`, `yelp_review_full`, etc.
- `TEXT_COLUMN` — which column contains the text. Default `sentence` (SST-2). Most datasets use `text`.
- `VAL_SPLIT` — name of the validation split. Default `validation`. Some datasets use `test` or `dev`.

Also fixed a bug: `train.py` had `MODEL_NAME = os.getenv("MODEL_NAME", MODEL_NAME)` — referencing itself before being defined. Fixed to `"distilbert-base-uncased"` as the default string.

**What an engineer changes to deploy on a new task:**
Only the K8s manifest env vars — no Python code changes required. Example for IMDb sentiment:
```
MODEL_NAME=roberta-base
NUM_LABELS=2
DATASET_NAME=imdb
TEXT_COLUMN=text
VAL_SPLIT=test
```

**Interview angle:**
This is the line between a script and a platform. The training loop, tokenizer, inference server, drift evaluation, and model registry are all task-agnostic now. Swapping datasets is config, not code. This mirrors how real MLOps platforms (SageMaker Pipelines, Vertex AI) work — the infrastructure is fixed, the model and data are parameters.

---

## Step 20 — Production-Grade Training Loop

**What changed:**

**LABEL_MAP env var** — `app.py` was hardcoded to return "positive"/"negative" for any model. Now reads from `LABEL_MAP={"0": "negative", "1": "positive"}`. For AG News: `{"0": "World", "1": "Sports", "2": "Business", "3": "Sci/Tech"}`. Without this, NUM_LABELS=4 would serve wrong labels.

**Full hyperparameter configurability** — every training knob is now an env var with a production-tested default:
- `MAX_LENGTH=128`, `EPOCHS=3`, `BATCH_SIZE=16`, `LEARNING_RATE=2e-5`
- `ACCURACY_THRESHOLD=0.80`, `WEIGHT_DECAY=0.01`, `WARMUP_STEPS=100`, `GRAD_CLIP=1.0`

**Best model checkpoint** — previously saved the last epoch's model. Now saves only when val accuracy improves. If epoch 2 peaks and epoch 3 overfits, you get epoch 2's weights.

**Linear warmup + LR decay** — flat LR is not how transformers are fine-tuned. Added `get_linear_schedule_with_warmup`: ramps from 0 → peak LR over `WARMUP_STEPS` steps, then decays linearly to ~0 by end of training. Warmup prevents the randomly initialized classification head from destroying pretrained DistilBERT weights early in training.

**Gradient clipping** — `clip_grad_norm_(model.parameters(), GRAD_CLIP)` called between `loss.backward()` and `optimizer.step()`. If a gradient norm exceeds 1.0, it's scaled down. Acts as a safety net — does nothing during normal training, prevents catastrophic weight updates on bad batches.

**Weighted F1 score** — `f1_score(all_labels, all_preds, average="weighted")` from sklearn, logged alongside accuracy in both `evaluate.py` and `retrain_if_needed.py`. Accuracy is misleading on imbalanced datasets. Weighted F1 accounts for class distribution.

**Bug fixed** — `ACCURACY_THRESHOLD` was defined but never used in the actual decision logic in `retrain_if_needed.py`. Lines 80 and 85 still had `0.80` hardcoded. Fixed to use `ACCURACY_THRESHOLD`.

**Interview angle:**
The training loop now matches what you'd see in a production fine-tuning job — warmup + decay LR, gradient clipping, weight regularization, best checkpoint, and F1 evaluation. Flat LR with no regularization is the first thing an ML engineer notices in a review.

---

## Step 21 — Early Stopping + /predict_batch Endpoint

**Early stopping (`PATIENCE=3`):**
Added `epochs_no_improve` counter inside the training loop. If val accuracy doesn't improve for `PATIENCE` consecutive epochs, training stops early. Best model is already saved from the epoch it peaked. Saves GPU compute, prevents overfitting on longer runs. Configurable via `PATIENCE` env var.

```
Epoch 1 — accuracy 0.82 → improved, save, epochs_no_improve=0
Epoch 2 — accuracy 0.84 → improved, save, epochs_no_improve=0
Epoch 3 — accuracy 0.83 → no improvement, epochs_no_improve=1
Epoch 4 — accuracy 0.83 → no improvement, epochs_no_improve=2
Epoch 5 — accuracy 0.82 → no improvement, epochs_no_improve=3 → STOP
```

**`/predict_batch` endpoint:**
Accepts a list of texts, runs one forward pass for the whole batch, returns predictions for all. Full PostgreSQL logging via `bulk_save_objects` — one DB transaction for the entire batch. Prometheus counter incremented per prediction. `MAX_BATCH_SIZE=32` guard — rejects oversized batches before any tensor is created, keeps the pod alive.

```json
POST /predict_batch
{"texts": ["great movie", "terrible film", "okay I guess"]}
← {"predictions": [{"prediction": "positive", "confidence": 0.97}, ...]}
```

**Why bulk_save_objects over individual db.add():**
One `INSERT` statement for all rows vs N separate statements. Same atomicity — one commit, all or nothing. Correct approach for batch operations.

**MAX_BATCH_SIZE guard:**
Without it, a client sending 10,000 texts would create a massive tensor and OOM-kill the pod, taking down `/predict` for everyone. The guard rejects oversized requests with HTTP 400 before any computation — pod stays alive.

**Interview angle:**
Batch inference is standard in production — data pipelines, bulk classification jobs, backend services that buffer requests. The key design decisions: one DB transaction for the whole batch (not N inserts), MAX_BATCH_SIZE to protect pod memory, full audit trail identical to single predictions.

---

## Step 22 — Full Platform Audit: No Hardcoded Values Anywhere

**Goal:** Make the platform truly deploy-agnostic. Any engineer should be able to fork, set env vars, and deploy to their own AWS account — zero code changes.

**What was hardcoded and how it was fixed:**

**`docker-compose.yml`** — postgres credentials (`postgres/postgres`) and Grafana admin password (`admin`) were hardcoded. Replaced with `${VAR:-default}` syntax — docker-compose reads these from `.env`, falls back to the same safe defaults if not set.

**`.github/workflows/ci.yml`** — AWS region (`us-east-1`), ECR repo names (`ml-pipeline-inference`, `ml-pipeline-training`), and EKS cluster name (`ml-pipeline`) were hardcoded. Replaced with GitHub repository variables (`vars.AWS_REGION`, `vars.ECR_INFERENCE_REPO`, etc.). Any engineer deploying to their own AWS sets these in GitHub Settings — no code changes needed.

**`prometheus.yml`** — job_name was `"sentiment-api"`. Renamed to `"ml-pipeline-api"` — last place the old project name appeared in infrastructure config.

**Grafana Slack webhook** — `contact-points.yml` is gitignored (Grafana provisioning doesn't support env var substitution here). Added `contact-points.yml.example` as a template and documented the setup step in README. Engineers copy the file, paste their webhook URL — one manual step, done once.

**`.env.example`** — created to document every env var docker-compose reads from the environment, so there's no guessing what to set.

**Interview angle:**
A platform that only works with your specific AWS account isn't a platform — it's a personal project with a hard-coded deploy target. The fix is separation of config from code: infrastructure references live in CI/CD variables and env files, never in source. This is the Twelve-Factor App principle applied to ML infrastructure.

---

## Step 23 — Helm Chart: Deploy Entire Platform with One Command

**Goal:** Package the platform so any engineer can deploy it to their own Kubernetes cluster by editing one file and running one command.

**What Helm does:**
Helm is a templating engine for Kubernetes YAML. Instead of 8 separate YAML files with hardcoded values, you get a chart — templates with `{{ .Values.xxx }}` placeholders filled in from `values.yaml` at deploy time.

**Structure:**
```
helm/ml-pipeline/
├── Chart.yaml          ← chart metadata (name, version)
├── values.yaml         ← all configurable defaults (committed)
├── values.secret.yaml  ← real passwords (gitignored, never committed)
└── templates/          ← k8s YAMLs with {{ .Values.xxx }} placeholders
```

**Deploy command:**
```bash
helm install ml-pipeline ./helm/ml-pipeline \
  -f helm/ml-pipeline/values.yaml \
  -f helm/ml-pipeline/values.secret.yaml
```

That single command deploys inference, training CronJob, MLflow, Prometheus, Grafana, and PostgreSQL to any cluster.

**values.yaml covers:**
- AWS: region, ECR account ID, S3 bucket
- Model: name, num_labels, label_map, dataset, text_column
- Training: schedule, epochs, batch_size, lr, weight_decay, warmup, grad_clip, patience
- Drift detection: all thresholds
- Postgres: user, db
- Grafana: admin password placeholder, Slack webhook placeholder
- Inference HPA: min/max replicas, CPU threshold

**Secrets pattern:**
`values.yaml` has empty placeholders for passwords. `values.secret.yaml` (gitignored) has the real values. Helm merges them at deploy time — secrets never touch git.

**Interview angle:**
Helm is the standard packaging format for Kubernetes applications — it's how real ML platforms are shipped. The difference between a personal project and a platform is whether someone else can deploy it without touching your code. Helm makes that possible: clone, fill in `values.yaml`, one command, done.
