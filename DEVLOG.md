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

## Up Next

- Download full Colab model and replace local models/
- Delete cluster after testing to avoid AWS charges (`eksctl delete cluster --name ml-pipeline`)

### Future Improvement: S3 Model Storage

Currently using `kubectl cp` to manually copy model weights into the EBS volume — not production-grade.

**Production pattern:**
1. `train.py` uploads weights to S3 after training (`boto3.upload_file`)
2. Add an **init container** to the inference Deployment — runs before the main container, downloads weights from S3 into the shared volume
3. Main container starts only after init container completes

```
train.py → s3://bucket/models/best_model.pt
pod starts → init container: aws s3 cp s3://bucket/models/ /app/models/
main container starts → model already on disk, no manual copy needed
```

This means every retrain automatically makes the new weights available on next pod restart — zero manual intervention, true closed loop.
