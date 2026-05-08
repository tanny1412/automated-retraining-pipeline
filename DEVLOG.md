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

## Up Next

- Download full Colab model and replace local models/
- Kubernetes (optional — understand mapping: Compose → Deployments/Services/CronJobs)
