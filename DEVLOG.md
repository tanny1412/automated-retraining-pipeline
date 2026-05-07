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

## Up Next

- **Step 2:** Wrap training with MLflow — track every run's metrics, params, and artifacts
- **Step 3:** Build the retraining trigger — detect when to retrain and kick off `train.py` automatically
- **Step 4:** Drift detection — monitor incoming data distribution vs training distribution
- **Step 5:** Docker + CI/CD
