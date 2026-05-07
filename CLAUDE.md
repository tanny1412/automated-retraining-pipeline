# Automated ML Retraining Pipeline

## What This Project Is
A personal MLOps project built step-by-step for interview preparation. The goal is an automated ML retraining pipeline — not just a working system, but one Tanish can explain end-to-end in interviews.

## Collaboration Rules (CRITICAL)
- Build in **small incremental pieces** — one logical chunk at a time
- After each piece: explain what it does, why it exists, and the interview angle
- **Never build a full file in one shot** — break into: imports → config → data → model → loop → save
- Wait for confirmation before moving to the next piece
- Tanish does not write code by hand — Claude builds each piece, Tanish understands it

## Current Phase: Step 1 — Core ML Loop
Goal: train → save → load → serve predictions → replace automatically

### What's been built
- `requirements.txt` — all dependencies installed
- `models/` directory — created
- `train.py` — imports + argparse skeleton only (epochs, batch-size, lr)

### What's next in train.py
1. Dataset loading (SST-2 from HuggingFace)
2. Tokenization
3. Model setup (DistilBERT)
4. Training loop
5. Evaluation + save

### After train.py is done
- Build `app.py` (FastAPI: /health + /predict)
- Then Step 2: MLflow

## Tech Stack
- Model: `distilbert-base-uncased` fine-tuned on SST-2 (binary sentiment)
- Training: PyTorch manual loop (not HuggingFace Trainer — teaches more)
- Device: Apple M2 — use `torch.device("mps" if torch.backends.mps.is_available() else "cpu")`
- Serving: FastAPI + uvicorn
- Python: 3.11.8 via pyenv at `~/.pyenv/versions/3.11.8/`

## Project Structure
```
project/
├── CLAUDE.md          ← you are here
├── requirements.txt
├── train.py           ← in progress
├── app.py             ← not started
├── models/
│   ├── best_model.pt  ← created by train.py
│   └── tokenizer/     ← created by train.py
```

## Key Design Decisions (for interviews)
- Inference is **separate** from training — app.py only loads and serves, never retrains
- Model is parameterized via CLI args so retraining can be triggered automatically with different hyperparameters
- DistilBERT chosen for speed (40% smaller than BERT, ~97% accuracy)

## Step Roadmap
| Step | Focus | Status |
|------|-------|--------|
| 1 | train.py + app.py core loop | In progress |
| 2 | MLflow experiment tracking | Not started |
| 3 | Orchestration (trigger retraining) | Not started |
| 4 | Monitoring + drift detection | Not started |
| 5 | Docker + CI/CD | Not started |
