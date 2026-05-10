import os
import argparse
import logging
import mlflow
from mlflow.tracking import MlflowClient

MODEL_NAME = os.getenv("MODEL_NAME", "distilbert-base-uncased")
NUM_LABELS = int(os.getenv("NUM_LABELS", "2"))
DATASET_NAME = os.getenv("DATASET_NAME", "sst2")
TEXT_COLUMN = os.getenv("TEXT_COLUMN", "sentence")
VAL_SPLIT = os.getenv("VAL_SPLIT", "validation")
MAX_LENGTH = int(os.getenv("MAX_LENGTH", "128"))
WEIGHT_DECAY = float(os.getenv("WEIGHT_DECAY", "0.01"))
WARMUP_STEPS = int(os.getenv("WARMUP_STEPS", "100"))
GRAD_CLIP = float(os.getenv("GRAD_CLIP", "1.0"))
PATIENCE = int(os.getenv("PATIENCE", "3"))
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=int(os.getenv("EPOCHS", "3")))
    parser.add_argument("--batch-size", type=int, default=int(os.getenv("BATCH_SIZE", "16")))
    parser.add_argument("--lr", type=float, default=float(os.getenv("LEARNING_RATE", "2e-5")))
    parser.add_argument("--max-samples", type=int, default=None)
    return parser.parse_args()

def load_data(max_samples=None):
    dataset = load_dataset(DATASET_NAME)
    train_data = dataset["train"]
    val_data = dataset[VAL_SPLIT]
    if max_samples:
        train_data = train_data.select(range(max_samples))
        val_data = val_data.select(range(min(max_samples // 5, len(val_data))))
    logger.info(f"Train size: {len(train_data)}, Val size: {len(val_data)}")
    return train_data, val_data

def tokenize_data(train_data, val_data):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize(batch):
        return tokenizer(batch[TEXT_COLUMN], truncation=True, padding="max_length", max_length=MAX_LENGTH)

    train_tokenized = train_data.map(tokenize, batched=True)
    val_tokenized = val_data.map(tokenize, batched=True)

    train_tokenized.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    val_tokenized.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    return tokenizer, train_tokenized, val_tokenized

def get_model_and_loaders(train_tokenized, val_tokenized, batch_size):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info(f"Using device: {device}")

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)
    model = model.to(device)

    train_loader = DataLoader(train_tokenized, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_tokenized, batch_size=batch_size)

    return model, train_loader, val_loader, device

def train(model, tokenizer, train_loader, val_loader, device, args):
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=total_steps)
    best_val_accuracy = 0.0
    epochs_no_improve = 0

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0

        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", unit="batch")
        for batch in progress:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            progress.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating", leave=False):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                preds = outputs.logits.argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        accuracy = correct / total
        logger.info(f"Epoch {epoch+1}/{args.epochs} — loss: {avg_loss:.4f}, val_accuracy: {accuracy:.4f}")
        mlflow.log_metrics({"loss": avg_loss, "val_accuracy": accuracy}, step=epoch + 1)

        if accuracy > best_val_accuracy:
            best_val_accuracy = accuracy
            epochs_no_improve = 0
            save_model(model, tokenizer)
            logger.info(f"New best model saved — val_accuracy: {best_val_accuracy:.4f}")
        else:
            epochs_no_improve += 1
            logger.info(f"No improvement for {epochs_no_improve}/{PATIENCE} epochs")
            if epochs_no_improve >= PATIENCE:
                logger.info(f"Early stopping triggered — no improvement for {PATIENCE} epochs")
                break

def save_model(model, tokenizer):
    torch.save(model.state_dict(), "models/best_model.pt")
    tokenizer.save_pretrained("models/tokenizer")
    logger.info("Model and tokenizer saved to models/")

if __name__ == "__main__":
    args = get_args()
    logger.info(f"epochs={args.epochs}, batch_size={args.batch_size}, lr={args.lr}")

    mlflow.set_experiment("sentiment-classifier")
    with mlflow.start_run() as run:
        mlflow.log_params({
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "max_samples": args.max_samples,
            "model": MODEL_NAME,
            "dataset": DATASET_NAME,
            "num_labels": NUM_LABELS,
            "text_column": TEXT_COLUMN,
            "weight_decay": WEIGHT_DECAY,
            "warmup_steps": WARMUP_STEPS,
            "grad_clip": GRAD_CLIP,
            "patience": PATIENCE,
        })

        train_data, val_data = load_data(args.max_samples)
        tokenizer, train_tokenized, val_tokenized = tokenize_data(train_data, val_data)
        model, train_loader, val_loader, device = get_model_and_loaders(train_tokenized, val_tokenized, args.batch_size)
        train(model, tokenizer, train_loader, val_loader, device, args)
        mlflow.log_artifacts("models/", artifact_path="model")
        client = MlflowClient()
        try:
            client.create_registered_model("sentiment-classifier")
        except mlflow.exceptions.MlflowException:
            pass  # registered model already exists from a prior run
        mv = client.create_model_version(
            name="sentiment-classifier",
            source=mlflow.get_artifact_uri("model"),
            run_id=run.info.run_id,
        )
        logger.info(f"Registered model version {mv.version} in MLflow Registry")
    logger.info("Training complete")
    exit(0)
