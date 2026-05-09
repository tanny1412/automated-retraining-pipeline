import argparse
import logging
import mlflow
from mlflow.tracking import MlflowClient
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-samples", type=int, default=None)
    return parser.parse_args()

def load_data(max_samples=None):
    dataset = load_dataset("sst2")
    train_data = dataset["train"]
    val_data = dataset["validation"]
    if max_samples:
        train_data = train_data.select(range(max_samples))
        val_data = val_data.select(range(min(max_samples // 5, len(val_data))))
    logger.info(f"Train size: {len(train_data)}, Val size: {len(val_data)}")
    return train_data, val_data

def tokenize_data(train_data, val_data):
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    def tokenize(batch):
        return tokenizer(batch["sentence"], truncation=True, padding="max_length", max_length=128)

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

    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
    model = model.to(device)

    train_loader = DataLoader(train_tokenized, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_tokenized, batch_size=batch_size)

    return model, train_loader, val_loader, device

def train(model, train_loader, val_loader, device, args):
    optimizer = AdamW(model.parameters(), lr=args.lr)

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
            optimizer.step()

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
            "model": "distilbert-base-uncased",
            "dataset": "sst2",
        })

        train_data, val_data = load_data(args.max_samples)
        tokenizer, train_tokenized, val_tokenized = tokenize_data(train_data, val_data)
        model, train_loader, val_loader, device = get_model_and_loaders(train_tokenized, val_tokenized, args.batch_size)
        train(model, train_loader, val_loader, device, args)
        save_model(model, tokenizer)
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
