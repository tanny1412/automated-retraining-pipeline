import argparse
import logging
import mlflow
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.90)
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()

def load_model(model_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    if model_path:
        tokenizer = AutoTokenizer.from_pretrained(f"{model_path}/tokenizer")
        model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
        model.load_state_dict(torch.load(f"{model_path}/best_model.pt", map_location=device))
        logger.info(f"Model loaded from disk: {model_path}")
    else:
        artifact_path = mlflow.artifacts.download_artifacts("models:/sentiment-classifier@Production")
        tokenizer = AutoTokenizer.from_pretrained(f"{artifact_path}/tokenizer")
        model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
        model.load_state_dict(torch.load(f"{artifact_path}/best_model.pt", map_location=device))
        logger.info("Model loaded from Registry (Production)")
    model.to(device)
    model.eval()
    return model, tokenizer, device

def evaluate(model, tokenizer, device, batch_size):
    dataset = load_dataset("sst2")
    val_data = dataset["validation"]

    def tokenize(batch):
        return tokenizer(batch["sentence"], truncation=True, padding="max_length", max_length=128)

    val_tokenized = val_data.map(tokenize, batched=True)
    val_tokenized.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    val_loader = DataLoader(val_tokenized, batch_size=batch_size)

    correct, total = 0, 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = outputs.logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    logger.info(f"Validation accuracy: {accuracy:.4f}")
    return accuracy

if __name__ == "__main__":
    args = get_args()
    model, tokenizer, device = load_model()
    accuracy = evaluate(model, tokenizer, device, args.batch_size)

    if accuracy < args.threshold:
        logger.info(f"Accuracy {accuracy:.4f} below threshold {args.threshold} — retraining needed")
        exit(1)
    else:
        logger.info(f"Accuracy {accuracy:.4f} above threshold {args.threshold} — model is healthy")
        exit(0)
