import argparse
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from torch.optim import AdamW

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    return parser.parse_args()

def load_data():
    dataset = load_dataset("sst2")
    train_data = dataset["train"]
    val_data = dataset["validation"]
    print(f"Train size: {len(train_data)}, Val size: {len(val_data)}")
    return train_data, val_data

def tokenize_data(train_data, val_data):
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    def tokenize(batch):
        return tokenizer(batch["sentence"], truncation=True, padding="max_length", max_length=128)

    train_tokenized = train_data.map(tokenize, batched=True)
    val_tokenized = val_data.map(tokenize, batched=True)

    train_tokenized.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    val_tokenized.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    print(f"Sample token ids: {train_tokenized[0]['input_ids'][:10]}")
    return tokenizer, train_tokenized, val_tokenized

def train(model, train_loader, val_loader, device, args):
    optimizer = AdamW(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # validation
        model.eval()
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
        print(f"Epoch {epoch+1}/{args.epochs} — loss: {avg_loss:.4f}, val_accuracy: {accuracy:.4f}")

def get_model_and_loaders(train_tokenized, val_tokenized, batch_size):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
    model = model.to(device)

    train_loader = DataLoader(train_tokenized, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_tokenized, batch_size=batch_size)

    return model, train_loader, val_loader, device

if __name__ == "__main__":
    args = get_args()
    print(f"epochs={args.epochs}, batch_size={args.batch_size}, lr={args.lr}")
    train_data, val_data = load_data()
    tokenizer, train_tokenized, val_tokenized = tokenize_data(train_data, val_data)
    model, train_loader, val_loader, device = get_model_and_loaders(train_tokenized, val_tokenized, args.batch_size)
    train(model, train_loader, val_loader, device, args)
