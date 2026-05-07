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

if __name__ == "__main__":
    args = get_args()
    print(f"epochs={args.epochs}, batch_size={args.batch_size}, lr={args.lr}")
    train_data, val_data = load_data()
    tokenizer, train_tokenized, val_tokenized = tokenize_data(train_data, val_data)
