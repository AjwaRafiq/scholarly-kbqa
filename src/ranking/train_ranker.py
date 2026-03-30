import json
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from tqdm import tqdm
import os
import numpy as np

os.makedirs("models/bert_ranker", exist_ok=True)
os.makedirs("results", exist_ok=True)

class RankerDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        question = item["question"]
        path = " > ".join(item["path"])

        encoding = self.tokenizer(
            question, path,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "token_type_ids": encoding["token_type_ids"].squeeze(),
            "label": torch.tensor(item["label"], dtype=torch.long)
        }

def train_ranker(train_file, dev_file, output_dir, epochs=10, batch_size=32, lr=2e-5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    with open(train_file) as f:
        train_data = json.load(f)
    with open(dev_file) as f:
        dev_data = json.load(f)

    print(f"Train size: {len(train_data)} | Dev size: {len(dev_data)}")

    # Count labels
    train_labels = [d["label"] for d in train_data]
    n_pos = sum(train_labels)
    n_neg = len(train_labels) - n_pos
    print(f"Train positives: {n_pos} | negatives: {n_neg} | ratio: 1:{n_neg//max(n_pos,1)}")

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # Use class weights to handle imbalance properly
    pos_weight = n_neg / max(n_pos, 1)
    class_weights = torch.tensor([1.0, pos_weight], dtype=torch.float).to(device)
    print(f"Class weights: neg=1.0, pos={pos_weight:.1f}")

    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=2,
        hidden_dropout_prob=0.2,       # dropout to prevent overfitting
        attention_probs_dropout_prob=0.2
    ).to(device)

    train_dataset = RankerDataset(train_data, tokenizer)
    dev_dataset = RankerDataset(dev_data, tokenizer)

    # Weighted sampler — oversample positives during training
    sample_weights = [pos_weight if l == 1 else 1.0 for l in train_labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size)

    # Lower LR + weight decay to reduce overfitting
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    # Weighted loss function
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

    best_f1 = 0
    patience = 2
    no_improve = 0

    for epoch in range(epochs):
        # Training
        model.train()
        total_loss = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                token_type_ids=batch["token_type_ids"]
            )
            loss = loss_fn(outputs.logits, batch["label"])
            total_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        avg_loss = total_loss / len(train_loader)

        # Evaluation
        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in dev_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    token_type_ids=batch["token_type_ids"]
                )
                preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
                labels = batch["label"].cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels)

        f1  = f1_score(all_labels, all_preds, zero_division=0)
        acc = accuracy_score(all_labels, all_preds)
        pre = precision_score(all_labels, all_preds, zero_division=0)
        rec = recall_score(all_labels, all_preds, zero_division=0)

        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f} | Acc={acc:.4f} | "
              f"P={pre:.4f} | R={rec:.4f} | F1={f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            no_improve = 0
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            print(f"  -> New best model saved (F1={f1:.4f})")
        else:
            no_improve += 1
            print(f"  -> No improvement ({no_improve}/{patience})")
            if no_improve >= patience:
                print("Early stopping triggered!")
                break

    print(f"\nTraining complete. Best F1: {best_f1:.4f}")

if __name__ == "__main__":
    train_ranker(
        "data/processed/ranker_training.json",
        "data/processed/ranker_dev.json",
        "models/bert_ranker",
        epochs=5,
        batch_size=32,
        lr=2e-5
    )
