import json
import torch
import os
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration, get_linear_schedule_with_warmup
from torch.optim import AdamW
from tqdm import tqdm

os.makedirs("models/t5_generator", exist_ok=True)

class T5Dataset(Dataset):
    def __init__(self, data, tokenizer, max_input_len=256, max_output_len=256):
        self.data = data
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        input_enc = self.tokenizer(
            item["input"],
            max_length=self.max_input_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        output_enc = self.tokenizer(
            item["output"],
            max_length=self.max_output_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        labels = output_enc["input_ids"].squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_enc["input_ids"].squeeze(),
            "attention_mask": input_enc["attention_mask"].squeeze(),
            "labels": labels
        }

def train_t5(train_file, dev_file, output_dir, epochs=10, batch_size=8, lr=3e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    with open(train_file) as f:
        train_data = json.load(f)
    with open(dev_file) as f:
        dev_data = json.load(f)

    print(f"Train: {len(train_data)} | Dev: {len(dev_data)}")

    local_model_path = "models/t5_base_cache/models--t5-base/snapshots/a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1"
    tokenizer = T5Tokenizer.from_pretrained(local_model_path, local_files_only=True)
    model = T5ForConditionalGeneration.from_pretrained(local_model_path, local_files_only=True).to(device)

    train_dataset = T5Dataset(train_data, tokenizer)
    dev_dataset = T5Dataset(dev_data, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    best_loss = float("inf")
    patience = 3
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
                labels=batch["labels"]
            )
            loss = outputs.loss
            total_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        avg_train_loss = total_loss / len(train_loader)

        # Evaluation
        model.eval()
        total_dev_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in dev_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"]
                )
                total_dev_loss += outputs.loss.item()

                generated = model.generate(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    max_length=256,
                    num_beams=5
                )

                pred_texts = tokenizer.batch_decode(generated, skip_special_tokens=True)
                label_ids = batch["labels"].clone()
                label_ids[label_ids == -100] = tokenizer.pad_token_id
                gold_texts = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

                for pred, gold in zip(pred_texts, gold_texts):
                    if pred.strip() == gold.strip():
                        correct += 1
                    total += 1

        avg_dev_loss = total_dev_loss / len(dev_loader)
        em = correct / total if total > 0 else 0

        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f} | "
              f"Dev Loss={avg_dev_loss:.4f} | EM={em:.4f}")

        if avg_dev_loss < best_loss:
            best_loss = avg_dev_loss
            no_improve = 0
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            print(f"  -> New best model saved (Loss={avg_dev_loss:.4f})")
        else:
            no_improve += 1
            print(f"  -> No improvement ({no_improve}/{patience})")
            if no_improve >= patience:
                print("Early stopping triggered!")
                break

    print(f"\nTraining complete. Best dev loss: {best_loss:.4f}")

if __name__ == "__main__":
    train_t5(
        "data/processed/t5_train.json",
        "data/processed/t5_dev.json",
        "models/t5_generator",
        epochs=10,
        batch_size=8,
        lr=3e-4
    )
