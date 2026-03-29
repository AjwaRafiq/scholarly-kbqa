import json
import random
import torch
import numpy as np
import os
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

os.makedirs("models/entity_linker", exist_ok=True)
os.makedirs("data/embeddings", exist_ok=True)

def train_entity_linker(el_data_file, entities_file, output_dir):
    with open(el_data_file) as f:
        el_data = json.load(f)
    with open(entities_file) as f:
        entities = json.load(f)

    train_examples = []

    for item in el_data:
        question = item["question"]
        entity_name = item["entity_name"]

        # Positive pair
        train_examples.append(InputExample(
            texts=[question, entity_name],
            label=1.0
        ))

        # Hard negative
        entity_type = entities[item["entity_id"]]["type"]
        negatives = [
            e["name"] for eid, e in entities.items()
            if e["type"] == entity_type and eid != item["entity_id"]
        ]

        if negatives:
            neg_name = random.choice(negatives[:50])
            train_examples.append(InputExample(
                texts=[question, neg_name],
                label=0.0
            ))

    print(f"Training examples: {len(train_examples)}")

    model = SentenceTransformer("all-MiniLM-L6-v2")
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=32)
    train_loss = losses.CosineSimilarityLoss(model)

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=10,
        warmup_steps=100,
        output_path=output_dir,
        show_progress_bar=True
    )

    print(f"Entity linker saved to {output_dir}")
    return model

def build_entity_index(entities_file, model_dir, output_file):
    with open(entities_file) as f:
        entities = json.load(f)

    model = SentenceTransformer(model_dir)

    entity_names = []
    entity_ids = []

    for eid, einfo in entities.items():
        entity_names.append(einfo["name"])
        entity_ids.append(eid)

    print(f"Encoding {len(entity_names)} entities...")
    embeddings = model.encode(
        entity_names,
        show_progress_bar=True,
        batch_size=256
    )

    np.savez(
        output_file,
        embeddings=embeddings,
        entity_ids=entity_ids,
        entity_names=entity_names
    )
    print(f"Entity index saved to {output_file}")

if __name__ == "__main__":
    print("Step 1: Training entity linker...")
    train_entity_linker(
        "data/processed/el_training.json",
        "data/kb/entities.json",
        "models/entity_linker"
    )

    print("\nStep 2: Building entity index...")
    build_entity_index(
        "data/kb/entities.json",
        "models/entity_linker",
        "data/embeddings/entity_index.npz"
    )

    print("\nDone!")
