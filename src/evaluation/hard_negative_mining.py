import json

def mine_hard_negatives(eval_results_file, original_train_file, output_file):
    with open(eval_results_file) as f:
        eval_data = json.load(f)
    with open(original_train_file) as f:
        original_train = json.load(f)

    errors = eval_data["errors"]
    print(f"Found {len(errors)} errors to mine")

    hard_negatives = []
    for error in errors:
        hard_negatives.append({
            "question": error["question"],
            "predicted_wrong": error["predicted"],
            "gold_correct": error["gold"],
            "error_type": error["type"],
            "source": error["source"]
        })

    augmented_train = original_train + hard_negatives

    with open(output_file, "w") as f:
        json.dump(augmented_train, f, indent=2)

    print(f"Augmented training data: {len(augmented_train)} examples "
          f"({len(original_train)} original + {len(hard_negatives)} hard negatives)")

if __name__ == "__main__":
    mine_hard_negatives(
        "results/evaluation_results.json",
        "data/processed/ranker_training.json",
        "data/processed/ranker_training_augmented.json"
    )
