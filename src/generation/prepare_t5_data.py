import json
import os

os.makedirs("data/processed", exist_ok=True)

def prepare_t5_data(qa_file, output_file):
    with open(qa_file) as f:
        qa_pairs = json.load(f)

    t5_data = []

    for qa in qa_pairs:
        input_text = f"translate question to sparql: {qa['question']}"
        output_text = qa["sparql"].strip()

        t5_data.append({
            "input": input_text,
            "output": output_text,
            "id": qa["id"],
            "type": qa["type"]
        })

    with open(output_file, "w") as f:
        json.dump(t5_data, f, indent=2)

    print(f"Created {len(t5_data)} T5 training examples from {qa_file}")

if __name__ == "__main__":
    prepare_t5_data("data/golden_qa/train.json", "data/processed/t5_train.json")
    prepare_t5_data("data/golden_qa/dev.json", "data/processed/t5_dev.json")
