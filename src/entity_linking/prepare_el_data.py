import json
import re
import os

os.makedirs("data/processed", exist_ok=True)

def prepare_el_training_data(qa_file, entities_file, output_file):
    with open(qa_file) as f:
        qa_pairs = json.load(f)
    with open(entities_file) as f:
        entities = json.load(f)

    # Build name-to-id mapping
    name_to_id = {}
    for eid, einfo in entities.items():
        name_lower = einfo["name"].lower().strip()
        name_to_id[name_lower] = eid

    el_data = []

    for qa in qa_pairs:
        question = qa["question"]

        for entity_mention in qa["entities"]:
            mention_lower = entity_mention.lower().strip()
            entity_id = None

            # Exact match first
            if mention_lower in name_to_id:
                entity_id = name_to_id[mention_lower]
            else:
                # Fuzzy match
                for name, eid in name_to_id.items():
                    if mention_lower in name or name in mention_lower:
                        entity_id = eid
                        break

            if entity_id:
                match = re.search(re.escape(entity_mention), question, re.IGNORECASE)
                if match:
                    el_data.append({
                        "question": question,
                        "mention": entity_mention,
                        "span_start": match.start(),
                        "span_end": match.end(),
                        "entity_id": entity_id,
                        "entity_name": entities[entity_id]["name"]
                    })

    with open(output_file, "w") as f:
        json.dump(el_data, f, indent=2)

    print(f"Created {len(el_data)} entity linking examples")
    return el_data

if __name__ == "__main__":
    prepare_el_training_data(
        "data/golden_qa/all_generated.json",
        "data/kb/entities.json",
        "data/processed/el_training.json"
    )
