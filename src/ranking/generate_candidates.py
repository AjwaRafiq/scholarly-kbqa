import json
import random
from collections import defaultdict

class PathEnumerator:
    def __init__(self, triples_file, entities_file):
        self.outgoing = defaultdict(list)
        self.incoming = defaultdict(list)

        with open(triples_file) as f:
            for line in f:
                t = json.loads(line)
                self.outgoing[t["subject"]].append((t["predicate"], t["object"]))
                self.incoming[t["object"]].append((t["predicate"], t["subject"]))

        with open(entities_file) as f:
            self.entities = json.load(f)

        print(f"Loaded KB: {len(self.entities)} entities, "
              f"{sum(len(v) for v in self.outgoing.values())} outgoing edges")

    def enumerate_paths(self, entity_ids, max_hops=2):
        candidates = []

        for eid in entity_ids:
            # 1-hop paths
            for rel1, target1 in self.outgoing.get(eid, []):
                candidates.append({
                    "path": [rel1],
                    "path_str": f"({eid})-[{rel1}]->({target1})",
                    "answer": target1,
                    "answer_name": self.entities.get(target1, {}).get("name", target1),
                    "hops": 1
                })

            if max_hops >= 2:
                # 2-hop paths
                for rel1, mid in self.outgoing.get(eid, []):
                    for rel2, target2 in self.outgoing.get(mid, []):
                        if target2 != eid:
                            candidates.append({
                                "path": [rel1, rel2],
                                "path_str": f"({eid})-[{rel1}]->({mid})-[{rel2}]->({target2})",
                                "answer": target2,
                                "answer_name": self.entities.get(target2, {}).get("name", target2),
                                "hops": 2
                            })

        # Deduplicate by answer
        seen = set()
        unique = []
        for c in candidates:
            key = (tuple(c["path"]), c["answer"])
            if key not in seen:
                seen.add(key)
                unique.append(c)

        return unique

def generate_training_candidates(qa_file, triples_file, entities_file, output_file):
    enumerator = PathEnumerator(triples_file, entities_file)

    with open(qa_file) as f:
        qa_pairs = json.load(f)

    training_data = []

    for qa in qa_pairs:
        entity_ids = []
        for mention in qa["entities"]:
            for eid, einfo in enumerator.entities.items():
                if mention.lower() in einfo["name"].lower():
                    entity_ids.append(eid)
                    break

        if not entity_ids:
            continue

        candidates = enumerator.enumerate_paths(entity_ids, max_hops=2)

        gold_answer = qa["answer"].lower().strip()
        gold_parts = [p.strip() for p in gold_answer.split(",")]

        for cand in candidates:
            answer_name = cand["answer_name"].lower()
            is_correct = (
                gold_answer in answer_name or
                answer_name in gold_answer or
                any(part in answer_name for part in gold_parts if len(part) > 3)
            )
            training_data.append({
                "question": qa["question"],
                "path": cand["path"],
                "path_str": cand["path_str"],
                "answer": cand["answer_name"],
                "label": 1 if is_correct else 0,
                "qa_id": qa["id"]
            })

    positives = [d for d in training_data if d["label"] == 1]
    negatives = [d for d in training_data if d["label"] == 0]

    random.shuffle(negatives)
    max_neg = len(positives) * 5
    balanced = positives + negatives[:max_neg]
    random.shuffle(balanced)

    with open(output_file, "w") as f:
        json.dump(balanced, f, indent=2)

    print(f"Total: {len(balanced)} ({len(positives)} positive, {min(len(negatives), max_neg)} negative)")

if __name__ == "__main__":
    print("Generating training candidates...")
    generate_training_candidates(
        "data/golden_qa/train.json",
        "data/kb/triples.jsonl",
        "data/kb/entities.json",
        "data/processed/ranker_training.json"
    )
    print("\nGenerating dev candidates...")
    generate_training_candidates(
        "data/golden_qa/dev.json",
        "data/kb/triples.jsonl",
        "data/kb/entities.json",
        "data/processed/ranker_dev.json"
    )
