import json

def check_coverage(qa_file, entities_file, triples_file):
    with open(qa_file) as f:
        qa_pairs = json.load(f)
    with open(entities_file) as f:
        entities = json.load(f)
    with open(triples_file) as f:
        triples = [json.loads(line) for line in f]

    entity_names = {e["name"].lower() for e in entities.values()}
    triple_relations = {t["predicate"] for t in triples}

    # citation_count is a property on entities, not a triple relation
    # so we treat it as always available
    property_relations = {"citation_count"}

    covered = 0
    uncovered = []

    for qa in qa_pairs:
        entities_found = all(
            any(ent.lower() in name for name in entity_names)
            for ent in qa["entities"]
        )
        relations_found = all(
            rel in triple_relations or rel in property_relations
            for rel in qa["relations"]
        )
        if entities_found and relations_found:
            covered += 1
        else:
            uncovered.append({
                "question": qa["question"],
                "missing_entities": [e for e in qa["entities"]
                    if not any(e.lower() in n for n in entity_names)],
                "missing_relations": [r for r in qa["relations"]
                    if r not in triple_relations and r not in property_relations]
            })

    coverage = covered / len(qa_pairs) * 100
    print(f"KB Coverage: {coverage:.1f}% ({covered}/{len(qa_pairs)})")

    if coverage >= 95:
        print("Coverage is good! Ready for Phase 2.")
    else:
        print(f"WARNING: Coverage below 95%. {len(uncovered)} uncovered questions.")
        for item in uncovered[:10]:
            print(f"  Q: {item['question']}")
            if item["missing_entities"]:
                print(f"     Missing entities: {item['missing_entities']}")
            if item["missing_relations"]:
                print(f"     Missing relations: {item['missing_relations']}")

if __name__ == "__main__":
    check_coverage(
        "data/golden_qa/all_generated.json",
        "data/kb/entities.json",
        "data/kb/triples.jsonl"
    )
