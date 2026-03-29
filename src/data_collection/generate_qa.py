import json
import os
import random
from collections import defaultdict

ENTITIES_FILE = "data/kb/entities.json"
TRIPLES_FILE = "data/kb/triples.jsonl"
OUTPUT_DIR = "data/golden_qa"
os.makedirs(OUTPUT_DIR, exist_ok=True)

random.seed(42)

def load_kb():
    print("Loading entities...")
    with open(ENTITIES_FILE) as f:
        entities = json.load(f)
    print("Loading triples...")
    triples = []
    with open(TRIPLES_FILE) as f:
        for line in f:
            triples.append(json.loads(line))
    return entities, triples

def build_indexes(entities, triples):
    subj_index = defaultdict(list)
    obj_index = defaultdict(list)
    for t in triples:
        subj_index[t["subject"]].append((t["predicate"], t["object"]))
        obj_index[t["object"]].append((t["predicate"], t["subject"]))
    return subj_index, obj_index

def get_papers_with_metadata(entities):
    papers = {}
    for eid, e in entities.items():
        if (e.get("type") == "Paper" and
            e.get("year") and
            e.get("name") and
            e.get("venue")):
            papers[eid] = e
    return papers

def generate_single_hop(entities, subj_index, papers, n=120):
    qa_pairs = []
    paper_ids = list(papers.keys())
    random.shuffle(paper_ids)

    for pid in paper_ids:
        if len(qa_pairs) >= n:
            break
        paper = papers[pid]
        title = paper["name"]
        relations = subj_index[pid]

        # authored_by
        authors = [entities[oid]["name"] for pred, oid in relations
                   if pred == "authored_by" and oid in entities]
        if authors and len(qa_pairs) < n:
            qa_pairs.append({
                "id": f"q_single_{len(qa_pairs)+1:03d}",
                "question": f"Who are the authors of the paper '{title}'?",
                "type": "single_hop",
                "answer": ", ".join(authors),
                "sparql": f"SELECT ?author WHERE {{ ?p title '{title}' . ?p authored_by ?a . ?a name ?author }}",
                "entities": [title],
                "relations": ["authored_by"],
                "difficulty": "easy",
                "verified": False
            })

        # published_year
        year = paper.get("year")
        if year and len(qa_pairs) < n:
            qa_pairs.append({
                "id": f"q_single_{len(qa_pairs)+1:03d}",
                "question": f"When was the paper '{title}' published?",
                "type": "single_hop",
                "answer": str(year),
                "sparql": f"SELECT ?year WHERE {{ ?p title '{title}' . ?p published_year ?year }}",
                "entities": [title],
                "relations": ["published_year"],
                "difficulty": "easy",
                "verified": False
            })

        # published_in
        venue = paper.get("venue")
        if venue and len(qa_pairs) < n:
            qa_pairs.append({
                "id": f"q_single_{len(qa_pairs)+1:03d}",
                "question": f"Where was the paper '{title}' published?",
                "type": "single_hop",
                "answer": venue,
                "sparql": f"SELECT ?venue WHERE {{ ?p title '{title}' . ?p published_in ?venue }}",
                "entities": [title],
                "relations": ["published_in"],
                "difficulty": "easy",
                "verified": False
            })

    return qa_pairs[:n]

def generate_two_hop(entities, subj_index, obj_index, papers, n=80):
    qa_pairs = []
    paper_ids = list(papers.keys())
    random.shuffle(paper_ids)

    for pid in paper_ids:
        if len(qa_pairs) >= n:
            break
        paper = papers[pid]
        title = paper["name"]
        relations = subj_index[pid]

        authors = [(oid, entities[oid]["name"]) for pred, oid in relations
                   if pred == "authored_by" and oid in entities]
        venue = paper.get("venue")

        if authors and venue and len(qa_pairs) < n:
            author_id, author_name = authors[0]
            author_papers = [entities[sid]["name"] for pred, sid in obj_index[author_id]
                             if pred == "authored_by" and sid in entities
                             and entities[sid].get("venue") == venue
                             and sid != pid]
            if author_papers:
                qa_pairs.append({
                    "id": f"q_two_{len(qa_pairs)+1:03d}",
                    "question": f"What papers by {author_name} were published in {venue}?",
                    "type": "two_hop",
                    "answer": ", ".join(author_papers[:5]),
                    "sparql": f"SELECT ?title WHERE {{ ?p authored_by ?a . ?a name '{author_name}' . ?p published_in ?v . ?v name '{venue}' . ?p title ?title }}",
                    "entities": [author_name, venue],
                    "relations": ["authored_by", "published_in"],
                    "difficulty": "medium",
                    "verified": False
                })

        topics = [entities[oid]["name"] for pred, oid in relations
                  if pred == "has_topic" and oid in entities]
        citing = [entities[sid]["name"] for pred, sid in obj_index[pid]
                  if pred == "cites" and sid in entities
                  and entities[sid].get("year")]

        if topics and citing and len(qa_pairs) < n:
            qa_pairs.append({
                "id": f"q_two_{len(qa_pairs)+1:03d}",
                "question": f"Which papers cite '{title}' and are about {topics[0]}?",
                "type": "two_hop",
                "answer": ", ".join(citing[:5]),
                "sparql": f"SELECT ?p WHERE {{ ?p cites ?ref . ?ref title '{title}' . ?p has_topic ?t . ?t name '{topics[0]}' }}",
                "entities": [title, topics[0]],
                "relations": ["cites", "has_topic"],
                "difficulty": "medium",
                "verified": False
            })

    return qa_pairs[:n]

def generate_aggregation(entities, subj_index, obj_index, papers, n=50):
    qa_pairs = []

    # Group papers by author
    author_papers = defaultdict(list)
    for pid in papers:
        for pred, oid in subj_index[pid]:
            if pred == "authored_by" and oid in entities:
                author_papers[oid].append(pid)

    for aid, pids in list(author_papers.items()):
        if len(qa_pairs) >= n // 2:
            break
        if len(pids) >= 2:
            author_name = entities[aid]["name"]
            qa_pairs.append({
                "id": f"q_agg_{len(qa_pairs)+1:03d}",
                "question": f"How many papers has {author_name} published in this dataset?",
                "type": "aggregation",
                "answer": str(len(pids)),
                "sparql": f"SELECT (COUNT(?p) AS ?cnt) WHERE {{ ?p authored_by ?a . ?a name '{author_name}' }}",
                "entities": [author_name],
                "relations": ["authored_by"],
                "difficulty": "medium",
                "verified": False
            })

    # Most cited paper per venue
    venue_papers = defaultdict(list)
    for pid, paper in papers.items():
        venue = paper.get("venue")
        if venue:
            venue_papers[venue].append((pid, paper.get("citation_count", 0), paper["name"]))

    for venue, vpapers in list(venue_papers.items()):
        if len(qa_pairs) >= n:
            break
        if len(vpapers) >= 2:
            top = max(vpapers, key=lambda x: x[1])
            qa_pairs.append({
                "id": f"q_agg_{len(qa_pairs)+1:03d}",
                "question": f"What is the most cited paper published in {venue}?",
                "type": "aggregation",
                "answer": top[2],
                "sparql": f"SELECT ?title WHERE {{ ?p published_in ?v . ?v name '{venue}' . ?p title ?title }} ORDER BY DESC(?citations) LIMIT 1",
                "entities": [venue],
                "relations": ["published_in"],
                "difficulty": "medium",
                "verified": False
            })

    return qa_pairs[:n]

def generate_comparison(entities, subj_index, papers, n=50):
    qa_pairs = []
    paper_list = list(papers.items())
    random.shuffle(paper_list)

    # Papers published after year X
    years = sorted(set(p["year"] for _, p in paper_list if p.get("year")))
    if years:
        mid_year = years[len(years)//2]
        recent = [p["name"] for _, p in paper_list if p.get("year", 0) > mid_year]
        if recent and len(qa_pairs) < n:
            qa_pairs.append({
                "id": f"q_comp_{len(qa_pairs)+1:03d}",
                "question": f"Which papers in the dataset were published after {mid_year}?",
                "type": "comparison",
                "answer": ", ".join(recent[:5]) + f" (and {len(recent)-5} more)" if len(recent) > 5 else ", ".join(recent),
                "sparql": f"SELECT ?title WHERE {{ ?p published_year ?y . FILTER(?y > {mid_year}) . ?p title ?title }}",
                "entities": [str(mid_year)],
                "relations": ["published_year"],
                "difficulty": "medium",
                "verified": False
            })

    # Compare citation counts
    for i in range(0, len(paper_list)-1, 2):
        if len(qa_pairs) >= n:
            break
        pid1, p1 = paper_list[i]
        pid2, p2 = paper_list[i+1]
        c1 = p1.get("citation_count", 0) or 0
        c2 = p2.get("citation_count", 0) or 0
        if c1 > 0 and c2 > 0:
            winner = p1["name"] if c1 >= c2 else p2["name"]
            qa_pairs.append({
                "id": f"q_comp_{len(qa_pairs)+1:03d}",
                "question": f"Which paper has more citations: '{p1['name']}' or '{p2['name']}'?",
                "type": "comparison",
                "answer": winner,
                "sparql": f"SELECT ?title WHERE {{ VALUES ?title {{'{p1['name']}' '{p2['name']}'}} ?p title ?title }} ORDER BY DESC(?citations) LIMIT 1",
                "entities": [p1["name"], p2["name"]],
                "relations": ["citation_count"],
                "difficulty": "hard",
                "verified": False
            })

    return qa_pairs[:n]

def split_and_save(all_qa):
    random.shuffle(all_qa)
    train = all_qa[:200]
    dev = all_qa[200:250]
    test = all_qa[250:300]

    for split, data in [("train", train), ("dev", dev), ("test", test)]:
        path = os.path.join(OUTPUT_DIR, f"{split}.json")
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"  Saved {len(data)} pairs to {path}")

    # Also save all for review
    with open(os.path.join(OUTPUT_DIR, "all_generated.json"), "w") as f:
        json.dump(all_qa, f, indent=2)

if __name__ == "__main__":
    entities, triples = load_kb()
    subj_index, obj_index = build_indexes(entities, triples)
    papers = get_papers_with_metadata(entities)

    print(f"\nPapers with full metadata: {len(papers)}")

    print("\nGenerating single-hop questions (120)...")
    single = generate_single_hop(entities, subj_index, papers, n=120)
    print(f"  Generated: {len(single)}")

    print("Generating two-hop questions (80)...")
    two = generate_two_hop(entities, subj_index, obj_index, papers, n=80)
    print(f"  Generated: {len(two)}")

    print("Generating aggregation questions (50)...")
    agg = generate_aggregation(entities, subj_index, obj_index, papers, n=50)
    print(f"  Generated: {len(agg)}")

    print("Generating comparison questions (50)...")
    comp = generate_comparison(entities, subj_index, papers, n=50)
    print(f"  Generated: {len(comp)}")

    all_qa = single + two + agg + comp
    print(f"\nTotal generated: {len(all_qa)}")

    print("\nSplitting and saving...")
    split_and_save(all_qa)

    print("\nDone! Review data/golden_qa/all_generated.json to verify questions.")
