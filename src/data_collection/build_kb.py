import json
import os
from collections import defaultdict

INPUT_FILE = "data/raw/papers/all_papers.json"
KB_DIR = "data/kb"
os.makedirs(KB_DIR, exist_ok=True)

def build_kb(papers_dict):
    triples = []
    entities = {}
    relation_counts = defaultdict(int)

    for pid, paper in papers_dict.items():
        title = paper.get("title", "").strip()
        if not title:
            continue

        # Register paper entity
        entities[pid] = {
            "type": "Paper",
            "name": title,
            "year": paper.get("year"),
            "venue": paper.get("venue", ""),
            "citation_count": paper.get("citationCount", 0),
            "abstract": paper.get("abstract", ""),
        }

        # Paper -> authored_by -> Author
        for author in paper.get("authors", []):
            aid = author.get("authorId")
            aname = author.get("name", "").strip()
            if aid and aname:
                entities[aid] = {"type": "Author", "name": aname}
                triples.append({
                    "subject": pid,
                    "predicate": "authored_by",
                    "object": aid
                })
                relation_counts["authored_by"] += 1

        # Paper -> published_year -> Year
        year = paper.get("year")
        if year:
            year_id = f"year_{year}"
            entities[year_id] = {"type": "Year", "name": str(year)}
            triples.append({
                "subject": pid,
                "predicate": "published_year",
                "object": year_id
            })
            relation_counts["published_year"] += 1

        # Paper -> published_in -> Venue
        venue = paper.get("venue", "").strip()
        if venue:
            venue_id = f"venue_{venue.lower().replace(' ', '_')}"
            entities[venue_id] = {"type": "Venue", "name": venue}
            triples.append({
                "subject": pid,
                "predicate": "published_in",
                "object": venue_id
            })
            relation_counts["published_in"] += 1

        # Paper -> cites -> Paper
        for ref in paper.get("references", []) or []:
            ref_id = ref.get("paperId")
            ref_title = ref.get("title", "").strip()
            if ref_id and ref_title:
                if ref_id not in entities:
                    entities[ref_id] = {"type": "Paper", "name": ref_title}
                triples.append({
                    "subject": pid,
                    "predicate": "cites",
                    "object": ref_id
                })
                relation_counts["cites"] += 1

        # Paper -> has_topic -> Topic
        for field in paper.get("fieldsOfStudy", []) or []:
            topic_id = f"topic_{field.lower().replace(' ', '_')}"
            entities[topic_id] = {"type": "Topic", "name": field}
            triples.append({
                "subject": pid,
                "predicate": "has_topic",
                "object": topic_id
            })
            relation_counts["has_topic"] += 1

    return triples, entities, dict(relation_counts)

if __name__ == "__main__":
    print("Loading papers...")
    with open(INPUT_FILE) as f:
        papers = json.load(f)

    print(f"Building KB from {len(papers)} papers...")
    triples, entities, rel_counts = build_kb(papers)

    # Save triples
    with open(os.path.join(KB_DIR, "triples.jsonl"), "w") as f:
        for t in triples:
            f.write(json.dumps(t) + "\n")

    # Save entities
    with open(os.path.join(KB_DIR, "entities.json"), "w") as f:
        json.dump(entities, f, indent=2)

    # Save relations
    with open(os.path.join(KB_DIR, "relations.json"), "w") as f:
        json.dump(rel_counts, f, indent=2)

    print(f"\nKB Statistics:")
    print(f"  Entities : {len(entities)}")
    print(f"  Triples  : {len(triples)}")
    print(f"  Relations:")
    for rel, count in sorted(rel_counts.items(), key=lambda x: -x[1]):
        print(f"    {rel}: {count}")
