import requests
import json
import time
import os

API_BASE = "https://api.semanticscholar.org/graph/v1"
OUTPUT_DIR = "data/raw/papers"
os.makedirs(OUTPUT_DIR, exist_ok=True)

API_KEY = "i5xXoovL6o8We81Rsmzzj87tYkTPBksy6MpAxKz9"

HEADERS = {"x-api-key": API_KEY}

FIELDS = (
    "paperId,title,abstract,year,venue,citationCount,"
    "authors.authorId,authors.name,"
    "references.paperId,references.title,"
    "fieldsOfStudy"
)

DOMAIN_QUERIES = [
    "transformer architecture",
    "natural language processing",
    "knowledge graph question answering",
    "BERT pre-training",
    "graph neural networks",
    "reinforcement learning",
    "computer vision object detection",
    "large language models",
    "attention mechanism",
    "transfer learning",
    "neural machine translation",
    "question answering systems",
    "information retrieval",
    "entity linking",
    "text classification",
]

def search_papers(query, limit=100, offset=0, retries=3):
    url = f"{API_BASE}/paper/search"
    params = {
        "query": query,
        "limit": min(limit, 100),
        "offset": offset,
        "fields": FIELDS
    }
    for attempt in range(retries):
        try:
            response = requests.get(url, params=params, headers=HEADERS, timeout=30)
            if response.status_code == 429:
                print(f"Rate limited. Waiting 10s...")
                time.sleep(10)
                continue
            response.raise_for_status()
            return response.json()
        except requests.exceptions.Timeout:
            print(f"Timeout on attempt {attempt+1}. Waiting 30s...")
            time.sleep(30)
        except Exception as e:
            print(f"Error on attempt {attempt+1}: {e}. Waiting 15s...")
            time.sleep(15)
    return {"data": []}

def load_existing():
    out = os.path.join(OUTPUT_DIR, "all_papers.json")
    if os.path.exists(out) and os.path.getsize(out) > 0:
        with open(out) as f:
            return json.load(f)
    return {}

def save_papers(papers):
    out = os.path.join(OUTPUT_DIR, "all_papers.json")
    with open(out, "w") as f:
        json.dump(papers, f)
    print(f"  Saved {len(papers)} papers so far...")

def load_progress():
    prog = os.path.join(OUTPUT_DIR, "progress.json")
    if os.path.exists(prog):
        with open(prog) as f:
            return json.load(f)
    return {"completed_queries": []}

def save_progress(progress):
    prog = os.path.join(OUTPUT_DIR, "progress.json")
    with open(prog, "w") as f:
        json.dump(progress, f)

if __name__ == "__main__":
    all_papers = load_existing()
    progress = load_progress()
    completed = progress["completed_queries"]

    print(f"Resuming: {len(all_papers)} papers already collected")
    print(f"Completed queries: {completed}\n")

    for query in DOMAIN_QUERIES:
        if query in completed:
            print(f"Skipping '{query}' (already done)")
            continue

        print(f"\nCollecting: '{query}'")
        collected = 0
        offset = 0
        papers_per_query = 200

        while collected < papers_per_query:
            batch_size = min(100, papers_per_query - collected)
            result = search_papers(query, limit=batch_size, offset=offset)
            papers = result.get("data", [])

            if not papers:
                break

            for paper in papers:
                pid = paper.get("paperId")
                if pid and pid not in all_papers:
                    all_papers[pid] = paper

            collected += len(papers)
            offset += batch_size
            time.sleep(3)

        save_papers(all_papers)
        completed.append(query)
        save_progress({"completed_queries": completed})
        print(f"  Done '{query}': {collected} fetched | Total unique: {len(all_papers)}")

    print(f"\nCollection complete! Total papers: {len(all_papers)}")
