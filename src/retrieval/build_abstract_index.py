import json
import numpy as np
import os
from sentence_transformers import SentenceTransformer

def build_abstract_index(papers_file, output_dir):
    with open(papers_file) as f:
        papers = json.load(f)

    papers_with_abstracts = []
    for pid, pinfo in papers.items():
        abstract = pinfo.get("abstract", "")
        if abstract and len(abstract) > 50:
            papers_with_abstracts.append({
                "id": pid,
                "title": pinfo.get("title", ""),
                "abstract": abstract,
                "year": pinfo.get("year"),
                "venue": pinfo.get("venue", "")
            })

    print(f"Papers with abstracts: {len(papers_with_abstracts)}")
    model = SentenceTransformer("models/entity_linker")
    texts = [f"{p['title']}. {p['abstract'][:500]}" for p in papers_with_abstracts]

    print("Encoding abstracts...")
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=128)

    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "abstract_embeddings.npy"), embeddings)

    with open(os.path.join(output_dir, "abstract_metadata.json"), "w") as f:
        json.dump(papers_with_abstracts, f, indent=2)

    print(f"Saved {len(embeddings)} abstract embeddings to {output_dir}")

if __name__ == "__main__":
    build_abstract_index(
        "data/raw/papers/all_papers.json",
        "data/embeddings/abstracts"
    )
