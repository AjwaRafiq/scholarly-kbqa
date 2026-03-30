import json
import numpy as np
from sentence_transformers import SentenceTransformer

class AbstractRetriever:
    def __init__(self, embeddings_dir, model_dir="models/entity_linker"):
        self.model = SentenceTransformer(model_dir)
        self.embeddings = np.load(f"{embeddings_dir}/abstract_embeddings.npy")

        with open(f"{embeddings_dir}/abstract_metadata.json") as f:
            self.metadata = json.load(f)

        # Normalize
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        self.embeddings_norm = self.embeddings / (norms + 1e-8)

        print(f"Loaded {len(self.metadata)} abstract embeddings")

    def retrieve(self, question, top_k=5):
        q_emb = self.model.encode([question])
        q_norm = q_emb / (np.linalg.norm(q_emb) + 1e-8)
        scores = np.dot(q_norm, self.embeddings_norm.T)[0]
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            meta = self.metadata[idx]
            results.append({
                "title": meta["title"],
                "abstract": meta["abstract"][:500],
                "score": float(scores[idx]),
                "year": meta.get("year"),
                "venue": meta.get("venue", ""),
                "paper_id": meta["id"]
            })

        return results

# Test it
if __name__ == "__main__":
    retriever = AbstractRetriever("data/embeddings/abstracts")

    test_questions = [
        "What is the attention mechanism in transformers?",
        "How does BERT use bidirectional training?",
        "What are graph neural networks used for?"
    ]

    for q in test_questions:
        print(f"\nQ: {q}")
        results = retriever.retrieve(q, top_k=3)
        for r in results:
            print(f"  -> [{r['score']:.3f}] {r['title']} ({r['year']})")
