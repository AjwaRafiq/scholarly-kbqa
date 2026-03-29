import numpy as np
from sentence_transformers import SentenceTransformer

class EntityLinker:
    def __init__(self, model_dir, index_file):
        self.model = SentenceTransformer(model_dir)
        data = np.load(index_file, allow_pickle=True)
        self.embeddings = data["embeddings"]
        self.entity_ids = list(data["entity_ids"])
        self.entity_names = list(data["entity_names"])

        # Normalize for cosine similarity
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        self.embeddings_norm = self.embeddings / (norms + 1e-8)

    def link(self, question, top_k=5, threshold=0.5):
        q_emb = self.model.encode([question])
        q_norm = q_emb / (np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-8)
        scores = np.dot(q_norm, self.embeddings_norm.T)[0]
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            score = float(scores[idx])
            if score >= threshold:
                results.append((
                    self.entity_ids[idx],
                    self.entity_names[idx],
                    score
                ))
        return results

if __name__ == "__main__":
    linker = EntityLinker("models/entity_linker", "data/embeddings/entity_index.npz")

    test_questions = [
        "Who wrote the Attention Is All You Need paper?",
        "When was BERT published?",
        "What papers cite GPT-3?",
        "What papers by Yoshua Bengio were published in NeurIPS?",
        "How many papers has Yann LeCun published?"
    ]

    for q in test_questions:
        print(f"\nQ: {q}")
        entities = linker.link(q)
        if entities:
            for eid, ename, score in entities:
                print(f"  -> {ename} (score: {score:.3f})")
        else:
            print("  -> No entities found above threshold")
