import json
import sys
import re
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import BertTokenizer, BertForSequenceClassification

sys.path.insert(0, ".")
from src.entity_linking.linker import EntityLinker
from src.ranking.generate_candidates import PathEnumerator
from src.pipeline.executor import KBExecutor
from src.retrieval.retriever import AbstractRetriever

class ScholarlyKBQA:
    def __init__(self, config):
        print("Loading Entity Linker...")
        self.entity_linker = EntityLinker(
            config["entity_linker_dir"],
            config["entity_index_file"]
        )

        print("Loading Path Enumerator...")
        self.path_enumerator = PathEnumerator(
            config["triples_file"],
            config["entities_file"]
        )

        print("Loading KB Executor...")
        self.executor = KBExecutor(
            config["triples_file"],
            config["entities_file"]
        )

        print("Loading BERT Ranker...")
        self.ranker_tokenizer = BertTokenizer.from_pretrained(config["bert_ranker_dir"])
        self.ranker_model = BertForSequenceClassification.from_pretrained(
            config["bert_ranker_dir"]
        )
        self.ranker_model.eval()

        print("Loading T5 Generator...")
        self.t5_tokenizer = T5Tokenizer.from_pretrained(
            config["t5_generator_dir"], local_files_only=True
        )
        self.t5_model = T5ForConditionalGeneration.from_pretrained(
            config["t5_generator_dir"], local_files_only=True
        )
        self.t5_model.eval()

        print("Loading Abstract Retriever...")
        self.retriever = AbstractRetriever(config["abstract_embeddings_dir"])

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ranker_model.to(self.device)
        self.t5_model.to(self.device)

        print(f"Pipeline ready! (device: {self.device})")

    def detect_question_intent(self, question):
        q = question.lower()
        if any(w in q for w in ["who wrote", "who are the authors", "authored by", "who published"]):
            return "authored_by"
        if any(w in q for w in ["when was", "what year", "published in year", "year was"]):
            return "published_year"
        if any(w in q for w in ["where was", "what venue", "what conference", "what journal"]):
            return "published_in"
        if "cite" in q and ("about" in q or "topic" in q or "field" in q):
            return "cites_topic"
        if any(w in q for w in ["cite", "cites", "citing", "cited by"]):
            return "cites"
        if "papers by" in q and ("published in" in q or "conference" in q or "journal" in q):
            return "author_venue"
        if any(w in q for w in ["how many papers", "count", "number of papers"]):
            return "count"
        if any(w in q for w in ["most cited", "highest citation"]):
            return "most_cited"
        if any(w in q for w in ["more citations", "more cited"]):
            return "compare_citations"
        if any(w in q for w in ["topic", "field", "area", "about"]):
            return "has_topic"
        return None

    def answer(self, question, verbose=False):
        result = {
            "question": question,
            "answer": None,
            "confidence": 0.0,
            "source": None,
            "evidence": [],
            "debug": {}
        }

        intent = self.detect_question_intent(question)
        if verbose:
            print(f"  Intent: {intent}")

        if intent == "count":
            return self._handle_count(question, result, verbose)

        if intent == "most_cited":
            return self._handle_most_cited(question, result, verbose)

        if intent == "compare_citations":
            return self._handle_comparison(question, result, verbose)

        if intent == "cites_topic":
            return self._handle_cites_topic(question, result, verbose)

        if intent == "author_venue":
            return self._handle_author_venue(question, result, verbose)

        # STEP 1: Entity Linking
        entities = self.entity_linker.link(question, top_k=5, threshold=0.4)
        if verbose:
            print(f"  Entities: {[(e[1], f'{e[2]:.2f}') for e in entities]}")

        result["debug"]["entities"] = [
            {"id": e[0], "name": e[1], "score": e[2]} for e in entities
        ]

        if not entities:
            return self._fallback_abstract(question, result, verbose)

        entity_ids = [e[0] for e in entities[:3]]
        candidates = self.path_enumerator.enumerate_paths(entity_ids, max_hops=2)

        if verbose:
            print(f"  Candidate paths: {len(candidates)}")

        if not candidates:
            return self._fallback_abstract(question, result, verbose)

        top_path, top_entity_id, kb_confidence = self._select_path(
            question, intent, candidates, entity_ids
        )

        if verbose:
            print(f"  Top path: {top_path} (score: {kb_confidence:.3f})")

        if len(top_path) == 1:
            all_results = self.executor.execute_1hop(top_entity_id, top_path[0])
        elif len(top_path) == 2:
            all_results = self.executor.execute_2hop(top_entity_id, top_path[0], top_path[1])
        else:
            all_results = []

        if all_results:
            kb_answer = ", ".join([r["name"] for r in all_results[:10]])
        else:
            kb_answer = ""

        if kb_answer and kb_confidence >= 0.5:
            result["answer"] = kb_answer
            result["natural_answer"] = kb_answer
            result["confidence"] = kb_confidence
            result["source"] = "knowledge_base"
            result["evidence"] = [{"type": "kb_triple", "path": top_path, "answer": kb_answer}]
            return result
        else:
            return self._fallback_abstract(question, result, verbose,
                                           kb_answer=kb_answer, kb_score=kb_confidence)

    def _handle_cites_topic(self, question, result, verbose):
        """Handle 'which papers cite X and are about topic Y'."""
        # Extract cited paper from quotes
        matches = re.findall(r"'([^']+)'", question)
        if not matches:
            return self._fallback_abstract(question, result, verbose)

        cited_title = matches[0]
        cited_id = self.executor.resolve_entity(cited_title)

        if not cited_id:
            return self._fallback_abstract(question, result, verbose)

        # Find all papers that cite this paper (incoming cites triples)
        citing_papers = [
            t["subject"] for t in self.executor.incoming.get(cited_id, [])
            if t["predicate"] == "cites"
        ]

        if not citing_papers:
            return self._fallback_abstract(question, result, verbose)

        # Get names of citing papers
        paper_names = [
            self.executor.entities.get(pid, {}).get("name", "")
            for pid in citing_papers
            if self.executor.entities.get(pid, {}).get("name")
        ]

        if paper_names:
            answer = ", ".join(paper_names[:5])
            result["answer"] = answer
            result["natural_answer"] = answer
            result["confidence"] = 0.85
            result["source"] = "knowledge_base"
            result["evidence"] = [{"type": "kb_cites", "cited": cited_title}]
            return result

        return self._fallback_abstract(question, result, verbose)

    def _handle_author_venue(self, question, result, verbose):
        """Handle 'papers by author X published in venue Y'."""
        # Extract author name — typically before "were published"
        author_match = re.search(r"papers by ([^']+?) (?:were published|published)", question, re.I)
        venue_match = re.search(r"published in (.+?)(?:\?|$)", question, re.I)

        if not author_match or not venue_match:
            return self._fallback_abstract(question, result, verbose)

        author_name = author_match.group(1).strip()
        venue_name = venue_match.group(1).strip()

        author_id = self.executor.resolve_entity(author_name)
        venue_id = self.executor.resolve_entity(venue_name)

        if not author_id:
            return self._fallback_abstract(question, result, verbose)

        # Find papers by this author
        author_papers = set(
            t["subject"] for t in self.executor.incoming.get(author_id, [])
            if t["predicate"] == "authored_by"
        )

        # Find papers in this venue
        if venue_id:
            venue_papers = set(
                t["subject"] for t in self.executor.incoming.get(venue_id, [])
                if t["predicate"] == "published_in"
            )
            matching = author_papers & venue_papers
        else:
            matching = author_papers

        paper_names = [
            self.executor.entities.get(pid, {}).get("name", "")
            for pid in matching
            if self.executor.entities.get(pid, {}).get("name")
        ]

        if paper_names:
            answer = ", ".join(paper_names[:5])
            result["answer"] = answer
            result["natural_answer"] = answer
            result["confidence"] = 0.85
            result["source"] = "knowledge_base"
            result["evidence"] = [{"type": "kb_author_venue"}]
            return result

        return self._fallback_abstract(question, result, verbose)

    def _select_path(self, question, intent, candidates, entity_ids):
        intent_to_relation = {
            "authored_by": "authored_by",
            "published_year": "published_year",
            "published_in": "published_in",
            "cites": "cites",
            "has_topic": "has_topic"
        }

        target_relation = intent_to_relation.get(intent)

        if target_relation:
            intent_candidates = [c for c in candidates if c["path"][0] == target_relation]
            if intent_candidates:
                scored = self._rank_candidates(question, intent_candidates)
                top = scored[0]
                return top["path"], entity_ids[0], top["score"]

        scored = self._rank_candidates(question, candidates)
        top = scored[0]
        return top["path"], entity_ids[0], top["score"]

    def _handle_count(self, question, result, verbose):
        entities = self.entity_linker.link(question, top_k=3, threshold=0.4)
        if not entities:
            return self._fallback_abstract(question, result, verbose)

        for eid, ename, escore in entities:
            incoming = self.executor.incoming.get(eid, [])
            papers = [t for t in incoming if t["predicate"] == "authored_by"]
            count = len(papers)
            if count > 0:
                result["answer"] = str(count)
                result["natural_answer"] = str(count)
                result["confidence"] = escore
                result["source"] = "knowledge_base"
                result["evidence"] = [{"type": "kb_count", "entity": ename, "count": count}]
                return result

        return self._fallback_abstract(question, result, verbose)

    def _handle_most_cited(self, question, result, verbose):
        entities = self.entity_linker.link(question, top_k=5, threshold=0.4)
        if not entities:
            return self._fallback_abstract(question, result, verbose)

        for eid, ename, escore in entities:
            incoming = self.executor.incoming.get(eid, [])
            papers = [
                {
                    "id": t["subject"],
                    "name": self.executor.entities.get(t["subject"], {}).get("name", ""),
                    "citation_count": self.executor.entities.get(t["subject"], {}).get("citation_count", 0) or 0
                }
                for t in incoming if t["predicate"] == "published_in"
            ]

            if papers:
                top = max(papers, key=lambda x: x["citation_count"])
                answer = top.get("name", "")
                if answer:
                    result["answer"] = answer
                    result["natural_answer"] = answer
                    result["confidence"] = escore
                    result["source"] = "knowledge_base"
                    result["evidence"] = [{"type": "kb_most_cited", "venue": ename}]
                    return result

        return self._fallback_abstract(question, result, verbose)

    def _handle_comparison(self, question, result, verbose):
        matches = re.findall(r"'([^']+)'", question)
        if len(matches) < 2:
            return self._fallback_abstract(question, result, verbose)

        title1, title2 = matches[0], matches[1]
        eid1 = self.executor.resolve_entity(title1)
        eid2 = self.executor.resolve_entity(title2)

        c1 = self.executor.entities.get(eid1, {}).get("citation_count", 0) or 0 if eid1 else 0
        c2 = self.executor.entities.get(eid2, {}).get("citation_count", 0) or 0 if eid2 else 0

        if c1 == 0 and c2 == 0:
            return self._fallback_abstract(question, result, verbose)

        answer = title1 if c1 >= c2 else title2
        result["answer"] = answer
        result["natural_answer"] = answer
        result["confidence"] = 0.9
        result["source"] = "knowledge_base"
        result["evidence"] = [{"type": "kb_comparison", "c1": c1, "c2": c2}]
        return result

    def _rank_candidates(self, question, candidates):
        scored = []
        for cand in candidates:
            path_str = " > ".join(cand["path"])
            encoding = self.ranker_tokenizer(
                question, path_str,
                max_length=128, padding="max_length",
                truncation=True, return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.ranker_model(**encoding)
                probs = torch.softmax(outputs.logits, dim=-1)
                score = probs[0][1].item()

            cand["score"] = score
            scored.append(cand)

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored

    def _fallback_abstract(self, question, result, verbose,
                            kb_answer=None, kb_score=0):
        if verbose:
            print("  -> Falling back to abstract retrieval")

        abstracts = self.retriever.retrieve(question, top_k=3)

        evidence = []
        if kb_answer and kb_score > 0.3:
            evidence.append({"type": "kb_partial", "answer": kb_answer, "confidence": kb_score})

        for abstract in abstracts:
            evidence.append({
                "type": "abstract",
                "title": abstract["title"],
                "text": abstract["abstract"][:300],
                "score": abstract["score"]
            })

        result["answer"] = self._synthesize_answer(question, kb_answer, evidence)
        result["natural_answer"] = result["answer"]
        result["confidence"] = max(kb_score, abstracts[0]["score"] if abstracts else 0)
        result["source"] = "hybrid" if kb_answer else "abstract_retrieval"
        result["evidence"] = evidence
        return result

    def _synthesize_answer(self, question, kb_answer, evidence):
        if kb_answer:
            return kb_answer
        for e in evidence:
            if e["type"] == "abstract":
                return e["title"]
        return "I could not find an answer to this question."


DEFAULT_CONFIG = {
    "entity_linker_dir": "models/entity_linker",
    "entity_index_file": "data/embeddings/entity_index.npz",
    "triples_file": "data/kb/triples.jsonl",
    "entities_file": "data/kb/entities.json",
    "bert_ranker_dir": "models/bert_ranker",
    "t5_generator_dir": "models/t5_generator",
    "abstract_embeddings_dir": "data/embeddings/abstracts",
}

if __name__ == "__main__":
    pipeline = ScholarlyKBQA(DEFAULT_CONFIG)

    test_questions = [
        "Who wrote the Attention Is All You Need paper?",
        "When was BERT published?",
        "How many papers has Ashish Vaswani published in this dataset?",
        "Which paper has more citations: 'BERT' or 'GPT-3'?",
        "Which papers cite 'Graph Neural Networks for Anomaly Detection in Industrial Internet of Things' and are about Computer Science?",
    ]

    for q in test_questions:
        print(f"\n{'='*60}")
        print(f"Q: {q}")
        result = pipeline.answer(q, verbose=True)
        print(f"A: {result.get('natural_answer', result.get('answer', 'N/A'))}")
        print(f"Confidence: {result['confidence']:.2f}")
        print(f"Source: {result['source']}")
