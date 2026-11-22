"""The chosen RAG configuration (Phase 7, round 4; selected on dev by docs/rag_design.md §6a).

Selection run 2026-11-23 (data/rag/runs/selection_*.json): recall@6 on the 8 answerable
dev questions was bm25 5/8, dense 7/8, hybrid 7/8, hybrid+glossary 7/8 and
hybrid+paragraphs 7/8, so the tie went to the simplest candidate, dense. Dev max_cosine
did not separate answerable from unanswerable questions (lowest answerable 0.6599 for
D08, highest unanswerable 0.7233 for D04), so tau is disabled: refusal relies on
NO_ANSWER and the validator. Frozen in docs/rag_preregistration.md.
"""

from src.rag.glossary import expand_query
from src.rag.retrieve import Retriever

RETRIEVAL = {"mode": "dense", "glossary": False, "paragraphs": False, "k": 6, "rrf_k": 60}
TAU = None


def build_retriever(frame, embeddings, embedder, config: dict = RETRIEVAL) -> Retriever:
    return Retriever(frame.chunk_id.tolist(), frame.embed_text.tolist(), embeddings, embedder,
                     rrf_k=config["rrf_k"], mode=config["mode"],
                     expand=expand_query if config["glossary"] else None)
