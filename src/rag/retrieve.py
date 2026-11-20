"""Hybrid retrieval: dense cosine + BM25, fused by reciprocal-rank fusion (Phase 7).

RRF score of a chunk = sum over the two rankings of 1 / (RRF_K + rank). It is
rank-based: the top chunk always gets a high fused score, even when nothing in
the corpus is relevant. So the refusal signal is NOT the fused score (decision
C1): it is `max_cosine`, the dense cosine of the best chunk in the whole corpus,
with `max_bm25` reported as an optional second condition. Both thresholds are
set on the dev questions only.
"""

from dataclasses import dataclass

import numpy as np

from src.rag.bm25 import BM25

RRF_K = 60
TOP_K = 6


@dataclass
class Hit:
    chunk_id: str
    rank: int              # 1-based rank after fusion
    rrf: float
    cosine: float
    bm25: float
    dense_rank: int
    bm25_rank: int | None  # None when the chunk shares no term with the query


@dataclass
class Retrieval:
    query: str
    hits: list[Hit]
    max_cosine: float      # refusal signal (C1): best dense similarity over ALL chunks
    max_bm25: float


def ranks(scores: np.ndarray, eligible: np.ndarray | None = None) -> dict[int, int]:
    """index -> 1-based rank by descending score; ties broken by index (deterministic)."""
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
    if eligible is not None:
        order = [i for i in order if eligible[i]]
    return {i: r for r, i in enumerate(order, start=1)}


def rrf_fuse(rank_maps: list[dict[int, int]], k: int = RRF_K) -> dict[int, float]:
    fused: dict[int, float] = {}
    for rank_map in rank_maps:
        for i, r in rank_map.items():
            fused[i] = fused.get(i, 0.0) + 1.0 / (k + r)
    return fused


class Retriever:
    def __init__(self, chunk_ids: list[str], embed_texts: list[str], embeddings: np.ndarray, embedder,
                 rrf_k: int = RRF_K):
        if len(chunk_ids) != len(embed_texts) or len(chunk_ids) != embeddings.shape[0]:
            raise ValueError("chunk ids, texts and embeddings differ in length")
        self.chunk_ids, self.embeddings, self.embedder, self.rrf_k = chunk_ids, embeddings, embedder, rrf_k
        self.bm25 = BM25(embed_texts)

    def search(self, query: str, k: int = TOP_K) -> Retrieval:
        cos = self.embeddings @ self.embedder.embed_query(query)
        bm = self.bm25.scores(query)
        dense_r, bm_r = ranks(cos), ranks(bm, eligible=bm > 0)
        fused = rrf_fuse([dense_r, bm_r], self.rrf_k)
        order = sorted(fused, key=lambda i: (-fused[i], i))[:k]
        hits = [Hit(self.chunk_ids[i], n, fused[i], float(cos[i]), float(bm[i]), dense_r[i], bm_r.get(i))
                for n, i in enumerate(order, start=1)]
        return Retrieval(query, hits, float(cos.max()), float(bm.max()))
