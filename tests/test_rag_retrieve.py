"""BM25, reciprocal-rank fusion, hybrid retrieval and the index files (FakeEmbedder, no network)."""

import json

import numpy as np
import pytest

from src.rag.bm25 import BM25, tokenize
from src.rag.chunk import chunk_sections
from src.rag.embed import FakeEmbedder
from src.rag.index import IndexMismatch, build_index, check_embedder, load_index
from src.rag.parse import Line, Section
from src.rag.retrieve import Retriever, ranks, rrf_fuse

DOCS = [
    "A bank shall file a SAR no later than 30 calendar days after initial detection.",
    "Each financial institution shall file a report of each transaction in currency of more than $10,000.",
    "Records of funds transfers of $3,000 or more must be retained for five years.",
    "The Secretary may issue administrative rulings on request.",
]


def test_tokenize_keeps_section_numbers_and_amounts():
    assert tokenize("See § 1020.320(b)(3) for $10,000 CTRs.") == ["see", "1020.320", "b", "3", "10,000", "ctrs"]


def test_bm25_ranks_exact_terms_first_and_is_zero_without_overlap():
    bm = BM25(DOCS)
    assert int(np.argmax(bm.scores("10,000 currency report"))) == 1
    assert bm.scores("zebra giraffe").tolist() == [0, 0, 0, 0]
    assert all(v >= 0 for v in bm.idf.values())


def test_ranks_ties_and_eligibility():
    assert ranks(np.array([0.5, 0.9, 0.5])) == {1: 1, 0: 2, 2: 3}
    assert ranks(np.array([0.0, 2.0, 1.0]), eligible=np.array([False, True, True])) == {1: 1, 2: 2}


def test_rrf_fuse_math():
    fused = rrf_fuse([{0: 1, 1: 2}, {1: 1}], k=60)
    assert fused[0] == pytest.approx(1 / 61)
    assert fused[1] == pytest.approx(1 / 62 + 1 / 61)


def retriever():
    emb = FakeEmbedder()
    vectors = emb.embed_documents(DOCS).vectors
    return Retriever([f"c{i}" for i in range(len(DOCS))], DOCS, vectors, emb)


def test_hybrid_search_returns_top_k_with_scores():
    result = retriever().search("When must a bank file a SAR after detection?", k=2)
    assert [h.rank for h in result.hits] == [1, 2]
    assert result.hits[0].chunk_id == "c0"
    assert result.hits[0].rrf >= result.hits[1].rrf


def test_refusal_signal_is_max_cosine_not_rrf():
    """Decision C1: with an unrelated query the fused score of the top hit stays high (rank-based),
    while max_cosine (dense similarity of the best chunk) is what can drop below tau."""
    r = retriever()
    off_topic = r.search("zebra giraffe savanna", k=1)
    on_topic = r.search("file a SAR 30 calendar days after detection", k=1)
    assert off_topic.hits[0].rrf >= 1 / 61  # still ranked #1 on dense: RRF cannot tell
    assert off_topic.max_bm25 == 0
    assert off_topic.max_cosine < on_topic.max_cosine
    cos_all = r.embeddings @ r.embedder.embed_query("zebra giraffe savanna")
    assert off_topic.max_cosine == pytest.approx(float(cos_all.max()))


def test_length_mismatch_rejected():
    with pytest.raises(ValueError):
        Retriever(["a"], ["x", "y"], np.zeros((2, 4)), FakeEmbedder())


def sections():
    return [Section("cfr31_1020", "1020.320", "Reports by banks of suspicious transactions", "",
                    [Line(DOCS[0], 10, "1197")]),
            Section("cfr31_1020", "1020.410", "Records to be made and retained by banks", "",
                    [Line(DOCS[2], 12, "1199")])]


def test_index_round_trip_and_tamper_detection(tmp_path):
    chunks = chunk_sections(sections())
    emb = FakeEmbedder()
    manifest = build_index(chunks, emb, tmp_path, {"cfr31_1020": "f" * 64})
    assert manifest["n_chunks"] == 2 and manifest["embedder"]["dim"] == 256
    assert manifest["embedder"]["doc_prefix"] == "search_document: "
    frame, vectors, loaded = load_index(tmp_path, {"cfr31_1020": "f" * 64})
    assert frame.chunk_id.tolist() == [c.chunk_id for c in chunks]
    assert vectors.shape == (2, 256)
    assert {"page_start", "page_end", "section", "heading", "embed_text", "text_sha256"} <= set(frame.columns)
    check_embedder(loaded, FakeEmbedder())
    with pytest.raises(IndexMismatch):
        check_embedder(loaded, FakeEmbedder(dim=128))
    with pytest.raises(IndexMismatch):
        load_index(tmp_path, {"cfr31_1020": "0" * 64})
    np.save(tmp_path / "embeddings.npy", vectors * 2)
    with pytest.raises(IndexMismatch):
        load_index(tmp_path)


def test_index_rejects_changed_chunker(tmp_path):
    build_index(chunk_sections(sections()), FakeEmbedder(), tmp_path, {})
    path = tmp_path / "index_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["chunker"]["overlap_words"] = 10
    path.write_text(json.dumps(manifest))
    with pytest.raises(IndexMismatch):
        load_index(tmp_path)
