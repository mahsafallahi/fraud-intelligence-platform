"""Round-4 retrieval candidates, glossary, paragraph splitting and the tau rule (no network)."""

import pytest

from src.rag.chunk import chunk_sections, chunker_params, paragraph_groups
from src.rag.config import RETRIEVAL, TAU
from src.rag.embed import FakeEmbedder
from src.rag.glossary import GLOSSARY, expand_query
from src.rag.parse import Line, Section
from src.rag.retrieve import Retriever
from tests.conftest import load_script

DOCS = [
    "A bank shall file a SAR no later than 30 calendar days after initial detection.",
    "Each financial institution shall file a report of each transaction in currency of more than $10,000.",
    "Records of funds transfers of $3,000 or more must be retained for five years.",
]


def retriever(mode, expand=None):
    emb = FakeEmbedder()
    return Retriever([f"c{i}" for i in range(3)], DOCS, emb.embed_documents(DOCS).vectors, emb,
                     mode=mode, expand=expand)


def test_modes_use_only_their_rankings():
    q = "currency transaction 10,000"
    bm25 = retriever("bm25").search(q, k=3)
    assert all(h.bm25_rank is not None for h in bm25.hits)          # only chunks sharing a term
    dense = retriever("dense").search(q, k=3)
    assert [h.chunk_id for h in dense.hits] == [h.chunk_id for h in sorted(dense.hits, key=lambda h: h.dense_rank)]
    assert dense.max_cosine == retriever("hybrid").search(q, k=3).max_cosine
    with pytest.raises(ValueError):
        retriever("sparse")


def test_glossary_expands_whole_terms_only():
    assert expand_query("What is the CTR threshold?") == (
        "What is the CTR threshold? (currency transaction report, report of transactions in currency)")
    assert expand_query("A cashier's check") == "A cashier's check"          # "cash" is a whole word only
    assert expand_query("314(b) and wire transfer").count("transmittal of funds") == 1
    assert "travel rule" in GLOSSARY and len(GLOSSARY) == 20
    r = retriever("hybrid", expand=expand_query)
    assert r.search("cash", k=1).query == "cash"


def lines(*texts):
    return [Line(t, 1, "1") for t in texts]


def test_paragraph_groups_follow_the_letter_sequence():
    groups = paragraph_groups(lines("Intro.", "(a) First.", "(1) sub.", "(b) Second.", "(i) roman, not (i) paragraph.",
                                    "(c) Third."))
    assert [[ln.text for ln in g] for g in groups] == [
        ["Intro."], ["(a) First.", "(1) sub."], ["(b) Second.", "(i) roman, not (i) paragraph."], ["(c) Third."]]


def test_paragraph_chunking_only_for_cfr_and_recorded_in_params():
    cfr = Section("cfr31_1020", "1020.320", "Reports", "", lines("(a) One two three.", "(b) Four five six."))
    faq = Section("fincen_sar_faq_2025", "Q1", "Question 1", "", lines("(a) One.", "(b) Two."))
    chunks = chunk_sections([cfr, faq], paragraphs=True)
    assert [c.chunk_id for c in chunks] == ["cfr31_1020:1020.320:001", "cfr31_1020:1020.320:002",
                                            "fincen_sar_faq_2025:Q1:001"]
    assert chunker_params(True)["paragraphs"] is True and "paragraphs" not in chunker_params()


def test_tau_rule_midpoint_or_disabled():
    choose_tau = load_script("rag_select_retrieval").choose_tau
    assert choose_tau([0.80, 0.85], [0.70, 0.75]) == 0.775
    assert choose_tau([0.66, 0.85], [0.70, 0.72]) is None


def test_chosen_config_matches_the_dev_selection():
    assert RETRIEVAL == {"mode": "dense", "glossary": False, "paragraphs": False, "k": 6, "rrf_k": 60}
    assert TAU is None


def test_raw_citation_counter():
    raw = load_script("rag_eval").raw_citations
    assert raw("A [1]. B [2, 7]. C [C:x:y:001].", 6) == (2, 4)
