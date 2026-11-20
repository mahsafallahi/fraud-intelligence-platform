"""Chunking: sizes, overlap, section boundaries, pages, ids and citations."""

import pytest

from src.rag.chunk import chunk_section, chunk_sections, citation, split_spans
from src.rag.parse import Line, Section


def make_section(n_words, section="1020.320", doc_id="cfr31_1020", words_per_page=100, sentence_every=10):
    words = [f"w{i}." if (i + 1) % sentence_every == 0 else f"w{i}" for i in range(n_words)]
    lines = [Line(" ".join(words[i:i + 10]), 1 + i // words_per_page, str(1000 + i // words_per_page))
             for i in range(0, n_words, 10)]
    return Section(doc_id, section, "Reports by banks of suspicious transactions", "Subpart C", lines)


def test_short_section_is_one_chunk():
    chunks = chunk_section(make_section(450))
    assert len(chunks) == 1 and chunks[0].n_words == 450


def test_long_section_cuts_at_sentence_ends_with_overlap():
    chunks = chunk_section(make_section(1000))
    assert len(chunks) > 1
    for c in chunks[:-1]:
        assert c.text.endswith(".") and 280 <= c.n_words <= 350
    for a, b in zip(chunks, chunks[1:]):
        assert a.text.split()[-50:] == b.text.split()[:50]
    covered = {w for c in chunks for w in c.text.split()}
    assert len(covered) == 1000


def test_cut_without_sentence_end_falls_back_to_target():
    spans = split_spans([False] * 1000)
    assert spans[0] == (0, 350) and spans[1][0] == 300


def test_bad_parameters_rejected():
    with pytest.raises(ValueError):
        split_spans([False] * 10, overlap=300)


def test_pages_ids_and_hash():
    chunks = chunk_section(make_section(1000, words_per_page=100))
    first = chunks[0]
    assert first.chunk_id == "cfr31_1020:1020.320:001" and chunks[1].chunk_id.endswith(":002")
    assert (first.page_start, first.label_start) == (1, "1000")
    assert first.page_end == 1 + (first.n_words - 1) // 100
    assert len(first.text_sha256) == 64
    assert first.embed_text.startswith("31 CFR § 1020.320 Reports by banks of suspicious transactions\n")


def test_chunks_never_cross_sections_and_ids_are_unique():
    chunks = chunk_sections([make_section(500, "1020.315"), make_section(500, "1020.320")])
    assert {c.section for c in chunks} == {"1020.315", "1020.320"}
    assert all(c.chunk_id.split(":")[1] == c.section for c in chunks)
    with pytest.raises(ValueError):
        chunk_sections([make_section(20), make_section(20)])  # same section twice


def test_citation_is_rendered_from_metadata():
    rec = chunk_section(make_section(150))[0].to_record()
    assert citation(rec) == ("31 CFR § 1020.320 (Reports by banks of suspicious transactions), "
                             "PDF p. 1–2, printed p. 1000–1001")
    faq = chunk_section(Section("fincen_sar_faq_2025", "Q3", "Question 3: Timeline", "",
                                [Line("one two three.", 3, "3")]))[0].to_record()
    assert citation(faq) == "FinCEN SAR FAQs (Oct. 9, 2025), Question 3: Timeline, PDF p. 3, printed p. 3"
