"""Mechanical checks of the question set (fixtures), and the committed questions.jsonl."""

import pytest

from src.rag.questions import (
    QUESTIONS, SEEN_QUERIES, check_questions, corpus_hits, labelling_order, load_questions,
)

TEXTS = {
    "cfr31_1020:1020.320": "31 CFR § 1020.320 Reports\nA bank shall file a SAR no later than 30 calendar days after detection.",
    "ffiec_appf:employees": "FFIEC ... Employees\n• Employee is reluctant to take a vacation.",
}


def ans(rid, split="dev", q="When is a SAR due?", **kw):
    r = {"id": rid, "split": split, "question": q, "answerable": True,
         "expected_sources": [["cfr31_1020", "1020.320"]], "key_facts": ["30 calendar days"],
         "author": "claude", "wording": "source_terms"}
    r.update(kw)
    return r


def unans(rid, split="dev", terms=("GDPR", "personal data breach", "72 hours")):
    return {"id": rid, "split": split, "question": "GDPR breach deadline?", "answerable": False,
            "expected_sources": [], "key_facts": [], "author": "claude", "absence_terms": list(terms)}


def full_set():
    recs = [ans(f"D{i:02d}", wording="paraphrase", q="When is the report due?", avoided_terms=["SAR"])
            for i in range(1, 9)]
    recs += [unans(f"D{i:02d}") for i in range(9, 13)]
    recs += [ans(f"T{i:02d}", split="test") for i in range(1, 23)]
    recs += [unans(f"T{i:02d}", split="test") for i in range(23, 31)]
    for r in recs[12:19]:          # 8 dev + 7 test paraphrased = 15 of 30
        r.update(wording="paraphrase", avoided_terms=["SAR"], question="When is the report due?")
    return recs


def problems_of(recs):
    return check_questions(recs, TEXTS)[0]


def test_valid_set_has_no_problems_and_reports_absence_counts():
    problems, report = check_questions(full_set(), TEXTS)
    assert problems == []
    assert report["counts"]["dev"] == (8, 4) and report["counts"]["test"] == (22, 8)
    assert report["absence"]["D09"] == {"GDPR": 0, "personal data breach": 0, "72 hours": 0}


def test_wrong_split_counts_and_duplicate_ids():
    recs = full_set()
    recs[-1] = ans("T01", split="test")
    problems = problems_of(recs)
    assert "duplicate id T01" in problems
    assert any(p.startswith("test: 23 answerable") for p in problems)


def test_seen_query_in_test_split_is_rejected():
    recs = full_set()
    recs[20]["question"] = SEEN_QUERIES[0]
    assert any("seen sanity query" in p for p in problems_of(recs))


def test_missing_section_and_missing_fact():
    recs = full_set()
    recs[20]["expected_sources"] = [["cfr31_1020", "1020.999"]]
    recs[21]["key_facts"] = ["45 calendar days"]
    problems = problems_of(recs)
    assert any("1020.999 is not in the index" in p for p in problems)
    assert any("'45 calendar days' not found" in p for p in problems)


def test_every_expected_section_must_contain_a_key_fact():
    recs = full_set()
    recs[20]["expected_sources"] = [["cfr31_1020", "1020.320"], ["ffiec_appf", "employees"]]
    assert "T09: expected section ffiec_appf:employees contains none of the key facts" in problems_of(recs)


def test_paraphrase_must_avoid_its_terms_and_half_must_be_paraphrased():
    recs = full_set()
    recs[0]["question"] = "When is the SAR due?"
    assert any("uses avoided term 'SAR'" in p for p in problems_of(recs))
    recs = full_set()
    for r in recs:
        if r["answerable"]:
            r["wording"] = "source_terms"
    assert any("are paraphrased (need half)" in p for p in problems_of(recs))


def test_absence_term_present_in_corpus_is_a_problem():
    recs = full_set()
    recs[8]["absence_terms"] = ["vacation", "GDPR", "72 hours"]
    assert any("absence term 'vacation' found in 1" in p for p in problems_of(recs))


def test_corpus_hits_is_literal_and_case_insensitive():
    assert corpus_hits("RELUCTANT to take", list(TEXTS.values())) == 1
    assert corpus_hits("reluct.nt", list(TEXTS.values())) == 0


def test_labelling_order_is_a_fixed_permutation():
    ids = [f"T{i:02d}" for i in range(1, 23)]
    order = labelling_order(ids)
    assert sorted(order) == ids and order == labelling_order(list(reversed(ids)))
    assert order != ids


def test_committed_question_file_structure():
    recs = load_questions(QUESTIONS)
    assert len(recs) == 42 and len({r["id"] for r in recs}) == 42
    seen = {q.casefold() for q in SEEN_QUERIES}
    assert not [r["id"] for r in recs if r["split"] == "test" and r["question"].casefold() in seen]
    assert {r["question"] for r in recs if r.get("seen")} == set(SEEN_QUERIES)
    assert all(r["author"] == "claude" for r in recs)  # option B: Claude drafts all, the user reviews labels
    assert labelling_order([r["id"] for r in recs if r["split"] == "test" and r["answerable"]])[:10] == [
        "T05", "T02", "T07", "T15", "T08", "T20", "T04", "T22", "T10", "T13"]


@pytest.mark.corpus
def test_committed_questions_pass_all_checks_on_the_real_corpus():
    from src.rag.chunk import chunk_sections
    from src.rag.corpus import RAW_DIR, load_manifest
    from src.rag.parse import load_document
    texts: dict[str, list[str]] = {}
    for d in load_manifest()["documents"]:
        for c in chunk_sections(load_document(d["doc_id"], RAW_DIR / d["filename"])[0]):
            texts.setdefault(f"{c.doc_id}:{c.section}", []).append(c.embed_text)
    problems, report = check_questions(load_questions(QUESTIONS), {k: "\n".join(v) for k, v in texts.items()})
    assert problems == []
    assert all(n == 0 for hits in report["absence"].values() for n in hits.values())
    assert report["counts"]["paraphrase"][0] * 2 >= report["counts"]["paraphrase"][1]
