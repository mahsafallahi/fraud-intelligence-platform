"""Mechanical checks of the hand-labelled question set (Phase 7, data/rag/eval/questions.jsonl).

Each record: id, split (dev|test), question, answerable, expected_sources
([[doc_id, section], ...]), key_facts (literal strings), author, wording
("paraphrase" | "source_terms"), avoided_terms (paraphrase only), absence_terms
(not-answerable only). The checks need no model: they only search the corpus text.
"""

import json
from collections import Counter
from pathlib import Path

from src.rag.corpus import PROJECT_ROOT

QUESTIONS = PROJECT_ROOT / "data" / "rag" / "eval" / "questions.jsonl"
SPLITS = {"dev": (8, 4), "test": (22, 8)}     # (answerable, not answerable), decision C5

# The five round-1 sanity queries have been seen: dev only, never in the test split.
SEEN_QUERIES = (
    "What is the deadline for a bank to file a suspicious activity report?",
    "What is the currency transaction report threshold for banks?",
    "What does the US travel rule require for funds transfers of $3,000 or more?",
    "What is the GDPR time limit to report a personal data breach to the supervisory authority?",
    "How long must a bank keep a copy of a SAR and its supporting documentation?",
)


def load_questions(path: Path = QUESTIONS) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _norm(text: str) -> str:
    text = text.replace("’", "'").replace("“", '"').replace("”", '"')
    return " ".join(text.split()).casefold()


def corpus_hits(term: str, texts: list[str]) -> int:
    """Number of texts containing `term` (case-insensitive, whitespace-normalised, literal)."""
    t = _norm(term)
    return sum(t in _norm(x) for x in texts)


def check_questions(records: list[dict], section_texts: dict[str, str]) -> tuple[list[str], dict]:
    """(problems, report). `section_texts` maps "doc_id:section" -> full section text (with heading)."""
    problems, report = [], {"facts_also_in": {}, "absence": {}, "counts": {}}
    all_texts = list(section_texts.values())

    ids = [r["id"] for r in records]
    for dup, n in Counter(ids).items():
        if n > 1:
            problems.append(f"duplicate id {dup}")
    for split, (n_ans, n_not) in SPLITS.items():
        rs = [r for r in records if r["split"] == split]
        got = (sum(r["answerable"] for r in rs), sum(not r["answerable"] for r in rs))
        report["counts"][split] = got
        if got != (n_ans, n_not):
            problems.append(f"{split}: {got[0]} answerable + {got[1]} not, expected {n_ans} + {n_not}")
    seen = {_norm(q) for q in SEEN_QUERIES}
    for r in records:
        rid = r["id"]
        if r["split"] not in SPLITS:
            problems.append(f"{rid}: unknown split {r['split']!r}")
        if r["split"] == "test" and _norm(r["question"]) in seen:
            problems.append(f"{rid}: a seen sanity query is in the test split")
        if r.get("author") not in ("claude", "user"):
            problems.append(f"{rid}: author must be 'claude' or 'user'")
        if r["answerable"]:
            keys = [f"{d}:{s}" for d, s in r["expected_sources"]]
            if not keys:
                problems.append(f"{rid}: answerable without expected_sources")
            for k in keys:
                if k not in section_texts:
                    problems.append(f"{rid}: expected section {k} is not in the index")
            if not r.get("key_facts"):
                problems.append(f"{rid}: no key_facts")
            for fact in r.get("key_facts", []):
                where = [k for k in keys if k in section_texts and corpus_hits(fact, [section_texts[k]])]
                if not where:
                    problems.append(f"{rid}: key fact {fact!r} not found in any expected section")
                others = sorted(k for k, t in section_texts.items()
                                if k not in keys and corpus_hits(fact, [t]))
                report["facts_also_in"][(rid, fact)] = others
            if r.get("wording") not in ("paraphrase", "source_terms"):
                problems.append(f"{rid}: wording must be 'paraphrase' or 'source_terms'")
            if r.get("wording") == "paraphrase":
                if not r.get("avoided_terms"):
                    problems.append(f"{rid}: paraphrase without avoided_terms")
                for term in r.get("avoided_terms", []):
                    if _norm(term) in _norm(r["question"]):
                        problems.append(f"{rid}: paraphrase uses avoided term {term!r}")
        else:
            if r.get("expected_sources"):
                problems.append(f"{rid}: not-answerable question has expected_sources")
            terms = r.get("absence_terms", [])
            if len(terms) < 3:
                problems.append(f"{rid}: needs at least 3 absence_terms")
            hits = {t: corpus_hits(t, all_texts) for t in terms}
            report["absence"][rid] = hits
            for t, n in hits.items():
                if n:
                    problems.append(f"{rid}: absence term {t!r} found in {n} section(s)")
    answerable = [r for r in records if r["answerable"]]
    para = sum(r.get("wording") == "paraphrase" for r in answerable)
    report["counts"]["paraphrase"] = (para, len(answerable))
    if answerable and para * 2 < len(answerable):
        problems.append(f"only {para}/{len(answerable)} answerable questions are paraphrased (need half)")
    return problems, report


def section_texts_from_chunks(frame) -> dict[str, str]:
    """Join a section's chunks (heading + text; overlaps repeat, which is harmless for searching)."""
    out: dict[str, list[str]] = {}
    for row in frame.itertuples():
        out.setdefault(f"{row.doc_id}:{row.section}", []).append(row.embed_text)
    return {k: "\n".join(v) for k, v in out.items()}


LABEL_SEED = 20261121
LABEL_SAMPLE_SIZE = 10


def labelling_order(answerable_test_ids: list[str], seed: int = LABEL_SEED) -> list[str]:
    """Option B: a fixed-seed random ORDER of the answerable test questions, drawn before the test run.

    The human claim-labelling sample is the first LABEL_SAMPLE_SIZE questions in this
    order that the system actually answers; a refused one is replaced by the next.
    """
    import random
    order = sorted(answerable_test_ids)
    random.Random(seed).shuffle(order)
    return order

