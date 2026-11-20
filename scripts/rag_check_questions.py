"""Check data/rag/eval/questions.jsonl against the built index (no model, no retrieval).

    python scripts/rag_check_questions.py

Prints every problem (exit 1 if any), the absence search for each not-answerable
question (term -> sections hit; all must be 0), and for each key fact the other
sections that also contain it (candidates for extra expected sources).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.rag.index import INDEX_DIR, load_index  # noqa: E402
from src.rag.questions import (  # noqa: E402
    LABEL_SAMPLE_SIZE, LABEL_SEED, QUESTIONS, check_questions, labelling_order, load_questions,
    section_texts_from_chunks,
)


def main() -> int:
    frame, _, manifest = load_index(INDEX_DIR)
    records = load_questions(QUESTIONS)
    texts = section_texts_from_chunks(frame)
    problems, report = check_questions(records, texts)

    print(f"index: {manifest['n_chunks']} chunks, {len(texts)} sections; questions: {len(records)}")
    print(f"counts (answerable, not): {report['counts']}")
    print("\nabsence checks (literal, case-insensitive, over all sections):")
    for rid, hits in report["absence"].items():
        print(f"  {rid}: " + ", ".join(f"{t!r}={n}" for t, n in hits.items()))
    print("\nkey facts also found outside the expected sections:")
    for (rid, fact), others in report["facts_also_in"].items():
        if others:
            print(f"  {rid} {fact!r}: {len(others)} -> {others[:6]}{' ...' if len(others) > 6 else ''}")
    test_ids = [r["id"] for r in records if r["split"] == "test" and r["answerable"]]
    order = labelling_order(test_ids)
    print(f"\nhuman claim-labelling order (seed {LABEL_SEED}; sample = first {LABEL_SAMPLE_SIZE} answered): {order}")
    print(f"\nproblems: {len(problems)}")
    for p in problems:
        print("  - " + p)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
