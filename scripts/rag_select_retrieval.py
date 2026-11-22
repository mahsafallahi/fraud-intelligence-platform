"""Round-4 retrieval selection on the DEV split only (docs/rag_design.md §6a). No LLM.

    python scripts/rag_select_retrieval.py

For each candidate: recall@6 on the 8 answerable dev questions and the dense max_cosine
of all 12 dev questions. Picks the best recall (ties -> simplest, in the doc's order),
then sets tau by the current rule (midpoint if dev separates, else disabled).
Writes data/rag/runs/selection_<UTC>.json. Never reads the test split.
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.rag.corpus import PROJECT_ROOT, load_manifest  # noqa: E402
from src.rag.embed import OllamaEmbedder  # noqa: E402
from src.rag.glossary import expand_query  # noqa: E402
from src.rag.index import INDEX_DIR, check_embedder, load_index  # noqa: E402
from src.rag.questions import QUESTIONS, load_questions  # noqa: E402
from src.rag.retrieve import TOP_K, Retriever  # noqa: E402

RUNS_DIR = PROJECT_ROOT / "data" / "rag" / "runs"
PARAGRAPH_INDEX = INDEX_DIR / "variants" / "paragraphs"
# (name, mode, glossary, paragraph index) in the doc's order: simplest first (ties go to the earlier one)
CANDIDATES = [
    ("bm25", "bm25", False, False),
    ("dense", "dense", False, False),
    ("hybrid", "hybrid", False, False),
    ("hybrid+glossary", "hybrid", True, False),
    ("hybrid+paragraphs", "hybrid", False, True),
]


def choose_tau(answerable: list[float], unanswerable: list[float]) -> float | None:
    """Midpoint if every answerable max_cosine is above every unanswerable one; otherwise None."""
    if answerable and unanswerable and min(answerable) > max(unanswerable):
        return round((min(answerable) + max(unanswerable)) / 2, 4)
    return None


def main() -> int:
    hashes = {d["doc_id"]: d["sha256"] for d in load_manifest()["documents"]}
    dev = [q for q in load_questions(QUESTIONS) if q["split"] == "dev"]
    assert len(dev) == 12, "dev split must have 12 questions"
    embedder = OllamaEmbedder()
    indexes = {}
    for para in (False, True):
        frame, emb, manifest = load_index(PARAGRAPH_INDEX if para else INDEX_DIR, hashes, paragraphs=para)
        check_embedder(manifest, embedder)
        indexes[para] = (frame, emb)

    results = []
    for name, mode, gloss, para in CANDIDATES:
        frame, emb = indexes[para]
        section_of = dict(zip(frame.chunk_id, frame.doc_id + ":" + frame.section))
        r = Retriever(frame.chunk_id.tolist(), frame.embed_text.tolist(), emb, embedder, mode=mode,
                      expand=expand_query if gloss else None)
        per_q = []
        for q in dev:
            res = r.search(q["question"], k=TOP_K)
            got = [section_of[h.chunk_id] for h in res.hits]
            expected = {f"{d}:{s}" for d, s in q["expected_sources"]}
            per_q.append({"id": q["id"], "answerable": q["answerable"], "max_cosine": round(res.max_cosine, 4),
                          "hit": bool(expected & set(got)) if q["answerable"] else None,
                          "top_sections": got})
        recall = sum(p["hit"] for p in per_q if p["answerable"])
        results.append({"name": name, "recall": recall, "per_question": per_q})
        misses = [p["id"] for p in per_q if p["answerable"] and not p["hit"]]
        print(f"{name:18s} recall@6 = {recall}/8   misses: {misses}")

    best = max(r["recall"] for r in results)
    chosen = next(r for r in results if r["recall"] == best)      # first = simplest among ties
    ans = [p["max_cosine"] for p in chosen["per_question"] if p["answerable"]]
    una = [p["max_cosine"] for p in chosen["per_question"] if not p["answerable"]]
    tau = choose_tau(ans, una)
    print(f"\nchosen: {chosen['name']} (recall {best}/8; ties go to the simplest)")
    print("dev max_cosine, answerable:   " + ", ".join(f"{p['id']}={p['max_cosine']:.4f}"
                                                       for p in chosen["per_question"] if p["answerable"]))
    print("dev max_cosine, unanswerable: " + ", ".join(f"{p['id']}={p['max_cosine']:.4f}"
                                                       for p in chosen["per_question"] if not p["answerable"]))
    print(f"min answerable = {min(ans):.4f}, max unanswerable = {max(una):.4f} -> tau = {tau}"
          + ("" if tau is not None else " (disabled: dev does not separate)"))

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    out = RUNS_DIR / f"selection_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    out.write_text(json.dumps({"candidates": results, "chosen": chosen["name"], "tau": tau,
                               "embedder": embedder.describe()}, indent=2) + "\n", encoding="utf-8")
    print(f"record: {out.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
