"""Evaluate the full answering pipeline on the DEV split (Phase 7, round 4).

    python scripts/rag_eval.py            # dev only; the test split is refused here

Per question: retrieved sections vs expected, answered/refused, validator attempts,
mechanical number check, time. Totals: recall@6, raw citation validity, mechanical
pass rate, correct and false refusals, and rule C4. The run record (with the full
fingerprint: Ollama version, digests, prompt hashes, index manifest and questions
SHA-256, retrieval config) goes to data/rag/runs/eval_dev_<UTC>.json.
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.rag.answer import Answerer  # noqa: E402
from src.rag.citations import CITE_LIKE, NUM_CITE, numbers_in  # noqa: E402
from src.rag.config import RETRIEVAL, TAU, build_retriever  # noqa: E402
from src.rag.corpus import PROJECT_ROOT, load_manifest  # noqa: E402
from src.rag.embed import OllamaEmbedder  # noqa: E402
from src.rag.index import INDEX_DIR, check_embedder, load_index  # noqa: E402
from src.rag.llm import OllamaChat  # noqa: E402
from src.rag.questions import QUESTIONS, load_questions  # noqa: E402
from src.rag.runmeta import fingerprint, ollama_version  # noqa: E402

RUNS_DIR = PROJECT_ROOT / "data" / "rag" / "runs"


def raw_citations(reply: str, k: int) -> tuple[int, int]:
    """(valid, total) citations in one model reply, before the validator (C4 'raw citation validity')."""
    valid = total = 0
    for raw in CITE_LIKE.findall(reply):
        if not NUM_CITE.fullmatch(raw):
            total += 1
            continue
        for n in numbers_in(raw):
            total += 1
            valid += 1 <= n <= k
    return valid, total


def main() -> int:
    if "--split" in sys.argv and "test" in sys.argv:
        print("STOP: the test run is not approved; this script evaluates dev only", file=sys.stderr)
        return 2
    hashes = {d["doc_id"]: d["sha256"] for d in load_manifest()["documents"]}
    frame, embeddings, manifest = load_index(INDEX_DIR, hashes, paragraphs=RETRIEVAL["paragraphs"])
    embedder = OllamaEmbedder()
    check_embedder(manifest, embedder)
    records = frame.set_index("chunk_id", drop=False).to_dict("index")
    section_of = {cid: f"{r['doc_id']}:{r['section']}" for cid, r in records.items()}
    llm = OllamaChat()
    answerer = Answerer(build_retriever(frame, embeddings, embedder), llm, records, tau=TAU, k=RETRIEVAL["k"])
    fp = fingerprint(llm.describe(), embedder.describe(), ollama_version(requests.Session()), RETRIEVAL)

    dev = [q for q in load_questions(QUESTIONS) if q["split"] == "dev"]
    rows, raw_valid, raw_total = [], 0, 0
    for q in dev:
        out = answerer.answer(q["question"])
        expected = {f"{d}:{s}" for d, s in q["expected_sources"]}
        got = [section_of[c] for c in out.retrieved]
        for reply in out.replies:
            if "NO_ANSWER" not in reply.upper().replace(" ", "_"):
                v, t = raw_citations(reply, RETRIEVAL["k"])
                raw_valid, raw_total = raw_valid + v, raw_total + t
        cited_hit = bool(expected & {section_of[c] for c in out.cited}) if out.status == "answered" else None
        row = {"id": q["id"], "answerable": q["answerable"], "status": out.status, "reason": out.reason,
               "attempts": out.attempts, "retrieval_hit": bool(expected & set(got)) if q["answerable"] else None,
               "cited_expected": cited_hit, "retrieved_sections": got, "expected": sorted(expected),
               "claim_checks": out.claim_checks, "seconds": round(out.seconds, 1),
               "validator": [{"ok": v.ok, "errors": v.errors} for v in out.validations],
               "answer": out.rendered, "replies": out.replies}
        rows.append(row)
        mech = f"{sum(c['ok'] for c in out.claim_checks)}/{len(out.claim_checks)}" if out.status == "answered" else "-"
        print(f"{q['id']} {'ans' if q['answerable'] else 'NOT'} hit={row['retrieval_hit']!s:5} "
              f"{out.status:8s} {out.reason:17s} attempts={out.attempts} "
              f"validator={['PASS' if v.ok else 'FAIL' for v in out.validations]} mech={mech} "
              f"cited_expected={cited_hit} {row['seconds']}s")

    ans = [r for r in rows if r["answerable"]]
    una = [r for r in rows if not r["answerable"]]
    checks = [c for r in rows if r["status"] == "answered" for c in r["claim_checks"]]
    mech_rate = sum(c["ok"] for c in checks) / len(checks) if checks else None
    raw_rate = raw_valid / raw_total if raw_total else None
    totals = {
        "recall_at_6": f"{sum(r['retrieval_hit'] for r in ans)}/{len(ans)}",
        "answered_answerable": f"{sum(r['status'] == 'answered' for r in ans)}/{len(ans)}",
        "false_refusals": f"{sum(r['status'] == 'refused' for r in ans)}/{len(ans)}",
        "correct_refusals": f"{sum(r['status'] == 'refused' for r in una)}/{len(una)}",
        "citation_correctness": f"{sum(bool(r['cited_expected']) for r in ans if r['status'] == 'answered')}/"
                                f"{sum(r['status'] == 'answered' for r in ans)}",
        "raw_citation_validity": f"{raw_valid}/{raw_total}" + (f" = {raw_rate:.3f}" if raw_rate is not None else ""),
        "mechanical_check": f"{sum(c['ok'] for c in checks)}/{len(checks)}"
                            + (f" = {mech_rate:.3f}" if mech_rate is not None else ""),
    }
    c4_rag = {"raw_citation_validity_ok": raw_rate is not None and raw_rate >= 0.95,
              "mechanical_check_ok": mech_rate is not None and mech_rate >= 0.95}
    print("\n" + "\n".join(f"{k}: {v}" for k, v in totals.items()))
    print(f"C4 (RAG part): {c4_rag} -> "
          + ("llama3.1 stays (agent part of C4 is checked in part B)" if all(c4_rag.values())
             else "C4 TRIGGERED: qwen2.5 would be needed; STOP and ask before any pull"))

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    path = RUNS_DIR / f"eval_dev_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    path.write_text(json.dumps({"fingerprint": fp, "tau": TAU, "totals": totals, "c4_rag": c4_rag,
                                "rows": rows}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"record: {path.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
