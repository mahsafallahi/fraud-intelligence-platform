"""Answer questions with the local LLM and validated citations (Phase 7).

    python scripts/rag_answer.py "question" ["question" ...]
    python scripts/rag_answer.py --seen          # the 5 seen sanity queries (dev only)

Prints each answer with rendered citations, the validator result and the time per
answer, and writes a run record (model digest, options, prompt hashes, index
manifest, results) to data/rag/runs/<UTC timestamp>.json (git-ignored).
tau is left unset until it is chosen on the dev set (decision C1).
"""

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.rag.answer import Answerer, prompt_hashes  # noqa: E402
from src.rag.corpus import PROJECT_ROOT, load_manifest  # noqa: E402
from src.rag.embed import OllamaEmbedder  # noqa: E402
from src.rag.index import INDEX_DIR, check_embedder, load_index  # noqa: E402
from src.rag.llm import OllamaChat  # noqa: E402
from src.rag.questions import SEEN_QUERIES  # noqa: E402
from src.rag.retrieve import Retriever  # noqa: E402

RUNS_DIR = PROJECT_ROOT / "data" / "rag" / "runs"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("questions", nargs="*")
    parser.add_argument("--seen", action="store_true", help="the 5 seen sanity queries")
    parser.add_argument("--cpu", action="store_true", help="force CPU (num_gpu=0) for timing")
    args = parser.parse_args(argv)
    questions = list(args.questions) + (list(SEEN_QUERIES) if args.seen else [])
    if not questions:
        parser.error("no questions")

    hashes = {d["doc_id"]: d["sha256"] for d in load_manifest()["documents"]}
    frame, embeddings, manifest = load_index(INDEX_DIR, hashes)
    embedder = OllamaEmbedder()
    check_embedder(manifest, embedder)
    retriever = Retriever(frame.chunk_id.tolist(), frame.embed_text.tolist(), embeddings, embedder)
    records = frame.set_index("chunk_id", drop=False).to_dict("index")
    llm = OllamaChat(options={"num_gpu": 0} if args.cpu else None)
    answerer = Answerer(retriever, llm, records)

    run = {"started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "llm": llm.describe(), "embedder": embedder.describe(), "prompts": prompt_hashes(),
           "index": {k: manifest[k] for k in ("created_utc", "corpus", "chunker", "n_chunks", "files")},
           "tau": answerer.tau, "results": []}
    for q in questions:
        out = answerer.answer(q)
        print(f"\nQ: {q}\n   status={out.status}{' (' + out.reason + ')' if out.reason else ''}  "
              f"attempts={out.attempts}  seconds={out.seconds:.1f}  llm={[round(s, 1) for s in out.llm_seconds]}  "
              f"max_cosine={out.max_cosine:.4f}")
        for n, v in enumerate(out.validations, start=1):
            print(f"   validation {n}: {'PASS' if v.ok else 'FAIL'} {v.errors}")
        if out.status == "answered":
            ok = sum(c["ok"] for c in out.claim_checks)
            print(f"   mechanical number check: {ok}/{len(out.claim_checks)} sentences pass"
                  + "".join(f"\n      missing {c['missing']} in: {c['sentence'][:90]!r}"
                            for c in out.claim_checks if not c["ok"]))
        print("   " + out.rendered.replace("\n", "\n   "))
        rec = asdict(out)
        rec["validations"] = [{"ok": v.ok, "errors": v.errors, "cited": v.cited} for v in out.validations]
        run["results"].append(rec)

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    path = RUNS_DIR / f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    path.write_text(json.dumps(run, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nrun record: {path.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
