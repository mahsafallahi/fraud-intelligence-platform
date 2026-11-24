"""The ONE-TIME test run of the frozen RAG configuration (Phase 7, approved 2026-11-24).

    python scripts/rag_test_run.py --check-only     # verify the frozen fingerprint, run nothing
    python scripts/rag_test_run.py                  # verify, then run the 30 test questions once

Stops (exit 1) before any question if:
- any fingerprint field differs from data/rag/eval/preregistration.json (Ollama version,
  model and embedder digests, options, prompt hashes, index manifest, questions, retrieval),
  or tau or the agent prompt hashes differ;
- src/rag differs from the freeze commit (code that could change an answer);
- the working tree has uncommitted changes to tracked files;
- a test run already exists: the output file is created exclusively and never overwritten.
Each record is written and flushed as soon as its question is answered, so a crash or a
scoring bug never needs a second run. There is no scoring here: see scripts/rag_test_report.py.
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.agent.writer import prompt_hashes as agent_prompt_hashes  # noqa: E402
from src.rag.answer import Answerer  # noqa: E402
from src.rag.config import RETRIEVAL, TAU, build_retriever  # noqa: E402
from src.rag.corpus import PROJECT_ROOT, load_manifest  # noqa: E402
from src.rag.embed import OllamaEmbedder  # noqa: E402
from src.rag.index import INDEX_DIR, check_embedder, load_index  # noqa: E402
from src.rag.llm import OllamaChat  # noqa: E402
from src.rag.questions import QUESTIONS, load_questions  # noqa: E402
from src.rag.runmeta import fingerprint, fingerprint_diff, ollama_version  # noqa: E402

PREREG = PROJECT_ROOT / "data" / "rag" / "eval" / "preregistration.json"
OUT = PROJECT_ROOT / "data" / "rag" / "runs" / "test_run.jsonl"
FREEZE_COMMIT = "09e866b"


def git(*args) -> str:
    return subprocess.run(["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True).stdout.strip()


def differences(frozen: dict, current: dict, tau, agent_prompts: dict, rag_diff: str, dirty: str) -> list[str]:
    """Everything that is not exactly as frozen (empty list: the run may start)."""
    out = [f"{k}: frozen {a!r}, now {b!r}" for k, (a, b) in fingerprint_diff(frozen["fingerprint"], current).items()]
    if frozen["tau"] != tau:
        out.append(f"tau: frozen {frozen['tau']!r}, now {tau!r}")
    if frozen["agent_prompts"] != agent_prompts:
        out.append("agent prompt hashes differ")
    if rag_diff:
        out.append(f"src/rag differs from the freeze commit: {rag_diff}")
    if dirty:
        out.append(f"uncommitted changes to tracked files: {dirty}")
    return out


def answer_record(q: dict, out) -> dict:
    """One raw run record; nothing is scored here."""
    return {"id": q["id"], "question": q["question"], "answerable": q["answerable"],
            "expected_sources": q["expected_sources"], "status": out.status, "reason": out.reason,
            "text": out.text, "rendered": out.rendered, "cited": out.cited, "retrieved": out.retrieved,
            "max_cosine": out.max_cosine, "attempts": out.attempts, "replies": out.replies,
            "validations": [{"ok": v.ok, "errors": v.errors, "cited": v.cited, "sentences": v.sentences}
                            for v in out.validations],
            "claim_checks": out.claim_checks, "seconds": round(out.seconds, 2)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)
    frozen = json.loads(PREREG.read_text(encoding="utf-8"))
    hashes = {d["doc_id"]: d["sha256"] for d in load_manifest()["documents"]}
    frame, embeddings, manifest = load_index(INDEX_DIR, hashes, paragraphs=RETRIEVAL["paragraphs"])
    embedder = OllamaEmbedder()
    check_embedder(manifest, embedder)
    llm = OllamaChat(model=frozen["fingerprint"]["llm_model"])
    current = fingerprint(llm.describe(), embedder.describe(), ollama_version(requests.Session()), RETRIEVAL)
    diffs = differences(frozen, current, TAU, agent_prompt_hashes(),
                        git("diff", "--name-only", FREEZE_COMMIT, "--", "src/rag", "data/rag/eval/questions.jsonl",
                            "data/rag/eval/preregistration.json"),
                        git("status", "--porcelain", "--untracked-files=no"))
    for d in diffs:
        print(f"DIFFERENT {d}", file=sys.stderr)
    if diffs:
        print("STOP: not the frozen configuration; nothing was run", file=sys.stderr)
        return 1
    test = [q for q in load_questions(QUESTIONS) if q["split"] == "test"]
    if (len(test), sum(q["answerable"] for q in test)) != (30, 22):
        print("STOP: the test split is not 22 + 8 questions", file=sys.stderr)
        return 1
    print(f"frozen fingerprint verified: {current['llm_model']} {current['llm_digest'][:12]}, "
          f"Ollama {current['ollama_version']}, all {len(current)} fields equal, src/rag unchanged since {FREEZE_COMMIT}")
    if OUT.exists():
        print(f"STOP: {OUT.relative_to(PROJECT_ROOT)} exists; the test run is done once only", file=sys.stderr)
        return 1
    if args.check_only:
        print("check only: no question was run")
        return 0

    records = frame.set_index("chunk_id", drop=False).to_dict("index")
    answerer = Answerer(build_retriever(frame, embeddings, embedder), llm, records, tau=TAU, k=RETRIEVAL["k"])
    with OUT.open("x", encoding="utf-8", newline="\n") as f:        # "x": never overwrite
        header = {"fingerprint": current, "tau": TAU, "commit": git("rev-parse", "HEAD"),
                  "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        f.write(json.dumps(header) + "\n")
        for q in test:
            rec = answer_record(q, answerer.answer(q["question"]))
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            print(f"{q['id']} {'ans' if q['answerable'] else 'NOT'} {rec['status']:8s} {rec['reason']:17s} "
                  f"attempts={rec['attempts']} {rec['seconds']}s", flush=True)
        f.write(json.dumps({"finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}) + "\n")
    print(f"record: {OUT.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
