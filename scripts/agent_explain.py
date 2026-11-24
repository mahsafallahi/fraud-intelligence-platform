"""Run the deterministic baseline and the tool-calling agent on the fixed cases (Phase 7).

    python scripts/agent_explain.py --kinds alert unknown_card      # API running
    python scripts/agent_explain.py --kinds api_down                 # API stopped

Prints the mechanical checks per case and mode, writes the notes (Markdown, card token
added by code) and a run record with the fingerprint to data/agent/runs/ (git-ignored).
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.agent.checks import check_note  # noqa: E402
from src.agent.loop import run_agent  # noqa: E402
from src.agent.note import render_markdown  # noqa: E402
from src.agent.tools import ScoreTool  # noqa: E402
from src.agent.workflow import run_baseline  # noqa: E402
from src.agent.writer import prompt_hashes  # noqa: E402
from src.rag.config import RETRIEVAL, build_retriever  # noqa: E402
from src.rag.corpus import PROJECT_ROOT, load_manifest  # noqa: E402
from src.rag.embed import OllamaEmbedder  # noqa: E402
from src.rag.index import INDEX_DIR, check_embedder, load_index  # noqa: E402
from src.rag.llm import OllamaChat  # noqa: E402
from src.rag.runmeta import fingerprint, ollama_version  # noqa: E402

CASES = PROJECT_ROOT / "data" / "agent" / "cases.json"
RUNS = PROJECT_ROOT / "data" / "agent" / "runs"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--kinds", nargs="+", default=["alert", "unknown_card"])
    parser.add_argument("--api", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True, help="the chat model chosen by rule C4")
    parser.add_argument("--modes", nargs="+", default=["baseline", "agent"], choices=["baseline", "agent"])
    args = parser.parse_args(argv)
    cases = json.loads(CASES.read_text(encoding="utf-8"))["cases"]
    hashes = {d["doc_id"]: d["sha256"] for d in load_manifest()["documents"]}
    frame, embeddings, manifest = load_index(INDEX_DIR, hashes)
    embedder = OllamaEmbedder()
    check_embedder(manifest, embedder)
    records = frame.set_index("chunk_id", drop=False).to_dict("index")
    retriever = build_retriever(frame, embeddings, embedder)
    llm = OllamaChat(model=args.model)
    session = requests.Session()
    fp = {**fingerprint(llm.describe(), embedder.describe(), ollama_version(session), RETRIEVAL),
          "agent_prompts": prompt_hashes()}
    tool = ScoreTool(cases, api_url=args.api, session=session)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = RUNS / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for ref, case in cases.items():
        if case["kind"] not in args.kinds:
            continue
        api_response = tool(ref).response if case["kind"] == "alert" else None   # reference for the checks
        for mode, fn in (("baseline", run_baseline), ("agent", run_agent)):
            if mode not in args.modes:
                continue
            res = fn(ref, tool, retriever, llm, records)
            checks = check_note(res, case, api_response)
            md = render_markdown(res.note, records, card_key=case["request"]["card_key"])
            (out_dir / f"{ref}_{mode}.md").write_text(md + "\n", encoding="utf-8")
            row = {"ref": ref, "kind": case["kind"], "mode": mode, "status": res.note.status,
                   "passed": all(checks.values()), "checks": checks, "tool_calls": res.tool_calls,
                   "turns": res.turns, "seconds": round(res.seconds, 1),
                   "claim_checks": res.written.claim_checks if res.written else [],
                   "errors": res.written.errors if res.written else []}
            rows.append(row)
            failed = [k for k, v in checks.items() if not v]
            mech = res.written.claim_checks if res.written else []
            print(f"{ref} {mode:8s} {res.note.status:26s} pass={row['passed']!s:5} failed={failed} "
                  f"tools={res.tool_calls} turns={res.turns} mech={sum(c['ok'] for c in mech)}/{len(mech)} "
                  f"{row['seconds']}s")
    (out_dir / "run.json").write_text(json.dumps({"fingerprint": fp, "kinds": args.kinds, "rows": rows},
                                                 indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for mode in args.modes:
        r = [x for x in rows if x["mode"] == mode]
        print(f"{mode}: {sum(x['passed'] for x in r)}/{len(r)} passed")
    print(f"notes and record: {out_dir.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
