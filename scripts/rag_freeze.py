"""Write docs/rag_preregistration.md (Phase 7 freeze, decision C7) from the dev evaluation record.

    python scripts/rag_freeze.py data/rag/runs/eval_dev_<model>_<UTC>.json

Takes the fingerprint of the chosen dev run (Ollama version, model and embedder digests,
prompt hashes, index manifest and questions SHA-256, retrieval config), checks that the
current files still match it, and writes the frozen configuration and all thresholds.
A machine-readable copy goes to data/rag/eval/preregistration.json (committed).
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.rag.answer import prompt_hashes  # noqa: E402
from src.rag.config import RETRIEVAL, TAU  # noqa: E402
from src.rag.corpus import PROJECT_ROOT  # noqa: E402
from src.rag.index import INDEX_DIR, MANIFEST_FILE  # noqa: E402
from src.rag.questions import LABEL_SAMPLE_SIZE, LABEL_SEED, QUESTIONS, labelling_order, load_questions  # noqa: E402
from src.rag.runmeta import file_sha256  # noqa: E402

DOC = PROJECT_ROOT / "docs" / "rag_preregistration.md"
JSON_OUT = PROJECT_ROOT / "data" / "rag" / "eval" / "preregistration.json"

THRESHOLDS = [
    ("Retrieval recall@6", "answerable test questions with at least one expected section in the top 6", ">= 19/22"),
    ("Citation validity, raw", "cited excerpt numbers within 1..k, before the validator", ">= 0.95"),
    ("Citation validity, final", "after the validator (true by construction)", "= 1.00"),
    ("Citation correctness", "answered answerable questions with at least one citation in an expected section", ">= 0.80"),
    ("Claim support, human", "sentences of the 10-question human sample labelled supported / unsupported / "
                             "contradicted by their cited chunks", ">= 0.90 supported and 0 contradicted"),
    ("Claim support, mechanical", "on all answered test questions: sentences with a number, number word, dollar "
                                  "amount (with qualifier), day count or section reference whose tokens all occur "
                                  "in a cited chunk", ">= 0.95"),
    ("Correct refusal", "unanswerable test questions refused", ">= 7/8"),
    ("False refusal", "answerable test questions refused", "<= 2/22"),
    ("Agent, mechanical", "10 fixed alert cases: schema, status, reasons, citations, no PII to the LLM, limits",
     "10/10"),
]


def main(argv=None) -> int:
    argv = argv or sys.argv[1:]
    record = json.loads(Path(argv[0]).read_text(encoding="utf-8"))
    fp = record["fingerprint"]
    now = {"prompts": prompt_hashes(), "index_manifest_sha256": file_sha256(INDEX_DIR / MANIFEST_FILE),
           "questions_sha256": file_sha256(QUESTIONS), "retrieval": RETRIEVAL}
    diff = {k: (fp[k], v) for k, v in now.items() if fp[k] != v}
    if diff:
        print(f"STOP: the files changed since the dev run: {diff}", file=sys.stderr)
        return 1
    test_ids = [q["id"] for q in load_questions(QUESTIONS) if q["split"] == "test" and q["answerable"]]
    order = labelling_order(test_ids)
    frozen = {"fingerprint": fp, "tau": TAU, "dev_record": Path(argv[0]).name, "dev_totals": record["totals"],
              "c4": record.get("c4_rag"), "label_order": order, "label_sample_first": order[:LABEL_SAMPLE_SIZE],
              "thresholds": [{"metric": m, "definition": d, "pass": p} for m, d, p in THRESHOLDS]}
    JSON_OUT.write_text(json.dumps(frozen, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    p = fp["prompts"]
    lines = [
        "# Phase 7 RAG: pre-registration (frozen before the test run)",
        "",
        "Written by `scripts/rag_freeze.py` from the chosen dev run. It is committed and pushed before",
        "the test run (decision C7); the push gives a public timestamp. The machine-readable copy is",
        "`data/rag/eval/preregistration.json`. **If any value below changes before the test run, the dev",
        "evaluation is re-run and this file is regenerated.** Details and reasons: `docs/rag_design.md`.",
        "",
        "## Frozen configuration",
        "",
        "| Item | Value |",
        "|---|---|",
        f"| Chat model | `{fp['llm_model']}` |",
        f"| Chat model digest | `{fp['llm_digest']}` |",
        f"| Chat options | `{json.dumps(fp['llm_options'])}` |",
        f"| Embedding model | `{fp['embed_model']}` (task prefixes `search_document: ` / `search_query: `) |",
        f"| Embedding digest | `{fp['embed_digest']}` |",
        f"| Ollama version | `{fp['ollama_version']}` |",
        f"| Retrieval | `{json.dumps(fp['retrieval'])}` (dense only, chosen on dev) |",
        f"| tau (refusal threshold) | `{TAU}`: disabled; dev did not separate. Refusal = NO_ANSWER + validator |",
        f"| Index manifest SHA-256 | `{fp['index_manifest_sha256']}` |",
        f"| questions.jsonl SHA-256 | `{fp['questions_sha256']}` (CRLF normalised to LF) |",
    ]
    lines += [f"| Prompt `{n}` SHA-256 | `{h}` |" for n, h in p.items()]
    lines += [
        "",
        f"Dev run: `data/rag/runs/{Path(argv[0]).name}`. Dev totals: " + "; ".join(f"{k} {v}" for k, v in record["totals"].items()) + ".",
        "",
        "## Human claim-labelling sample (option B)",
        "",
        f"Order drawn with `random.Random({LABEL_SEED}).shuffle` over the sorted ids of the 22 answerable test",
        f"questions. The sample is the first {LABEL_SAMPLE_SIZE} that the system answers; a refused one is replaced by the next.",
        "",
        "First 10: " + ", ".join(f"`{i}`" for i in order[:LABEL_SAMPLE_SIZE]) + ".",
        "Replacements in order: " + ", ".join(f"`{i}`" for i in order[LABEL_SAMPLE_SIZE:]) + ".",
        "",
        "## Pass thresholds (test set: 22 answerable + 8 not answerable)",
        "",
        "| Metric | Definition | Pass |",
        "|---|---|---|",
    ]
    lines += [f"| {m} | {d} | **{x}** |" for m, d, x in THRESHOLDS]
    lines += [
        "",
        "Reporting: counts with Wilson 95% intervals. The test set is run once with the frozen configuration.",
        "There are no prompt or configuration changes after the test run, and a failed threshold is reported as a fail.",
    ]
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {DOC.relative_to(PROJECT_ROOT)} and {JSON_OUT.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
