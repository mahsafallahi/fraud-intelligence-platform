"""Write docs/rag_preregistration.md (Phase 7 freeze, decision C7) from the two dev evaluation records.

    python scripts/rag_freeze.py --llama eval_dev_<UTC>.json --qwen eval_dev_qwen2.5_<UTC>.json

Applies rule C4 and the tie rule (docs/rag_design.md 6b and 8) to choose the model; stops if
the two runs differ in anything but the model. If the chosen model itself fails C4 on dev,
the pre-registration states this as a known limitation (decision (b), 2026-11-24).
Takes the fingerprint of the chosen dev run (Ollama version, model and embedder digests,
prompt hashes, index manifest and questions SHA-256, retrieval config), checks that the
current files still match it, and writes the frozen configuration and all thresholds.
A machine-readable copy goes to data/rag/eval/preregistration.json (committed).
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.rag.answer import prompt_hashes  # noqa: E402
from src.rag.config import RETRIEVAL, TAU  # noqa: E402
from src.rag.corpus import PROJECT_ROOT  # noqa: E402
from src.rag.index import INDEX_DIR, MANIFEST_FILE  # noqa: E402
from src.rag.questions import LABEL_SAMPLE_SIZE, LABEL_SEED, QUESTIONS, labelling_order, load_questions  # noqa: E402
from src.rag.runmeta import file_sha256, fingerprint_diff  # noqa: E402
from src.agent.writer import prompt_hashes as agent_prompt_hashes  # noqa: E402

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


def mech(record: dict) -> tuple[int, int]:
    """(passed, total) sentences of the mechanical check over the answered dev questions."""
    checks = [c for r in record["rows"] if r["status"] == "answered" for c in r["claim_checks"]]
    return sum(c["ok"] for c in checks), len(checks)


def passes_c4(record: dict) -> bool:
    return all(record["c4_rag"].values())


def choose(llama: dict, qwen: dict) -> tuple[str, str]:
    """('llama'|'qwen', reason) by rule C4 and the rule for two failing models (exact fractions, no rounding)."""
    if passes_c4(llama):
        return "llama", "llama3.1 passes C4 on dev, so there is no switch"
    if passes_c4(qwen):
        return "qwen", "llama3.1 fails C4 on dev and qwen2.5 passes it"
    (a, n), (b, m) = mech(llama), mech(qwen)
    if n and m and b * n > a * m:
        return "qwen", "both models fail C4 on dev and qwen2.5 has the higher mechanical-check rate"
    if n and m and b * n == a * m:
        return "llama", "both models fail C4 on dev and tie on the mechanical check, so llama3.1 stays"
    return "llama", "both models fail C4 on dev and llama3.1 has the higher mechanical-check rate"


def limitation_lines(chosen: dict, other: dict) -> list[str]:
    """The known-limitation section; empty when the chosen model passes C4."""
    if passes_c4(chosen):
        return []
    ok, n = mech(chosen)
    failed = [k for k, v in chosen["c4_rag"].items() if not v]
    out = ["", "## Known limitation: rule C4 is not met on dev", "",
           f"Both models fail C4 on dev. The chosen model `{chosen['fingerprint']['llm_model']}` fails "
           f"{', '.join(f'`{k}`' for k in failed)}; its mechanical check is {ok}/{n} against the required 0.95.",
           f"The other model `{other['fingerprint']['llm_model']}`: {mech(other)[0]}/{mech(other)[1]}. No threshold is",
           "changed because of this: the test run is judged by the thresholds below, and a fail is reported as a fail.",
           "Failing dev sentences of the chosen model:", ""]
    for r in chosen["rows"]:
        if r["status"] == "answered":
            out += [f"- `{r['id']}`: {json.dumps(c, ensure_ascii=False)}" for c in r["claim_checks"] if not c["ok"]]
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--llama", required=True, type=Path)
    parser.add_argument("--qwen", required=True, type=Path)
    args = parser.parse_args(argv)
    paths = {"llama": args.llama, "qwen": args.qwen}
    recs = {k: json.loads(p.read_text(encoding="utf-8")) for k, p in paths.items()}
    for key, rec in recs.items():
        if key not in rec["fingerprint"]["llm_model"]:
            print(f"STOP: --{key} is a run of {rec['fingerprint']['llm_model']}", file=sys.stderr)
            return 1
    unequal = set(fingerprint_diff(recs["llama"]["fingerprint"], recs["qwen"]["fingerprint"])) - {"llm_model", "llm_digest"}
    if unequal:
        print(f"STOP: the two dev runs are not comparable, they differ in {sorted(unequal)}", file=sys.stderr)
        return 1
    winner, why = choose(recs["llama"], recs["qwen"])
    record, other = recs[winner], recs["qwen" if winner == "llama" else "llama"]
    record_name = paths[winner].name
    fp = record["fingerprint"]
    now = {"prompts": prompt_hashes(), "index_manifest_sha256": file_sha256(INDEX_DIR / MANIFEST_FILE),
           "questions_sha256": file_sha256(QUESTIONS), "retrieval": RETRIEVAL}
    diff = {k: (fp[k], v) for k, v in now.items() if fp[k] != v}
    if diff:
        print(f"STOP: the files changed since the dev run: {diff}", file=sys.stderr)
        return 1
    test_ids = [q["id"] for q in load_questions(QUESTIONS) if q["split"] == "test" and q["answerable"]]
    order = labelling_order(test_ids)
    frozen = {"fingerprint": fp, "tau": TAU, "dev_record": record_name, "dev_totals": record["totals"],
              "c4": record["c4_rag"], "choice_reason": why,
              "other_model": {"llm_model": other["fingerprint"]["llm_model"],
                              "llm_digest": other["fingerprint"]["llm_digest"],
                              "dev_totals": other["totals"], "c4": other["c4_rag"]},
              "agent_prompts": agent_prompt_hashes(),
              "known_limitation_c4": not passes_c4(record), "label_order": order, "label_sample_first": order[:LABEL_SAMPLE_SIZE],
              "thresholds": [{"metric": m, "definition": d, "pass": p} for m, d, p in THRESHOLDS]}
    JSON_OUT.write_text(json.dumps(frozen, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")

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
    lines += [f"| Agent prompt `{n}` SHA-256 | `{h}` |" for n, h in agent_prompt_hashes().items()]
    lines += [
        "",
        f"Dev run: `data/rag/runs/{record_name}`.",
        "",
        "## Model choice (rule C4, dev set)",
        "",
        f"| Dev metric | {recs['llama']['fingerprint']['llm_model']} | {recs['qwen']['fingerprint']['llm_model']} |",
        "|---|---|---|",
    ]
    lines += [f"| {k} | {recs['llama']['totals'][k]} | {recs['qwen']['totals'][k]} |" for k in record["totals"]]
    lines += [f"| C4 passed | {passes_c4(recs['llama'])} | {passes_c4(recs['qwen'])} |", "",
              f"Chosen: `{fp['llm_model']}`. Reason: {why}."]
    lines += limitation_lines(record, other)
    lines += [
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
    print(f"chosen: {fp['llm_model']} ({why}); known limitation: {not passes_c4(record)}")
    print(f"wrote {DOC.relative_to(PROJECT_ROOT)} and {JSON_OUT.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
