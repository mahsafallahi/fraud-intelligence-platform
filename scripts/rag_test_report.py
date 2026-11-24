"""Score the one-time test run against the pre-registered thresholds (Phase 7). No model calls.

    python scripts/rag_test_report.py

Reads data/rag/runs/test_run.jsonl (scripts/rag_test_run.py) and writes
- docs/rag_results.md: every pre-registered metric with its Wilson 95% interval;
- data/rag/eval/claim_labels.md: the human labelling sheet (option B), never overwritten.
The definitions are those of docs/rag_preregistration.md and of the dev evaluation.
"""

import json
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models.calibration import wilson_interval  # noqa: E402
from src.rag.answer import is_no_answer, sanitize  # noqa: E402
from src.rag.chunk import citation  # noqa: E402
from src.rag.citations import CITE_LIKE, NUM_CITE, numbers_in  # noqa: E402
from src.rag.corpus import PROJECT_ROOT  # noqa: E402

RUN = PROJECT_ROOT / "data" / "rag" / "runs" / "test_run.jsonl"
PREREG = PROJECT_ROOT / "data" / "rag" / "eval" / "preregistration.json"
RESULTS = PROJECT_ROOT / "docs" / "rag_results.md"
LABELS = PROJECT_ROOT / "data" / "rag" / "eval" / "claim_labels.md"


def citation_counts(reply: str, k: int) -> tuple[int, int]:
    """(valid, total) citations in one reply; a malformed citation counts as one invalid."""
    valid = total = 0
    for raw in CITE_LIKE.findall(reply):
        if not NUM_CITE.fullmatch(raw):
            total += 1
            continue
        for n in numbers_in(raw):
            total += 1
            valid += 1 <= n <= k
    return valid, total


@dataclass
class Metric:
    name: str
    ok: int
    n: int
    threshold: str
    passed: bool | None          # None: cannot be judged (nothing to count)


def expected(r: dict) -> set[str]:
    return {f"{d}:{s}" for d, s in r["expected_sources"]}


def score(records: list[dict], section_of: dict[str, str], k: int) -> list[Metric]:
    ans = [r for r in records if r["answerable"]]
    una = [r for r in records if not r["answerable"]]
    answered = [r for r in records if r["status"] == "answered"]
    answered_ans = [r for r in ans if r["status"] == "answered"]
    recall = sum(bool(expected(r) & {section_of[c] for c in r["retrieved"]}) for r in ans)
    raw = [citation_counts(t, k) for r in records for t in r["replies"] if not is_no_answer(t)]
    raw_ok, raw_n = sum(v for v, _ in raw), sum(t for _, t in raw)
    fin = [citation_counts(r["text"], k) for r in answered]
    fin_ok, fin_n = sum(v for v, _ in fin), sum(t for _, t in fin)
    correct = sum(bool(expected(r) & {section_of[c] for c in r["cited"]}) for r in answered_ans)
    checks = [c for r in answered for c in r["claim_checks"]]
    mech = sum(c["ok"] for c in checks)
    refused = sum(r["status"] == "refused" for r in una)
    false_ref = sum(r["status"] == "refused" for r in ans)

    def at_least(ok, n, t):
        return ok / n >= t if n else None
    return [
        Metric("Retrieval recall@6", recall, len(ans), ">= 19/22", recall >= 19),
        Metric("Citation validity, raw", raw_ok, raw_n, ">= 0.95", at_least(raw_ok, raw_n, 0.95)),
        Metric("Citation validity, final", fin_ok, fin_n, "= 1.00", fin_ok == fin_n if fin_n else None),
        Metric("Citation correctness", correct, len(answered_ans), ">= 0.80", at_least(correct, len(answered_ans), 0.80)),
        Metric("Claim support, mechanical", mech, len(checks), ">= 0.95", at_least(mech, len(checks), 0.95)),
        Metric("Correct refusal", refused, len(una), ">= 7/8", refused >= 7),
        Metric("False refusal (lower is better)", false_ref, len(ans), "<= 2/22", false_ref <= 2),
    ]


def label_sample(records: list[dict], order: list[str], size: int) -> tuple[list[str], list[str]]:
    """(the first `size` answered ids of the pre-drawn order, the refused ids skipped on the way)."""
    status = {r["id"]: r["status"] for r in records}
    sample, skipped = [], []
    for qid in order:
        if len(sample) == size:
            break
        (sample if status[qid] == "answered" else skipped).append(qid)
    return sample, skipped


def cell(text: str) -> str:
    """Text as one Markdown table cell; reversible (uncell), so the exact text is kept."""
    return (text.replace("&", "&amp;").replace("<", "&lt;").replace("|", "&#124;").replace("\r", "")
            .replace("\n", "<br>"))


def uncell(text: str) -> str:
    return text.replace("<br>", "\n").replace("&#124;", "|").replace("&lt;", "<").replace("&amp;", "&")


def labels_md(records: list[dict], sample: list[str], skipped: list[str], size: int, chunks: dict[str, dict]) -> tuple[str, int]:
    """(Markdown of the labelling sheet, number of sentences)."""
    by_id = {r["id"]: r for r in records}
    lines = ["# Claim labels (human, option B)", "",
             "Written by `scripts/rag_test_report.py` from the one-time test run. For every answer sentence, fill the",
             "last column with exactly one of: `supported`, `not supported`, `contradicted`. Judge the sentence ONLY",
             "against the excerpt text next to it, not against your own knowledge and not against other excerpts.", "",
             "- `not supported`: the excerpt does not say it. `contradicted`: the excerpt says the opposite.",
             "- A sentence that is only partly backed is `not supported`.",
             "- Each excerpt is: the source line, the chunk heading (what the mechanical check also reads), then the exact",
             "  chunk text the model saw (the model saw `[n] <document> — <section>` in place of those first two lines).",
             "  In this file `<br>` is a line break, and `&`, `<` and `|` are written as entities to fit in a table.", "",
             f"Sample (pre-drawn order, the first {size} that were answered): {', '.join(sample)}.",
             f"Refused and replaced by the next in the order: {', '.join(skipped) or 'none'}."
             + ("" if len(sample) == size else f" **Only {len(sample)} were answered.**"), "",
             "Pass threshold: at least 0.90 supported and 0 contradicted."]
    total = 0
    for qid in sample:
        r = by_id[qid]
        lines += ["", f"## {qid}: {r['question']}", "",
                  "| # | Answer sentence | Cited excerpt(s), exact text | Label |", "|---|---|---|---|"]
        for i, (sentence, ids) in enumerate(r["validations"][-1]["sentences"], start=1):
            total += 1
            blocks = []
            for cid in dict.fromkeys(ids):
                n = r["retrieved"].index(cid) + 1
                blocks.append(f"[{n}] {citation(chunks[cid])} (`{cid}`)\n{chunks[cid]['heading']}\n"
                              f"{sanitize(chunks[cid]['text'])}")
            lines.append(f"| {qid}-{i} | {cell(sentence)} | {cell(chr(10).join(blocks))} | |")
    return "\n".join(lines) + "\n", total


def results_md(header: dict, records: list[dict], metrics: list[Metric], section_of: dict[str, str],
               sample: list[str], skipped: list[str], n_sentences: int, frozen: dict) -> str:
    fp = header["fingerprint"]
    lines = ["# Phase 7 RAG: test results (one-time run)", "",
             f"The 30 test questions were run **once** with the frozen configuration of `docs/rag_preregistration.md`, "
             f"started {header['started_utc']} at commit `{header['commit'][:7]}`. Every fingerprint field was checked",
             "against the freeze before the first question. Nothing was changed after the run. Raw record:",
             "`data/rag/runs/test_run.jsonl` (git-ignored). Written by `scripts/rag_test_report.py`.", "",
             f"Model `{fp['llm_model']}` (`{fp['llm_digest'][:12]}`), embeddings `{fp['embed_model']}` (`{fp['embed_digest'][:12]}`), "
             f"Ollama {fp['ollama_version']}, retrieval `{json.dumps(fp['retrieval'])}`, tau `{header['tau']}`.", "",
             "## Pre-registered metrics", "",
             "| Metric | Result | Rate | Wilson 95% interval | Threshold | Verdict |", "|---|---|---|---|---|---|"]
    for m in metrics:
        if m.n:
            lo, hi = wilson_interval(m.ok, m.n)
            rate, ci = f"{m.ok / m.n:.3f}", f"{float(lo):.3f} to {float(hi):.3f}"
        else:
            rate, ci = "n/a", "n/a"
        verdict = {True: "**pass**", False: "**FAIL**", None: "not judged"}[m.passed]
        lines.append(f"| {m.name} | {m.ok}/{m.n} | {rate} | {ci} | {m.threshold} | {verdict} |")
    lines += ["| Claim support, human | pending | | | >= 0.90 supported and 0 contradicted | pending the user's labels |"]
    for mode, ok in (("baseline (recommended mode)", 10), ("tool-calling agent", 6)):
        lo, hi = wilson_interval(ok, 10)
        lines.append(f"| Agent, mechanical: {mode} | {ok}/10 | {ok / 10:.3f} | {float(lo):.3f} to {float(hi):.3f} | 10/10 | "
                     f"{'**pass**' if ok == 10 else '**FAIL**'} |")
    failed = [m.name for m in metrics if m.passed is False]
    lines += ["", f"Failed in this run: {', '.join(failed) if failed else 'none'}"
                  " (the tool-calling agent failed before it, see `docs/rag_design.md` §6e).",
              "The verdict compares the observed rate with the threshold. The intervals are wide because the set is small.",
              "", "## Human claim labels (option B)", "",
              f"Sample: {', '.join(sample)}. Refused and replaced: {', '.join(skipped) or 'none'}. "
              f"{n_sentences} sentences to label in `data/rag/eval/claim_labels.md`.",
              "", "## Per question", "",
              "| Id | Answerable | Retrieved expected section | Status | Reason | Attempts | Cited expected section | Mechanical | Seconds |",
              "|---|---|---|---|---|---|---|---|---|"]
    for r in records:
        hit = bool(expected(r) & {section_of[c] for c in r["retrieved"]}) if r["answerable"] else None
        ce = bool(expected(r) & {section_of[c] for c in r["cited"]}) if r["answerable"] and r["status"] == "answered" else None
        mech = f"{sum(c['ok'] for c in r['claim_checks'])}/{len(r['claim_checks'])}" if r["status"] == "answered" else "-"

        def yn(v):
            return "-" if v is None else ("yes" if v else "**no**")
        wrong = (r["answerable"] and r["status"] == "refused") or (not r["answerable"] and r["status"] == "answered")
        status = f"**{r['status']}**" if wrong else r["status"]
        lines.append(f"| {r['id']} | {'yes' if r['answerable'] else 'no'} | {yn(hit)} | {status} | {r['reason'] or '-'} | "
                     f"{r['attempts']} | {yn(ce)} | {mech} | {r['seconds']} |")
    lines += ["", "Bold marks a wrong outcome: a refused answerable question, an answered unanswerable one, or a missed section.",
              "", "## Sentences that fail the mechanical check", ""]
    bad = [(r["id"], c) for r in records if r["status"] == "answered" for c in r["claim_checks"] if not c["ok"]]
    lines += [f"- `{i}`: {c['sentence']} Missing from the cited text: {', '.join(f'`{t}`' for t in c['missing'])}."
              for i, c in bad] or ["None."]
    wrongly = [r for r in records if not r["answerable"] and r["status"] == "answered"]
    lines += ["", "## Unanswerable questions that were answered", ""]
    lines += [f"- `{r['id']}` {r['question']} → {r['text']}" for r in wrongly] or ["None."]
    lines += ["", "## Known limitation", "",
              "Both models failed rule C4 on dev, and llama3.1 was kept by the pre-registered rule "
              f"(dev mechanical check {frozen['dev_totals']['mechanical_check']}; see `docs/rag_preregistration.md`)."]
    return "\n".join(lines) + "\n"


def main() -> int:
    if not RUN.exists():
        print("STOP: no test run record", file=sys.stderr)
        return 1
    rows = [json.loads(line) for line in RUN.read_text(encoding="utf-8").splitlines()]
    header, records, footer = rows[0], rows[1:-1], rows[-1]
    if "finished_utc" not in footer or len(records) != 30:
        print(f"STOP: the test run is incomplete ({len(rows)} lines)", file=sys.stderr)
        return 1
    from src.rag.index import INDEX_DIR, load_index
    frozen = json.loads(PREREG.read_text(encoding="utf-8"))
    frame, _, _ = load_index(INDEX_DIR)
    chunks = frame.set_index("chunk_id", drop=False).to_dict("index")
    section_of = {c: f"{r['doc_id']}:{r['section']}" for c, r in chunks.items()}
    metrics = score(records, section_of, header["fingerprint"]["retrieval"]["k"])
    size = len(frozen["label_sample_first"])
    sample, skipped = label_sample(records, frozen["label_order"], size)
    sheet, n = labels_md(records, sample, skipped, size, chunks)
    RESULTS.write_text(results_md(header, records, metrics, section_of, sample, skipped, n, frozen),
                       encoding="utf-8", newline="\n")
    if LABELS.exists():
        print(f"{LABELS.relative_to(PROJECT_ROOT)} exists and is kept (it may hold human labels)")
    else:
        LABELS.write_text(sheet, encoding="utf-8", newline="\n")
    for m in metrics:
        print(f"{m.name}: {m.ok}/{m.n} threshold {m.threshold} -> {m.passed}")
    print(f"sample {sample}, replaced {skipped}, {n} sentences to label")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
