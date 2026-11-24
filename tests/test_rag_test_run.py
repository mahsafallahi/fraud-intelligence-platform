"""One-time test run: fingerprint gate, scoring and label sheet, with made-up records (no Ollama)."""

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


run, rep = load("rag_test_run"), load("rag_test_report")
FROZEN = json.loads(run.PREREG.read_text(encoding="utf-8"))
CHUNK = {"doc_id": "cfr31_1020", "section": "1020.320", "section_title": "Reports | by banks", "subpart": "",
         "page_start": 10, "page_end": 10, "label_start": "1197", "label_end": "1197",
         "heading": "31 CFR § 1020.320", "text": "A bank <shall> file\nwithin 30 days | or & so."}


def test_gate_accepts_only_the_frozen_configuration():
    fp, prompts = FROZEN["fingerprint"], FROZEN["agent_prompts"]
    assert run.differences(FROZEN, fp, FROZEN["tau"], prompts, "", "") == []
    for key in fp:
        assert run.differences(FROZEN, {**fp, key: "changed"}, FROZEN["tau"], prompts, "", ""), key
    assert run.differences(FROZEN, fp, 0.7, prompts, "", "")
    assert run.differences(FROZEN, fp, FROZEN["tau"], {}, "", "")
    assert run.differences(FROZEN, fp, FROZEN["tau"], prompts, "src/rag/answer.py", "")
    assert run.differences(FROZEN, fp, FROZEN["tau"], prompts, "", " M src/rag/llm.py")


def rec(i, answerable=True, status="answered", text="It is 30 days [1].", cited=("c1",), checks=(True,), replies=None):
    return {"id": i, "question": f"q {i}", "answerable": answerable, "status": status, "reason": "", "attempts": 1,
            "expected_sources": [["cfr31_1020", "1020.320"]] if answerable else [], "retrieved": ["c1", "c2"],
            "cited": list(cited), "text": text, "rendered": text, "replies": replies or [text], "seconds": 1.0,
            "validations": [{"ok": True, "errors": [], "cited": list(cited), "sentences": [[text, list(cited)]]}],
            "claim_checks": [{"sentence": text, "tokens": ["30"], "missing": [] if ok else ["30"], "ok": ok} for ok in checks]}


SECTION = {"c1": "cfr31_1020:1020.320", "c2": "cfr31_1020:1020.315"}


def test_score_counts_and_verdicts():
    records = [rec(f"T{n:02d}") for n in range(1, 20)]                                       # 19 good
    records += [rec("T20", cited=("c2",), text="See [2].", checks=(False,)),                   # wrong section
                rec("T21", status="refused", text="refusal", cited=(), checks=(), replies=["NO_ANSWER [9]"]),
                rec("T22", replies=["Bad [9].", "It is 30 days [1]."])]                        # 1 invalid raw citation
    records += [rec(f"T{n}", False, "refused", "refusal", (), (), ["NO_ANSWER"]) for n in range(23, 30)]
    records += [rec("T30", False, "answered")]
    m = {x.name: x for x in rep.score(records, SECTION, 6)}
    assert (m["Retrieval recall@6"].ok, m["Retrieval recall@6"].n) == (22, 22)
    raw = m["Citation validity, raw"]                    # NO_ANSWER replies are not counted
    assert (raw.ok, raw.n, raw.passed) == (22, 23, True)
    assert m["Citation validity, final"].passed and m["Citation validity, final"].n == 22
    assert (m["Citation correctness"].ok, m["Citation correctness"].n) == (20, 21)
    mech = m["Claim support, mechanical"]                # the answered unanswerable question counts too
    assert (mech.ok, mech.n, mech.passed) == (21, 22, True)
    assert (m["Correct refusal"].ok, m["Correct refusal"].passed) == (7, True)
    assert (m["False refusal (lower is better)"].ok, m["False refusal (lower is better)"].passed) == (1, True)


@pytest.mark.parametrize("n_refused,passed", [(2, True), (3, False)])
def test_false_refusal_threshold_is_2_of_22(n_refused, passed):
    records = [rec(f"T{i}", status="refused" if i < n_refused else "answered") for i in range(22)]
    assert rep.score(records, SECTION, 6)[-1].passed is passed


def test_sample_skips_refused_and_takes_the_next_in_the_drawn_order():
    records = [rec("A"), rec("B", status="refused"), rec("C"), rec("D")]
    assert rep.label_sample(records, ["B", "D", "A", "C"], 2) == (["D", "A"], ["B"])


def test_label_sheet_keeps_the_exact_excerpt_text_and_an_empty_column():
    md, n = rep.labels_md([rec("A")], ["A"], ["B"], 1, {"c1": CHUNK})
    row = next(line for line in md.splitlines() if line.startswith("| A-1 |"))
    cells = [c.strip() for c in row.strip("|").split("|")]
    assert n == 1 and len(cells) == 4 and cells[3] == "" and cells[1] == "It is 30 days [1]."
    assert CHUNK["text"] in rep.uncell(cells[2])


def test_results_document_renders(tmp_path):
    records = [rec(f"T{n:02d}") for n in range(1, 23)] + [rec(f"T{n}", False, "refused", "r", (), (), ["NO_ANSWER"])
                                                          for n in range(23, 31)]
    header = {"fingerprint": FROZEN["fingerprint"], "tau": None, "commit": "abcdef123", "started_utc": "x"}
    md = rep.results_md(header, records, rep.score(records, SECTION, 6), SECTION, ["T01"], [], 1, FROZEN)
    assert "| Correct refusal | 8/8 | 1.000 | 0.676 to 1.000 | >= 7/8 | **pass** |" in md
    assert "pending the user's labels" in md and "6/10" in md
