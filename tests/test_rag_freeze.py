"""Model choice by rule C4 and the known-limitation text of scripts/rag_freeze.py (no Ollama)."""

import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "rag_freeze", Path(__file__).resolve().parents[1] / "scripts" / "rag_freeze.py")
freeze = importlib.util.module_from_spec(spec)
spec.loader.exec_module(freeze)


def rec(model, ok, total, raw_ok=True):
    checks = [{"ok": i < ok, "sentence": f"s{i}"} for i in range(total)]
    return {"fingerprint": {"llm_model": model}, "rows": [{"id": "D01", "status": "answered", "claim_checks": checks}],
            "c4_rag": {"raw_citation_validity_ok": raw_ok, "mechanical_check_ok": bool(total) and ok / total >= 0.95}}


@pytest.mark.parametrize("llama,qwen,winner", [
    ((10, 10), (10, 10), "llama"),      # llama3.1 passes: no switch
    ((9, 10), (12, 12), "qwen"),        # only qwen2.5 passes
    ((9, 10), (11, 12), "qwen"),        # both fail, qwen2.5 higher
    ((9, 10), (18, 20), "llama"),       # both fail, exact tie
    ((9, 10), (7, 8), "llama"),         # both fail, llama3.1 higher
    ((9, 10), (0, 0), "llama"),         # qwen2.5 has no checked sentence: it cannot win
])
def test_choice_follows_c4_and_the_tie_rule(llama, qwen, winner):
    assert freeze.choose(rec("llama3.1", *llama), rec("qwen2.5", *qwen))[0] == winner


def test_raw_citation_failure_also_fails_c4():
    assert freeze.choose(rec("llama3.1", 9, 10), rec("qwen2.5", 10, 10, raw_ok=False))[0] == "qwen"   # higher rate
    assert freeze.limitation_lines(rec("qwen2.5", 10, 10, raw_ok=False), rec("llama3.1", 9, 10))


def test_limitation_text_only_when_the_chosen_model_fails():
    assert freeze.limitation_lines(rec("qwen2.5", 10, 10), rec("llama3.1", 9, 10)) == []
    text = "\n".join(freeze.limitation_lines(rec("qwen2.5", 11, 12), rec("llama3.1", 9, 10)))
    assert "Known limitation" in text and "11/12" in text and "9/10" in text and "s11" in text
