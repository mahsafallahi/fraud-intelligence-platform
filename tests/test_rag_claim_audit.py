"""The human claim-support metric script (fixtures; no model)."""

from tests.conftest import load_script

audit_mod = load_script("rag_claim_audit")

SHEET = """# Claim labels
| # | Answer sentence | Cited excerpt(s), exact text | Label |
|---|---|---|---|
| T01-1 | A [1]. | [1] x | supported |
| T01-2 | B [1]. | [1] x | supported |
| T02-1 | C [2]. | [2] y | {label} |
"""


def run(tmp_path, label):
    p = tmp_path / "labels.md"
    p.write_text(SHEET.format(label=label), encoding="utf-8")
    return audit_mod.audit(audit_mod.read_labels(p))


def test_all_supported_passes(tmp_path):
    r = run(tmp_path, "supported")
    assert (r["total"], r["supported_rate"], r["contradicted"], r["passed"]) == (3, 1.0, 0, True)


def test_one_contradicted_fails_even_above_the_rate(tmp_path):
    p = tmp_path / "labels.md"
    p.write_text(SHEET.format(label="contradicted") + "".join(
        f"| T09-{i} | s [1]. | [1] z | supported |\n" for i in range(1, 10)), encoding="utf-8")
    r = audit_mod.audit(audit_mod.read_labels(p))
    assert r["supported_rate"] >= 0.9 and r["contradicted"] == 1 and r["passed"] is False


def test_unlabelled_or_unknown_label_is_invalid(tmp_path):
    assert run(tmp_path, "")["invalid"] == {"T02-1": ""}
    assert run(tmp_path, "maybe")["passed"] is False
