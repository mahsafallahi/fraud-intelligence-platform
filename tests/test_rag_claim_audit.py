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


# ---- second rater and agreement -------------------------------------------------

def test_agreement_raw_disagreements_and_kappa():
    r1 = {"a": "supported", "b": "supported", "c": "not supported", "d": "not supported"}
    r2 = {"a": "supported", "b": "not supported", "c": "not supported", "d": "supported"}
    a = audit_mod.agreement(r1, r2)
    assert (a["agree"], a["n"], a["disagreements"]) == (2, 4, ["b", "d"])
    assert a["p_e"] == 0.5 and a["kappa"] == 0.0


def test_kappa_perfect_and_undefined():
    two = {"a": "supported", "b": "not supported"}
    assert audit_mod.agreement(two, dict(two))["kappa"] == 1.0
    one = {"a": "supported", "b": "supported"}
    assert audit_mod.agreement(one, dict(one))["kappa"] is None          # 1 - p_e == 0


def test_single_category_rater_gives_uninformative_zero_kappa():
    r1 = {f"x{i}": "supported" for i in range(26)}
    r2 = {**r1, **{f"x{i}": "not supported" for i in range(6)}}
    a = audit_mod.agreement(r1, r2)
    assert (a["agree"], len(a["disagreements"])) == (20, 6)
    assert abs(a["kappa"]) < 1e-12 and a["single_category"] == {"rater1": True, "rater2": False}


def test_agreement_needs_the_same_ids():
    import pytest
    with pytest.raises(ValueError):
        audit_mod.agreement({"a": "supported"}, {"b": "supported"})


def test_rater2_reader_accepts_empty_reasons(tmp_path):
    p = tmp_path / "r2.md"
    p.write_text("| # | Rater-2 label | Rater-2 reason |\n|---|---|---|\n| T01-1 | supported | |\n"
                 "| T01-2 | not supported | Not in the excerpt. |\n", encoding="utf-8")
    assert audit_mod.read_rater2(p) == {"T01-1": "supported", "T01-2": "not supported"}


def test_committed_rater2_file_matches_the_expected_agreement():
    a = audit_mod.agreement(audit_mod.read_labels(), audit_mod.read_rater2())
    assert (a["agree"], a["n"]) == (20, 26)
    assert a["disagreements"] == ["T02-2", "T04-2", "T07-2", "T10-1", "T15-2", "T22-1"]
