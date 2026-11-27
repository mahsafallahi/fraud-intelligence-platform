"""Human claim-support metric (Phase 7, option B) from data/rag/eval/claim_labels.md. No model, no network.

    python scripts/rag_claim_audit.py

Reads the last (Label) column of every answer-sentence row ("| T02-1 | ..."), counts
supported / not supported / contradicted, and checks the pre-registered bar:
>= 0.90 supported and 0 contradicted. Exit 1 if a row is unlabelled or has an unknown label.
"""

import re
import sys
from collections import Counter
from pathlib import Path

LABELS_FILE = Path(__file__).resolve().parents[1] / "data" / "rag" / "eval" / "claim_labels.md"
ALLOWED = ("supported", "not supported", "contradicted")
ROW = re.compile(r"^\| (T\d\d-\d+) \| .* \| ([^|]*) \|$")


def read_labels(path: Path = LABELS_FILE) -> dict[str, str]:
    labels = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        m = ROW.match(line)
        if m:
            labels[m.group(1)] = m.group(2).strip()
    return labels


def audit(labels: dict[str, str]) -> dict:
    bad = {k: v for k, v in labels.items() if v not in ALLOWED}
    counts = Counter(labels.values())
    total = len(labels)
    rate = counts["supported"] / total if total else 0.0
    return {"total": total, "counts": dict(counts), "supported_rate": rate,
            "contradicted": counts["contradicted"], "invalid": bad,
            "questions": sorted({k.split("-")[0] for k in labels}),
            "passed": not bad and total > 0 and rate >= 0.90 and counts["contradicted"] == 0}


RATER2_FILE = LABELS_FILE.with_name("claim_labels_rater2.md")
RATER2_ROW = re.compile(r"^\| (T\d\d-\d+) \| ([^|]*?) \|([^|]*)\|$")      # the reason may be empty


def read_rater2(path: Path = RATER2_FILE) -> dict[str, str]:
    """Second rater (post hoc, LLM, not blind): {id: label} from "| id | label | reason |" rows."""
    labels = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        m = RATER2_ROW.match(line)
        if m:
            labels[m.group(1)] = m.group(2).strip()
    return labels


def agreement(r1: dict[str, str], r2: dict[str, str]) -> dict:
    """Raw agreement, disagreeing ids and Cohen's kappa over the ids both raters labelled.

    kappa = (p_o - p_e) / (1 - p_e); it is None when 1 - p_e == 0 (both raters used one and the same
    category). When ONE rater used a single category, p_e equals the other rater's share of that
    category, so kappa is 0 whatever the agreement: it is reported, but it is not informative.
    """
    if set(r1) != set(r2):
        raise ValueError(f"the raters labelled different ids: {sorted(set(r1) ^ set(r2))}")
    ids = sorted(r1)
    n = len(ids)
    agree = sum(r1[i] == r2[i] for i in ids)
    p_o = agree / n
    c1, c2 = Counter(r1.values()), Counter(r2.values())
    p_e = sum(c1[k] * c2[k] for k in set(c1) | set(c2)) / (n * n)
    kappa = None if 1 - p_e == 0 else (p_o - p_e) / (1 - p_e)
    return {"n": n, "agree": agree, "p_o": p_o, "p_e": p_e, "kappa": kappa,
            "disagreements": [i for i in ids if r1[i] != r2[i]],
            "single_category": {"rater1": len(c1) == 1, "rater2": len(c2) == 1}}


def _print_audit(name: str, r: dict) -> None:
    print(f"{name}: sentences {r['total']}  labels {r['counts']}")
    print(f"{name}: supported {r['counts'].get('supported', 0)}/{r['total']} = {r['supported_rate']:.2f}  "
          f"contradicted {r['contradicted']}  bar (>= 0.90 and 0 contradicted): {'PASS' if r['passed'] else 'FAIL'}")


def main() -> int:
    r1_labels = read_labels()
    r = audit(r1_labels)
    if r["invalid"]:
        print(f"STOP: unlabelled or unknown labels: {r['invalid']}", file=sys.stderr)
        return 1
    print(f"questions: {len(r['questions'])} ({', '.join(r['questions'])})")
    print(f"sentences: {r['total']}  labels: {r['counts']}")
    print(f"supported: {r['counts'].get('supported', 0)}/{r['total']} = {r['supported_rate']:.2f}  "
          f"contradicted: {r['contradicted']}")
    print(f"bar: >= 0.90 supported and 0 contradicted -> {'PASS' if r['passed'] else 'FAIL'}")
    if not RATER2_FILE.exists():
        return 0

    print("\nsecond rater (post hoc, LLM, not blind; does not change the official verdict above)")
    r2_labels = read_rater2()
    r2 = audit(r2_labels)
    if r2["invalid"]:
        print(f"STOP: rater 2 unlabelled or unknown labels: {r2['invalid']}", file=sys.stderr)
        return 1
    _print_audit("rater 2", r2)
    a = agreement(r1_labels, r2_labels)
    print(f"raw agreement: {a['agree']}/{a['n']} = {a['p_o']:.2f}")
    print(f"disagreements: {', '.join(a['disagreements']) or 'none'}")
    kappa = "undefined (1 - p_e = 0)" if a["kappa"] is None else f"{a['kappa']:.3f}"
    print(f"Cohen's kappa: {kappa} (p_o {a['p_o']:.3f}, p_e {a['p_e']:.3f})")
    if any(a["single_category"].values()):
        who = " and ".join(k.replace("rater", "rater ") for k, v in a["single_category"].items() if v)
        print(f"  caveat: {who} used a single category, so p_e equals the other rater's share of it and kappa "
              f"is 0 (or undefined) whatever the agreement; kappa is not informative here, use raw agreement")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
