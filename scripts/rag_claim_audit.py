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


def main() -> int:
    r = audit(read_labels())
    if r["invalid"]:
        print(f"STOP: unlabelled or unknown labels: {r['invalid']}", file=sys.stderr)
        return 1
    print(f"questions: {len(r['questions'])} ({', '.join(r['questions'])})")
    print(f"sentences: {r['total']}  labels: {r['counts']}")
    print(f"supported: {r['counts'].get('supported', 0)}/{r['total']} = {r['supported_rate']:.2f}  "
          f"contradicted: {r['contradicted']}")
    print(f"bar: >= 0.90 supported and 0 contradicted -> {'PASS' if r['passed'] else 'FAIL'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
