"""Pick the 10 fixed alert cases for the Phase 7 agent, WITHOUT labels (needs the running API).

    python scripts/agent_make_cases.py [--api http://127.0.0.1:8000]

Rule (fixed before running):
- candidates: validation-period transactions (2020-07-01 .. 2020-09-23) that match the EDA
  baseline rule (night hour 22-03 and amount > $136.33), read from Gold fraud_dev.duckdb
  read-only; only the request columns are selected, never is_fraud;
- order: md5(trans_num || 'phase7-agent'), ascending (fixed, label-free);
- score candidates through the API in that order and keep the first 8 with alert = true;
- 1 unknown-card case: the request of the 9th alert with a card_key that is not in the snapshot;
- 1 API-down case: the request of the 10th alert, run while the API is stopped.
Writes data/agent/cases.json (git-ignored: it holds card tokens and merchant coordinates).
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import duckdb
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.rag.corpus import PROJECT_ROOT  # noqa: E402

GOLD = PROJECT_ROOT / "data" / "gold" / "fraud_dev.duckdb"
SNAPSHOT = PROJECT_ROOT / "data" / "serving" / "serving.duckdb"
OUT = PROJECT_ROOT / "data" / "agent" / "cases.json"
CANDIDATES_SQL = """
select f.card_key, f.trans_ts, f.amt, m.category, f.merch_lat, f.merch_long
from fact_transactions f join dim_merchant m using (merchant_key)
where f.trans_ts >= timestamp '2020-07-01' and f.trans_ts < timestamp '2020-09-24'
  and (hour(f.trans_ts) >= 22 or hour(f.trans_ts) <= 3) and f.amt > 136.33
order by md5(f.trans_num || 'phase7-agent')
limit 400
"""


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--api", default="http://127.0.0.1:8000")
    args = parser.parse_args(argv)
    con = duckdb.connect(str(GOLD), read_only=True)
    rows = con.sql(CANDIDATES_SQL).fetchall()
    con.close()
    with requests.Session() as s:
        alerts, scored = [], 0
        for card_key, ts, amt, category, lat, lon in rows:
            req = {"card_key": card_key, "trans_ts": ts.isoformat(), "amt": float(amt), "category": category,
                   "merch_lat": float(lat), "merch_long": float(lon)}
            r = s.post(f"{args.api}/score", json=req, timeout=30)
            scored += 1
            if r.status_code == 200 and r.json()["alert"]:
                alerts.append(req)
                if len(alerts) == 10:
                    break
    if len(alerts) < 10:
        print(f"STOP: only {len(alerts)} alerts among {scored} candidates", file=sys.stderr)
        return 1
    snap = duckdb.connect(str(SNAPSHOT), read_only=True)
    known = {k for (k,) in snap.sql("select card_key from cards").fetchall()}
    snap.close()
    unknown = hashlib.sha256(b"phase7-unknown-card").hexdigest()[:32]
    assert unknown not in known
    cases = {f"A{i:02d}": {"kind": "alert", "request": a} for i, a in enumerate(alerts[:8], start=1)}
    cases["U01"] = {"kind": "unknown_card", "request": {**alerts[8], "card_key": unknown}}
    cases["D01"] = {"kind": "api_down", "request": alerts[9]}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"rule": __doc__.split("Rule (fixed before running):")[1].split("Writes")[0].strip(),
                               "candidates_scored": scored, "cases": cases}, indent=2) + "\n", encoding="utf-8")
    print(f"{scored} candidates scored, 10 alerts kept; wrote {OUT.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
