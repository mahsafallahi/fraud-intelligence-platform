"""Training-serving parity on REAL test-period transactions (local only).

Offline side: build_features() on the whole Gold frame (fraud_dev.duckdb),
scored in one batch by the bundle's booster, exactly as notebook 04 scored the
test period. Online side: the API, one request per transaction.

Sample (seed 42, chosen from offline scores only, never from labels):
  2,000 random test-period rows + every offline alert (raw prob >= threshold)
  + every row within +-5% of the threshold + every same-second row.
is_fraud and gender are dropped right after loading and never used.

Pass rules (decided 2026-11-19):
- in process (.venv): features exact; raw log-odds, raw and calibrated
  probability exact; alert flags and top-3 reasons 100% identical.
- container (Linux, API_URL set): raw log-odds exact; probabilities may differ
  only at float rounding level (<= 1e-12); flags and reasons 100% identical.
The measured maxima and the latency are printed (run with -s to see them).
"""

import json
import os
import time
import urllib.request

import duckdb
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api.bundle import load_bundle
from src.api.main import create_app
from src.api.scoring import score_features
from src.models.features import FINAL_COLUMNS, build_features, load_model_frame
from src.models.split import TEST, assign_split
from tests.conftest import BUNDLE_DIR, GOLD_DEV_DB, SERVING_DB

pytestmark = pytest.mark.gold

N_RANDOM = 2000
NEAR = 0.05            # relative band around the threshold
CONTAINER_PROB_TOL = 1e-12
P99_LIMIT_MS = 200


@pytest.fixture(scope="module")
def offline():
    bundle = load_bundle(BUNDLE_DIR)
    frame = load_model_frame(GOLD_DEV_DB).drop(columns=["is_fraud", "gender"])
    features = build_features(frame)
    test = (assign_split(frame["trans_ts"]) == TEST).to_numpy()
    X = features.loc[test, FINAL_COLUMNS]
    rows = frame.loc[test, ["trans_num", "card_key", "trans_ts", "amt", "category"]].copy()
    rows["raw_score"] = bundle.booster.predict(X, raw_score=True)
    rows["probability_raw"] = bundle.booster.predict(X)
    rows["probability_calibrated"] = bundle.calibrator.predict(rows["probability_raw"].to_numpy())
    with duckdb.connect(str(GOLD_DEV_DB), read_only=True) as con:
        coords = con.execute("""select trans_num, merch_lat, merch_long from fact_transactions
                                where trans_ts >= '2020-10-01'""").df()
    rows = rows.merge(coords, on="trans_num", how="left", validate="1:1").set_index(X.index)
    assert rows["merch_lat"].notna().all()

    thr = bundle.threshold
    rng = np.random.default_rng(42)
    pick = set(rng.choice(rows.index, size=N_RANDOM, replace=False))
    pick |= set(rows.index[rows["probability_raw"] >= thr])
    pick |= set(rows.index[(rows["probability_raw"] - thr).abs() <= NEAR * thr])
    pick |= set(rows.index[rows.duplicated(["card_key", "trans_ts"], keep=False)])
    sample = rows.loc[sorted(pick)]
    return bundle, X.loc[sample.index], sample


def request_body(r) -> dict:
    return {"card_key": r.card_key, "trans_ts": r.trans_ts.isoformat(), "amt": float(r.amt),
            "category": r.category, "merch_lat": float(r.merch_lat), "merch_long": float(r.merch_long)}


def offline_reasons(bundle, X, idx):
    return score_features(bundle, X.loc[[idx]]).reasons


def summarise(label, latencies, diffs, n):
    lat = np.array(latencies)
    print(f"\n[{label}] rows compared: {n:,}")
    for k, v in diffs.items():
        print(f"  max |diff| {k}: {v:.3e}")
    print(f"  latency ms: p50 {np.percentile(lat, 50):.1f} | p95 {np.percentile(lat, 95):.1f} | "
          f"p99 {np.percentile(lat, 99):.1f} | max {lat.max():.1f}")
    return np.percentile(lat, 99)


def test_in_process_api_equals_offline(offline):
    bundle, X, sample = offline
    from src.api.scoring import new_row_features

    app = create_app(BUNDLE_DIR, SERVING_DB)
    client = TestClient(app)
    store = app.state.store
    latencies, max_diff = [], {"raw_score": 0.0, "probability_raw": 0.0, "probability_calibrated": 0.0}
    try:
        for r in sample.itertuples():
            req = request_body(r)
            x = new_row_features(store, FINAL_COLUMNS, card_key=r.card_key,
                                 trans_ts=r.trans_ts.to_pydatetime(), amt=r.amt, category=r.category,
                                 merch_lat=r.merch_lat, merch_long=r.merch_long)
            pd.testing.assert_frame_equal(x.reset_index(drop=True), X.loc[[r.Index]].reset_index(drop=True),
                                          check_exact=True)
            t0 = time.perf_counter()
            resp = client.post("/score", json=req)
            latencies.append(1000 * (time.perf_counter() - t0))
            assert resp.status_code == 200, resp.text
            body = resp.json()
            for k in max_diff:
                max_diff[k] = max(max_diff[k], abs(body[k] - getattr(r, k)))
                assert body[k] == getattr(r, k), (r.trans_num, k, body[k], getattr(r, k))
            assert body["alert"] == bool(r.probability_raw >= bundle.threshold)
            assert body["reasons"] == offline_reasons(bundle, X, r.Index)
    finally:
        store.close()
    p99 = summarise("in process", latencies, max_diff, len(sample))
    print(f"  alerts in sample: {int((sample['probability_raw'] >= bundle.threshold).sum())}")
    assert p99 <= P99_LIMIT_MS


@pytest.mark.container
def test_container_api_equals_offline(offline):
    bundle, X, sample = offline
    url = os.environ["API_URL"].rstrip("/") + "/score"
    latencies, max_diff = [], {"raw_score": 0.0, "probability_raw": 0.0, "probability_calibrated": 0.0}
    flag_mismatch = reason_mismatch = 0
    for r in sample.itertuples():
        data = json.dumps(request_body(r)).encode()
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = json.loads(resp.read())
        latencies.append(1000 * (time.perf_counter() - t0))
        for k in max_diff:
            max_diff[k] = max(max_diff[k], abs(body[k] - getattr(r, k)))
        flag_mismatch += body["alert"] != bool(r.probability_raw >= bundle.threshold)
        off = offline_reasons(bundle, X, r.Index)
        same = [(a["feature"], a["value"]) for a in body["reasons"]] == [(b["feature"], b["value"]) for b in off]
        close = all(abs(a["contribution"] - b["contribution"]) <= CONTAINER_PROB_TOL
                    for a, b in zip(body["reasons"], off))
        reason_mismatch += not (same and close)
    p99 = summarise("container", latencies, max_diff, len(sample))
    print(f"  alert flag mismatches: {flag_mismatch} | reason mismatches: {reason_mismatch}")
    assert flag_mismatch == 0 and reason_mismatch == 0
    assert max_diff["raw_score"] == 0.0
    assert max_diff["probability_raw"] <= CONTAINER_PROB_TOL
    assert max_diff["probability_calibrated"] <= CONTAINER_PROB_TOL
    assert p99 <= P99_LIMIT_MS
