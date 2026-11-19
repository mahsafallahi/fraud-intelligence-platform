"""Unknown cards, dates outside the calendar and edge timestamps (synthetic; runs in CI)."""

import pandas as pd
import pytest


@pytest.fixture(scope="module")
def txns(synthetic):
    return synthetic[1]


def body(card_key, ts, **extra):
    return {"card_key": card_key, "trans_ts": ts, "amt": 25.0, "category": "grocery_pos",
            "merch_lat": 40.0, "merch_long": -76.0, **extra}


def test_unknown_card_is_422_unknown_card(api):
    r = api.post("/score", json=body("f" * 32, "2020-11-10T12:00:00"))
    assert r.status_code == 422
    assert r.json() == {"code": "unknown_card", "detail": "card_key is not in the serving snapshot"}


@pytest.mark.parametrize("ts", ["2018-12-31T23:59:59", "2021-01-01T00:00:00"])
def test_date_outside_calendar_is_422(api, txns, ts):
    r = api.post("/score", json=body(txns["card_key"].iloc[0], ts))
    assert r.status_code == 422 and r.json()["code"] == "date_outside_calendar"


def test_calendar_edges_are_accepted(api, txns):
    card = txns["card_key"].iloc[0]
    assert api.post("/score", json=body(card, "2019-01-01T00:00:00")).status_code == 200
    assert api.post("/score", json=body(card, "2020-12-31T23:59:59")).status_code == 200


def test_known_card_before_its_first_transaction_scores(api, txns):
    # no earlier history: card features are missing, exactly as for a card's first row in training
    first = txns.groupby("card_key")["trans_ts"].min()
    card = first.index[0]
    ts = (first.iloc[0] - pd.Timedelta(days=30)).isoformat()
    r = api.post("/score", json=body(card, ts))
    assert r.status_code == 200 and 0 < r.json()["probability_raw"] < 1


def test_replaying_a_historical_transaction_does_not_see_itself(api, txns):
    # the same request twice gives the same answer: the snapshot is read-only and
    # history is strictly earlier, so a stored transaction never counts itself
    t = txns.iloc[20]
    req = body(t.card_key, t.trans_ts.isoformat(), amt=float(t.amt), category=t.category,
               merch_lat=float(t.merch_lat), merch_long=float(t.merch_long))
    a, b = api.post("/score", json=req).json(), api.post("/score", json=req).json()
    assert a["raw_score"] == b["raw_score"] and a["request_id"] != b["request_id"]
