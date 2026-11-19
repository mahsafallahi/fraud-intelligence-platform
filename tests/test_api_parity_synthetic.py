"""Training-serving parity on synthetic data (runs in CI).

For EVERY synthetic transaction, the API path (history strictly before it +
the transaction, resolved by the snapshot, then build_features) must give
exactly the features that build_features gives on the whole synthetic frame,
and the same scores. Covers same-second pairs, bursts, cards with fewer than 5
earlier transactions and missing income.
"""

import numpy as np
import pandas as pd
import pytest

from src.api.scoring import new_row_features, score_features
from src.models.features import FINAL_COLUMNS, OUTPUT_COLUMNS, REQUIRED_COLUMNS, build_features

FULL_FRAME_SQL = """
    select h.card_key, h.trans_ts, h.amt, h.category, h.channel,
           c.birth_year, c.city_pop, c.median_household_income, c.has_income,
           h.distance_km, h.is_public_holiday_in_state,
           h.is_nationwide_public_holiday, h.is_bank_holiday
    from card_history h join cards c using (card_key)
"""


@pytest.fixture(scope="module")
def offline(api, synthetic):
    """Offline features: build_features on the whole synthetic frame, keyed by (card, time, amount)."""
    store = api.app.state.store
    full = store._con.cursor().execute(FULL_FRAME_SQL).df()
    features = build_features(full[REQUIRED_COLUMNS])
    features.index = pd.MultiIndex.from_frame(full[["card_key", "trans_ts", "amt"]])
    assert features.index.is_unique
    return features


def test_synthetic_data_covers_the_edge_cases(offline):
    assert offline["amt_vs_card_median"].isna().any()      # fewer than 5 earlier transactions
    assert offline["secs_since_prev"].isna().any()         # a card's first transaction
    assert (offline["card_txn_1h"] > 0).any() and (offline["card_txn_24h"] > 1).any()
    assert offline["is_public_holiday_in_state"].eq(1).any()
    ts = offline.index.to_frame(index=False)
    assert ts.duplicated(["card_key", "trans_ts"]).any()   # a same-second pair


def test_api_features_equal_offline_features_for_every_transaction(api, synthetic, offline):
    store, (_, txns) = api.app.state.store, synthetic
    rows = []
    for t in txns.itertuples():
        x = new_row_features(store, OUTPUT_COLUMNS, card_key=t.card_key, trans_ts=t.trans_ts.to_pydatetime(),
                             amt=t.amt, category=t.category, merch_lat=t.merch_lat, merch_long=t.merch_long)
        rows.append(x.reset_index(drop=True))
    got = pd.concat(rows, ignore_index=True)
    keys = list(zip(txns["card_key"], txns["trans_ts"], txns["amt"]))
    want = offline.loc[keys, OUTPUT_COLUMNS].reset_index(drop=True)
    assert len(got) == len(txns) == len(offline)
    pd.testing.assert_frame_equal(got, want, check_exact=True)


def test_api_scores_equal_offline_scores(api, synthetic, offline):
    bundle, store, (_, txns) = api.app.state.bundle, api.app.state.store, synthetic
    keys = list(zip(txns["card_key"], txns["trans_ts"], txns["amt"]))
    X_off = offline.loc[keys, FINAL_COLUMNS]
    raw_off = bundle.booster.predict(X_off, raw_score=True)
    p_off = bundle.booster.predict(X_off)
    for i, t in enumerate(txns.itertuples()):
        x = new_row_features(store, FINAL_COLUMNS, card_key=t.card_key, trans_ts=t.trans_ts.to_pydatetime(),
                             amt=t.amt, category=t.category, merch_lat=t.merch_lat, merch_long=t.merch_long)
        s = score_features(bundle, x)
        assert s.raw_score == raw_off[i]
        assert s.probability_raw == p_off[i]
        assert s.alert == bool(p_off[i] >= bundle.threshold)
    assert np.isfinite(raw_off).all()
