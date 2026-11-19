"""The real serving snapshot against Gold (local only: marker gold).

The builder already compares every row with Gold before writing the file;
these tests re-check the finished file and the API's read path on it.
"""

from datetime import datetime

import duckdb
import pytest

from src.api.history import DateOutsideCalendar, SnapshotStore, UnknownCard
from src.models.features import MODEL_FRAME_SQL, REQUIRED_COLUMNS
from tests.conftest import GOLD_DEV_DB, SERVING_DB

pytestmark = pytest.mark.gold


@pytest.fixture(scope="module")
def store():
    s = SnapshotStore(SERVING_DB)
    yield s
    s.close()


@pytest.fixture(scope="module")
def gold():
    con = duckdb.connect(str(GOLD_DEV_DB), read_only=True)
    yield con
    con.close()


def test_meta_records_the_source(store, gold):
    assert store.meta["source_file"] == "data/gold/fraud_dev.duckdb"
    max_ts = gold.execute("select max(trans_ts) from fact_transactions").fetchone()[0]
    assert store.meta["source_max_trans_ts"] == str(max_ts)
    assert int(store.meta["source_rows"]) == gold.execute(
        "select count(*) from fact_transactions").fetchone()[0]


def test_snapshot_holds_no_labels_or_identifiers(store):
    # one process may hold only one configuration per DuckDB file: reuse the store's connection
    cur = store._con.cursor()
    cols = {r[0] for r in cur.execute("select column_name from information_schema.columns").fetchall()}
    assert not cols & {"is_fraud", "gender", "job", "trans_num", "zip", "merchant", "card_hash"}


def test_model_frame_equals_gold_history_for_a_real_transaction(store, gold):
    row = gold.execute("""
        select f.trans_num, f.card_key, f.trans_ts, f.amt, m.category, f.merch_lat, f.merch_long
        from fact_transactions f join dim_merchant m using (merchant_key)
        where f.trans_ts >= '2020-10-01' order by f.trans_num limit 1
    """).fetchone()
    trans_num, card_key, ts, amt, category, mlat, mlong = row
    frame = store.model_frame(card_key, ts, amt, category, mlat, mlong)
    gold_frame = gold.execute(
        f"select * from ({MODEL_FRAME_SQL}) where card_key = ? and trans_ts <= ?", [card_key, ts]).df()
    # Gold rows strictly earlier + the transaction itself (same-second others excluded)
    expected = gold_frame[(gold_frame["trans_ts"] < ts) | (gold_frame["trans_num"] == trans_num)]
    got = frame.sort_values("trans_ts", kind="stable").reset_index(drop=True)[REQUIRED_COLUMNS]
    want = expected.sort_values("trans_ts", kind="stable").reset_index(drop=True)[REQUIRED_COLUMNS]
    assert got.equals(want)
    assert frame["is_new"].sum() == 1


def test_unknown_card_and_date_outside_calendar(store, gold):
    card = gold.execute("select card_key from dim_card limit 1").fetchone()[0]
    with pytest.raises(UnknownCard):
        store.model_frame("0" * 32, datetime(2020, 11, 1, 12), 10.0, "travel", 40.0, -75.0)
    with pytest.raises(DateOutsideCalendar):
        store.model_frame(card, datetime(2021, 1, 1, 0, 0, 1), 10.0, "travel", 40.0, -75.0)
