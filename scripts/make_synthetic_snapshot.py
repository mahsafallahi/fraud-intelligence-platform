"""Write a small synthetic serving snapshot for tests and the CI container smoke test.

    python scripts/make_synthetic_snapshot.py build/synthetic.duckdb

Same tables, columns and SQL types as scripts/build_serving_snapshot.py, but
made-up data: three cards, deterministic (fixed seed), no real person. History
rows get distance_km and the state-holiday flag from the SAME resolver SQL
(src/api/resolve.py) the API uses for new rows, so features computed by the API
can be compared exactly with build_features() on the whole synthetic frame.

Edge cases on purpose: a same-second pair, bursts inside 1 h and 24 h, a card
with fewer than 5 earlier transactions (no median yet), a card without
income, a state public holiday and a nationwide holiday.
"""

import csv
import hashlib
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import duckdb  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.api.resolve import SCHEMA_VERSION, distance_sql, state_holiday_sql  # noqa: E402
from src.models.features import CATEGORIES  # noqa: E402

SEED_CSV = PROJECT_ROOT / "src" / "dbt" / "seeds" / "category_mcc_map.csv"


def card_key(name: str) -> str:
    return hashlib.md5(name.encode("utf-8")).hexdigest()


CARDS = pd.DataFrame([
    {"card_key": card_key("synthetic-card-A"), "birth_year": 1961, "state": "PA", "city_pop": 1200,
     "median_household_income": 54000, "has_income": True, "home_lat": 40.30, "home_long": -76.10},
    {"card_key": card_key("synthetic-card-B"), "birth_year": 1990, "state": "NY", "city_pop": 250000,
     "median_household_income": 71000, "has_income": True, "home_lat": 42.70, "home_long": -73.80},
    {"card_key": card_key("synthetic-card-C"), "birth_year": 1978, "state": "TX", "city_pop": 48000,
     "median_household_income": None, "has_income": False, "home_lat": 31.10, "home_long": -97.40},
])
NATIONWIDE = {"2020-11-26": True, "2020-12-25": True}   # also bank holidays here
STATE_HOLIDAYS = [("2020-10-12", "PA"), ("2020-11-03", "TX")]


def transactions() -> pd.DataFrame:
    rng = np.random.default_rng(20261119)
    rows = []
    plan = {"synthetic-card-A": 40, "synthetic-card-B": 4, "synthetic-card-C": 14}
    start = pd.Timestamp("2020-10-01 08:00:00")
    for (name, n), home in zip(plan.items(), CARDS.itertuples()):
        gaps = rng.choice([30, 300, 2400, 7200, 50000, 120000], size=n, p=[.15, .15, .2, .2, .15, .15])
        ts = start + pd.to_timedelta(np.cumsum(gaps), unit="s")
        for i, t in enumerate(ts):
            rows.append({"card_key": card_key(name), "trans_ts": t.floor("s"),
                         "amt": float(round(rng.lognormal(3.5, 1.2), 2)),
                         "category": CATEGORIES[int(rng.integers(len(CATEGORIES)))],
                         "merch_lat": float(round(home.home_lat + rng.normal(0, 0.6), 6)),
                         "merch_long": float(round(home.home_long + rng.normal(0, 0.6), 6))})
            if name == "synthetic-card-A" and i == 10:  # same-second pair, different amount
                rows.append({**rows[-1], "amt": rows[-1]["amt"] + 7.5})
    # land a few transactions on the holidays
    out = pd.DataFrame(rows)
    out.loc[5, "trans_ts"] = pd.Timestamp("2020-10-12 13:00:00")
    out = out.sort_values(["card_key", "trans_ts"], kind="stable", ignore_index=True)
    return out


def make_synthetic_snapshot(path) -> pd.DataFrame:
    """Write the snapshot to path; return the transactions (with merchant coordinates)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)
    txns = transactions()
    with open(SEED_CSV, newline="", encoding="utf-8") as f:
        categories = pd.DataFrame([{"category": r["category"], "channel": r["channel"]}
                                   for r in csv.DictReader(f)])
    days = pd.DataFrame({"calendar_date": pd.date_range("2019-01-01", "2020-12-31", freq="D")})
    days["is_nationwide_public_holiday"] = days["calendar_date"].dt.strftime("%Y-%m-%d").isin(NATIONWIDE)
    days["is_bank_holiday"] = days["is_nationwide_public_holiday"]
    state_holidays = pd.DataFrame(STATE_HOLIDAYS, columns=["holiday_date", "state_code"])

    con = duckdb.connect(str(path))
    try:
        for name, df in [("cards_df", CARDS), ("txns_df", txns), ("categories_df", categories),
                         ("days_df", days), ("state_df", state_holidays)]:
            con.register(name, df)
        con.execute("""create table cards as select card_key, cast(birth_year as smallint) as birth_year,
                       state, cast(city_pop as bigint) as city_pop,
                       cast(median_household_income as bigint) as median_household_income,
                       cast(has_income as boolean) as has_income,
                       cast(home_lat as double) as home_lat, cast(home_long as double) as home_long
                       from cards_df order by card_key""")
        con.execute("create table categories as select category, channel from categories_df order by category")
        con.execute("""create table calendar as select cast(calendar_date as date) as calendar_date,
                       is_nationwide_public_holiday, is_bank_holiday from days_df order by 1""")
        con.execute("""create table state_holidays as select cast(holiday_date as date) as holiday_date,
                       state_code from state_df order by all""")
        dist = distance_sql("t.merch_lat", "t.merch_long", "c.home_lat", "c.home_long")
        hol = state_holiday_sql("d.calendar_date", "c.state", "d.is_nationwide_public_holiday")
        con.execute(f"""
            create table card_history as
            select t.card_key, cast(t.trans_ts as timestamp) as trans_ts, cast(t.amt as double) as amt,
                   t.category, k.channel, {dist} as distance_km, {hol} as is_public_holiday_in_state,
                   d.is_nationwide_public_holiday, d.is_bank_holiday
            from txns_df t
            join cards c using (card_key)
            join categories k using (category)
            join calendar d on d.calendar_date = cast(t.trans_ts as date)
            order by t.card_key, t.trans_ts
        """)
        con.execute("create table meta (key varchar primary key, value varchar)")
        con.executemany("insert into meta values (?, ?)", [
            ("schema_version", str(SCHEMA_VERSION)), ("source_file", "synthetic"),
            ("built_at", "synthetic"), ("source_rows", str(len(txns))), ("cards", str(len(CARDS)))])
        con.execute("checkpoint")
    finally:
        con.close()
    return txns


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: make_synthetic_snapshot.py <output.duckdb>")
    txns = make_synthetic_snapshot(sys.argv[1])
    print(f"wrote {sys.argv[1]}: {len(txns)} transactions, {len(CARDS)} cards")
    print(f"card A: {card_key('synthetic-card-A')}")
