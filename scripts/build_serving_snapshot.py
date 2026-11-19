"""Build the API's serving snapshot from Gold (Phase 6).

    .venv/Scripts/python.exe scripts/build_serving_snapshot.py   # -> data/serving/serving.duckdb

Source (decided 2026-11-19): data/gold/fraud_dev.duckdb, opened READ-ONLY, plus
the cardholder home coordinates from Silver transactions. data/gold/fraud.duckdb
(Airflow's file) is never opened. Gold and dbt are not changed: the state
holiday view int_state_public_holidays reads Silver through relative paths
that only resolve from src/dbt, so the build runs with that working directory.

Tables written (no labels, gender, job, ZIP, merchant names or trans_num):
  card_history  card_key, trans_ts, amt, category, channel, distance_km and the
                three holiday flags, sorted by card and time
  cards         card_key, birth_year, state, city_pop, median_household_income,
                has_income, home_lat, home_long
  calendar, state_holidays, categories, meta

Before the file is moved into place, the build stops unless:
- every card has exactly one home location and one location row;
- the resolver SQL of src/api/resolve.py, evaluated on EVERY historical row,
  reproduces Gold's distance_km and is_public_holiday_in_state exactly;
- the history frame the API reads equals features.MODEL_FRAME_SQL on Gold for
  every row (EXCEPT ALL in both directions, labels and ids left out).
"""

import os
import subprocess
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import duckdb  # noqa: E402

from src.api.resolve import SCHEMA_VERSION, distance_sql, state_holiday_sql  # noqa: E402

GOLD_DEV_DB = PROJECT_ROOT / "data" / "gold" / "fraud_dev.duckdb"
FORBIDDEN_DB = PROJECT_ROOT / "data" / "gold" / "fraud.duckdb"
SILVER_TRANSACTIONS = PROJECT_ROOT / "data" / "silver" / "transactions"
DBT_DIR = PROJECT_ROOT / "src" / "dbt"
OUT = PROJECT_ROOT / "data" / "serving" / "serving.duckdb"

# The model frame of features.MODEL_FRAME_SQL without trans_num, is_fraud, gender
GOLD_FRAME_SQL = """
    select f.card_key, f.trans_ts, f.amt, m.category, m.channel,
           c.birth_year, l.city_pop, l.median_household_income, l.has_income,
           f.distance_km, f.is_public_holiday_in_state,
           d.is_nationwide_public_holiday, d.is_bank_holiday
    from fraud_dev.fact_transactions f
    join fraud_dev.dim_merchant m using (merchant_key)
    join fraud_dev.dim_card     c using (card_key)
    join fraud_dev.dim_location l using (location_key)
    join fraud_dev.dim_date     d using (date_key)
"""
SNAPSHOT_FRAME_SQL = """
    select h.card_key, h.trans_ts, h.amt, h.category, h.channel,
           c.birth_year, c.city_pop, c.median_household_income, c.has_income,
           h.distance_km, h.is_public_holiday_in_state,
           h.is_nationwide_public_holiday, h.is_bank_holiday
    from card_history h join cards c using (card_key)
"""


@contextmanager
def working_directory(path: Path):
    before = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(before)


def one(con, sql: str):
    return con.execute(sql).fetchone()[0]


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"{'ok  ' if ok else 'FAIL'} {label} {detail}".rstrip())
    if not ok:
        raise SystemExit(f"stopped: {label} {detail}")


def git_commit() -> str:
    return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT,
                          capture_output=True, text=True, check=True).stdout.strip()


def build(con) -> None:
    silver = (SILVER_TRANSACTIONS / "**" / "*.parquet").as_posix()
    con.execute(f"""
        create table cards as
        with card_location as (
            select distinct card_key, location_key from fraud_dev.fact_transactions
        ),
        home as (
            select distinct md5(card_hash) as card_key, lat as home_lat, long as home_long
            from read_parquet('{silver}', hive_partitioning = false)
        )
        select c.card_key, c.birth_year, l.state, l.city_pop,
               l.median_household_income, l.has_income, h.home_lat, h.home_long
        from fraud_dev.dim_card c
        join card_location cl using (card_key)
        join fraud_dev.dim_location l using (location_key)
        join home h using (card_key)
        order by c.card_key
    """)
    con.execute("""
        create table card_history as
        select f.card_key, f.trans_ts, f.amt, m.category, m.channel, f.distance_km,
               f.is_public_holiday_in_state, d.is_nationwide_public_holiday, d.is_bank_holiday
        from fraud_dev.fact_transactions f
        join fraud_dev.dim_merchant m using (merchant_key)
        join fraud_dev.dim_date d using (date_key)
        order by f.card_key, f.trans_ts
    """)
    con.execute("""
        create table calendar as
        select calendar_date, is_nationwide_public_holiday, is_bank_holiday
        from fraud_dev.dim_date order by calendar_date
    """)
    con.execute("""
        create table state_holidays as
        select distinct holiday_date, state_code
        from fraud_dev.int_state_public_holidays order by all
    """)
    con.execute("""
        create table categories as
        select category, channel from fraud_dev.category_mcc_map order by category
    """)


def verify(con) -> None:
    n_fact = one(con, "select count(*) from fraud_dev.fact_transactions")
    n_cards = one(con, "select count(*) from fraud_dev.dim_card")
    check("card_history rows = Gold fact rows", one(con, "select count(*) from card_history") == n_fact,
          f"({n_fact:,})")
    check("cards = dim_card rows, one row per card",
          one(con, "select count(*) from cards") == n_cards
          and one(con, "select count(distinct card_key) from cards") == n_cards, f"({n_cards})")
    check("every card has history",
          one(con, "select count(*) from cards where card_key not in (select card_key from card_history)") == 0)
    check("no missing values in cards (except income)",
          one(con, """select count(*) from cards where birth_year is null or state is null
                      or city_pop is null or home_lat is null or home_long is null""") == 0)
    check("categories = the 14 of the seed", one(con, "select count(*) from categories") == 14)

    distance = distance_sql("f.merch_lat", "f.merch_long", "c.home_lat", "c.home_long")
    holiday = state_holiday_sql("d.calendar_date", "c.state", "d.is_nationwide_public_holiday")
    mismatches = con.execute(f"""
        select count(*) filter (where {distance} is distinct from f.distance_km),
               count(*) filter (where {holiday} is distinct from f.is_public_holiday_in_state),
               count(*)
        from fraud_dev.fact_transactions f
        join cards c using (card_key)
        join calendar d on d.calendar_date = cast(f.trans_ts as date)
    """).fetchone()
    check("resolver distance_km = Gold on every row", mismatches[0] == 0 and mismatches[2] == n_fact,
          f"({mismatches[0]} mismatches in {mismatches[2]:,} rows)")
    check("resolver is_public_holiday_in_state = Gold on every row", mismatches[1] == 0,
          f"({mismatches[1]} mismatches)")

    only_gold = one(con, f"select count(*) from ({GOLD_FRAME_SQL} except all {SNAPSHOT_FRAME_SQL})")
    only_snap = one(con, f"select count(*) from ({SNAPSHOT_FRAME_SQL} except all {GOLD_FRAME_SQL})")
    check("snapshot model frame = Gold model frame (both directions)", only_gold == 0 and only_snap == 0,
          f"(only in Gold {only_gold}, only in snapshot {only_snap})")


def main() -> None:
    check("source is not Airflow's fraud.duckdb", GOLD_DEV_DB.resolve() != FORBIDDEN_DB.resolve())
    check("source exists", GOLD_DEV_DB.is_file(), str(GOLD_DEV_DB.relative_to(PROJECT_ROOT)))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".duckdb.tmp")
    tmp.unlink(missing_ok=True)

    with working_directory(DBT_DIR):  # the dbt staging views use paths relative to src/dbt
        con = duckdb.connect(str(tmp))
        try:
            con.execute(f"attach '{GOLD_DEV_DB.as_posix()}' as fraud_dev (read_only)")
            build(con)
            verify(con)
            max_ts = one(con, "select max(trans_ts) from fraud_dev.fact_transactions")
            meta = {
                "schema_version": str(SCHEMA_VERSION),
                "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "builder_git_commit": git_commit(),
                "source_file": "data/gold/fraud_dev.duckdb",
                "source_max_trans_ts": str(max_ts),
                "source_rows": str(one(con, "select count(*) from card_history")),
                "cards": str(one(con, "select count(*) from cards")),
                "home_coordinates": "data/silver/transactions (lat, long per card)",
                "duckdb_version": duckdb.__version__,
            }
            con.execute("create table meta (key varchar primary key, value varchar)")
            con.executemany("insert into meta values (?, ?)", list(meta.items()))
            con.execute("detach fraud_dev")
            con.execute("checkpoint")
        finally:
            con.close()
    os.replace(tmp, OUT)
    print(f"wrote {OUT.relative_to(PROJECT_ROOT)} ({OUT.stat().st_size / 1e6:.1f} MB)")
    for k, v in meta.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
