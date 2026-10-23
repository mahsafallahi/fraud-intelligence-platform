"""Feature engineering for the fraud model.

Design (decided 2026-10-23, from the Phase 3 findings in docs/eda_findings.md):
- build_features() is a pure function: it takes the model frame and returns one
  row of features per transaction, with the same index. No I/O, so it can be
  unit-tested with tiny fake data and reused unchanged by the API in Phase 6.
- Past-only: every card-level feature uses only that card's STRICTLY EARLIER
  transactions. Two transactions of one card in the same second do not count
  as earlier for each other (the real data has 44 such pairs, none fraud,
  all with different amounts): a real-time scorer could not rely on a
  transaction arriving in the same second, and the result must not depend on
  row order.
- Features are built on the full timeline (so card history continues across
  split boundaries) and only then split. A leakage test enforces past-only:
  removing future rows must not change any past feature value.
- No imputation here: missing values stay NaN. If a model needs imputation,
  it is fitted on the training period only, inside the model pipeline.
- Fails loudly on missing columns, missing keys or unknown categories.
- Excluded on purpose: gender (fairness audit only), trans_num, card_key,
  merchant name, merchant coordinates, anything derived from labels.
"""

import duckdb
import numpy as np
import pandas as pd

CATEGORIES = [
    "entertainment", "food_dining", "gas_transport", "grocery_net", "grocery_pos",
    "health_fitness", "home", "kids_pets", "misc_net", "misc_pos",
    "personal_care", "shopping_net", "shopping_pos", "travel",
]
CHANNELS = ["in_person", "online", "unknown"]

NIGHT_HOURS = [22, 23, 0, 1, 2, 3]
MIN_PRIOR_FOR_MEDIAN = 5

CATEGORICAL_FEATURES = ["category", "channel"]
CARD_HISTORY_FEATURES = ["card_prev_txn_count"]
FEATURE_COLUMNS = [
    "amt", "hour", "is_night", "day_of_week", "category", "channel",
    "secs_since_prev", "amt_vs_card_median", "card_txn_1h", "card_txn_24h",
    "card_prev_txn_count", "age", "city_pop", "median_household_income",
    "has_income", "distance_km", "is_public_holiday_in_state",
    "is_nationwide_public_holiday", "is_bank_holiday",
]

REQUIRED_COLUMNS = [
    "card_key", "trans_ts", "amt", "category", "channel", "birth_year",
    "city_pop", "median_household_income", "has_income", "distance_km",
    "is_public_holiday_in_state", "is_nationwide_public_holiday", "is_bank_holiday",
]
FLAG_COLUMNS = [
    "has_income", "is_public_holiday_in_state",
    "is_nationwide_public_holiday", "is_bank_holiday",
]

MODEL_FRAME_SQL = """
    select f.trans_num, f.card_key, f.trans_ts, f.amt, f.is_fraud,
           f.distance_km, f.is_public_holiday_in_state,
           m.category, m.channel,
           c.gender, c.birth_year,
           l.city_pop, l.median_household_income, l.has_income,
           d.is_nationwide_public_holiday, d.is_bank_holiday
    from fact_transactions f
    join dim_merchant m using (merchant_key)
    join dim_card     c using (card_key)
    join dim_location l using (location_key)
    join dim_date     d using (date_key)
    order by f.trans_ts, f.trans_num
"""


def load_model_frame(db_path) -> pd.DataFrame:
    """Read the wide model frame from the Gold DuckDB file (read-only)."""
    with duckdb.connect(str(db_path), read_only=True) as con:
        return con.sql(MODEL_FRAME_SQL).df()


def _count_earlier_within(ts_ns, group_positions, window) -> np.ndarray:
    """Per row: number of the same card's earlier rows in (t - window, t).

    Rows are sorted by card and time. Same-second rows are fixed afterwards by
    _first_of_tie_block, so only the first row of a tie block must be right.
    """
    width = pd.Timedelta(window).value
    counts = np.zeros(len(ts_ns), dtype="int64")
    for pos in group_positions.values():
        t = ts_ns[pos]
        counts[pos] = np.arange(len(t)) - np.searchsorted(t, t - width, side="right")
    return counts


def _first_of_tie_block(values, is_first) -> np.ndarray:
    """Give every row of a same-card, same-second block the value of its first row.

    The first row of a block only looked at strictly earlier rows, so copying its
    value makes simultaneous transactions invisible to each other.
    """
    values = np.asarray(values)
    starts = np.flatnonzero(is_first)
    block = np.cumsum(is_first) - 1
    return values[starts][block]


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Build model features; returns FEATURE_COLUMNS with the same index as frame."""
    missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    if not frame.index.is_unique:
        raise ValueError("frame index must be unique")
    for col in ["card_key", "trans_ts", "amt", "birth_year"]:
        if frame[col].isna().any():
            raise ValueError(f"missing values in {col}")

    s = frame.assign(trans_ts=pd.to_datetime(frame["trans_ts"]))
    s = s.sort_values(["card_key", "trans_ts"], kind="stable")
    by_card = s.groupby("card_key", sort=False)
    is_first = (~s.duplicated(["card_key", "trans_ts"])).to_numpy()
    out = pd.DataFrame(index=s.index)

    # Transaction itself (categories are checked BEFORE conversion, so the
    # check never depends on how pandas treats unknown values)
    for col, allowed in [("category", CATEGORIES), ("channel", CHANNELS)]:
        has_missing = s[col].isna().any()
        bad = sorted(set(s[col].dropna().astype(str)) - set(allowed))
        if has_missing or bad:
            shown = bad + (["<missing>"] if has_missing else [])
            raise ValueError(f"unknown or missing {col} values: {shown}")
    out["amt"] = s["amt"].astype("float64")
    out["hour"] = s["trans_ts"].dt.hour.astype("int64")
    out["is_night"] = out["hour"].isin(NIGHT_HOURS).astype("int8")
    out["day_of_week"] = s["trans_ts"].dt.dayofweek.astype("int64")
    out["category"] = pd.Categorical(s["category"], categories=CATEGORIES)
    out["channel"] = pd.Categorical(s["channel"], categories=CHANNELS)

    # Card behaviour: strictly earlier transactions only
    secs = by_card["trans_ts"].diff().dt.total_seconds()
    out["secs_since_prev"] = _first_of_tie_block(secs, is_first)
    past_median = by_card["amt"].transform(
        lambda a: a.shift().expanding(min_periods=MIN_PRIOR_FOR_MEDIAN).median()
    )
    out["amt_vs_card_median"] = s["amt"].to_numpy() / _first_of_tie_block(past_median, is_first)
    ts_ns = s["trans_ts"].to_numpy(dtype="datetime64[ns]").astype("int64")
    for name, window in [("card_txn_1h", "1h"), ("card_txn_24h", "24h")]:
        counts = _count_earlier_within(ts_ns, by_card.indices, window)
        out[name] = _first_of_tie_block(counts, is_first)
    out["card_prev_txn_count"] = _first_of_tie_block(by_card.cumcount(), is_first)

    # Customer, location, holidays
    out["age"] = (s["trans_ts"].dt.year - s["birth_year"]).astype("int64")
    out["city_pop"] = s["city_pop"].astype("float64")
    out["median_household_income"] = s["median_household_income"].astype("float64")
    out["distance_km"] = s["distance_km"].astype("float64")
    for col in FLAG_COLUMNS:
        out[col] = s[col].astype("int8")

    return out.loc[frame.index, FEATURE_COLUMNS]