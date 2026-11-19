"""SQL shared by the serving-snapshot builder and the API (Phase 6).

The API must give build_features() exactly the model frame that Gold gives it
offline (features.MODEL_FRAME_SQL). For a NEW transaction two Gold columns do
not exist yet and are derived here, in DuckDB:
- distance_km: the haversine expression of fact_transactions.sql, copied
  character for character (same constant, same round(.., 3)), and evaluated by
  DuckDB, not numpy, so the same math library computes it.
- is_public_holiday_in_state: nationwide holiday OR a state public holiday for
  the cardholder's state, as in fact_transactions.sql.
The copy is not trusted: scripts/build_serving_snapshot.py evaluates BOTH
expressions on every historical row and stops unless they equal Gold exactly.

Everything else in the frame is a lookup: card profile (cards), calendar
flags (calendar) and the category's channel (categories).
"""

SCHEMA_VERSION = 1


def distance_sql(merch_lat: str, merch_long: str, lat: str, long: str) -> str:
    """The haversine distance of fact_transactions.sql, for the given column names."""
    return f"""round(
        2 * 6371.0088 * asin(least(1.0, sqrt(
            pow(sin(radians({merch_lat} - {lat}) / 2), 2)
            + cos(radians({lat}))
            * cos(radians({merch_lat}))
            * pow(sin(radians({merch_long} - {long}) / 2), 2)
        ))),
        3
    )"""


def state_holiday_sql(day: str, state: str, nationwide: str) -> str:
    """is_public_holiday_in_state for a date, a state and that date's nationwide flag."""
    return f"""(
        {nationwide}
        or exists (select 1 from state_holidays s
                   where s.holiday_date = {day} and s.state_code = {state})
    )"""


# The card's history in model-frame columns (features.REQUIRED_COLUMNS), strictly
# before the new transaction. Same-second rows are left out on purpose: in
# build_features they are invisible to each other anyway, so the new row's
# features are unchanged, and a replayed transaction never sees itself.
HISTORY_SQL = """
    select h.card_key, h.trans_ts, h.amt, h.category, h.channel,
           c.birth_year, c.city_pop, c.median_household_income, c.has_income,
           h.distance_km, h.is_public_holiday_in_state,
           h.is_nationwide_public_holiday, h.is_bank_holiday,
           false as is_new
    from card_history h
    join cards c using (card_key)
    where h.card_key = $card_key and h.trans_ts < $trans_ts
"""

# The new transaction in the same columns, resolved from the snapshot tables
NEW_ROW_SQL = f"""
    select c.card_key, $trans_ts as trans_ts, $amt as amt, k.category, k.channel,
           c.birth_year, c.city_pop, c.median_household_income, c.has_income,
           {distance_sql("$merch_lat", "$merch_long", "c.home_lat", "c.home_long")} as distance_km,
           {state_holiday_sql("d.calendar_date", "c.state", "d.is_nationwide_public_holiday")}
               as is_public_holiday_in_state,
           d.is_nationwide_public_holiday, d.is_bank_holiday,
           true as is_new
    from cards c
    cross join (select * from calendar where calendar_date = cast($trans_ts as date)) d
    cross join (select * from categories where category = $category) k
    where c.card_key = $card_key
"""

FRAME_SQL = f"""
    select * from ({HISTORY_SQL})
    union all
    select * from ({NEW_ROW_SQL})
"""


def frame_params(card_key, trans_ts, amt, category, merch_lat, merch_long) -> dict:
    """Named parameters for FRAME_SQL, with the SQL types Gold uses."""
    return {"card_key": card_key, "trans_ts": trans_ts, "amt": float(amt),
            "category": category, "merch_lat": float(merch_lat), "merch_long": float(merch_long)}
