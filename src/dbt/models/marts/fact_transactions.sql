-- fact_transactions: one row per card transaction (same row count as Silver).
-- Keys use the same expressions as the dimensions (md5 of the natural key),
-- so no join to the dimensions is needed; relationship tests prove that every
-- key finds exactly one dimension row.
-- Customer coordinates stay in Silver: Gold exposes only the distance.
-- Both joins below are to one-row-per-key tables, so rows cannot fan out.

with transactions as (

    select *
    from {{ ref('stg_transactions') }}

),

dates as (

    select
        date_key,
        calendar_date,
        is_nationwide_public_holiday
    from {{ ref('dim_date') }}

),

state_holidays as (

    select
        holiday_date,
        state_code
    from {{ ref('int_state_public_holidays') }}

)

select
    transactions.trans_num,
    md5(transactions.card_hash) as card_key,
    md5(transactions.merchant || '|' || transactions.category) as merchant_key,
    md5(transactions.zip) as location_key,
    dates.date_key,
    transactions.trans_ts,
    transactions.amt,
    transactions.is_fraud,
    transactions.merch_lat,
    transactions.merch_long,
    round(
        2 * 6371.0088 * asin(least(1.0, sqrt(
            pow(sin(radians(transactions.merch_lat - transactions."lat") / 2), 2)
            + cos(radians(transactions."lat"))
            * cos(radians(transactions.merch_lat))
            * pow(sin(radians(transactions.merch_long - transactions."long") / 2), 2)
        ))),
        3
    ) as distance_km,
    (
        dates.is_nationwide_public_holiday
        or state_holidays.holiday_date is not null
    ) as is_public_holiday_in_state,
    transactions._batch_date
from transactions
left join dates
    on cast(transactions.trans_ts as date) = dates.calendar_date
left join state_holidays
    on cast(transactions.trans_ts as date) = state_holidays.holiday_date
    and transactions.state = state_holidays.state_code