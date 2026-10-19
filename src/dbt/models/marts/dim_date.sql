-- dim_date: generated calendar, 2019-01-01 to 2020-12-31 (731 rows).
-- It is NOT derived from transaction dates: 2020-02-29 has no transactions
-- but must exist. Nationwide holidays are collapsed to one row per date
-- before joining, so the calendar can never fan out.
-- date_key is the readable integer YYYYMMDD (the usual convention for dates).

with calendar as (

    select cast(day as date) as calendar_date
    from generate_series(
        date '2019-01-01',
        date '2020-12-31',
        interval 1 day
    ) as t(day)

),

nationwide as (

    select
        holiday_date,
        bool_or(contains(types, 'Public')) as is_nationwide_public_holiday,
        bool_or(contains(types, 'Bank')) as is_bank_holiday,
        string_agg(name, ' / ' order by name) as nationwide_holiday_name
    from {{ ref('stg_holidays') }}
    where state_code is null
    group by holiday_date

)

select
    cast(strftime(calendar.calendar_date, '%Y%m%d') as integer) as date_key,
    calendar.calendar_date,
    year(calendar.calendar_date) as calendar_year,
    month(calendar.calendar_date) as calendar_month,
    day(calendar.calendar_date) as day_of_month,
    isodow(calendar.calendar_date) as day_of_week,
    dayname(calendar.calendar_date) as day_name,
    isodow(calendar.calendar_date) in (6, 7) as is_weekend,
    coalesce(nationwide.is_nationwide_public_holiday, false) as is_nationwide_public_holiday,
    coalesce(nationwide.is_bank_holiday, false) as is_bank_holiday,
    nationwide.nationwide_holiday_name
from calendar
left join nationwide
    on calendar.calendar_date = nationwide.holiday_date