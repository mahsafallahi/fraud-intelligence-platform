-- dim_date holds every day from 2019-01-01 to 2020-12-31 exactly once
-- (731 days): right first and last day, no gaps, no duplicates.
-- Assumes the date column is named calendar_date.
-- Fails (returns a row) otherwise.
with stats as (
    select
        min(calendar_date) as first_day,
        max(calendar_date) as last_day,
        count(*) as row_count,
        count(distinct calendar_date) as distinct_days
    from {{ ref('dim_date') }}
)
select *
from stats
where first_day is distinct from date '2019-01-01'
   or last_day is distinct from date '2020-12-31'
   or distinct_days <> 731
   or row_count <> 731
