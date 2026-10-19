-- int_state_public_holidays: one row per (date, state) that has a
-- state-level PUBLIC holiday. One state can have two holiday records on the
-- same day, so the rows are collapsed here; joining the raw rows to the
-- fact table would duplicate 948 transactions.
-- Observance, Optional and Authorities/School days are not days off: excluded.

select
    holiday_date,
    state_code,
    string_agg(name, ' / ' order by name) as state_holiday_name
from {{ ref('stg_holidays') }}
where state_code is not null
  and contains(types, 'Public')
group by
    holiday_date,
    state_code