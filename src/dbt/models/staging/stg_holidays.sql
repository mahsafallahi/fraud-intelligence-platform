-- One row per holiday and state (state_code is null for nationwide holidays),
-- exactly as in Silver (select only, no logic).
select
    holiday_date,
    name,
    country_code,
    state_code,
    is_global,
    types,
    _ingested_at,
    _source,
    _year
from {{ source('silver', 'holidays') }}
