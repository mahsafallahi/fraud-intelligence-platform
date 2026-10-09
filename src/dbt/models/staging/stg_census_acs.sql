-- One row per ZIP area and vintage, exactly as in Silver (select only, no logic).
select
    zip,
    state_fips,
    population,
    median_household_income,
    median_age,
    _ingested_at,
    _source,
    _vintage
from {{ source('silver', 'census_acs') }}
