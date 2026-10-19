-- dim_location: one row per ZIP code seen in transactions (985 rows).
-- City, state and city population are fully determined by ZIP.
-- Census data is left-joined: a ZIP missing from the reference keeps its row
-- with null attributes. Missing income stays null, with an explicit flag;
-- imputation is a modelling decision, not a warehouse one.

with locations as (

    select distinct
        zip,
        city,
        state,
        city_pop
    from {{ ref('stg_transactions') }}

),

acs as (

    select
        zip,
        population,
        median_household_income,
        median_age
    from {{ ref('stg_census_acs') }}

),

zcta as (

    select
        zip,
        land_area_m2,
        water_area_m2,
        centroid_lat,
        centroid_long
    from {{ ref('stg_census_zcta') }}

)

select
    md5(locations.zip) as location_key,
    locations.zip,
    locations.city,
    locations.state,
    locations.city_pop,
    acs.population,
    acs.median_household_income,
    acs.median_household_income is not null as has_income,
    acs.median_age,
    zcta.land_area_m2,
    zcta.water_area_m2,
    zcta.centroid_lat,
    zcta.centroid_long
from locations
left join acs
    on locations.zip = acs.zip
left join zcta
    on locations.zip = zcta.zip