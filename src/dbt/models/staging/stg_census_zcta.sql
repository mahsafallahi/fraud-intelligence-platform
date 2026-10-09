-- One row per ZIP area and vintage, exactly as in Silver (select only, no logic).
select
    zip,
    land_area_m2,
    water_area_m2,
    centroid_lat,
    centroid_long,
    _ingested_at,
    _source,
    _content_sha256,
    _vintage
from {{ source('silver', 'census_zcta') }}
