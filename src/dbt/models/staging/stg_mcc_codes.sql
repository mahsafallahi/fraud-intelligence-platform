-- One row per merchant category code, exactly as in Silver (select only, no logic).
select
    mcc,
    description
from {{ source('silver', 'mcc') }}
