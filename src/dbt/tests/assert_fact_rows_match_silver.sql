-- fact_transactions has exactly as many rows as Silver transactions.
-- Fails (returns a row) when the counts differ.
select fact_rows, silver_rows
from (
    select
        (select count(*) from {{ ref('fact_transactions') }}) as fact_rows,
        (select count(*) from {{ ref('stg_transactions') }}) as silver_rows
)
where fact_rows <> silver_rows
