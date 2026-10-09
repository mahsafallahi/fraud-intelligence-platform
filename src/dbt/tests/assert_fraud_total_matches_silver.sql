-- The fraud total stays 9,651: same as Silver, and the number in the design.
-- Fails (returns a row) when either differs.
select fact_fraud, silver_fraud
from (
    select
        (select sum(is_fraud) from {{ ref('fact_transactions') }}) as fact_fraud,
        (select sum(is_fraud) from {{ ref('stg_transactions') }}) as silver_fraud
)
where fact_fraud is distinct from silver_fraud
   or fact_fraud <> 9651
