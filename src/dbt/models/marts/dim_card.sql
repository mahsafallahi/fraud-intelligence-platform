-- dim_card: one row per card (Type 1, attributes never change in the data).
-- If an attribute ever changes, a card gets two rows and the unique test
-- on card_hash fails: that is the signal to add history tracking.

with cards as (

    select distinct
        card_hash,
        gender,
        birth_year,
        job
    from {{ ref('stg_transactions') }}

)

select
    md5(card_hash) as card_key,
    card_hash,
    gender,
    birth_year,
    job
from cards