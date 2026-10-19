-- dim_merchant: one row per merchant name + category (700 rows).
-- Seven names appear in two unrelated categories, so the name alone is not
-- a key. MCC and channel come from the 14-row seed; the MCC description is
-- joined from the reference table, never typed by hand.

with merchants as (

    select distinct
        merchant,
        category
    from {{ ref('stg_transactions') }}

),

mapping as (

    select
        category,
        mcc,
        channel
    from {{ ref('category_mcc_map') }}

),

codes as (

    select
        mcc,
        description
    from {{ ref('stg_mcc_codes') }}

)

select
    md5(merchants.merchant || '|' || merchants.category) as merchant_key,
    merchants.merchant,
    merchants.category,
    mapping.mcc,
    codes.description as mcc_description,
    mapping.channel
from merchants
left join mapping
    on merchants.category = mapping.category
left join codes
    on mapping.mcc = codes.mcc