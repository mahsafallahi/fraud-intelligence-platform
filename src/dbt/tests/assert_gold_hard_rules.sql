-- Hard data-quality rules for the Gold layer: any violation fails the build.
-- Each block returns one row (rule name + number of bad rows) only when its
-- rule is broken, so the test passes when the whole query returns nothing.

with fact as (select * from {{ ref('fact_transactions') }}),
cards as (select * from {{ ref('dim_card') }}),
locations as (select * from {{ ref('dim_location') }}),
merchants as (select * from {{ ref('dim_merchant') }})

select 'amount is not positive' as rule, count(*) as bad_rows
from fact where amt <= 0
having count(*) > 0

union all
select 'transaction outside 2019-2020', count(*)
from fact where trans_ts < timestamp '2019-01-01' or trans_ts >= timestamp '2021-01-01'
having count(*) > 0

union all
select 'row in the wrong daily partition', count(*)
from fact where cast(trans_ts as date) <> _batch_date
having count(*) > 0

union all
select 'is_fraud is not 0 or 1', count(*)
from fact where is_fraud not in (0, 1)
having count(*) > 0

union all
select 'distance missing or outside 0-200 km', count(*)
from fact where distance_km is null or distance_km < 0 or distance_km > 200
having count(*) > 0

union all
select 'merchant point outside the U.S. box', count(*)
from fact
where merch_lat not between 17 and 72 or merch_long not between -180 and -60
having count(*) > 0

union all
select 'gender is not F or M', count(*)
from cards where gender is null or gender not in ('F', 'M')
having count(*) > 0

union all
select 'birth year implausible', count(*)
from cards where birth_year not between 1900 and 2010
having count(*) > 0

union all
select 'has_income disagrees with income', count(*)
from locations where has_income <> (median_household_income is not null)
having count(*) > 0

union all
select 'negative population', count(*)
from locations where population < 0
having count(*) > 0

union all
select 'merchant without MCC or description', count(*)
from merchants where mcc is null or mcc_description is null
having count(*) > 0