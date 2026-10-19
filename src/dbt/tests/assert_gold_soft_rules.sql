{{ config(severity='warn') }}

-- Soft data-quality rules: a violation does not mean the data is broken,
-- but something changed and a person should look. They warn, never fail.
-- Bands are set around today's values (fraud 0.521%, no-income share 6.7%).

with fact as (select * from {{ ref('fact_transactions') }}),
locations as (select * from {{ ref('dim_location') }}),
dates as (select * from {{ ref('dim_date') }})

select 'overall fraud rate outside 0.3%-1%' as rule, avg(is_fraud) as observed
from fact
having avg(is_fraud) not between 0.003 and 0.01

union all
select 'share of transactions without income outside 5%-8%',
       avg(case when not locations.has_income then 1.0 else 0.0 end)
from fact
join locations on fact.location_key = locations.location_key
having avg(case when not locations.has_income then 1.0 else 0.0 end) not between 0.05 and 0.08

union all
select 'days without transactions (other than 2020-02-29)', count(*)
from dates
left join (select distinct date_key from fact) as days_with_data
    on dates.date_key = days_with_data.date_key
where days_with_data.date_key is null
  and dates.calendar_date <> date '2020-02-29'
having count(*) > 0