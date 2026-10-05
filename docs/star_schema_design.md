# Star Schema Design (Gold layer)

Status: design approved on 2026-10-05. Implementation (dbt) not started.
Evidence for every number below: `notebooks/02_star_schema_design.ipynb`,
run on the full Silver layer (731 daily partitions).

## Grain

One row in the fact table = one card transaction (1,852,394 rows, 9,651 fraud).

## Tables

| Table | Grain | Rows | Columns from Silver | Added in Gold |
|---|---|---|---|---|
| fact_transactions | one transaction | 1,852,394 | trans_num, trans_ts, amt, is_fraud, merch_lat, merch_long, _batch_date | card_key, merchant_key, location_key, date_key, distance_km, is_public_holiday_in_state |
| dim_card | one card | 999 | card_hash, gender, birth_year, job | card_key |
| dim_merchant | one merchant name + category | 700 | merchant, category | merchant_key, MCC code and description (from a 14-row seed) |
| dim_location | one ZIP code seen in transactions | 985 | zip, city, state, city_pop | location_key, population, median_household_income, has_income, median_age, land_area_m2, water_area_m2, centroid_lat, centroid_long |
| dim_date | one calendar day | 731 | derived from trans_ts | date_key, day of week, month, year, is_weekend, is_nationwide_public_holiday, is_bank_holiday |

Not carried into Gold: customer lat/long (Silver only; Gold exposes distance),
_ingested_at, _source, _source_file.

`trans_num` stays in the fact table as a degenerate dimension (the business
identifier of the transaction, used for tracing and uniqueness tests).

## Decisions and the evidence behind them

### 1. Card attributes are stable, so dim_card is a simple (Type 1) dimension
For all 999 cards, gender, birth year, job, city, state and ZIP never change
in the two years of data. No history tracking is needed.

### 2. Location attributes belong to the ZIP code, not to the card
City, state, coordinates and city population are fully determined by ZIP
(985 ZIPs; 14 ZIPs are shared by two cards, 985 + 14 = 999).
The customer coordinates in the source are a ZIP-level point, not a home address.

### 3. Star, not snowflake: the fact table carries its own location key
Rejected: fact -> card -> location. With a direct key, "fraud rate by state"
is one join, and a transaction keeps the customer's location as of that
moment even if a card's address changes in the future. The dimensions are
tiny, so snowflaking would save nothing.

### 4. Merchant identity is name + category
There are 700 merchant/category pairs but only 693 distinct names. Every one
of the 14 categories has exactly 50 merchants. Seven names appear in two
unrelated categories, both active for the whole period and with very
different fraud rates. A dimension keyed on name alone, with category as an
attribute, would duplicate 35,242 fact rows on join.
Assumption (cannot be proven in simulated data): these are different
businesses that share a generated name. Real data would have a merchant ID.

### 5. Merchant coordinates stay at the fact grain
One merchant has up to 6,262 different coordinate pairs: the simulator draws
a new point for every transaction. They are an attribute of the transaction,
not of the merchant.

### 6. dim_location is built from transaction ZIPs, with reference data left-joined
All 985 ZIPs exist in both Census tables. If a future ZIP is missing from the
reference data, the transaction is kept and its location attributes are null.
Rejected: loading all 33,144 U.S. ZIP areas (97% would never be used).

### 7. Missing income stays null, with an explicit flag
57 of 985 ZIPs have no median household income. They cover 124,334
transactions (6.7%) and 539 fraud cases. The missingness is not random:
39% of these ZIPs have under 100 residents, versus 1.6% of the others
(median population 147 vs 3,333). Imputation is a modelling decision and is
done later, inside the training pipeline, not in the warehouse.
Also: 12 ZIPs have no median age and 8 have zero population.

### 8. Holidays: two flags, collapsed before joining
Silver holidays has 160 rows (20 nationwide, 140 state level) and several
types. Only `Public` and `Bank` are flagged; `Observance`, `Optional` and
`Authorities,School` are not days off.
- Nationwide holidays (20 rows on 20 distinct dates) go into dim_date as
  is_nationwide_public_holiday and is_bank_holiday.
- State-level holidays are first collapsed to one row per (date, state),
  then joined to set is_public_holiday_in_state on the fact table
  (nationwide public OR state public).
A direct join to the raw holiday rows would duplicate 948 transactions,
because one state can have two holiday records on the same day.
Prototype result: 0 extra rows; 52,496 transactions on a public holiday
(46,917 nationwide + 5,579 state level, no overlap); 53,247 on a bank holiday.
Rejected: a bridge table left to analysts (easy to fan out), and a
date x state dimension of 37,281 rows (too heavy for one flag).

## Known limitations

- The holiday flag uses the customer's home state; the merchant's state is
  not in the data.
- The dataset is simulated (Sparkov). Reference data (Census, holidays, MCC)
  is real.
- Census income is censored at 2,499 and 250,001; none of the 985
  transaction ZIPs is affected.

## Checks the implementation must pass

- fact_transactions has exactly as many rows as Silver transactions.
- trans_num is unique; every foreign key finds exactly one dimension row.
- Row counts: dim_card 999, dim_merchant 700, dim_location 985, dim_date 731.
- Fraud total stays 9,651.

## Open items

- The 14-row category -> MCC mapping (dbt seed).
- Tooling setup for the Gold layer.