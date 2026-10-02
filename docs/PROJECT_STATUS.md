# Fraud Intelligence Platform — Project Status

**Last updated:** 2026-10-02
**Current step:** Phase 1 complete (Bronze + Silver + Airflow for all four sources). Next: Phase 2, star-schema warehouse with dbt and data quality with Great Expectations.

This file is the single source of truth for where the project stands. It is updated at the end of every working session.

---

## 1. Goal

An end-to-end financial fraud detection platform that demonstrates both **Data Engineering** (ingestion, orchestration, warehouse, data quality) and **Data Science** (statistics, causal inference, modelling, explainability), plus GenAI and MLOps layers.

---

## 2. Phase plan

| # | Phase | Main role | Status |
|---|---|---|---|
| 1 | Ingestion, Bronze and Silver layers, Airflow | Data Engineering | ✅ Complete: Bronze ✅, Silver ✅, Airflow ✅ (transactions, holidays, Census ACS, Census Gazetteer) |
| 2 | Star-schema warehouse (dbt) and data quality (Great Expectations) | Data Engineering | ⬜ Not started |
| 3 | Exploratory analysis and hypothesis testing | Data Science | ⬜ Not started |
| 4 | Fraud models: imbalance handling, SHAP, MLflow | Data Science | ⬜ Not started |
| 5 | Causal inference on fraud risk drivers | Data Science | ⬜ Not started |
| 6 | FastAPI service, Docker, CI/CD | MLOps | ⬜ Not started |
| 7 | RAG over AML regulations and an agentic decision layer | GenAI | ⬜ Not started |
| 8 | Dashboard, deployment, documentation | All | ⬜ Not started |

Each phase ends with its own commit, e.g. `Phase 1: ...`.

---

## 3. Data sources

| Source | Type | Coverage | Notes |
|---|---|---|---|
| Card transactions (Kaggle, kartik2112) | CSV → daily batches | 2019-01-01 to 2020-12-31, 1,852,394 rows, 9,651 fraud | Simulated with Sparkov (CC0 licence). Real bank data is never published for privacy reasons. |
| Public holidays (Nager.Date API) | REST API, no key | US, 2019–2020 | Enriches the date dimension. |
| Merchant category codes (MCC) | Open CSV (GitHub `greggles/mcc-codes`, Unlicense) | 981 codes | Pinned to commit `9675cfa`. Transactions have 14 `category` values, not MCC codes. A category → MCC mapping file will be written by hand as a dbt seed in phase 2. |
| Census ZIP areas (Gazetteer 2019) | Zip file, no key | 33,144 ZIP areas | Land/water area and centre point per ZIP. |
| Census ACS demographics (Census API) | REST API, key in `.env` | 33,120 ZIP areas (ACS 5-year, 2019) | Population, median household income, median age. Uses the sentinel `-666666666` for "not available". |

---

## 4. Storage conventions

- `data/raw/` — original downloads, never modified.
- `data/landing/` — daily files produced by the feed simulator (plays the role of the bank).
- `data/bronze/<source>/...` — data exactly as received, in Parquet, partitioned: transactions by `batch_date`, holidays by `country`/`year`, MCC by `version` (commit), Census Gazetteer and ACS by `vintage`.
- `data/silver/<source>/...` — cleaned, typed and validated data, one Parquet file per partition, same partitioning as Bronze: `transactions/batch_date=YYYY-MM-DD/`, `census_acs/vintage=YYYY/`, `census_zcta/vintage=YYYY/`, `holidays/country=XX/year=YYYY/`. MCC has no Silver layer (its category mapping comes as a dbt seed). Written by `src/pipeline/silver_jobs.py`: read one Bronze partition file, run the pure transform from `src/silver/`, write atomically; the transforms' dtypes (including nullable `Int64` and `date32`) are kept exactly.
- `data/gold/` — modelled layer (to come in phase 2).
- All data folders are git-ignored; only `.gitkeep` files are committed.
- Bronze keeps every business column as a string (CSV/file sources) or the raw response body in one `response_body` column (API sources: holidays, Census ACS), and adds `_`-prefixed metadata. All sources have `_ingested_at` (UTC) and `_source`; the rest depends on the source:
  - transactions: `_source_file`, `_batch_date`
  - holidays: `_request_url`, `_country`, `_year`
  - MCC: `_source_url`, `_content_sha256`
  - Census Gazetteer: `_source_url`, `_source_file`, `_content_sha256`
  - Census ACS: `_request_url` (API key removed), `_vintage`
- Writes are atomic: write to a temporary file, then rename. Each run overwrites its own partition, so re-runs never duplicate data.
- API/download sources are validated before writing, so a bad response never overwrites a good partition.

---

## 5. Orchestration (Airflow)

- Airflow 3.3.2 in Docker (custom image on the `slim` base, Python 3.12): api-server (UI and REST API; replaces the webserver in Airflow 3), scheduler, dag-processor, Postgres, and a one-off `airflow-init` that migrates the metadata DB. LocalExecutor (tasks run inside the scheduler, at most 4 at once), no Celery or Redis.
- UI at `http://localhost:8080`, no login. Acceptable only because it is bound to `127.0.0.1` for local development.
- `src/` and `data/` are mounted into the containers; code changes need no image rebuild.
- DAG `bronze_transactions_daily`: daily data-interval schedule (the run for day D starts after D ends) from 2019-01-01 to 2020-12-31 with catch-up; at most 4 runs at once. `ingest_transactions` (Bronze; 3 retries, 5 minutes apart) → `silver_transactions` (Silver for the same day; loads the PII key once per run).
- DAG `bronze_reference_data`: manual trigger only; holidays, MCC, Census ZIP and Census ACS run in parallel (2 retries, 2 minutes apart), each followed by its Silver task (`silver_holidays`, `silver_census_zcta`, `silver_census_acs`).
- Silver tasks: 1 retry (1 minute) for transient problems; a `SilverValidationError` is converted to `AirflowFailException` (`airflow.sdk.exceptions`) and fails at once.
- Full Bronze backfill completed: 731 daily runs, all successful, row counts verified against raw.
- Historical Silver built on 2026-10-02 inside Airflow (REST clear with `only_new`, see Key decisions): 731 transaction runs + the reference run, every task successful on its first try, Bronze byte-identical before and after. Verified:
  - transactions: 731 partitions, 1,852,394 rows, 9,651 fraud, 0 days where Silver rows ≠ Bronze rows, 999 distinct `card_hash` = 999 distinct Bronze `cc_num`, every `card_hash` 64 hex characters, every `zip` 5 digits, 0 duplicate `trans_num`, 1 schema across all partitions and equal to `SILVER_SCHEMA`, 2020-02-29 = 0 rows with the full schema;
  - Census ACS: 33,120 rows, 2,299 null `median_household_income`, 569 null `median_age`; Census Gazetteer: 33,144 rows; holidays: 160 rows (2019 + 2020);
  - `PII_HASH_KEY` and `CENSUS_API_KEY`: in 0 of 1,477 Airflow log files, 0 lines of container output, 0 of 735 Silver files.
- Memory while running: about 1.1 GB total for Bronze, about 1.2 GB for Silver (scheduler about 920 MiB), well under the per-container limits (5.5 GB in total).

---

## 6. Secrets

Stored only in `.env`, which is git-ignored. `.env.example` lists the names with empty values.

- `CENSUS_API_KEY` — Census API. Docker Compose passes it only to the scheduler, as the Airflow Variable `census_api_key`.
- `AIRFLOW_FERNET_KEY`, `AIRFLOW_JWT_SECRET` — generated for Airflow (Fernet encrypts connection passwords in the metadata DB; the JWT secret lets the Airflow 3 components trust each other).
- `PII_HASH_KEY` — secret key for hashing personal data in Silver: 64 hex characters, loaded as 32 bytes with `bytes.fromhex` by `load_pii_key()`; error messages never include the value. Docker Compose passes it only to the scheduler, as the Airflow Variable `pii_hash_secret`.

Keys are never printed in output or logs. The Census API only accepts the key as a URL parameter, so the code removes it from the stored URL, re-raises request errors with the key replaced by `***`, and attaches a redacting filter to the urllib3 loggers; in Airflow the key is also registered with `mask_secret`.

---

## 7. Key decisions

| Decision | Reason |
|---|---|
| Fraud detection as the domain | Stable, high demand in finance; lets one project show both DE and DS. |
| Kartik2112 dataset instead of the ULB credit card dataset | ULB columns are PCA-anonymised (V1–V28), so no star schema or causal analysis is possible. |
| Several sources, not one CSV | A warehouse's value is integrating different sources (file, API, reference data). |
| Daily batches with a landing-zone simulator | Simulates a real daily bank feed and incremental ingestion. |
| Train and test files merged into one stream | A real feed has no train/test split; it is recreated later with a date filter. |
| Bronze stores every column as a string (API sources: the raw response body) | Allows reprocessing, avoids silent corruption (ZIP leading zeros, card number precision), survives source format changes; a raw body also keeps the Bronze schema stable when an API field is null for a whole year. |
| Parquet instead of CSV | Columnar and compressed: 478 MiB of CSV became 234 MiB of Parquet. |
| Reference sources pinned to a version (MCC commit, Census vintage) | Every run downloads identical input; upgrading is a deliberate change and the old version stays in its own partition. |
| Data-interval schedule for the daily DAG | A run processes a day only after it has ended, never a day still in progress. |
| Airflow `slim` image plus Airflow's constraints file | 268 MB download instead of 670 MB; the constraints file pins every dependency to versions Airflow has tested. |
| Empty file for a day with no data | "Missing file" means an upstream outage and fails; "empty file" means the source confirmed no data. |
| Census Gazetteer now, Census API with a key afterwards | Progress was not blocked by the key; demographics add value for causal analysis. |
| Separate `requirements.txt` and `requirements-dev.txt` | Keeps the runtime Docker image smaller. |
| Card number hashed with a secret key; names and street dropped | The card number is the customer identifier needed for behavioural features; names have no analytical value (data minimisation). A plain hash of a card number can be reversed by brute force, so a keyed hash (HMAC) is used. `first`, `last`, `street` are dropped; `cc_num` becomes `card_hash` (HMAC-SHA256 with `PII_HASH_KEY`, 64 hex characters). |
| `dob` replaced by `birth_year` (int16); the full date is dropped | Full date of birth + ZIP + gender is a quasi-identifier that can single out a person; age is all the models need. |
| Customer `lat`/`long` kept on purpose | Needed for the customer-to-merchant distance. They locate the customer almost like a home address, so raw coordinates stay in Silver only; Gold, the API and the dashboard expose the distance, never the coordinates. |
| Silver transform is a pure function: `transform_transactions(bronze, key)` → DataFrame, no file I/O | Easy to test without files; the caller (notebook or Airflow task) reads Bronze and writes Silver. |
| Strict validation: any unacceptable value raises `SilverValidationError`; fixed 22-column schema | Bad data stops the batch instead of silently reaching Gold; the output schema is identical for every batch. |
| `_batch_date` stays `date32[pyarrow]` in Silver, as in Bronze | It is a day, not a timestamp. |
| No check that the date of `trans_ts` equals `_batch_date` | Late-arriving rows are normal in real feeds and must not fail a batch (this dataset has none: every Bronze row matched its partition date in the backfill check). |
| `fraud_` prefix removed from merchant names | Generator artefact with no meaning; a name that is empty after removal fails validation. |
| No `.wslconfig` change | Per-container memory limits are enough; avoids a system-wide change. |
| PII key reaches Airflow as the Variable `pii_hash_secret` | Airflow masks Variable values when the name contains a sensitive word such as `secret` (substring match); `pii_hash_key` would not match. Masking was proved with a throwaway dummy Variable (`probe_secret`), never with the real key: the dummy value appeared as `***` in the task log and in 0 log files. The task also calls `mask_secret`. |
| Historical Silver built with the REST clear `only_new` | Adding a task makes a new DAG version, and the 731 finished runs stayed on the old one. `POST /api/v2/dags/{dag_id}/dagRuns/{run_id}/clear` with `only_new: true` adds and runs only the new Silver task, so Bronze was not re-run: all 731 Bronze files kept the same SHA-256 and `_ingested_at`, and the Bronze task instances kept their state, try number and timestamps. A dry run first confirmed `['silver_transactions']` for every run. |
| Silver tasks: 1 retry, but `SilverValidationError` fails fast | A retry helps with transient problems (I/O, container restart); it cannot fix bad data, so validation errors become `AirflowFailException` and fail without retrying. |
| Expected card count is 999, not 1,000 | Measured, not assumed: 999 distinct `cc_num` in Bronze (983 in the train file plus cards that only appear in the test file). Checks use measured numbers, never round guesses. |

---

## 8. Data quality findings

- Zero nulls and zero blank strings in the transactions: a sign the data is simulated.
- `trans_num` is unique (candidate primary key); no duplicate rows.
- `Unnamed: 0` is a leftover row index.
- `unix_time` is exactly 2,556–2,557 days (about 7 years) earlier than `trans_date_trans_time`: unreliable, not used.
- One row in `fraudTrain.csv` is out of time order (on 2019-02-28, within the same day, so daily partitioning is unaffected).
- ZIP codes lost leading zeros: 58 of 985 transaction ZIPs have only 4 digits, so only 927 match Census without padding; all 985 match after padding to 5 digits.
- Gazetteer lines have trailing spaces (the last column name has about 130).
- Census ACS uses `-666666666` for "not available"; for median household income this affects 2,299 of 33,120 ZIP areas.
- Every merchant name starts with `fraud_`: a generator artefact, does not leak the label.
- No transactions on 2020-02-29 (the generator skipped the leap day).
- Fraud amounts are higher (median about $397 vs $47) but capped around $1,376, so a single amount threshold does not work.
- Fraud rate is below 1%: accuracy is not a useful metric; use precision, recall and PR-AUC.
- 999 distinct cards in total (983 in the train file); the same 999 appear as distinct `card_hash` values in Silver.

---

## 9. Silver layer to-do (transactions first)

- [x] Drop `Unnamed: 0` and `unix_time`
- [x] Drop personal data: `first`, `last`, `street`
- [x] Hash `cc_num` with HMAC using `PII_HASH_KEY`
- [x] Cast types: timestamp, `dob` → `birth_year` (int16), amounts and coordinates as float, `city_pop` as integer, `is_fraud` as 0/1
- [x] Pad `zip` to 5 digits, kept as a string
- [x] Write Silver partitions to `data/silver/transactions/` (the transform has no file I/O) and run it from Airflow
- [x] Holidays and Census ACS: parse the raw JSON `response_body` into one row per holiday / ZIP area
- [x] Census: turn `-666666666` into null
- [x] Gazetteer: trim whitespace in column names and values
- [x] Tests for the Silver transactions transform (31 tests)
- [x] Tests for the other Silver transformations, and for the Silver writer (13 tests); full suite 149 passed

---

## 10. Issues solved (lessons learned)

- **Schema consistency:** the empty leap-day partition was written with `null` column types, which would break readers such as dbt. Fixed by setting metadata types explicitly, with a test comparing schemas.
- **Test run blocked the backfill:** an `airflow dags test` run inside the scheduled range occupied the slot for 2019-01-05, so catch-up looped. Fixed by deleting that run record. Lesson: test with dates outside the real range, or delete test runs afterwards.
- **Empty batches in Silver:** the same risk as the Bronze leap day applies to Silver: a 0-row batch must still have the full typed schema. `transform_transactions` returns `empty_silver_frame()` for an empty batch; verified on 2020-02-29 (0 rows, dtypes identical to a normal day).
- **Paused DAGs do not run backfills:** queued runs start only after unpausing.
- **Data intervals:** a run at time T processes the last complete day before T.
- **Python 3.14 and old pins:** old pandas pins had no wheels for Python 3.14, so pip tried to compile from source. Fixed by pinning versions that have wheels.
- **Airflow constraints vs our pins:** Airflow 3.3.2's constraints file pins pandas 3.0.5 while we had pinned 3.0.6, so pip could not install both. Fixed by aligning `requirements.txt` to 3.0.5, so the venv and the Docker image run the same version.
- **HTTP 200 is not success:** without a key, the Census API answers `200` with an HTML "Missing Key" page. Fixed by validating the body (JSON, expected columns) before writing.
- **Line endings:** `.gitattributes` forces LF so shell scripts work inside Linux containers.
- **.gitignore gap:** the original pattern missed partitioned sub-folders; fixed before any data was committed.
- **A false "0" from Git Bash grep:** in Git Bash, `grep` aborts when `-i` and `-F` are combined and prints nothing. In `$(grep -ciF …)` that showed as an empty count, and piped into `wc -l` it showed as `0`, which looked like "secret not found". Fixed by counting in Python (or with Linux `grep` inside the container). Lesson: a zero can mean "nothing found" or "the check never ran", so every zero-count check now has a positive control that must succeed (e.g. a known ZIP must be found in the Silver values) before a zero is trusted.

---

## 11. Environment

- Windows 11, 32 GB RAM, PyCharm, Docker Desktop.
- Python 3.14.6 in `.venv`; pandas 3.0.5, pyarrow 25.0.1, requests 2.34.2, python-dotenv 1.2.3.
- The Airflow Docker image uses Python 3.12 with the same package versions. 3.12 was chosen as a mature, widely supported version for Airflow; images for 3.13 and 3.14 also exist.
- Dev tools: ipykernel, nbconvert, pytest. The test suite (149 tests) passed at the last run.

---

## 12. How to resume a session

Run from the project root (the folder that contains `docker-compose.yml`):

```
.venv\Scripts\activate
docker compose up -d
```

Airflow UI: `http://localhost:8080`. To stop Airflow and keep all data:

```
docker compose down
```

Never add `-v` to that command: it deletes the Airflow volumes.

---

## 13. Session log

| Date | Work done |
|---|---|
| 2026-09-18 | Project structure created and pushed to GitHub. |
| 2026-09-23 | Topic changed to fraud detection; repository renamed; environment and dataset set up. |
| 2026-09-24 | First look at the data in `00_first_look.ipynb`. |
| 2026-09-25 | Bronze ingestion for transactions (with feed simulator), holidays, MCC, Census ZIP and Census ACS; tests added. |
| 2026-09-26 | Airflow in Docker; full 731-day backfill verified against raw; Silver design notebook started. |
| 2026-09-27 | Status file created; Silver layer started. |
| 2026-09-28 | Silver transactions transform (`src/silver/transactions.py`, commit `74109ed`): pure function, strict validation, fixed 22-column schema, empty batches keep the schema. 31 tests (`tests/test_silver_transactions.py`, commit `f2ac29c`); full suite 74 passed. Verified in `01_silver_design.ipynb`: 2019-01-01 → 2,414 rows × 22 columns, 779 cards before and after hashing, hash length 64, ZIP length 5; 2020-02-29 → 0 rows with identical dtypes. |
| 2026-10-02 | Silver transforms for Census ACS, Census Gazetteer and holidays with tests (author, commit `7b842d2`). Silver writer `src/pipeline/silver_jobs.py` and Airflow Silver tasks with 13 tests (commit `bdd3216`); full suite 149 passed. Masking of `pii_hash_secret` proved with a dummy Variable. Historical Silver built in Airflow with the `only_new` clear (731 transaction runs + reference run), Bronze byte-identical. Verified: 731 partitions, 1,852,394 rows, 9,651 fraud, 999 cards; ACS 33,120 rows (2,299 / 569 nulls); Gazetteer 33,144; holidays 160; secrets in 0 logs and 0 Silver files. **Phase 1 complete.** |

---

## 14. Update routine

At the end of every session:

1. Add a row to the session log.
2. Update the phase status, the to-do list, and any new decisions or lessons.
3. Update "Last updated" and "Current step" at the top.
4. Commit and push.
