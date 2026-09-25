# Fraud Intelligence Platform

End-to-end Data Engineering + Data Science + GenAI + MLOps project.

## Status
🚧 Under active development — Phase 1 (Data Ingestion & Lake) in progress.

## Architecture
See `docs/architecture-diagram.png` (coming soon).

## Tech Stack
Python · PostgreSQL · Airflow · dbt · Great Expectations · Scikit-learn · XGBoost/LightGBM · Chroma · MLflow · Docker · GitHub Actions

## Data sources (Bronze layer)
| Source | Ingestion | Bronze location |
|---|---|---|
| Card transactions (Kaggle, kartik2112) | Daily batches from a simulated landing zone | `data/bronze/transactions/batch_date=YYYY-MM-DD/` |
| US public holidays (Nager.Date API) | One call per country and year | `data/bronze/holidays/country=US/year=YYYY/` |
| Merchant category codes (greggles/mcc-codes) | CSV pinned to a commit | `data/bronze/mcc/version=<sha>/` |
| ZIP geography (Census Gazetteer) | Zip file per vintage | `data/bronze/census_zcta/vintage=YYYY/` |
| ZIP demographics (Census ACS 5-year API, needs a key) | One call per vintage | `data/bronze/census_acs/vintage=YYYY/` |

Bronze keeps every source as received (strings or the raw response body) plus
`_`-prefixed ingestion metadata. Each run overwrites its own partition, so re-runs never duplicate data.

## How to Run

### 1. Local setup
```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements-dev.txt   # Windows; use .venv/bin/python on Linux/macOS
.venv/Scripts/python -m pytest
```
Download the Kaggle dataset into `data/raw/` (`fraudTrain.csv`, `fraudTest.csv`), then create the daily landing files:
```bash
.venv/Scripts/python -m src.ingestion.simulate_transaction_feed
```

### 2. Secrets
Copy `.env.example` to `.env` and fill in:
- `CENSUS_API_KEY`: free key from https://api.census.gov/data/key_signup.html
- `AIRFLOW_FERNET_KEY`: `python -c "import base64,os; print(base64.urlsafe_b64encode(os.urandom(32)).decode())"`
- `AIRFLOW_JWT_SECRET`: `python -c "import secrets; print(secrets.token_urlsafe(48))"`

`.env` is git-ignored. The Census key is sent as a URL parameter, so the code strips it from stored URLs,
error messages and HTTP-library logs, and Airflow masks it in task logs.

### 3. Airflow (Docker)
```bash
docker compose up -d --build     # UI: http://localhost:8080
docker compose down              # stop; the metadata DB volume is kept
```
Airflow 3.3 with LocalExecutor and Postgres. Memory limits per container add up to 5.5 GB.
`src/` and `data/` are mounted into the containers, so code changes need no rebuild.

| DAG | Schedule | What it does |
|---|---|---|
| `bronze_transactions_daily` | Daily, 2019-01-01 to 2020-12-31, catch-up on | Ingests one day of transactions per run |
| `bronze_reference_data` | Manual | Holidays, MCC, Census Gazetteer and Census ACS, in parallel |

DAGs start **paused**. Unpausing `bronze_transactions_daily` makes the scheduler catch up all 731 days.

Notes:
- The transactions DAG uses a data-interval schedule: the run for day D starts after D ends, at midnight on D+1.
  A **manual** trigger at time T therefore processes the last complete day before T
  (`airflow dags test bronze_transactions_daily 2019-01-05` ingests 2019-01-04).
- Local development only: the UI has no login (every user is admin) and is bound to `127.0.0.1`.
