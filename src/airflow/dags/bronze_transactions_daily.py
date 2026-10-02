"""Daily Bronze ingestion of the transactions feed (one landing file per day).

Schedule: a data-interval timetable. The run for 2019-01-01 covers the interval
[2019-01-01, 2019-01-02) and starts after that interval ends, at midnight on
2019-01-02, so it only processes a day that is complete. The task ingests the
day of data_interval_start.

catchup=True makes Airflow create a run for every past interval between
start_date and end_date: this is the backfill of the whole dataset.

Each run then builds the Silver partition for the same day (silver_transactions).
"""

from datetime import timedelta

import pendulum
from airflow.sdk import dag, task
from airflow.timetables.interval import CronDataIntervalTimetable


@dag(
    dag_id="bronze_transactions_daily",
    schedule=CronDataIntervalTimetable("@daily", timezone="UTC"),
    start_date=pendulum.datetime(2019, 1, 1, tz="UTC"),
    end_date=pendulum.datetime(2020, 12, 31, tz="UTC"),
    catchup=True,
    max_active_runs=4,
    default_args={
        # A landing file that arrives late is retried, not treated as a failure.
        "retries": 3,
        "retry_delay": timedelta(minutes=5),
    },
    tags=["bronze", "transactions"],
)
def bronze_transactions_daily():
    @task
    def ingest_transactions(data_interval_start=None) -> int:
        # Imported here, not at the top: DAG files are parsed often, so they
        # should not load pandas on every parse.
        from src.ingestion.bronze_transactions import ingest_day

        return ingest_day(data_interval_start.date())

    # 1 retry for transient problems (I/O, container restart); data errors
    # (SilverValidationError) become AirflowFailException and fail at once.
    @task(retries=1, retry_delay=timedelta(minutes=1))
    def silver_transactions(data_interval_start=None) -> int:
        from airflow.sdk import Variable
        from airflow.sdk.exceptions import AirflowFailException
        from airflow.sdk.log import mask_secret

        from src.pipeline.silver_jobs import fail_fast_on_validation_error, run_transactions_day
        from src.silver.transactions import load_pii_key

        # Loaded once per run. The Variable name contains "secret", so Airflow
        # masks it by name; mask_secret registers it explicitly as well.
        secret = Variable.get("pii_hash_secret")
        mask_secret(secret)
        key = load_pii_key({"PII_HASH_KEY": secret})

        run = fail_fast_on_validation_error(AirflowFailException)(run_transactions_day)
        return run(data_interval_start.date(), key)

    # Silver for a day runs only after Bronze for the same day succeeded.
    ingest_transactions() >> silver_transactions()


bronze_transactions_daily()
