"""Bronze ingestion of the reference sources: holidays, MCC codes, Census ZIP data.

Manual trigger only (schedule=None): each source is pinned to a version
(year, commit, vintage), so it is re-run deliberately when a version changes.
The four tasks are independent and run in parallel.
"""

from datetime import timedelta

import pendulum
from airflow.sdk import dag, task

# Years/vintage covered by the transactions data (2019-01-01 .. 2020-12-31).
HOLIDAY_YEARS = [2019, 2020]
CENSUS_VINTAGE = 2019


@dag(
    dag_id="bronze_reference_data",
    schedule=None,
    start_date=pendulum.datetime(2019, 1, 1, tz="UTC"),
    catchup=False,
    default_args={
        "retries": 2,
        "retry_delay": timedelta(minutes=2),
    },
    tags=["bronze", "reference"],
)
def bronze_reference_data():
    @task
    def holidays() -> int:
        from src.ingestion.bronze_holidays import ingest_year

        return sum(ingest_year(year) for year in HOLIDAY_YEARS)

    @task
    def mcc() -> int:
        from src.ingestion.bronze_mcc import ingest

        return ingest()

    @task
    def census_zcta() -> int:
        from src.ingestion.bronze_census_zcta import ingest

        return ingest(CENSUS_VINTAGE)

    @task
    def census_acs() -> int:
        from airflow.sdk import Variable
        from airflow.sdk.log import mask_secret

        from src.ingestion.bronze_census_acs import ingest

        api_key = Variable.get("census_api_key")
        mask_secret(api_key)  # explicit, in addition to Airflow's name-based masking
        return ingest(CENSUS_VINTAGE, api_key=api_key)

    holidays()
    mcc()
    census_zcta()
    census_acs()


bronze_reference_data()
