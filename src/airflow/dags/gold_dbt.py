"""Gold layer: rebuild the star schema with `dbt build` (seeds, models, tests).

Asset-triggered: runs when Silver transactions OR Silver reference data change.
Gold is rebuilt from the whole Silver layer, so one run at a time is enough
(max_active_runs=1). Design: docs/star_schema_design.md; dbt project: src/dbt.

dbt runs from its own virtualenv in the image (/opt/dbt-venv), with the
`airflow` target: Airflow is the only writer of data/gold/fraud.duckdb.
- a hard test or model error fails the task at once (no retry);
- a soft-rule warning does not fail it, but is logged as "SOFT RULE WARNING";
- dbt itself crashing is retried once.
"""

from datetime import timedelta

import pendulum
from airflow.sdk import Asset, dag, task

from src.pipeline.assets import GOLD_STAR_SCHEMA, SILVER_REFERENCE, SILVER_TRANSACTIONS


@dag(
    dag_id="gold_dbt",
    schedule=(Asset(SILVER_TRANSACTIONS) | Asset(SILVER_REFERENCE)),
    start_date=pendulum.datetime(2026, 10, 1, tz="UTC"),
    catchup=False,
    max_active_runs=1,
    default_args={
        "retries": 1,
        "retry_delay": timedelta(minutes=2),
    },
    tags=["gold", "dbt"],
)
def gold_dbt():
    @task(outlets=[Asset(GOLD_STAR_SCHEMA)])
    def dbt_build() -> dict:
        from airflow.sdk.exceptions import AirflowFailException

        from src.pipeline.gold_dbt import run_dbt_build

        return run_dbt_build(target="airflow", fail_with=AirflowFailException)

    dbt_build()


gold_dbt()
