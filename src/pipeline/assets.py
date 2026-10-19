"""Names of the Airflow Assets that connect the DAGs (Silver -> Gold).

Plain strings, so this module does not import Airflow and stays testable
locally. The DAGs wrap them in airflow.sdk.Asset.
"""

SILVER_TRANSACTIONS = "silver_transactions"  # emitted by silver_transactions
SILVER_REFERENCE = "silver_reference"        # emitted by the four reference Silver tasks
GOLD_STAR_SCHEMA = "gold_star_schema"        # emitted by gold_dbt (for the API / dashboard later)
