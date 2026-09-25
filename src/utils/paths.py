"""Central place for data-lake paths.

DATA_DIR defaults to <project>/data but can be overridden with the DATA_DIR
environment variable (e.g. when the code runs inside an Airflow container).
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.getenv("DATA_DIR", PROJECT_ROOT / "data"))

RAW_DIR = DATA_DIR / "raw"
LANDING_DIR = DATA_DIR / "landing"
BRONZE_DIR = DATA_DIR / "bronze"
