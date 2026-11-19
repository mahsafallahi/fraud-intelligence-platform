"""Shared pytest setup.

Markers (registered in pyproject.toml):
- gold:      needs the Gold dev DuckDB file and/or the serving snapshot
- mlflow:    needs the local MLflow store
- container: needs a running API container (API_URL)
Locally such a test is SKIPPED with the reason when its files are missing, so
pytest -rs shows exactly what did not run. CI deselects them with -m instead,
so they never look like passes there.
"""

import os
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GOLD_DEV_DB = PROJECT_ROOT / "data" / "gold" / "fraud_dev.duckdb"
SERVING_DB = PROJECT_ROOT / "data" / "serving" / "serving.duckdb"
MLFLOW_DB = PROJECT_ROOT / "mlflow.db"


def pytest_collection_modifyitems(config, items):
    missing = {
        "gold": [p for p in (GOLD_DEV_DB, SERVING_DB) if not p.exists()],
        "mlflow": [p for p in (MLFLOW_DB,) if not p.exists()],
    }
    for item in items:
        for marker, paths in missing.items():
            if marker in item.keywords and paths:
                names = ", ".join(str(p.relative_to(PROJECT_ROOT)) for p in paths)
                item.add_marker(pytest.mark.skip(reason=f"{marker}: missing {names}"))
        if "container" in item.keywords and not os.environ.get("API_URL"):
            item.add_marker(pytest.mark.skip(reason="container: API_URL is not set"))
