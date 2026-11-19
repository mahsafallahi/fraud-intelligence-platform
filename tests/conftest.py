"""Shared pytest setup.

Markers (registered in pyproject.toml):
- gold:      needs the Gold dev DuckDB file and/or the serving snapshot
- mlflow:    needs the local MLflow store
- container: needs a running API container (API_URL)
Locally such a test is SKIPPED with the reason when its files are missing, so
pytest -rs shows exactly what did not run. CI deselects them with -m instead,
so they never look like passes there.
"""

import importlib.util
import os
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GOLD_DEV_DB = PROJECT_ROOT / "data" / "gold" / "fraud_dev.duckdb"
SERVING_DB = PROJECT_ROOT / "data" / "serving" / "serving.duckdb"
MLFLOW_DB = PROJECT_ROOT / "mlflow.db"
BUNDLE_DIR = PROJECT_ROOT / "models" / "F"


def load_script(name: str):
    """Import a module from scripts/ (not a package) by file path."""
    spec = importlib.util.spec_from_file_location(name, PROJECT_ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def synthetic(tmp_path_factory):
    """(snapshot path, transactions with merchant coordinates) of the synthetic snapshot."""
    path = tmp_path_factory.mktemp("serving") / "synthetic.duckdb"
    txns = load_script("make_synthetic_snapshot").make_synthetic_snapshot(path)
    return path, txns


@pytest.fixture(scope="session")
def api(synthetic):
    """The real app with the committed bundle and the synthetic snapshot."""
    from fastapi.testclient import TestClient

    from src.api.main import create_app
    app = create_app(BUNDLE_DIR, synthetic[0])
    yield TestClient(app)
    app.state.store.close()


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
