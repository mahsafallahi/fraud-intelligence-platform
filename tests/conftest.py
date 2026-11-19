"""Shared pytest setup.

Markers (registered in pyproject.toml):
- gold:      needs the Gold dev DuckDB file and/or the serving snapshot
- mlflow:    needs the local MLflow store
- container: needs a running API container (API_URL)
Locally such a test is SKIPPED with the reason when its files are missing, so
pytest -rs shows exactly what did not run. CI deselects them with -m instead,
so they never look like passes there.
"""

import asyncio
import importlib.util
import json
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


class AsgiClient:
    """Minimal in-process HTTP client for an ASGI app, standard library only.

    FastAPI's TestClient needs httpx, which is not a project dependency; this
    sends one request through the app (middleware and handlers included) and
    returns (status, json_body, headers).
    """

    def __init__(self, app):
        self.app = app

    def request(self, method: str, path: str, body=None):
        raw = b"" if body is None else (body if isinstance(body, bytes) else json.dumps(body).encode())
        scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
                 "method": method, "scheme": "http", "path": path, "raw_path": path.encode(),
                 "root_path": "", "query_string": b"", "client": ("test", 1), "server": ("test", 80),
                 "headers": [(b"host", b"test"), (b"content-type", b"application/json"),
                             (b"content-length", str(len(raw)).encode())]}
        messages = [{"type": "http.request", "body": raw, "more_body": False}]
        sent = {"status": None, "headers": {}, "body": b""}

        async def receive():
            return messages.pop(0) if messages else {"type": "http.disconnect"}

        async def send(message):
            if message["type"] == "http.response.start":
                sent["status"] = message["status"]
                sent["headers"] = {k.decode(): v.decode() for k, v in message["headers"]}
            elif message["type"] == "http.response.body":
                sent["body"] += message.get("body", b"")

        asyncio.run(self.app(scope, receive, send))
        return sent["status"], json.loads(sent["body"]) if sent["body"] else None, sent["headers"]

    def get(self, path):
        return self.request("GET", path)

    def post(self, path, body):
        return self.request("POST", path, body)


@pytest.fixture(scope="session")
def synthetic(tmp_path_factory):
    """(snapshot path, transactions with merchant coordinates) of the synthetic snapshot."""
    path = tmp_path_factory.mktemp("serving") / "synthetic.duckdb"
    txns = load_script("make_synthetic_snapshot").make_synthetic_snapshot(path)
    return path, txns


@pytest.fixture(scope="session")
def api(synthetic):
    """The real app with the committed bundle and the synthetic snapshot."""
    from src.api.main import create_app
    app = create_app(BUNDLE_DIR, synthetic[0])
    yield AsgiClient(app)
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
