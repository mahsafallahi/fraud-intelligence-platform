"""FastAPI app of the fraud scoring API (Phase 6).

Run (the app is built by a factory, so importing this module loads nothing):

    uvicorn src.api.main:create_app_from_env --factory --port 8000

Environment: BUNDLE_DIR (default models/F), SNAPSHOT_PATH (default
data/serving/serving.duckdb). Both are loaded and verified at startup; a bad
bundle or snapshot stops the process instead of serving a wrong model.

Endpoints: GET /health, GET /model-info, POST /score (one transaction; no
batch endpoint, see docs/api_design.md).

Logging: one JSON line per request (request_id, path, status, latency, error
code; for /score also model_version, alert and the rounded probability).
Never logged: card_key, amount, time, coordinates, client_ref, request bodies.
"""

import json
import logging
import os
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from src.api.bundle import load_bundle
from src.api.history import DateOutsideCalendar, SnapshotStore, UnknownCard
from src.api.schemas import ScoreRequest, ScoreResponse
from src.api.scoring import score_transaction

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BUNDLE_DIR = PROJECT_ROOT / "models" / "F"
DEFAULT_SNAPSHOT = PROJECT_ROOT / "data" / "serving" / "serving.duckdb"

logger = logging.getLogger("fraud_api")


def _setup_logging() -> None:
    if not any(getattr(h, "_fraud_api", False) for h in logger.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        handler._fraud_api = True
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def _log(event: dict) -> None:
    logger.info(json.dumps(event, separators=(",", ":")))


def _error(status: int, code: str, detail=None) -> JSONResponse:
    body = {"code": code}
    if detail is not None:
        body["detail"] = detail
    return JSONResponse(status_code=status, content=body)


def create_app(bundle_dir=DEFAULT_BUNDLE_DIR, snapshot_path=DEFAULT_SNAPSHOT) -> FastAPI:
    _setup_logging()
    bundle = load_bundle(bundle_dir)
    store = SnapshotStore(snapshot_path)
    app = FastAPI(title="Fraud scoring API", version=bundle.model_version)
    app.state.bundle, app.state.store = bundle, store
    _log({"event": "startup", "model_version": bundle.model_version,
          "snapshot_built_at": store.meta.get("built_at")})

    @app.middleware("http")
    async def request_log(request: Request, call_next):
        request.state.request_id = str(uuid.uuid4())
        request.state.log = {}
        start = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception as exc:  # unexpected: answer 500, log only the exception type
            request.state.log["error"] = f"internal_error:{type(exc).__name__}"
            response = _error(500, "internal_error")
        response.headers["X-Request-ID"] = request.state.request_id
        _log({"event": "request", "request_id": request.state.request_id,
              "method": request.method, "path": request.url.path, "status": response.status_code,
              "latency_ms": round(1000 * (time.perf_counter() - start), 2), **request.state.log})
        return response

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        request.state.log["error"] = "validation_error"
        # loc / msg / type only: pydantic's "input" would echo the submitted values
        errors = [{"loc": list(e["loc"]), "msg": e["msg"], "type": e["type"]} for e in exc.errors()]
        return _error(422, "validation_error", errors)

    @app.exception_handler(UnknownCard)
    async def unknown_card(request: Request, exc: UnknownCard):
        request.state.log["error"] = "unknown_card"
        return _error(422, "unknown_card", "card_key is not in the serving snapshot")

    @app.exception_handler(DateOutsideCalendar)
    async def outside_calendar(request: Request, exc: DateOutsideCalendar):
        request.state.log["error"] = "date_outside_calendar"
        return _error(422, "date_outside_calendar", str(exc))

    @app.get("/health")
    def health():
        try:
            store.ping()
        except Exception:
            return JSONResponse(status_code=503, content={"status": "unavailable"})
        return {"status": "ok"}

    @app.get("/model-info")
    def model_info():
        m = bundle.manifest
        return {
            "model_version": bundle.model_version,
            "bundle_id": bundle.bundle_id,
            "columns": bundle.columns,
            "threshold": {"value": bundle.threshold, "applies_to": "probability_raw", "rule": ">="},
            "calibrator": {"method": bundle.calibrator.method},
            "files": m["files"],
            "mlflow": m["mlflow"],
            "export": m["export"],
            "versions": m["versions"],
            "snapshot": store.meta,
        }

    @app.post("/score", response_model=ScoreResponse)
    def score(req: ScoreRequest, request: Request):
        s = score_transaction(bundle, store, card_key=req.card_key, trans_ts=req.trans_ts,
                              amt=req.amt, category=req.category,
                              merch_lat=req.merch_lat, merch_long=req.merch_long)
        request.state.log.update({"model_version": bundle.model_version, "alert": s.alert,
                                  "probability_calibrated": round(s.probability_calibrated, 3)})
        return ScoreResponse(
            request_id=request.state.request_id, client_ref=req.client_ref,
            model_version=bundle.model_version, raw_score=s.raw_score,
            probability_raw=s.probability_raw, probability_calibrated=s.probability_calibrated,
            alert=s.alert, threshold=bundle.threshold, reasons=s.reasons)

    return app


def create_app_from_env() -> FastAPI:
    return create_app(os.environ.get("BUNDLE_DIR", DEFAULT_BUNDLE_DIR),
                      os.environ.get("SNAPSHOT_PATH", DEFAULT_SNAPSHOT))
