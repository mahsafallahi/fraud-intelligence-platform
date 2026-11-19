# Scoring API — design and results (Phase 6)

Model F (LightGBM, 16 columns) served as a real-time scoring API with FastAPI,
packaged in Docker and tested by GitHub Actions. Design approved on 2026-11-19;
built and verified the same day.

## 1. What a request does

```
POST /score {card_key, trans_ts, amt, category, merch_lat, merch_long, client_ref?}
  │
  ├─ snapshot (data/serving/serving.duckdb, read-only)
  │    card's transactions strictly before trans_ts + the new transaction,
  │    in the columns of features.MODEL_FRAME_SQL (src/api/resolve.py)
  ├─ build_features()  ← the same function as offline, unchanged
  ├─ booster (models/F/model.txt): raw log-odds, probability, SHAP contributions
  ├─ calibrator (models/F/calibrator.json): Platt on the raw probability
  └─ alert = probability_raw >= 0.2940968658100669
```

## 2. Decisions

| Decision | Rejected alternative and why |
|---|---|
| **Bundle of plain files** in `models/F/`: native LightGBM text model, calibrator JSON, column and category lists, threshold, manifest (SHA-256 per file, `bundle_id`, git commit, MLflow run ids). No MLflow, pickle or skops at serving. | MLflow pyfunc at serving: heavy dependency plus a tracking DB. Pickle/skops: code execution or extra dependencies. ONNX: categorical support needs conversion; parity no longer native. |
| **Bundle committed to git** (1.8 MB), `models/** -text` in `.gitattributes` so git never changes bytes (hashes stay valid). | Ignoring it: CI could not smoke-test the real model and the image would not build from git alone. |
| **Fail fast at startup**: wrong hash, wrong columns, wrong category mapping, missing snapshot table or schema version → the process exits. | Starting "degraded" with 503s: hides a broken deployment. |
| **Alert rule = raw probability ≥ the exact validation threshold** (the 1,000th highest validation score in notebook 04). On the test period this gives 969 alerts in 92 days (10.5 per day). | **1,082 was an offline top-N budget only** (1,000 alerts × 92/85 days, used in notebook 04 to compare models at equal workload). A real-time scorer sees one transaction at a time and cannot rank a whole quarter, so a budget cannot be applied online. |
| **Training-serving parity by reuse**: the API builds the model frame and calls the unchanged `build_features`. | Re-implementing the features per request (incremental per-card state): faster, but a second copy of the feature logic that can drift. |
| **History = strictly earlier transactions** (`trans_ts < t`). Same-second transactions are invisible to each other in `build_features`, so the new row's features do not change, and a replayed historical transaction never sees itself. | Including same-second rows and de-duplicating by `trans_num`: more code, same result. |
| **Serving snapshot** (DuckDB file, 25 MB) built from `fraud_dev.duckdb` (read-only) + Silver home coordinates; mounted read-only, never in the image. `fraud.duckdb` (Airflow's file) is never opened. | Querying Gold directly: Airflow's writer lock, a 5-way join per request (18–26 ms), and no home coordinates. Parquet files: several artefacts instead of one. |
| **Scored transactions do not update the snapshot.** | Write-back: concurrency and idempotency to solve, and replay results would depend on request order. Consequence: in a live replay, a burst on one card does not see its own earlier requests, so `card_txn_1h` / `card_txn_24h` undercount. A later phase could add an append table keyed by `trans_num`. |
| **Derived Gold columns for the new row** (`distance_km`, `is_public_holiday_in_state`) use the exact SQL of `fact_transactions.sql`, evaluated in DuckDB. The builder evaluates them on all 1,852,394 historical rows and stops unless they equal Gold exactly. | Computing them in numpy: another math library, possible last-digit differences. Client-sent distance or holiday flags: the client would control feature values (trust boundary). |
| **State holidays** read from the view `int_state_public_holidays` with the working directory set to `src/dbt` (its staging views use relative paths). dbt and Gold are not changed. | Materialising the view in dbt (rejected by the owner). Re-filtering Silver holidays in Python: a second copy of dbt logic. |
| **Home lat/long in the snapshot** (approved 2026-11-19): needed for `distance_km`. The file is git-ignored, mounted read-only, never in the image; coordinates are never returned or logged. | Without them `distance_km` cannot be computed for a new transaction. |
| **Request = what is known at authorisation**: card token (`card_key`, pseudonymised upstream), time, amount, category, merchant location. Everything else is looked up. | MCC alone: 5411, 5999 and 5311 each map to two categories. Card number + hashing in the API: the HMAC secret would have to be in the container. |
| **422 for every input problem**, with a `code`: `validation_error`, `unknown_card`, `date_outside_calendar`. | 404 for an unknown card: reads as a missing route. Defaulting unknown holiday flags to 0: silently wrong features. |
| **Reasons = top 3 positive SHAP contributions** (`pred_contrib`, log-odds) with the feature value; categories as names. | Top 3 by absolute value: negative contributions argue against fraud. |
| **No batch endpoint.** | Two transactions of one card in a batch would have to see each other, which reopens parity; offline batch scoring stays in the notebooks. |
| **Logging**: one JSON line per request (request_id, path, status, latency, error code; for `/score` model version, alert, calibrated probability rounded to 3 decimals). Never: card_key, amount, time, coordinates, `client_ref`, bodies; uvicorn access log off; unexpected errors log only the exception type. | Logging the request for debugging: personal data in logs. |
| **`requirements-api.txt` + hashed `requirements-api.lock`** (Linux, CPython 3.14, all versions equal `.venv`). | Adding FastAPI to `requirements.txt`: the Airflow image installs that file under Airflow's constraints (Airflow 3 pins its own FastAPI). |
| **No scikit-learn and no pyarrow in the image**: `calibration.py` imports scikit-learn only inside the fit functions; container parity proves pandas behaves the same without pyarrow. | Installing them: about +45 MB and +120 MB for nothing used. |
| **Docker**: `python:3.14.6-slim-trixie` pinned by digest (= `.venv`), single stage, `libgomp1`, hashed lock with `--only-binary --no-deps --require-hashes`, non-root uid 10001, `OMP_NUM_THREADS=1`, read-only root filesystem works. | Multi-stage: nothing compiles, so it would only drop pip (about 10 MB). |
| **Per-Dockerfile ignore** `docker/api/Dockerfile.dockerignore` (allowlist). | Widening the root `.dockerignore`: it belongs to the Airflow build. |
| **CI**: tests (local-data markers deselected, warnings are errors), image build, size gate, container smoke test on a synthetic snapshot with the real bundle; actions pinned by SHA; read-only token; no secrets. | Running Gold-dependent tests in CI: the data is git-ignored and must not leave the machine. |

## 3. Results (2026-11-19)

| Check | Result |
|---|---|
| Bundle export | Text model = MLflow model on all 228,187 validation rows (raw score and probability identical). Threshold 0.2940968658100669 → exactly 1,000 validation alerts; equals the logged `threshold.f_prob` 0.2940968658. `bundle_id` be26ca8810f6…, exported from a clean tree. |
| Snapshot build | 1,852,394 rows, 999 cards. Resolver distance and state-holiday flag = Gold on every row (0 mismatches); snapshot model frame = Gold model frame in both directions. |
| Parity, in process | 3,000 test-period transactions (2,000 random, all 969 offline alerts, 23 near the threshold, 26 same-second; chosen without labels): features exact; raw log-odds, raw and calibrated probability max difference 0; flags and reasons 100% identical. |
| Parity, container (Linux) | Same 3,000: raw log-odds max difference 0; probabilities max 1.11e-16 (limit 1e-12); 0 flag and 0 reason mismatches. |
| Latency (laptop, one worker) | In process p50 39.3 ms, p99 71.5 ms; container over HTTP p50 51.1 ms, p99 86.3 ms (limit 200 ms). |
| Image | Layers 546 MB (filesystem 516 MB), 168 MB compressed. Docker Desktop's "disk usage" shows 714 MB because the containerd store counts the unpacked layers plus the compressed blobs. |
| Logs | 0 card keys in 3,015 container log lines. |

The parity test never uses labels: `is_fraud` and `gender` are dropped right
after loading, so it is an engineering check, not a second test evaluation.

## 4. How to run

```bash
# 1. Bundle (only after a model change; needs MLflow, dev env)
.venv/Scripts/python.exe scripts/export_bundle.py
# 2. Serving snapshot (after every Gold rebuild of fraud_dev.duckdb)
.venv/Scripts/python.exe scripts/build_serving_snapshot.py
# 3a. API locally
.venv/Scripts/python.exe -m uvicorn src.api.main:create_app_from_env --factory --port 8000
# 3b. or in Docker (profile "api" is not started by plain `docker compose up`)
docker compose --profile api up -d --build api
# 4. Tests: everything local, then the container parity against a running API
.venv/Scripts/python.exe -m pytest -rs
API_URL=http://127.0.0.1:8000 .venv/Scripts/python.exe -m pytest -rs -s tests/test_api_parity.py -k container
```

Example request (OpenAPI page at `http://localhost:8000/docs`):

```json
{"card_key": "<32 hex>", "trans_ts": "2020-11-03T14:22:05", "amt": 123.45,
 "category": "shopping_net", "merch_lat": 40.12, "merch_long": -75.21, "client_ref": "optional"}
```

Response fields: `request_id`, `client_ref`, `model_version`, `raw_score`
(log-odds), `probability_raw`, `probability_calibrated`, `alert`, `threshold`,
`reasons` (feature, value, contribution).

Test markers: `gold` (needs `data/gold/fraud_dev.duckdb` and the snapshot),
`mlflow` (needs `mlflow.db`), `container` (needs `API_URL`). Locally they are
skipped with the reason when the files are missing; CI deselects them.

Regenerating `requirements-api.lock`: resolve `requirements-api.txt` with
`pip install --dry-run --ignore-installed --only-binary=:all: --platform manylinux_2_28_x86_64 --platform manylinux_2_17_x86_64 --platform manylinux2014_x86_64 --python-version 3.14 --implementation cp --target <tmp> --report <report.json>`,
constrained to the `.venv` versions, and write `name==version \` +
`--hash=sha256:…` per package (checked by `tests/test_requirements_pins.py`).

## 5. Limitations

- **Replay only:** the calendar covers 2019-01-01 to 2020-12-31; other dates get `date_outside_calendar`.
- **Read-only history:** requests do not see each other (see section 2).
- **Stale snapshot:** after a Gold rebuild the snapshot must be rebuilt; `/model-info` shows its `built_at` and `source_max_trans_ts`.
- **Known cards only:** a card without a profile in the snapshot cannot be scored (`unknown_card`).
- **Model limitations from Phase 4 still apply:** daytime fraud and women's fraud are caught less (`docs/model_results.md`); the data is simulated.
- **F's provenance:** the MLflow run of F carries `git_dirty=True` (recorded as is in the manifest; see the Phase 4 notes).
