"""Export model F from MLflow into the plain-file bundle the API serves (Phase 6).

Run once, in the dev environment (.venv, which has MLflow and skops):

    .venv/Scripts/python.exe scripts/export_bundle.py            # writes models/F/

What it does, and the checks that stop it before anything is written:
1. Finds the runs in experiment fraud-phase4 by name: F_coarse_profile (must be
   the one run tagged selected=yes), F_calibrator (must calibrate that run) and
   test_evaluation (must evaluate that run).
2. Loads F as logged, writes its booster as native LightGBM text, reloads the
   text and requires IDENTICAL raw scores and probabilities on every validation
   row (array_equal, no tolerance).
3. Recomputes the alert threshold exactly as notebook 04 did (the 1,000th
   highest raw validation probability) and requires: exactly 1,000 validation
   alerts, and the same value as the param logged in test_evaluation (which was
   written with 10 significant digits).
4. Copies calibrator.json byte for byte from the F_calibrator run.
5. Writes features.json, threshold.json and manifest.json (SHA-256 per file),
   then loads the result with src.api.bundle.load_bundle as a final check.

Test-period rows are never loaded: features are past-only, so validation
features do not depend on them (the same shortcut notebook 04 takes).
"""

import json
import platform
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import lightgbm as lgb  # noqa: E402
import mlflow  # noqa: E402
import numpy as np  # noqa: E402

from src.api.bundle import BUNDLE_FORMAT, MANIFEST, bundle_id, load_bundle, sha256_file  # noqa: E402
from src.models.features import (  # noqa: E402
    CATEGORIES, CHANNELS, FINAL_COLUMNS, build_features, load_model_frame,
)
from src.models.split import TEST, VALIDATION, assign_split  # noqa: E402

EXPERIMENT = "fraud-phase4"
MODEL_RUN, CALIBRATOR_RUN, TEST_RUN = "F_coarse_profile", "F_calibrator", "test_evaluation"
BUDGET = 1000  # validation alerts that define the threshold (notebook 04, step H)
GOLD_DB = PROJECT_ROOT / "data" / "gold" / "fraud_dev.duckdb"  # the file notebook 04 used
OUT = PROJECT_ROOT / "models" / "F"


def git(*args) -> str:
    return subprocess.run(["git", *args], cwd=PROJECT_ROOT, capture_output=True,
                          text=True, check=True).stdout.strip()


def write_json(path: Path, obj) -> None:
    path.write_bytes((json.dumps(obj, indent=2) + "\n").encode("utf-8"))


def find_runs() -> dict:
    runs = mlflow.search_runs(experiment_names=[EXPERIMENT])  # active runs only
    names = runs["tags.mlflow.runName"]
    if names.duplicated().any():
        raise SystemExit(f"duplicate run names: {sorted(set(names[names.duplicated()]))}")
    by_name = runs.set_index("tags.mlflow.runName")
    selected = list(runs.loc[runs["tags.selected"] == "yes", "tags.mlflow.runName"])
    if selected != [MODEL_RUN]:
        raise SystemExit(f"expected exactly {MODEL_RUN} selected, found {selected}")
    for name in (MODEL_RUN, CALIBRATOR_RUN, TEST_RUN):
        if by_name.loc[name, "status"] != "FINISHED":
            raise SystemExit(f"{name} is not FINISHED")
    model_id = by_name.loc[MODEL_RUN, "run_id"]
    if by_name.loc[CALIBRATOR_RUN, "tags.calibrates"] != model_id:
        raise SystemExit(f"{CALIBRATOR_RUN} does not calibrate {MODEL_RUN}")
    if by_name.loc[TEST_RUN, "tags.evaluates"] != model_id:
        raise SystemExit(f"{TEST_RUN} does not evaluate {MODEL_RUN}")
    return {name: by_name.loc[name] for name in (MODEL_RUN, CALIBRATOR_RUN, TEST_RUN)}


def main() -> None:
    mlflow.set_tracking_uri(f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}")
    runs = find_runs()
    model_id = runs[MODEL_RUN]["run_id"]
    calibrator_id = runs[CALIBRATOR_RUN]["run_id"]
    test_id = runs[TEST_RUN]["run_id"]
    client = mlflow.MlflowClient()
    logged_threshold = client.get_run(test_id).data.params["threshold.f_prob"]

    # Validation features, exactly as notebook 04 built them (test rows dropped first)
    frame = load_model_frame(GOLD_DB)
    frame = frame[assign_split(frame["trans_ts"]) != TEST]
    features = build_features(frame)
    val = (assign_split(frame["trans_ts"]) == VALIDATION).to_numpy()
    X_val = features.loc[val, FINAL_COLUMNS]
    print(f"validation rows: {len(X_val):,}")

    model = mlflow.lightgbm.load_model(f"runs:/{model_id}/model")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "model.txt").write_bytes(model.booster_.model_to_string().encode("utf-8"))
        booster = lgb.Booster(model_file=str(tmp / "model.txt"))
        raw_logged = model.predict(X_val, raw_score=True)
        raw_text = booster.predict(X_val, raw_score=True)
        if not np.array_equal(raw_logged, raw_text):
            raise SystemExit(f"text model differs: max {np.abs(raw_logged - raw_text).max():.3e}")
        p_logged = model.predict_proba(X_val)[:, 1]
        p_val = booster.predict(X_val)
        if not np.array_equal(p_logged, p_val):
            raise SystemExit("text model probabilities differ from the logged model")
        print(f"text model == logged model on {len(X_val):,} validation rows (raw score and probability, exact)")

        threshold = float(np.sort(p_val)[::-1][BUDGET - 1])
        n_alerts = int((p_val >= threshold).sum())
        if n_alerts != BUDGET:
            raise SystemExit(f"threshold gives {n_alerts} validation alerts, expected {BUDGET}")
        if f"{threshold:.10g}" != logged_threshold:
            raise SystemExit(f"threshold {threshold:.10g} != logged {logged_threshold}")
        print(f"threshold {threshold!r} -> {n_alerts} validation alerts; matches logged {logged_threshold}")

        cal_path = mlflow.artifacts.download_artifacts(
            run_id=calibrator_id, artifact_path="calibrator.json", dst_path=str(tmp / "cal"))
        shutil.copyfile(cal_path, tmp / "calibrator.json")

        write_json(tmp / "features.json", {
            "columns": FINAL_COLUMNS,
            "categories": {"category": CATEGORIES, "channel": CHANNELS},
        })
        write_json(tmp / "threshold.json", {
            "applies_to": "raw_probability",
            "rule": ">=",
            "value": threshold,
            "source": f"the {BUDGET}th highest raw probability of F on the validation period "
                      "(notebook 04, step H); about 10.5 alerts per day on the test period",
            "validation_alerts": n_alerts,
        })

        files = {name: sha256_file(tmp / name)
                 for name in ("model.txt", "calibrator.json", "features.json", "threshold.json")}
        model_tags = client.get_run(model_id).data.tags
        write_json(tmp / MANIFEST, {
            "bundle_format": BUNDLE_FORMAT,
            "model_name": MODEL_RUN,
            "bundle_id": bundle_id(files),
            "files": files,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "export": {"git_commit": git("rev-parse", "--short", "HEAD"),
                       "git_dirty": bool(git("status", "--porcelain"))},
            "mlflow": {
                "experiment": EXPERIMENT,
                "model_run_id": model_id,
                "model_run_git_commit": model_tags.get("git_commit"),
                "model_run_git_dirty": model_tags.get("git_dirty"),
                "calibrator_run_id": calibrator_id,
                "test_evaluation_run_id": test_id,
            },
            "versions": {"lightgbm": lgb.__version__, "python": platform.python_version()},
        })

        load_bundle(tmp)  # the same checks the API runs at startup
        OUT.mkdir(parents=True, exist_ok=True)
        for name in (*files, MANIFEST):
            shutil.copyfile(tmp / name, OUT / name)
    bundle = load_bundle(OUT)
    print(f"wrote {OUT.relative_to(PROJECT_ROOT)}: {bundle.model_version}")
    for name in (*files, MANIFEST):
        print(f"  {name:16s} {(OUT / name).stat().st_size:>10,} bytes")


if __name__ == "__main__":
    main()
