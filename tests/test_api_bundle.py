"""The committed model bundle (models/F) and its loader (src/api/bundle.py)."""

import json
import shutil

import numpy as np
import pandas as pd
import pytest

from src.api.bundle import BUNDLE_FILES, MANIFEST, BundleError, bundle_id, load_bundle, sha256_file
from src.models.features import CATEGORIES, CHANNELS, FINAL_COLUMNS
from tests.conftest import PROJECT_ROOT

BUNDLE_DIR = PROJECT_ROOT / "models" / "F"


@pytest.fixture(scope="module")
def bundle():
    return load_bundle(BUNDLE_DIR)


@pytest.fixture
def copy(tmp_path):
    dst = tmp_path / "F"
    shutil.copytree(BUNDLE_DIR, dst)
    return dst


def test_manifest_hashes_match_the_committed_files():
    manifest = json.loads((BUNDLE_DIR / MANIFEST).read_text(encoding="utf-8"))
    assert sorted(manifest["files"]) == sorted(BUNDLE_FILES)
    for name, expected in manifest["files"].items():
        assert sha256_file(BUNDLE_DIR / name) == expected, name
    assert manifest["bundle_id"] == bundle_id(manifest["files"])


def test_bundle_loads_with_the_expected_contents(bundle):
    assert bundle.columns == FINAL_COLUMNS
    assert bundle.booster.feature_name() == FINAL_COLUMNS
    assert bundle.booster.pandas_categorical == [CATEGORIES, CHANNELS]
    assert bundle.calibrator.method == "platt"
    assert 0 < bundle.threshold < 1
    assert bundle.manifest["model_name"] == "F_coarse_profile"
    assert bundle.model_version == f"F_coarse_profile-{bundle.bundle_id[:12]}"
    assert bundle.booster.num_trees() == 1012  # n_trees logged for F in MLflow


def test_threshold_is_the_exact_logged_value(bundle):
    # test_evaluation logged threshold.f_prob = 0.2940968658 (10 significant digits)
    assert f"{bundle.threshold:.10g}" == "0.2940968658"


def test_model_scores_a_row_and_contributions_add_up(bundle):
    row = pd.DataFrame([{
        "amt": 950.0, "hour": 23, "is_night": 1, "day_of_week": 2,
        "category": "shopping_net", "channel": "online",
        "secs_since_prev": 40.0, "amt_vs_card_median": 12.0, "card_txn_1h": 3, "card_txn_24h": 6,
        "distance_km": 80.0, "is_public_holiday_in_state": 0, "is_nationwide_public_holiday": 0,
        "is_bank_holiday": 0, "age_band": 50, "urban": 0,
    }])
    for col in ("category", "channel"):
        row[col] = pd.Categorical(row[col], categories=CATEGORIES if col == "category" else CHANNELS)
    raw = bundle.booster.predict(row, raw_score=True)
    contrib = bundle.booster.predict(row, pred_contrib=True)
    assert contrib.shape == (1, len(FINAL_COLUMNS) + 1)
    assert np.isclose(contrib.sum(), raw[0], atol=1e-9)


def test_altered_file_is_refused(copy):
    with open(copy / "threshold.json", "ab") as f:
        f.write(b" ")
    with pytest.raises(BundleError, match="hash mismatch for threshold.json"):
        load_bundle(copy)


def test_missing_file_is_refused(copy):
    (copy / "calibrator.json").unlink()
    with pytest.raises(BundleError, match="missing bundle file: calibrator.json"):
        load_bundle(copy)


def test_missing_manifest_is_refused(copy):
    (copy / MANIFEST).unlink()
    with pytest.raises(BundleError, match="no manifest.json"):
        load_bundle(copy)


def test_manifest_with_an_edited_hash_is_refused(copy):
    manifest = json.loads((copy / MANIFEST).read_text(encoding="utf-8"))
    manifest["files"]["model.txt"] = "0" * 64
    (copy / MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(BundleError, match="hash mismatch for model.txt"):
        load_bundle(copy)


def test_consistently_rehashed_but_wrong_columns_are_refused(copy):
    # Someone edits features.json AND updates the manifest: the hash check passes,
    # the content check against features.py must still refuse it.
    features = json.loads((copy / "features.json").read_text(encoding="utf-8"))
    features["columns"] = list(reversed(features["columns"]))
    (copy / "features.json").write_bytes(json.dumps(features).encode("utf-8"))
    manifest = json.loads((copy / MANIFEST).read_text(encoding="utf-8"))
    manifest["files"]["features.json"] = sha256_file(copy / "features.json")
    manifest["bundle_id"] = bundle_id(manifest["files"])
    (copy / MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(BundleError, match="columns differ from FINAL_COLUMNS"):
        load_bundle(copy)
