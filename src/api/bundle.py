"""Load and verify the model bundle of the scoring API (Phase 6).

Design (decided 2026-11-19, docs/api_design.md):
- The bundle is a folder of plain files: the native LightGBM text model, the
  calibrator JSON, the column and category lists, the alert threshold and a
  manifest with a SHA-256 per file. No MLflow, no pickle, no skops at serving.
- Every file is hashed at load time and compared with the manifest; the model
  must have exactly FINAL_COLUMNS and the category lists of features.py. Any
  mismatch raises BundleError and the API does not start (fail fast, never a
  degraded service with a wrong model).
- The alert threshold applies to F's RAW probability (validation top 1,000 in
  notebook 04), compared with >=.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb

from src.models.calibration import Calibrator
from src.models.features import CATEGORIES, CHANNELS, FINAL_COLUMNS

BUNDLE_FORMAT = 1
BUNDLE_FILES = ("model.txt", "calibrator.json", "features.json", "threshold.json")
MANIFEST = "manifest.json"


class BundleError(RuntimeError):
    """The bundle is incomplete, altered or does not match this code."""


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bundle_id(file_hashes: dict) -> str:
    """One hash for the whole bundle: SHA-256 of 'name:hash' lines, sorted by name."""
    text = "".join(f"{name}:{file_hashes[name]}\n" for name in sorted(file_hashes))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Bundle:
    booster: lgb.Booster
    calibrator: Calibrator
    columns: list
    threshold: float
    manifest: dict

    @property
    def bundle_id(self) -> str:
        return self.manifest["bundle_id"]

    @property
    def model_version(self) -> str:
        return f"{self.manifest['model_name']}-{self.bundle_id[:12]}"


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_files(folder: Path) -> dict:
    """Check the manifest's file list and every hash; return the manifest."""
    folder = Path(folder)
    if not (folder / MANIFEST).is_file():
        raise BundleError(f"no {MANIFEST} in {folder}")
    manifest = _read_json(folder / MANIFEST)
    if manifest.get("bundle_format") != BUNDLE_FORMAT:
        raise BundleError(f"unsupported bundle_format: {manifest.get('bundle_format')}")
    hashes = manifest.get("files", {})
    if sorted(hashes) != sorted(BUNDLE_FILES):
        raise BundleError(f"manifest lists {sorted(hashes)}, expected {sorted(BUNDLE_FILES)}")
    for name, expected in hashes.items():
        path = folder / name
        if not path.is_file():
            raise BundleError(f"missing bundle file: {name}")
        actual = sha256_file(path)
        if actual != expected:
            raise BundleError(f"hash mismatch for {name}: manifest {expected[:12]}, file {actual[:12]}")
    if manifest.get("bundle_id") != bundle_id(hashes):
        raise BundleError("bundle_id does not match the file hashes")
    return manifest


def load_bundle(folder) -> Bundle:
    """Verify, then load the bundle. Raises BundleError on any mismatch."""
    folder = Path(folder)
    manifest = verify_files(folder)

    features = _read_json(folder / "features.json")
    if features.get("columns") != FINAL_COLUMNS:
        raise BundleError("features.json columns differ from FINAL_COLUMNS")
    if features.get("categories") != {"category": CATEGORIES, "channel": CHANNELS}:
        raise BundleError("features.json category lists differ from features.py")

    booster = lgb.Booster(model_file=str(folder / "model.txt"))
    if booster.feature_name() != FINAL_COLUMNS:
        raise BundleError(f"model columns {booster.feature_name()} differ from FINAL_COLUMNS")
    if booster.pandas_categorical != [CATEGORIES, CHANNELS]:
        raise BundleError("model category mapping differs from CATEGORIES / CHANNELS")

    calibrator = Calibrator.from_dict(_read_json(folder / "calibrator.json"))

    thr = _read_json(folder / "threshold.json")
    if thr.get("applies_to") != "raw_probability" or thr.get("rule") != ">=":
        raise BundleError(f"unexpected threshold definition: {thr}")
    value = float(thr["value"])
    if not 0 < value < 1:
        raise BundleError(f"threshold must be in (0, 1), got {value}")

    return Bundle(booster=booster, calibrator=calibrator, columns=list(FINAL_COLUMNS),
                  threshold=value, manifest=manifest)
