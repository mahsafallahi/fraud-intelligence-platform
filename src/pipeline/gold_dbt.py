"""Run `dbt build` for the Gold layer and turn its result into a task outcome.

dbt lives in its own virtualenv, so it is run as a subprocess, not imported.
Its output is streamed to the log line by line.

Outcome by dbt exit code:
- 0: success, including soft-rule warnings (severity: warn does not change the
  exit code). Every warning is logged on its own "SOFT RULE WARNING" line so it
  is easy to spot.
- 1: a model error or a failed hard test. Raised as `fail_with` (in Airflow:
  AirflowFailException), so the task fails at once: retrying cannot fix data.
- anything else (2: dbt itself crashed): DbtRuntimeError, a normal failure, so
  the task's retry applies.

dbt's build output (target/) and logs are written outside the mounted project
folder, so the container never overwrites the files of a local dbt run.
Only a small allow-list of environment variables reaches dbt: the Airflow
container holds secrets (e.g. the PII key) that dbt does not need.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
from collections import Counter
from collections.abc import Callable, Mapping
from pathlib import Path

from src.utils.paths import PROJECT_ROOT

log = logging.getLogger(__name__)

DBT_PROJECT_DIR = PROJECT_ROOT / "src" / "dbt"
DBT_EXECUTABLE = "/opt/dbt-venv/bin/dbt"
DBT_TARGET_PATH = Path("/tmp/dbt-target")
DBT_LOG_PATH = Path("/opt/airflow/logs/dbt")

# Variables dbt needs (paths, locale, temp dirs); everything else is withheld.
ENV_ALLOWLIST = frozenset({
    "PATH", "HOME", "USERPROFILE", "SYSTEMROOT", "TEMP", "TMP", "TMPDIR",
    "LANG", "LC_ALL", "DATA_DIR",
})

Runner = Callable[[list[str], Path, dict[str, str]], int]


class DbtRuntimeError(RuntimeError):
    """dbt could not finish (not a data problem); the task may retry."""


def dbt_env(environ: Mapping[str, str] | None = None) -> dict[str, str]:
    source = os.environ if environ is None else environ
    return {k: v for k, v in source.items() if k in ENV_ALLOWLIST or k.startswith("DBT_")}


def build_command(
    dbt_executable: str, project_dir: Path, target: str, target_path: Path, log_path: Path
) -> list[str]:
    return [
        dbt_executable, "--no-use-colors", "build",
        "--project-dir", str(project_dir),
        "--profiles-dir", str(project_dir),
        "--target", target,
        "--target-path", str(target_path),
        "--log-path", str(log_path),
    ]


def stream_command(cmd: list[str], cwd: Path, env: dict[str, str]) -> int:
    """Run cmd, log its output line by line, return the exit code."""
    with subprocess.Popen(
        cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace",
    ) as process:
        for line in process.stdout:
            log.info("dbt | %s", line.rstrip())
        return process.wait()


def read_run_results(target_path: Path) -> list[dict]:
    path = target_path / "run_results.json"
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8")).get("results", [])


def run_dbt_build(
    target: str,
    fail_with: type[Exception],
    project_dir: Path = DBT_PROJECT_DIR,
    dbt_executable: str = DBT_EXECUTABLE,
    target_path: Path = DBT_TARGET_PATH,
    log_path: Path = DBT_LOG_PATH,
    runner: Runner = stream_command,
) -> dict[str, int]:
    """Run `dbt build`; return the count of node results by status (pass, warn, ...)."""
    results_file = target_path / "run_results.json"
    if results_file.exists():
        results_file.unlink()  # never report an older run's results

    cmd = build_command(dbt_executable, project_dir, target, target_path, log_path)
    exit_code = runner(cmd, project_dir, dbt_env())
    results = read_run_results(target_path)
    summary = dict(Counter(r.get("status", "unknown") for r in results))

    if exit_code == 0:
        if not results:
            raise DbtRuntimeError("dbt build exited 0 but wrote no run_results.json")
        for r in results:
            if r.get("status") == "warn":
                log.warning("SOFT RULE WARNING: %s (%s row(s))", r.get("unique_id"), r.get("failures"))
        log.info("dbt build succeeded: %s", summary)
        return summary

    if exit_code == 1:
        failed = [f"{r.get('unique_id')} ({r.get('status')})" for r in results
                  if r.get("status") in ("fail", "error")]
        raise fail_with(f"dbt build failed (hard test or model error): {failed or summary}")

    raise DbtRuntimeError(f"dbt exited with code {exit_code}: {summary}")
