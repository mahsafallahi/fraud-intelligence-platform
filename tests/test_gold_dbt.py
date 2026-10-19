"""Tests for the Gold dbt wrapper (src/pipeline/gold_dbt.py), with a fake runner."""

import json
import logging
from pathlib import Path

import pytest

from src.pipeline.gold_dbt import (
    DbtRuntimeError,
    build_command,
    dbt_env,
    run_dbt_build,
)


class FakeFailException(Exception):
    """Stands in for airflow.sdk.exceptions.AirflowFailException."""


def result(unique_id, status, failures=0):
    return {"unique_id": unique_id, "status": status, "failures": failures}


def fake_runner(exit_code, results=None):
    """A runner that writes run_results.json like dbt would, then returns exit_code."""
    calls = []

    def run(cmd, cwd, env):
        calls.append((cmd, cwd, env))
        if results is not None:
            target_path = cmd[cmd.index("--target-path") + 1]
            (Path(target_path) / "run_results.json").write_text(
                json.dumps({"results": results}), encoding="utf-8")
        return exit_code

    run.calls = calls
    return run


def build(tmp_path, runner):
    return run_dbt_build(
        target="airflow", fail_with=FakeFailException, project_dir=tmp_path / "proj",
        dbt_executable="dbt", target_path=tmp_path, log_path=tmp_path / "logs", runner=runner,
    )


PASSING = [result("model.fraud_gold.fact_transactions", "success"),
           result("test.fraud_gold.assert_gold_hard_rules", "pass")]


def test_command_targets_airflow_and_writes_outside_the_project(tmp_path):
    cmd = build_command("dbt", tmp_path / "proj", "airflow", tmp_path / "t", tmp_path / "l")
    assert cmd[:3] == ["dbt", "--no-use-colors", "build"]
    assert cmd[cmd.index("--target") + 1] == "airflow"
    assert cmd[cmd.index("--profiles-dir") + 1] == str(tmp_path / "proj")
    assert cmd[cmd.index("--target-path") + 1] == str(tmp_path / "t")
    assert cmd[cmd.index("--log-path") + 1] == str(tmp_path / "l")


def test_exit_0_returns_summary(tmp_path):
    runner = fake_runner(0, PASSING)
    assert build(tmp_path, runner) == {"success": 1, "pass": 1}
    assert runner.calls[0][1] == tmp_path / "proj"  # runs in the project folder


def test_exit_0_with_soft_warning_succeeds_and_logs_it(tmp_path, caplog):
    results = PASSING + [result("test.fraud_gold.assert_gold_soft_rules", "warn", failures=2)]
    with caplog.at_level(logging.WARNING):
        summary = build(tmp_path, fake_runner(0, results))
    assert summary["warn"] == 1
    assert "SOFT RULE WARNING: test.fraud_gold.assert_gold_soft_rules (2 row(s))" in caplog.text


def test_exit_1_raises_fail_exception_with_failing_nodes(tmp_path):
    results = PASSING + [result("test.fraud_gold.assert_gold_hard_rules_x", "fail", failures=3)]
    with pytest.raises(FakeFailException, match="assert_gold_hard_rules_x \\(fail\\)"):
        build(tmp_path, fake_runner(1, results))


@pytest.mark.parametrize("exit_code", [2, 3, -9])
def test_other_exit_codes_are_retryable_runtime_errors(tmp_path, exit_code):
    with pytest.raises(DbtRuntimeError, match=f"code {exit_code}"):
        build(tmp_path, fake_runner(exit_code, None))


def test_exit_0_without_results_is_a_runtime_error(tmp_path):
    with pytest.raises(DbtRuntimeError, match="no run_results.json"):
        build(tmp_path, fake_runner(0, None))


def test_results_of_an_older_run_are_never_reported(tmp_path):
    (tmp_path / "run_results.json").write_text(json.dumps({"results": PASSING}), encoding="utf-8")
    with pytest.raises(DbtRuntimeError, match="no run_results.json"):
        build(tmp_path, fake_runner(0, None))  # dbt wrote nothing this time


def test_only_allowlisted_environment_reaches_dbt():
    env = dbt_env({
        "PATH": "/bin", "DATA_DIR": "/data", "DBT_LOG_FORMAT": "text",
        "AIRFLOW_VAR_PII_HASH_SECRET": "fake-secret", "AIRFLOW_VAR_CENSUS_API_KEY": "fake-key",
        "AIRFLOW__CORE__FERNET_KEY": "fake-fernet",
    })
    assert env == {"PATH": "/bin", "DATA_DIR": "/data", "DBT_LOG_FORMAT": "text"}


def test_runner_receives_the_filtered_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("AIRFLOW_VAR_PII_HASH_SECRET", "fake-secret")
    monkeypatch.setenv("DATA_DIR", "/data")
    runner = fake_runner(0, PASSING)
    build(tmp_path, runner)
    env = runner.calls[0][2]
    assert "AIRFLOW_VAR_PII_HASH_SECRET" not in env
    assert env["DATA_DIR"] == "/data"
