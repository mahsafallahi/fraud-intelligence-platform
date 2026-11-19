"""Request logs carry no personal or transaction data (synthetic; runs in CI)."""

import json
import logging

import pytest


@pytest.fixture
def request_body(synthetic):
    t = synthetic[1].iloc[3]
    return {"card_key": t.card_key, "trans_ts": "2020-12-01T10:11:12", "amt": 4321.98,
            "category": "travel", "merch_lat": 41.123456, "merch_long": -77.654321,
            "client_ref": "client-secret-ref"}


def records(caplog):
    return [r.getMessage() for r in caplog.records if r.name == "fraud_api"]


def assert_clean(messages, request_body):
    text = "\n".join(messages)
    for value in (request_body["card_key"], "4321.98", "41.123456", "-77.654321",
                  "client-secret-ref", "2020-12-01"):
        assert value not in text, value


def test_score_log_is_one_json_line_without_request_data(api, caplog, request_body):
    with caplog.at_level(logging.INFO, logger="fraud_api"):
        status, body, _ = api.post("/score", request_body)
    assert status == 200
    lines = records(caplog)
    assert len(lines) == 1
    event = json.loads(lines[0])
    assert event["event"] == "request" and event["status"] == 200
    assert event["request_id"] == body["request_id"]
    assert set(event) == {"event", "request_id", "method", "path", "status", "latency_ms",
                          "model_version", "alert", "probability_calibrated"}
    assert_clean(lines, request_body)


@pytest.mark.parametrize("change, code", [
    ({"card_key": "e" * 32}, "unknown_card"),
    ({"amt": -1}, "validation_error"),
    ({"trans_ts": "2021-06-01T00:00:00"}, "date_outside_calendar"),
])
def test_error_logs_carry_only_the_code(api, caplog, request_body, change, code):
    with caplog.at_level(logging.INFO, logger="fraud_api"):
        status, _, _ = api.post("/score", {**request_body, **change})
    assert status == 422
    lines = records(caplog)
    assert json.loads(lines[-1])["error"] == code
    assert_clean(lines, request_body)
    assert "e" * 32 not in "\n".join(lines)  # the unknown card key is not logged either


def test_unexpected_error_is_500_and_logs_only_the_type(api, caplog, request_body, monkeypatch):
    import src.api.main as main

    def boom(*args, **kwargs):
        raise RuntimeError(f"secret detail {request_body['card_key']}")

    monkeypatch.setattr(main, "score_transaction", boom)
    with caplog.at_level(logging.INFO, logger="fraud_api"):
        status, body, _ = api.post("/score", request_body)
    assert status == 500 and body == {"code": "internal_error"}
    lines = records(caplog)
    assert json.loads(lines[-1])["error"] == "internal_error:RuntimeError"
    assert_clean(lines, request_body)
