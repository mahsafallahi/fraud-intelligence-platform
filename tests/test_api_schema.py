"""Request validation, response shape and the endpoints (synthetic snapshot; runs in CI)."""

import math

import pytest

from src.models.features import CATEGORIES, FINAL_COLUMNS


@pytest.fixture(scope="module")
def good(synthetic):
    t = synthetic[1].iloc[-1]
    return {"card_key": t.card_key, "trans_ts": "2020-12-20T21:15:00", "amt": 912.40,
            "category": "shopping_net", "merch_lat": 40.9, "merch_long": -75.6, "client_ref": "abc-123"}


def test_score_returns_every_field(api, good):
    status, body, headers = api.post("/score", good)
    assert status == 200, body
    assert set(body) == {"request_id", "client_ref", "model_version", "raw_score", "probability_raw",
                         "probability_calibrated", "alert", "threshold", "reasons"}
    assert body["client_ref"] == "abc-123"
    assert headers["x-request-id"] == body["request_id"]
    assert 0 < body["probability_raw"] < 1 and 0 < body["probability_calibrated"] < 1
    assert math.isclose(body["probability_raw"], 1 / (1 + math.exp(-body["raw_score"])), rel_tol=1e-12)
    assert body["alert"] == (body["probability_raw"] >= body["threshold"])
    assert body["model_version"].startswith("F_coarse_profile-")


def test_reasons_are_positive_top3_with_values_and_category_names(api, good):
    _, body, _ = api.post("/score", {**good, "amt": 1500.0, "trans_ts": "2020-12-20T23:30:00"})
    reasons = body["reasons"]
    assert 1 <= len(reasons) <= 3
    contributions = [r["contribution"] for r in reasons]
    assert all(c > 0 for c in contributions) and contributions == sorted(contributions, reverse=True)
    assert all(r["feature"] in FINAL_COLUMNS for r in reasons)
    for r in reasons:
        if r["feature"] == "category":
            assert r["value"] in CATEGORIES
        if r["feature"] == "amt":
            assert r["value"] == 1500.0


def test_health_and_model_info(api):
    assert api.get("/health")[:2] == (200, {"status": "ok"})
    status, info, _ = api.get("/model-info")
    assert status == 200
    assert info["columns"] == FINAL_COLUMNS
    assert info["threshold"]["applies_to"] == "probability_raw"
    assert info["calibrator"] == {"method": "platt"}
    assert info["mlflow"]["model_run_id"] == "d6449e06aeea4c7bb7999c313f5d165e"
    assert info["snapshot"]["source_file"] == "synthetic"


@pytest.mark.parametrize("change, field", [
    ({"card_key": "XYZ"}, "card_key"),
    ({"card_key": "A" * 32}, "card_key"),
    ({"amt": 0}, "amt"),
    ({"amt": -5}, "amt"),
    ({"amt": "NaN"}, "amt"),
    ({"merch_lat": 91}, "merch_lat"),
    ({"merch_long": -181}, "merch_long"),
    ({"trans_ts": "2020-12-20T21:15:00+01:00"}, "trans_ts"),
    ({"trans_ts": "not a time"}, "trans_ts"),
    ({"client_ref": "x" * 65}, "client_ref"),
    ({"is_fraud": 0}, "is_fraud"),            # extra fields are refused
])
def test_invalid_requests_are_422_with_the_field(api, good, change, field):
    status, body, _ = api.post("/score", {**good, **change})
    assert status == 422
    assert body["code"] == "validation_error"
    assert any(e["loc"][-1] == field for e in body["detail"]), body


def test_missing_field_is_422(api, good):
    body = {k: v for k, v in good.items() if k != "amt"}
    status, out, _ = api.post("/score", body)
    assert status == 422 and out["detail"][0]["loc"] == ["body", "amt"]


def test_unknown_category_lists_the_allowed_values(api, good):
    status, body, _ = api.post("/score", {**good, "category": "casino"})
    assert status == 422 and body["code"] == "validation_error"
    msg = body["detail"][0]["msg"]
    assert "shopping_net" in msg and "travel" in msg


def test_validation_errors_do_not_echo_input(api, good):
    _, body, _ = api.post("/score", {**good, "amt": -123.45})
    assert all(set(e) == {"loc", "msg", "type"} for e in body["detail"])


def test_malformed_json_is_422(api):
    status, body, _ = api.request("POST", "/score", b"{not json")
    assert status == 422 and body["code"] == "validation_error"
