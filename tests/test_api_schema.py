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
    r = api.post("/score", json=good)
    body = r.json()
    assert r.status_code == 200, body
    assert set(body) == {"request_id", "client_ref", "model_version", "raw_score", "probability_raw",
                         "probability_calibrated", "alert", "threshold", "reasons"}
    assert body["client_ref"] == "abc-123"
    assert r.headers["x-request-id"] == body["request_id"]
    assert 0 < body["probability_raw"] < 1 and 0 < body["probability_calibrated"] < 1
    assert math.isclose(body["probability_raw"], 1 / (1 + math.exp(-body["raw_score"])), rel_tol=1e-12)
    assert body["alert"] == (body["probability_raw"] >= body["threshold"])
    assert body["model_version"].startswith("F_coarse_profile-")


def test_reasons_are_positive_top3_with_values_and_category_names(api, good):
    body = api.post("/score", json={**good, "amt": 1500.0, "trans_ts": "2020-12-20T23:30:00"}).json()
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
    r = api.get("/health")
    assert (r.status_code, r.json()) == (200, {"status": "ok"})
    r = api.get("/model-info")
    assert r.status_code == 200
    info = r.json()
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
    r = api.post("/score", json={**good, **change})
    body = r.json()
    assert r.status_code == 422
    assert body["code"] == "validation_error"
    assert any(e["loc"][-1] == field for e in body["detail"]), body


def test_missing_field_is_422(api, good):
    body = {k: v for k, v in good.items() if k != "amt"}
    r = api.post("/score", json=body)
    assert r.status_code == 422 and r.json()["detail"][0]["loc"] == ["body", "amt"]


def test_unknown_category_lists_the_allowed_values(api, good):
    r = api.post("/score", json={**good, "category": "casino"})
    body = r.json()
    assert r.status_code == 422 and body["code"] == "validation_error"
    msg = body["detail"][0]["msg"]
    assert "shopping_net" in msg and "travel" in msg


def test_validation_errors_do_not_echo_input(api, good):
    body = api.post("/score", json={**good, "amt": -123.45}).json()
    assert all(set(e) == {"loc", "msg", "type"} for e in body["detail"])


def test_malformed_json_is_422(api):
    r = api.post("/score", content=b"{not json", headers={"content-type": "application/json"})
    assert r.status_code == 422 and r.json()["code"] == "validation_error"
