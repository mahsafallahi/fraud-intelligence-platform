"""Tests for src/models/calibration.py."""

import json

import numpy as np
import pytest
from sklearn.isotonic import IsotonicRegression

from src.models.calibration import (
    Calibrator,
    brier,
    fit_isotonic,
    fit_platt,
    log_loss_clipped,
    logit,
    prior_shift,
    reliability_table,
    wilson_interval,
)


def overconfident_scores(n=20000, seed=0):
    """True probabilities, and model scores that are too extreme (twice the log-odds)."""
    rng = np.random.default_rng(seed)
    z = rng.normal(-4, 1.5, n)
    true_p = 1 / (1 + np.exp(-z))
    y = (rng.random(n) < true_p).astype(int)
    score = 1 / (1 + np.exp(-2 * z))
    return y, score, true_p


def test_platt_recovers_a_known_logit_scaling():
    y, score, _ = overconfident_scores()
    cal = fit_platt(y, score)
    assert cal.params["a"] == pytest.approx(0.5, abs=0.05)
    assert cal.params["b"] == pytest.approx(0.0, abs=0.25)


def test_platt_keeps_the_ranking():
    y, score, _ = overconfident_scores()
    cal = fit_platt(y, score)
    assert (np.diff(cal.predict(np.sort(score))) >= 0).all()


def test_calibration_improves_log_loss_and_brier():
    y, score, _ = overconfident_scores()
    for fit in (fit_platt, fit_isotonic):
        p = fit(y, score).predict(score)
        assert log_loss_clipped(y, p) < log_loss_clipped(y, score)
        assert brier(y, p) < brier(y, score)


def test_isotonic_matches_sklearn_exactly():
    y, score, _ = overconfident_scores(5000)
    cal = fit_isotonic(y, score)
    ref = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(score, y)
    new = np.concatenate([score[:100], [0.0, 1.0, 1e-9]])
    np.testing.assert_allclose(cal.predict(new), ref.predict(new), rtol=0, atol=1e-12)


def test_isotonic_creates_ties():
    y, score, _ = overconfident_scores(5000)
    p = fit_isotonic(y, score).predict(score)
    assert len(np.unique(p)) < len(np.unique(score))


def test_json_round_trip_gives_identical_predictions():
    y, score, _ = overconfident_scores(5000)
    for fit in (fit_platt, fit_isotonic):
        cal = fit(y, score)
        back = Calibrator.from_dict(json.loads(json.dumps(cal.to_dict())))
        assert np.array_equal(back.predict(score), cal.predict(score))


def test_unknown_method_and_bad_inputs_fail_loudly():
    with pytest.raises(ValueError, match="unknown method"):
        Calibrator("beta", {})
    with pytest.raises(ValueError, match="labels"):
        fit_platt([0, 2], [0.1, 0.2])
    with pytest.raises(ValueError, match="probabilities"):
        fit_isotonic([0, 1], [0.1, 1.5])
    with pytest.raises(ValueError, match="same length"):
        brier([0, 1], [0.1])
    with pytest.raises(ValueError, match="scores"):
        fit_platt([0, 1], [0.1, 0.9]).predict([np.nan])


def test_log_loss_is_finite_when_a_fraud_gets_probability_zero():
    assert np.isfinite(log_loss_clipped([1, 0], [0.0, 0.0]))
    assert log_loss_clipped([1, 0], [0.0, 0.0]) == pytest.approx(-np.log(1e-7) / 2, rel=1e-6)


def test_brier_known_value():
    assert brier([1, 0], [0.75, 0.25]) == pytest.approx(0.0625)


def test_logit_clips_at_zero_and_one():
    assert np.isfinite(logit([0.0, 1.0])).all()


def test_reliability_table_counts_and_bins():
    y, score, _ = overconfident_scores()
    table = reliability_table(y, score, n_bins=15)
    assert table["n"].sum() == len(y)
    assert table["n_fraud"].sum() == y.sum()
    assert (table["n"] > 0).all()
    assert (table["bin_low"] < table["bin_high"]).all()
    inside = (table["mean_pred"] >= table["bin_low"]) & (table["mean_pred"] <= table["bin_high"])
    assert inside.all()


def test_reliability_table_first_bin_takes_tiny_scores():
    table = reliability_table([0, 0, 1], [0.0, 1e-9, 0.9], n_bins=15)
    assert table["bin_low"].iloc[0] == 0.0 and table["n"].iloc[0] == 2


def test_wilson_interval_known_value():
    low, high = wilson_interval(5, 10)
    assert low == pytest.approx(0.2366, abs=1e-4) and high == pytest.approx(0.7634, abs=1e-4)


def test_prior_shift_identity_and_direction():
    p = np.array([0.001, 0.01, 0.5, 0.9])
    np.testing.assert_allclose(prior_shift(p, 0.004, 0.004), p, rtol=1e-12)
    assert (prior_shift(p, 0.004, 0.008) > p).all()
    assert (prior_shift(p, 0.004, 0.002) < p).all()
    with pytest.raises(ValueError, match="base rates"):
        prior_shift(p, 0.0, 0.5)


def test_prior_shift_matches_the_odds_formula():
    p, a, b = 0.2, 0.004, 0.006
    odds = p / (1 - p) * (b / (1 - b)) / (a / (1 - a))
    assert prior_shift(np.array([p]), a, b)[0] == pytest.approx(odds / (1 + odds))
