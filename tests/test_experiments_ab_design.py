"""Tests for src/experiments/ab_design.py."""

import numpy as np
import pandas as pd
import pytest

from src.experiments.ab_design import (
    breakeven_abandonment, cuped, decide, harm_pvalue, holm, mde, obf_boundaries, peeking_fpr,
    per_card_metrics, rate_above_pvalue, ratio_delta_ci, required_n, srm_pvalue, threshold_table,
    variance_reduction, welch_analysis,
)


def small_tx():
    return pd.DataFrame({
        "card_key": ["a", "a", "a", "b", "b"],
        "ts": pd.to_datetime(["2020-10-01 23:00", "2020-10-02 12:00", "2020-10-03 01:00",
                              "2020-10-01 10:00", "2020-10-02 10:00"]),
        "amt": [100.0, 20.0, 50.0, 10.0, 5.0],
        "is_fraud": [1, 0, 1, 0, 0],
        "score": [0.9, 0.3, 0.05, 0.5, 0.01],
    })


def test_per_card_metrics_known_values_and_zero_rows():
    m = per_card_metrics(small_tx(), threshold=0.2, cards=["a", "b", "c"])
    assert m.loc["a", "fraud_loss"] == 150.0 and m.loc["a", "fraud_above"] == 100.0
    assert m.loc["a", "legit_above"] == 20.0 and m.loc["b", "legit_above"] == 10.0
    assert m.loc["a", "night_share"] == pytest.approx(2 / 3)
    assert m.loc["c"].eq(0).all()


def test_per_card_metrics_requires_columns():
    with pytest.raises(ValueError, match="missing columns"):
        per_card_metrics(small_tx().drop(columns="score"), 0.2)


def test_threshold_table_counts():
    t = threshold_table(small_tx(), [0.2], days=2).iloc[0]
    assert t["stepups"] == 3 and t["fraud_tx"] == 1 and t["legit_tx"] == 2
    assert t["fraud_dollar_share"] == pytest.approx(100 / 150)
    assert t["cards_touched"] == 2 and t["stepups_per_day"] == 1.5


def test_required_n_known_value_and_inverse_of_mde():
    assert required_n(1.0, 0.1) == pytest.approx(1569.8, abs=0.5)
    assert mde(1.0, required_n(1.0, 0.1)) == pytest.approx(0.1)
    with pytest.raises(ValueError):
        required_n(0, 1)


def test_cuped_keeps_the_mean_and_removes_rho_squared():
    rng = np.random.default_rng(0)
    x = rng.normal(size=20000)
    y = 0.6 * x + rng.normal(size=20000)
    adj, theta = cuped(y, x)
    assert adj.mean() == pytest.approx(y.mean())
    assert theta == pytest.approx(0.6, abs=0.03)
    assert 1 - adj.var() / y.var() == pytest.approx(variance_reduction(y, x), abs=1e-3)


def test_srm_pvalue():
    assert srm_pvalue(500, 500) == pytest.approx(1.0)
    assert srm_pvalue(600, 400) < 1e-9


def test_welch_analysis_recovers_a_known_difference():
    rng = np.random.default_rng(1)
    treat = rng.random(4000) < 0.5
    y = rng.normal(10, 2, 4000) - 1.0 * treat
    r = welch_analysis(y, treat)
    assert r["ci_low"] < 1.0 < r["ci_high"] and r["p"] < 1e-6


def test_welch_matches_scipy_and_handles_a_constant_arm():
    from scipy import stats
    rng = np.random.default_rng(8)
    treat = rng.random(500) < 0.5
    y = rng.lognormal(3, 1, 500)
    ref = stats.ttest_ind(y[~treat], y[treat], equal_var=False)
    r = welch_analysis(y, treat)
    ci = ref.confidence_interval(0.95)
    assert r["p"] == pytest.approx(ref.pvalue) and r["ci_low"] == pytest.approx(ci.low)
    y2 = np.where(treat, rng.normal(5, 1, 500), 0.0)  # control constant at 0
    assert harm_pvalue(y2, treat, margin=1.0) < 1e-6
    assert welch_analysis(np.zeros(10), np.arange(10) < 5)["p"] == 1.0


def test_harm_pvalue_respects_the_margin():
    rng = np.random.default_rng(2)
    treat = rng.random(4000) < 0.5
    y = rng.normal(0, 1, 4000) + 0.5 * treat
    assert harm_pvalue(y, treat, margin=0.0) < 1e-6
    assert harm_pvalue(y, treat, margin=1.0) > 0.5


def test_ratio_delta_ci_and_rate_test():
    rng = np.random.default_rng(3)
    den = rng.integers(1, 10, 5000)
    num = rng.binomial(den, 0.2)
    d = ratio_delta_ci(num, den)
    assert d["ci_low"] < 0.2 < d["ci_high"]
    assert rate_above_pvalue(num, den, 0.1) < 1e-6 and rate_above_pvalue(num, den, 0.3) > 0.99
    with pytest.raises(ValueError):
        ratio_delta_ci([1], [1])


def test_holm_known_values():
    assert holm([0.01, 0.04, 0.03]).tolist() == pytest.approx([0.03, 0.06, 0.06])


def test_peeking_inflates_false_positives_but_the_fixed_horizon_does_not():
    rng = np.random.default_rng(4)
    cum = np.cumsum(rng.normal(size=(400, 10)), axis=1)
    r = peeking_fpr(cum, n_sims=400, seed=5)
    assert r["any_look"] > r["fixed_horizon"] + 0.05
    assert 0.02 <= r["fixed_horizon"] <= 0.09


def test_obf_boundaries_shape_and_overall_alpha():
    b = obf_boundaries(5, n_sims=100_000, seed=1)
    assert (np.diff(b) < 0).all() and b[-1] == pytest.approx(2.04, abs=0.05)
    rng = np.random.default_rng(6)
    z = np.cumsum(rng.standard_normal((100_000, 5)), axis=1) / np.sqrt(np.arange(1, 6))
    assert (np.abs(z) > b).any(axis=1).mean() == pytest.approx(0.05, abs=0.005)


def test_breakeven_known_values():
    assert breakeven_abandonment(800, 1000, 0.5, 1.0) == pytest.approx(0.4)
    assert breakeven_abandonment(800, 0, 0.5, 1.0) == float("inf")


def test_decide_rule():
    assert decide(0.5, 10.0, [0.5, 0.5]) == "ship"
    assert decide(0.0001, 10.0, [0.5, 0.5]) == "invalid (SRM)"
    assert decide(0.5, -1.0, [0.5, 0.5]) == "don't ship"
    assert decide(0.5, 10.0, [0.01, 0.5]) == "don't ship"
    assert decide(0.5, -1.0, [0.5, 0.5], enrolment_short=True) == "extend"
