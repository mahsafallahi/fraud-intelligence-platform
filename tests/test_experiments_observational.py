"""Tests for src/experiments/observational.py (positive controls with a known effect)."""

import numpy as np
import pandas as pd
import pytest

from src.experiments.observational import (
    compare_estimators, doubly_robust, fit_propensity, ipw, make_semi_synthetic, naive_diff,
    overlap_summary, psm_att, smd, summarise_estimators,
)


def fake_covariates(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "night_share": rng.beta(2, 8, n), "max_score": rng.beta(1, 5, n),
        "log_spend": rng.normal(9, 1, n), "age_band": rng.choice(np.arange(10, 100, 10), n),
        "urban": (rng.random(n) < 0.2).astype(float),
    })


LOSSES = np.random.default_rng(9).lognormal(8, 0.8, 500)


def test_semi_synthetic_estimands_are_exact():
    d = make_semi_synthetic(fake_covariates(), 20000, LOSSES, seed=1)
    assert d["ate"] == pytest.approx((d["y0"] - d["y1"]).mean())
    assert d["att"] == pytest.approx((d["y0"] - d["y1"])[d["t"]].mean())
    assert (d["y0"] > 0).mean() == pytest.approx(0.10, abs=0.01)
    assert d["t"].mean() == pytest.approx(0.30, abs=0.01)


def test_confounding_biases_the_naive_difference():
    d = make_semi_synthetic(fake_covariates(), 50000, LOSSES, seed=2)
    assert abs(-naive_diff(d["y"], d["t"]) - d["ate"]) > 0.3 * d["ate"]


def test_ipw_and_doubly_robust_recover_the_truth_with_measured_confounders():
    d = make_semi_synthetic(fake_covariates(), 60000, LOSSES, seed=3)
    X = d["X"][["night_share", "max_score", "log_spend", "age_band", "urban"]]
    ps = fit_propensity(X, d["t"])
    w = ipw(d["y"], d["t"], ps, "ate", trim=None)
    dr = doubly_robust(d["y"], d["t"], X, ps)
    assert -w["est"] == pytest.approx(d["ate"], abs=4 * w["se"])
    assert -dr["est"] == pytest.approx(d["ate"], abs=4 * dr["se"])


def test_weighting_balances_covariates():
    d = make_semi_synthetic(fake_covariates(), 40000, LOSSES, seed=4)
    X = d["X"][["night_share", "max_score", "log_spend", "age_band", "urban"]]
    ps = fit_propensity(X, d["t"])
    w = np.where(d["t"], 1 / ps, 1 / (1 - ps))
    assert smd(X, d["t"]).abs().max() > 0.2
    assert smd(X, d["t"], w).abs().max() < 0.1


def test_matching_targets_the_matched_treated():
    d = make_semi_synthetic(fake_covariates(), 40000, LOSSES, seed=5)
    X = d["X"][["night_share", "max_score", "log_spend", "age_band", "urban"]]
    ps = fit_propensity(X, d["t"])
    m = psm_att(d["y"], d["t"], ps)
    truth = (d["y0"] - d["y1"])[m["matched_treated"]].mean()
    assert -m["att"] == pytest.approx(truth, rel=0.25)
    assert m["matched_treated"].sum() + m["n_unmatched"] == d["t"].sum()


def test_overlap_summary_shape():
    rng = np.random.default_rng(6)
    ps = rng.uniform(0.01, 0.99, 1000)
    t = rng.random(1000) < ps
    s = overlap_summary(ps, t)
    assert list(s.index) == ["treated", "control"] and s["n"].sum() == 1000


def test_hidden_confounder_leaves_bias_that_measured_confounders_remove():
    cov = fake_covariates(4000, seed=7)
    full = summarise_estimators(compare_estimators(cov, LOSSES, 30000, n_reps=8, seed=100))
    hidden = summarise_estimators(compare_estimators(cov, LOSSES, 30000, n_reps=8, seed=100,
                                                     observed=["max_score", "log_spend", "age_band", "urban"]))
    dr_full, dr_hidden = full.loc["doubly robust"], hidden.loc["doubly robust"]
    assert abs(dr_full["bias"]) < 0.1 * dr_full["mean_truth"]
    assert abs(dr_hidden["bias"]) > 2 * abs(dr_full["bias"])
    assert abs(full.loc["naive", "bias"]) > abs(dr_full["bias"])
