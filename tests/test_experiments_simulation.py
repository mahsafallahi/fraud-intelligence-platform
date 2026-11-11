"""Tests for src/experiments/simulation.py."""

import numpy as np
import pandas as pd
import pytest

from src.experiments.simulation import (
    apply_stepup, assign_arms, prepare_stepup, resample_population, simulate_experiments, true_effects,
)


def fake_tx(n_cards=600, seed=0):
    """Cards with legitimate spend; about 15% have a fraud episode that F scores high."""
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(n_cards):
        for _ in range(rng.integers(5, 15)):
            rows.append((f"c{c}", float(rng.lognormal(3.5, 1)), 0, float(rng.beta(1, 8))))
        if rng.random() < 0.15:
            for _ in range(rng.integers(2, 6)):
                rows.append((f"c{c}", float(rng.lognormal(5.5, 0.6)), 1, float(rng.beta(8, 2))))
    tx = pd.DataFrame(rows, columns=["card_key", "amt", "is_fraud", "score"])
    return tx, tx["card_key"].unique()


def test_prepare_stepup_triggers_and_totals():
    tx, cards = fake_tx()
    d = prepare_stepup(tx, 0.2, cards)
    assert d.fraud_loss.sum() == pytest.approx(tx.loc[tx.is_fraud == 1, "amt"].sum())
    trig = tx[tx.score >= 0.2].groupby("card_key").size().reindex(cards).fillna(0).to_numpy() > 0
    assert (d.triggered == trig).all()


def test_no_effect_changes_nothing():
    tx, cards = fake_tx()
    d = prepare_stepup(tx, 0.2, cards)
    rng = np.random.default_rng(1)
    out = apply_stepup(d, assign_arms(len(cards), rng), 0.0, 0.0, 1.0, rng)
    assert np.array_equal(out["fraud_loss"].to_numpy(), d.fraud_loss)
    assert out["lost_revenue"].eq(0).all()


def test_full_block_removes_all_stepped_up_fraud_on_treated_cards_only():
    tx, cards = fake_tx()
    d = prepare_stepup(tx, 0.2, cards)
    rng = np.random.default_rng(2)
    treat = assign_arms(len(cards), rng)
    out = apply_stepup(d, treat, 1.0, 0.0, 1.0, rng)
    np.testing.assert_allclose(out["fraud_loss"].to_numpy()[treat], (d.fraud_loss - d.fraud_above)[treat])
    np.testing.assert_allclose(out["fraud_loss"].to_numpy()[~treat], d.fraud_loss[~treat])


def test_same_seed_same_result_and_bad_rates_rejected():
    tx, cards = fake_tx()
    d = prepare_stepup(tx, 0.2, cards)
    a = simulate_experiments(d, 0.5, 0.05, 1.0, n_sims=5, seed=7)
    b = simulate_experiments(d, 0.5, 0.05, 1.0, n_sims=5, seed=7)
    pd.testing.assert_frame_equal(a, b)
    with pytest.raises(ValueError):
        apply_stepup(d, assign_arms(len(cards), np.random.default_rng(0)), 1.5, 0, 1, np.random.default_rng(0))


def test_a_known_effect_is_recovered_and_an_aa_is_not_significant():
    tx, cards = fake_tx(n_cards=800)
    d = prepare_stepup(tx, 0.2, cards)
    truth = true_effects(d, 0.6, 0.0, 1.0, d.triggered)["fraud_loss_reduction"]
    sims = simulate_experiments(d, 0.6, 0.0, 1.0, n_sims=300, seed=11)
    assert sims["primary_diff"].mean() == pytest.approx(truth, rel=0.1)
    assert (sims["primary_p"] < 0.05).mean() > 0.8
    aa = simulate_experiments(d, 0.0, 0.0, 1.0, n_sims=400, seed=12)
    assert 0.02 <= (aa["primary_p"] < 0.05).mean() <= 0.09
    assert (aa["srm_p"] < 0.001).mean() <= 0.01


def test_abandonment_beyond_the_tolerance_blocks_shipping():
    tx, cards = fake_tx(n_cards=800)
    d = prepare_stepup(tx, 0.2, cards)
    sims = simulate_experiments(d, 0.6, 0.4, 1.0, n_sims=100, seed=13)
    assert (sims["decision"] == "don't ship").mean() > 0.8


def test_resample_population_incidence():
    rng = np.random.default_rng(3)
    v = np.where(rng.random(5000) < 0.1, rng.lognormal(7, 1, 5000), 0.0)
    out = resample_population(v, 200_000, rng, incidence=0.01)
    assert (out > 0).mean() == pytest.approx(0.01, abs=0.001)
    assert len(resample_population(v, 10, rng)) == 10
    with pytest.raises(ValueError):
        resample_population(v, 10, rng, incidence=1.5)
