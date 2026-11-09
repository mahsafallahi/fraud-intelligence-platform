"""Tests for src/models/ranking.py."""

import numpy as np
import pandas as pd
import pytest

from src.models.evaluate import card_bootstrap, top_n_mask
from src.models.ranking import (
    card_bootstrap_many,
    expected_loss,
    group_alert_metrics,
    recall_tie_range,
)


def test_expected_loss_is_probability_times_amount():
    assert expected_loss([0.5, 0.1], [100.0, 1000.0]).tolist() == [50.0, 100.0]


def test_expected_loss_can_reorder_alerts():
    p, amt = np.array([0.9, 0.2]), np.array([10.0, 1000.0])
    assert top_n_mask(p, 1).tolist() == [True, False]
    assert top_n_mask(expected_loss(p, amt), 1).tolist() == [False, True]


def test_expected_loss_rejects_bad_inputs():
    with pytest.raises(ValueError, match="same shape"):
        expected_loss([0.1], [1.0, 2.0])
    with pytest.raises(ValueError, match="probabilities"):
        expected_loss([1.2], [1.0])
    with pytest.raises(ValueError, match="amounts"):
        expected_loss([0.1], [-1.0])


def test_group_alert_metrics_known_values():
    y = [1, 1, 0, 0, 1, 0, 0]
    flagged = [1, 0, 1, 0, 1, 0, 1]
    groups = ["F", "F", "F", "F", "M", "M", "M"]
    t = group_alert_metrics(y, flagged, groups)
    assert t.loc["F", "recall"] == 0.5 and t.loc["M", "recall"] == 1.0
    assert t.loc["F", "false_alarm_rate"] == 0.5 and t.loc["M", "false_alarm_rate"] == 0.5
    assert t["n"].sum() == 7 and t["n_fraud"].tolist() == [2, 1]


def test_group_without_fraud_has_missing_recall():
    t = group_alert_metrics([0, 1], [1, 1], ["a", "b"])
    assert np.isnan(t.loc["a", "recall"]) and t.loc["b", "recall"] == 1.0


def test_group_alert_metrics_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        group_alert_metrics([1, 0], [1], ["a", "b"])


def fake_cards(n=600, seed=0):
    rng = np.random.default_rng(seed)
    cards = rng.integers(0, 60, n)
    y = (rng.random(n) < 0.2).astype(int)
    score = rng.random(n) + 0.5 * y
    return cards, y, score


def test_card_bootstrap_many_matches_card_bootstrap_draw_for_draw():
    cards, y, score = fake_cards()
    stat = lambda idx: float(y[idx][top_n_mask(score[idx], 50)].mean())
    one = card_bootstrap(cards, stat, n_boot=200)
    many = card_bootstrap_many(cards, lambda idx: {"precision": stat(idx)}, n_boot=200)
    assert many.loc["precision", "estimate"] == pytest.approx(one["estimate"], abs=0)
    assert many.loc["precision", "low"] == pytest.approx(one["low"], abs=0)
    assert many.loc["precision", "high"] == pytest.approx(one["high"], abs=0)


def test_card_bootstrap_many_returns_every_statistic_with_ordered_bounds():
    cards, y, score = fake_cards()
    out = card_bootstrap_many(cards, lambda idx: {"a": y[idx].mean(), "b": score[idx].mean()}, n_boot=100)
    assert list(out.index) == ["a", "b"]
    assert (out["low"] <= out["estimate"]).all() and (out["estimate"] <= out["high"]).all()


def test_card_bootstrap_many_pairs_differences():
    """A statistic minus itself is exactly zero in every draw, so its interval is [0, 0]."""
    cards, y, _ = fake_cards()
    out = card_bootstrap_many(cards, lambda idx: {"diff": y[idx].mean() - y[idx].mean()}, n_boot=50)
    assert out.loc["diff"].tolist() == [0.0, 0.0, 0.0]


def test_card_bootstrap_many_rejects_missing_cards():
    with pytest.raises(ValueError, match="missing card keys"):
        card_bootstrap_many(pd.Series(["a", None]), lambda idx: {"x": 1.0}, n_boot=5)


def test_recall_tie_range_without_ties_is_a_single_value():
    y, score = [1, 0, 1, 0], [0.9, 0.8, 0.7, 0.1]
    assert recall_tie_range(y, score, 2) == (0.5, 0.5)


def test_recall_tie_range_with_ties_at_the_cut():
    # the top 3 must take 2 of the 3 rows tied at 0.5 (one fraud, two legitimate)
    y, score = [1, 1, 0, 0, 1], [0.9, 0.5, 0.5, 0.5, 0.1]
    low, high = recall_tie_range(y, score, 3)
    assert (low, high) == (pytest.approx(1 / 3), pytest.approx(2 / 3))
    caught = np.asarray(y, bool)[top_n_mask(score, 3)].sum() / 3
    assert low <= caught <= high


def test_recall_tie_range_rejects_bad_n():
    with pytest.raises(ValueError, match="n must be"):
        recall_tie_range([1, 0], [0.5, 0.4], 0)
