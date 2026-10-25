"""Tests for src/models/evaluate.py."""

import numpy as np
import pandas as pd
import pytest

from src.models.evaluate import (
    alert_metrics,
    card_bootstrap,
    pr_auc,
    rule_baseline,
    top_n_mask,
)


def test_rule_baseline_strictly_above_amount_at_night():
    features = pd.DataFrame({
        "is_night": [1, 1, 0, 1],
        "amt": [136.33, 136.34, 500.0, 1000.0],
    })
    assert list(rule_baseline(features)) == [0, 1, 0, 1]


def test_pr_auc_known_value():
    # Fraud at ranks 1 and 3: precision 1/1 and 2/3 -> average 0.8333
    assert pr_auc([1, 0, 1, 0], [0.9, 0.8, 0.7, 0.1]) == pytest.approx(5 / 6)
    assert pr_auc([1, 0, 1, 0], [0.9, 0.2, 0.8, 0.1]) == pytest.approx(1.0)


def test_pr_auc_rejects_missing_scores():
    with pytest.raises(ValueError, match="missing scores"):
        pr_auc([1, 0], [0.5, np.nan])


def test_top_n_mask_picks_highest():
    assert list(top_n_mask([0.1, 0.9, 0.5, 0.7], 2)) == [False, True, False, True]
    assert not top_n_mask([0.1, 0.9], 0).any()


def test_top_n_mask_ties_follow_row_order():
    assert list(top_n_mask([0.5, 0.5, 0.5], 2)) == [True, True, False]


def test_top_n_mask_rejects_bad_n():
    with pytest.raises(ValueError, match="n must be between"):
        top_n_mask([0.1, 0.2], 3)


def test_alert_metrics_counts_and_dollars():
    m = alert_metrics(
        y=[1, 1, 0, 0, 1],
        flagged=[1, 0, 1, 0, 1],
        amt=[100.0, 50.0, 10.0, 10.0, 300.0],
    )
    assert m["alerts"] == 3
    assert m["fraud_caught"] == 2
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["dollars_caught"] == pytest.approx(400.0)
    assert m["dollar_recall"] == pytest.approx(400 / 450)


def test_alert_metrics_without_alerts():
    m = alert_metrics(y=[1, 0], flagged=[0, 0], amt=[10.0, 5.0])
    assert m["alerts"] == 0
    assert m["recall"] == 0
    assert np.isnan(m["precision"])


def test_bootstrap_resamples_whole_cards():
    # Card a has 2 rows, card b has 1: two draws can only give 2, 3 or 4 rows.
    result = card_bootstrap(["a", "a", "b"], lambda idx: len(idx), n_boot=200)
    assert set(result["samples"]) <= {2, 3, 4}
    assert {2, 4} <= set(result["samples"])


def test_bootstrap_is_reproducible():
    values = np.array([1.0, 5.0, 2.0, 8.0, 3.0])
    cards = ["a", "b", "c", "d", "e"]
    stat = lambda idx: values[idx].mean()
    first = card_bootstrap(cards, stat, n_boot=100, seed=1)["samples"]
    again = card_bootstrap(cards, stat, n_boot=100, seed=1)["samples"]
    other = card_bootstrap(cards, stat, n_boot=100, seed=2)["samples"]
    assert np.array_equal(first, again)
    assert not np.array_equal(first, other)


def test_bootstrap_estimate_and_interval():
    values = np.array([1.0, 2.0, 3.0, 4.0])
    result = card_bootstrap(["a", "b", "c", "d"], lambda idx: values[idx].mean())
    assert result["estimate"] == pytest.approx(2.5)
    assert result["low"] <= 2.5 <= result["high"]


def test_bootstrap_rejects_missing_cards():
    with pytest.raises(ValueError, match="missing card keys"):
        card_bootstrap([None, "a"], lambda idx: len(idx))