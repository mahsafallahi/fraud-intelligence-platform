"""Alert ranking and grouped evaluation (Phase 4, steps E and F).

Design (decided 2026-11-09):
- expected_loss = calibrated probability x amount: ranks alerts by the dollars a
  case is expected to cost, instead of by probability alone.
- group_alert_metrics gives recall and false-alarm rate per group (gender, age
  band, day/night) for one set of alerts.
- card_bootstrap_many resamples whole cards exactly like
  src.models.evaluate.card_bootstrap (same seed, same draws), but computes many
  statistics from each resample. Statistics from one resample are paired, so
  their differences are valid.
- recall_tie_range: when several rows share the score at the alert cut-off, the
  rows that get in depend on tie-breaking. It returns the lowest and highest
  recall any tie-breaking could give.
"""

import numpy as np
import pandas as pd


def expected_loss(p, amt) -> np.ndarray:
    """Calibrated probability times amount, per transaction."""
    p = np.asarray(p, dtype="float64")
    amt = np.asarray(amt, dtype="float64")
    if p.shape != amt.shape:
        raise ValueError("p and amt must have the same shape")
    if np.isnan(p).any() or ((p < 0) | (p > 1)).any():
        raise ValueError("p must be probabilities in [0, 1]")
    if np.isnan(amt).any() or (amt < 0).any():
        raise ValueError("amounts must be non-negative without missing values")
    return p * amt


def group_alert_metrics(y, flagged, groups) -> pd.DataFrame:
    """Recall and false-alarm rate per group for one set of alerts."""
    y = np.asarray(y).astype(bool)
    flagged = np.asarray(flagged).astype(bool)
    groups = np.asarray(groups)
    if not (len(y) == len(flagged) == len(groups)):
        raise ValueError("y, flagged and groups must have the same length")
    keys, codes = np.unique(groups, return_inverse=True)
    k = len(keys)
    n = np.bincount(codes, minlength=k)
    n_fraud = np.bincount(codes, weights=y, minlength=k)
    caught = np.bincount(codes, weights=y & flagged, minlength=k)
    false_alarms = np.bincount(codes, weights=~y & flagged, minlength=k)
    n_legit = n - n_fraud
    with np.errstate(invalid="ignore", divide="ignore"):
        recall = np.where(n_fraud > 0, caught / n_fraud, np.nan)
        far = np.where(n_legit > 0, false_alarms / n_legit, np.nan)
    return pd.DataFrame({
        "n": n, "n_fraud": n_fraud.astype(int), "fraud_caught": caught.astype(int), "recall": recall,
        "n_legit": n_legit.astype(int), "false_alarms": false_alarms.astype(int), "false_alarm_rate": far,
    }, index=pd.Index(keys, name="group"))


def card_bootstrap_many(cards, stat_fn, n_boot: int = 500, seed: int = 42,
                        level: float = 0.95) -> pd.DataFrame:
    """Card-level bootstrap for several statistics at once.

    stat_fn receives row positions (rows of a card drawn twice appear twice) and
    returns a dict of numbers. Returns one row per statistic: the full-data
    estimate, a percentile interval and defined_draws, the number of draws in
    which the statistic had a value. defined_draws = 0 flags a statistic that is
    never defined (for example recall of a group without fraud): its bounds are
    NaN. Draws are identical to card_bootstrap.
    """
    codes, uniques = pd.factorize(np.asarray(cards))
    if (codes < 0).any():
        raise ValueError("missing card keys")
    order = np.argsort(codes, kind="stable")
    counts = np.bincount(codes, minlength=len(uniques))
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    blocks = [order[starts[c]:starts[c] + counts[c]] for c in range(len(uniques))]

    estimate = pd.Series(stat_fn(np.arange(len(codes))), dtype="float64")
    rng = np.random.default_rng(seed)
    samples = np.empty((n_boot, len(estimate)))
    for b in range(n_boot):
        drawn = rng.integers(0, len(uniques), size=len(uniques))
        s = pd.Series(stat_fn(np.concatenate([blocks[c] for c in drawn])), dtype="float64")
        samples[b] = s.reindex(estimate.index).to_numpy()

    # A statistic can be undefined in every draw (recall of a group with no fraud).
    # It gets NaN bounds explicitly and defined_draws = 0, instead of a numpy warning.
    tail = 100 * (1 - level) / 2
    defined = np.isfinite(samples).sum(axis=0)
    low = np.full(samples.shape[1], np.nan)
    high = np.full(samples.shape[1], np.nan)
    ok = defined > 0
    if ok.any():
        low[ok], high[ok] = np.nanpercentile(samples[:, ok], [tail, 100 - tail], axis=0)
    return pd.DataFrame({"estimate": estimate.to_numpy(), "low": low, "high": high,
                         "defined_draws": defined}, index=estimate.index)


def recall_tie_range(y, score, n: int) -> tuple:
    """Lowest and highest recall among the n top scores over every tie-breaking."""
    y = np.asarray(y).astype(bool)
    score = np.asarray(score, dtype="float64")
    if not 0 < n <= len(score):
        raise ValueError(f"n must be between 1 and {len(score)}")
    cut = np.sort(score)[::-1][n - 1]
    above = score > cut
    tied = score == cut
    free = n - int(above.sum())
    fraud_above = int((y & above).sum())
    fraud_tied, legit_tied = int((y & tied).sum()), int((~y & tied).sum())
    lowest = fraud_above + max(0, free - legit_tied)
    highest = fraud_above + min(free, fraud_tied)
    total = int(y.sum())
    return lowest / total, highest / total
