"""A/B experiment design for step-up verification (Phase 5).

Design (decided 2026-11-12):
- Randomisation unit and analysis unit are the card, so per-card metrics are plain
  means over cards and need no delta method. Only a pooled rate (a ratio of sums
  across cards) needs one: ratio_delta_ci.
- Primary test: Welch's t-test on per-card dollars (the business estimand is the
  mean). Mann-Whitney is not used: with about 90% zeros it tests a shift in the
  distribution, not mean dollars.
- CUPED with a pre-period covariate; variance reduction equals the squared
  correlation.
- Guardrails are one-sided harm tests against a pre-registered margin, combined
  with Holm's correction.
- Peeking and O'Brien-Fleming boundaries are simulated, without extra libraries.
"""

import numpy as np
import pandas as pd
from scipy import stats

NIGHT_HOURS = (22, 23, 0, 1, 2, 3)
SRM_ALPHA = 0.001


def per_card_metrics(tx: pd.DataFrame, threshold: float, cards=None) -> pd.DataFrame:
    """Per-card fraud loss, spend and step-up exposure for one period.

    tx needs card_key, ts, amt, is_fraud and score. With `cards`, every listed card
    gets a row (zeros for cards without transactions).
    """
    missing = {"card_key", "ts", "amt", "is_fraud", "score"} - set(tx.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    fraud = tx["is_fraud"].to_numpy().astype(bool)
    above = (tx["score"] >= threshold).to_numpy()
    amt = tx["amt"].to_numpy(dtype="float64")
    g = pd.DataFrame({
        "card_key": tx["card_key"].to_numpy(),
        "fraud_loss": amt * fraud, "legit_spend": amt * ~fraud,
        "legit_above": amt * (~fraud & above), "fraud_above": amt * (fraud & above),
        "stepups": above, "legit_stepups": ~fraud & above,
        "score": tx["score"].to_numpy(dtype="float64"),
        "night": pd.to_datetime(tx["ts"]).dt.hour.isin(NIGHT_HOURS).to_numpy(),
    })
    out = g.groupby("card_key").agg(
        fraud_loss=("fraud_loss", "sum"), legit_spend=("legit_spend", "sum"), n_tx=("fraud_loss", "size"),
        legit_above=("legit_above", "sum"), fraud_above=("fraud_above", "sum"),
        stepups=("stepups", "sum"), legit_stepups=("legit_stepups", "sum"),
        max_score=("score", "max"), night_share=("night", "mean"))
    if cards is not None:
        out = out.reindex(pd.Index(cards, name="card_key")).fillna(0)
    return out


def threshold_table(tx: pd.DataFrame, thresholds, days: int) -> pd.DataFrame:
    """Step-ups, fraud and legitimate exposure at each threshold."""
    fraud = tx["is_fraud"].to_numpy().astype(bool)
    amt = tx["amt"].to_numpy(dtype="float64")
    rows = []
    for thr in thresholds:
        hit = (tx["score"] >= thr).to_numpy()
        fr, lg = hit & fraud, hit & ~fraud
        rows.append({
            "threshold": thr, "stepups": int(hit.sum()), "stepups_per_day": hit.sum() / days,
            "fraud_tx": int(fr.sum()), "fraud_tx_share": fr.sum() / fraud.sum(),
            "fraud_dollars": amt[fr].sum(), "fraud_dollar_share": amt[fr].sum() / amt[fraud].sum(),
            "legit_tx": int(lg.sum()), "legit_tx_per_day": lg.sum() / days,
            "legit_dollars": amt[lg].sum(), "legit_dollar_share": amt[lg].sum() / amt[~fraud].sum(),
            "cards_touched": int(tx.loc[hit, "card_key"].nunique()),
        })
    return pd.DataFrame(rows)


def _z(alpha: float, power: float) -> float:
    return stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)


def required_n(sd: float, delta: float, alpha: float = 0.05, power: float = 0.8) -> float:
    """Units per arm for a two-sided Welch test of a mean difference `delta`."""
    if sd <= 0 or delta <= 0:
        raise ValueError("sd and delta must be positive")
    return 2 * _z(alpha, power) ** 2 * sd ** 2 / delta ** 2


def mde(sd: float, n_per_arm: float, alpha: float = 0.05, power: float = 0.8) -> float:
    """Minimum detectable mean difference with n units per arm (inverse of required_n)."""
    if sd <= 0 or n_per_arm <= 0:
        raise ValueError("sd and n_per_arm must be positive")
    return _z(alpha, power) * sd * np.sqrt(2 / n_per_arm)


def cuped(y, x) -> tuple:
    """CUPED-adjusted metric and theta; the mean of y is unchanged."""
    y = np.asarray(y, dtype="float64")
    x = np.asarray(x, dtype="float64")
    if y.shape != x.shape:
        raise ValueError("y and x must have the same shape")
    vx = x.var(ddof=1)
    theta = 0.0 if vx == 0 else np.cov(y, x, ddof=1)[0, 1] / vx
    return y - theta * (x - x.mean()), float(theta)


def variance_reduction(y, x) -> float:
    """Share of the variance of y that CUPED with covariate x removes (= correlation squared)."""
    return float(np.corrcoef(np.asarray(y, float), np.asarray(x, float))[0, 1] ** 2)


def srm_pvalue(n_a: int, n_b: int, ratio: float = 0.5) -> float:
    """Chi-square test that the arm counts match the planned split."""
    total = n_a + n_b
    return float(stats.chisquare([n_a, n_b], [total * ratio, total * (1 - ratio)]).pvalue)


def _welch(a, b) -> tuple:
    """Mean of a minus mean of b, its Welch standard error and degrees of freedom.

    Computed directly (same formula as scipy's Welch test) so that an arm with zero
    variance, such as lost revenue in the control arm, is handled exactly.
    """
    na, nb = len(a), len(b)
    va, vb = a.var(ddof=1) / na, b.var(ddof=1) / nb
    se2 = va + vb
    diff = float(a.mean() - b.mean())
    if se2 == 0:
        return diff, 0.0, np.inf
    df = se2 ** 2 / ((va ** 2 / (na - 1) if va else 0.0) + (vb ** 2 / (nb - 1) if vb else 0.0))
    return diff, float(np.sqrt(se2)), float(df)


def welch_analysis(y, treat, x=None, alpha: float = 0.05) -> dict:
    """Control minus treatment mean with a Welch interval; optionally CUPED-adjusted with x."""
    y = np.asarray(y, dtype="float64")
    treat = np.asarray(treat, dtype=bool)
    if x is not None:
        y, _ = cuped(y, x)
    c, t = y[~treat], y[treat]
    diff, se, df = _welch(c, t)
    if se == 0:
        p, half = (1.0 if diff == 0 else 0.0), 0.0
    else:
        p = float(2 * stats.t.sf(abs(diff / se), df))
        half = float(stats.t.ppf(1 - alpha / 2, df) * se)
    return {"diff": diff, "ci_low": diff - half, "ci_high": diff + half, "p": p,
            "n_control": int(len(c)), "n_treatment": int(len(t))}


def harm_pvalue(y, treat, margin: float) -> float:
    """One-sided Welch p-value for H0: treatment mean - control mean <= margin (harm beyond the margin)."""
    y = np.asarray(y, dtype="float64")
    treat = np.asarray(treat, dtype=bool)
    diff, se, df = _welch(y[treat], y[~treat])
    if se == 0:
        return 0.0 if diff > margin else 1.0
    return float(stats.t.sf((diff - margin) / se, df))


def ratio_delta_ci(num, den, alpha: float = 0.05) -> dict:
    """Pooled ratio sum(num) / sum(den) over units, with a delta-method interval."""
    num = np.asarray(num, dtype="float64")
    den = np.asarray(den, dtype="float64")
    n = len(num)
    if n < 2 or den.sum() <= 0:
        raise ValueError("need at least two units and a positive denominator")
    r = num.sum() / den.sum()
    resid = num - r * den
    se = np.sqrt(resid.var(ddof=1) / n) / den.mean()
    z = stats.norm.ppf(1 - alpha / 2)
    return {"ratio": float(r), "se": float(se), "ci_low": float(r - z * se), "ci_high": float(r + z * se)}


def rate_above_pvalue(num, den, tolerance: float) -> float:
    """One-sided p-value for H0: pooled rate <= tolerance (delta method)."""
    d = ratio_delta_ci(num, den)
    if d["se"] == 0:
        return 0.0 if d["ratio"] > tolerance else 1.0
    return float(1 - stats.norm.cdf((d["ratio"] - tolerance) / d["se"]))


def holm(pvalues) -> np.ndarray:
    """Holm-adjusted p-values (strong family-wise error control)."""
    p = np.asarray(pvalues, dtype="float64")
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj


def peeking_fpr(cum_metric, n_sims: int, seed: int, alpha: float = 0.05) -> dict:
    """A/A false-positive rate when stopping at the first look with p < alpha vs at the last look.

    cum_metric: units x looks array of the cumulative metric at each look.
    """
    cum = np.asarray(cum_metric, dtype="float64")
    rng = np.random.default_rng(seed)
    n = len(cum)
    any_look = last = 0
    for _ in range(n_sims):
        t = rng.permutation(n) < n // 2
        p = stats.ttest_ind(cum[t], cum[~t], equal_var=False, axis=0).pvalue
        p = np.nan_to_num(p, nan=1.0)
        any_look += p.min() < alpha
        last += p[-1] < alpha
    return {"any_look": any_look / n_sims, "fixed_horizon": last / n_sims, "looks": cum.shape[1], "n_sims": n_sims}


def obf_boundaries(k: int, alpha: float = 0.05, n_sims: int = 200_000, seed: int = 0) -> np.ndarray:
    """Two-sided O'Brien-Fleming z boundaries for k equally spaced looks, found by simulation."""
    rng = np.random.default_rng(seed)
    looks = np.arange(1, k + 1)
    z = np.cumsum(rng.standard_normal((n_sims, k)), axis=1) / np.sqrt(looks)
    scaled = np.abs(z) * np.sqrt(looks / k)
    c = np.quantile(scaled.max(axis=1), 1 - alpha)
    return c * np.sqrt(k / looks)


def breakeven_abandonment(fraud_above: float, legit_above: float, block_rate: float, margin: float) -> float:
    """Abandonment rate at which blocked fraud dollars equal lost legitimate revenue."""
    if legit_above <= 0 or margin <= 0:
        return float("inf")
    return block_rate * fraud_above / (margin * legit_above)


def decide(srm_p: float, total_loss_ci_low: float, guardrail_p_holm, alpha: float = 0.05,
           enrolment_short: bool = False) -> str:
    """The pre-registered decision rule: ship, don't ship, extend, or invalid (sample-ratio mismatch)."""
    if srm_p < SRM_ALPHA:
        return "invalid (SRM)"
    breach = bool(np.any(np.asarray(guardrail_p_holm) < alpha))
    if not breach and total_loss_ci_low > 0:
        return "ship"
    if enrolment_short and not breach:
        return "extend"
    return "don't ship"
