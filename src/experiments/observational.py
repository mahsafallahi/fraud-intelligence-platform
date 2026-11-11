"""Observational causal methods on a semi-synthetic rollout with a KNOWN true effect (Phase 5).

Design (decided 2026-11-12):
- Semi-synthetic, in the style of the IHDP benchmark: real per-card covariates,
  a simulated fraud outcome and a simulated, confounded rollout. Both potential
  outcomes are known for every card, so every estimand is known exactly.
- Every estimator is compared with ITS OWN estimand: matching with a caliper
  estimates the effect on the matched treated cards; inverse probability weighting
  with trimming estimates the effect on the trimmed population.
- Matching is reported with bias and RMSE only: a naive bootstrap is invalid for
  matching estimators (Abadie & Imbens 2008), and the Abadie-Imbens standard error
  with an estimated propensity score is out of scope.
- Caliper: 0.2 standard deviations of the logit of the propensity score.
- Gender is never used in the rollout or the outcome; it is a balance check only.
"""

import numpy as np
import pandas as pd
from scipy import optimize
from sklearn.linear_model import LogisticRegression

OUTCOME_WEIGHTS = {"night_share": 0.6, "max_score": 0.5, "log_spend": 0.4, "age_band": -0.2, "urban": 0.2}
ASSIGN_WEIGHTS = {"night_share": 0.8, "max_score": 0.8, "log_spend": 0.6, "age_band": -0.2, "urban": 0.2}


def _sigmoid(z):
    return 1 / (1 + np.exp(-z))


def _standardise(X: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame:
    sd = ref.std(ddof=0).replace(0, 1)
    return (X - ref.mean()) / sd


def make_semi_synthetic(covariates: pd.DataFrame, n: int, loss_sizes, seed: int, block_effect=(0.65, 0.15),
                        confounding: float = 1.0, incidence: float = 0.10, treated_share: float = 0.3) -> dict:
    """Resample real card covariates and simulate fraud loss, a confounded rollout and both potential outcomes.

    The effect is heterogeneous: treatment removes a share tau = b0 + b1 * z(night_share)
    (clipped to [0, 1]) of the card's fraud loss, so the effect on the treated differs
    from the average effect. Returns the data and the exact estimands.
    """
    needed = set(OUTCOME_WEIGHTS) | set(ASSIGN_WEIGHTS)
    if missing := needed - set(covariates.columns):
        raise ValueError(f"missing covariates: {sorted(missing)}")
    rng = np.random.default_rng(seed)
    X = covariates.iloc[rng.integers(0, len(covariates), n)].reset_index(drop=True)
    Z = _standardise(X[list(needed)], covariates[list(needed)])
    risk = sum(w * Z[c] for c, w in OUTCOME_WEIGHTS.items()).to_numpy()
    c0 = optimize.brentq(lambda c: _sigmoid(c + risk).mean() - incidence, -20, 20)
    has_fraud = rng.random(n) < _sigmoid(c0 + risk)
    y0 = np.where(has_fraud, rng.choice(np.asarray(loss_sizes, float), n), 0.0)
    tau = np.clip(block_effect[0] + block_effect[1] * Z["night_share"].to_numpy(), 0, 1)
    y1 = y0 * (1 - tau)
    assign = confounding * sum(w * Z[c] for c, w in ASSIGN_WEIGHTS.items()).to_numpy()
    a0 = optimize.brentq(lambda a: _sigmoid(a + assign).mean() - treated_share, -20, 20)
    ps_true = _sigmoid(a0 + assign)
    t = rng.random(n) < ps_true
    y = np.where(t, y1, y0)
    effect = y0 - y1
    return {"X": X, "t": t, "y": y, "y0": y0, "y1": y1, "ps_true": ps_true,
            "ate": float(effect.mean()), "att": float(effect[t].mean()), "effect": effect}


def fit_propensity(X: pd.DataFrame, t) -> np.ndarray:
    """Logistic-regression propensity scores on standardised covariates."""
    Z = _standardise(X, X)
    model = LogisticRegression(C=1e6, max_iter=2000).fit(Z, np.asarray(t, dtype=int))
    return model.predict_proba(Z)[:, 1]


def smd(X: pd.DataFrame, t, w=None) -> pd.Series:
    """Standardised mean differences (treated - control), optionally weighted; pooled unweighted SD."""
    t = np.asarray(t, dtype=bool)
    w = np.ones(len(t)) if w is None else np.asarray(w, dtype="float64")
    out = {}
    for c in X.columns:
        x = X[c].to_numpy(dtype="float64")
        m1 = np.average(x[t], weights=w[t])
        m0 = np.average(x[~t], weights=w[~t])
        sd = np.sqrt((x[t].var(ddof=1) + x[~t].var(ddof=1)) / 2)
        out[c] = 0.0 if sd == 0 else (m1 - m0) / sd
    return pd.Series(out)


def overlap_summary(ps, t, trim=(0.05, 0.95)) -> pd.DataFrame:
    """Propensity-score quantiles per arm and the share of each arm outside the trimming range."""
    ps = np.asarray(ps, dtype="float64")
    t = np.asarray(t, dtype=bool)
    rows = {}
    for name, m in (("treated", t), ("control", ~t)):
        q = np.quantile(ps[m], [0, 0.05, 0.5, 0.95, 1])
        rows[name] = {"n": int(m.sum()), "min": q[0], "p05": q[1], "median": q[2], "p95": q[3], "max": q[4],
                      "outside_trim": float(((ps[m] < trim[0]) | (ps[m] > trim[1])).mean())}
    return pd.DataFrame(rows).T


def naive_diff(y, t) -> float:
    """Treated mean minus control mean (ignores confounding)."""
    y = np.asarray(y, dtype="float64")
    t = np.asarray(t, dtype=bool)
    return float(y[t].mean() - y[~t].mean())


def psm_att(y, t, ps, caliper_sd: float = 0.2) -> dict:
    """1:1 nearest-neighbour matching on the logit of the propensity score, with replacement and a caliper.

    The estimand is the effect on the MATCHED treated units (the caliper drops some
    treated units). No standard error: see the module docstring.
    """
    y = np.asarray(y, dtype="float64")
    t = np.asarray(t, dtype=bool)
    lg = np.log(np.clip(ps, 1e-12, 1 - 1e-12) / (1 - np.clip(ps, 1e-12, 1 - 1e-12)))
    caliper = caliper_sd * lg.std(ddof=1)
    c_idx = np.flatnonzero(~t)
    order = np.argsort(lg[c_idx])
    c_sorted, c_lg = c_idx[order], lg[c_idx][order]
    tr = np.flatnonzero(t)
    pos = np.clip(np.searchsorted(c_lg, lg[tr]), 1, len(c_lg) - 1)
    left, right = c_lg[pos - 1], c_lg[pos]
    take_left = np.abs(lg[tr] - left) <= np.abs(lg[tr] - right)
    match = np.where(take_left, c_sorted[pos - 1], c_sorted[pos])
    dist = np.abs(lg[tr] - lg[match])
    ok = dist <= caliper
    matched = np.zeros(len(t), dtype=bool)
    matched[tr[ok]] = True
    return {"att": float((y[tr[ok]] - y[match[ok]]).mean()), "matched_treated": matched,
            "n_unmatched": int((~ok).sum()), "caliper": float(caliper), "controls": match[ok]}


def ipw(y, t, ps, estimand: str = "ate", trim=(0.05, 0.95)) -> dict:
    """Hajek inverse probability weighting with trimming; influence-function standard error.

    The standard error treats the propensity score as known (conservative for an
    estimated score). With trimming the estimand is the effect on the kept units.
    """
    if estimand not in ("ate", "att"):
        raise ValueError("estimand must be 'ate' or 'att'")
    y = np.asarray(y, dtype="float64")
    t = np.asarray(t, dtype=bool)
    ps = np.asarray(ps, dtype="float64")
    keep = np.ones(len(t), dtype=bool) if trim is None else (ps >= trim[0]) & (ps <= trim[1])
    y, t, ps = y[keep], t[keep], ps[keep]
    if estimand == "ate":
        w1, w0 = t / ps, (~t) / (1 - ps)
    else:
        w1, w0 = t.astype(float), (~t) * ps / (1 - ps)
    mu1, mu0 = np.sum(w1 * y) / w1.sum(), np.sum(w0 * y) / w0.sum()
    phi = w1 * (y - mu1) / w1.mean() - w0 * (y - mu0) / w0.mean()
    est = mu1 - mu0
    se = phi.std(ddof=1) / np.sqrt(len(y))
    return {"est": float(est), "se": float(se), "keep": keep}


def _ols_predict(X: np.ndarray, y: np.ndarray, Xnew: np.ndarray) -> np.ndarray:
    A = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    return np.column_stack([np.ones(len(Xnew)), Xnew]) @ beta


def regression_adjustment(y, t, X: pd.DataFrame, estimand: str = "ate") -> float:
    """G-computation with a separate linear outcome model per arm. Bias and RMSE only (no SE)."""
    y = np.asarray(y, dtype="float64")
    t = np.asarray(t, dtype=bool)
    Z = _standardise(X, X).to_numpy()
    m1 = _ols_predict(Z[t], y[t], Z)
    m0 = _ols_predict(Z[~t], y[~t], Z)
    d = m1 - m0
    return float(d.mean() if estimand == "ate" else d[t].mean())


def doubly_robust(y, t, X: pd.DataFrame, ps, clip=(0.01, 0.99)) -> dict:
    """Augmented IPW estimate of the average effect, with an influence-function standard error."""
    y = np.asarray(y, dtype="float64")
    t = np.asarray(t, dtype=bool)
    ps = np.clip(np.asarray(ps, dtype="float64"), *clip)
    Z = _standardise(X, X).to_numpy()
    m1 = _ols_predict(Z[t], y[t], Z)
    m0 = _ols_predict(Z[~t], y[~t], Z)
    phi = m1 - m0 + t * (y - m1) / ps - (~t) * (y - m0) / (1 - ps)
    return {"est": float(phi.mean()), "se": float(phi.std(ddof=1) / np.sqrt(len(y)))}


def compare_estimators(covariates: pd.DataFrame, loss_sizes, n: int, n_reps: int, seed: int,
                       observed=None, confounding: float = 1.0, randomised: bool = False) -> pd.DataFrame:
    """Repeat the semi-synthetic rollout and record every estimator next to its own true estimand.

    `observed`: covariates the analyst can use (default: all rollout covariates);
    leaving one out creates an unmeasured confounder. randomised=True replaces the
    rollout with a coin flip (the randomised-experiment benchmark).
    """
    observed = list(observed or ASSIGN_WEIGHTS)
    rows = []
    for r in range(n_reps):
        d = make_semi_synthetic(covariates, n, loss_sizes, seed + r, confounding=0.0 if randomised else confounding)
        X, t, y, eff = d["X"][observed], d["t"], d["y"], d["effect"]
        ps = fit_propensity(X, t)
        m = psm_att(y, t, ps)
        ia, it = ipw(y, t, ps, "ate"), ipw(y, t, ps, "att")
        dr = doubly_robust(y, t, X, ps)
        keep = ia["keep"]
        nv = naive_diff(y, t)
        se_naive = np.sqrt(y[t].var(ddof=1) / t.sum() + y[~t].var(ddof=1) / (~t).sum())
        rows += [
            {"rep": r, "method": "naive", "estimand": "ATE", "est": -nv, "truth": d["ate"], "se": se_naive},
            {"rep": r, "method": "matching (caliper)", "estimand": "ATT, matched treated",
             "est": -m["att"], "truth": float(eff[m["matched_treated"]].mean()), "se": np.nan},
            {"rep": r, "method": "IPW (trimmed)", "estimand": "ATE, trimmed",
             "est": -ia["est"], "truth": float(eff[keep].mean()), "se": ia["se"]},
            {"rep": r, "method": "IPW, ATT weights (trimmed)", "estimand": "ATT, trimmed",
             "est": -it["est"], "truth": float(eff[it["keep"] & t].mean()), "se": it["se"]},
            {"rep": r, "method": "regression adjustment", "estimand": "ATE",
             "est": -regression_adjustment(y, t, X), "truth": d["ate"], "se": np.nan},
            {"rep": r, "method": "doubly robust", "estimand": "ATE", "est": -dr["est"], "truth": d["ate"], "se": dr["se"]},
        ]
    return pd.DataFrame(rows)


def summarise_estimators(results: pd.DataFrame) -> pd.DataFrame:
    """Bias, RMSE and 95% interval coverage per method (coverage only where a valid SE exists)."""
    def agg(g):
        err = g["est"] - g["truth"]
        has_se = g["se"].notna().all()
        cover = (np.abs(err) <= 1.959964 * g["se"]).mean() if has_se else np.nan
        return pd.Series({"estimand": g["estimand"].iloc[0], "mean_truth": g["truth"].mean(),
                          "mean_estimate": g["est"].mean(), "bias": err.mean(),
                          "rmse": np.sqrt((err ** 2).mean()), "coverage_95": cover})
    return results.groupby("method", sort=False)[["estimand", "est", "truth", "se"]].apply(agg)
