"""Probability calibration for the fraud model (Phase 4, step D).

Design (decided 2026-11-09):
- Two methods: Platt scaling (a logistic regression on logit(p), unpenalised)
  and isotonic regression on p. Platt keeps the ranking; isotonic can create
  ties.
- A fitted calibrator is stored as plain JSON parameters (Platt a and b, or the
  isotonic thresholds), not as a pickle: readable and safe to load.
  Calibrator.predict reproduces scikit-learn's isotonic prediction exactly
  (linear interpolation between thresholds, clipped at both ends).
- Log loss is computed on probabilities clipped to [1e-7, 1 - 1e-7], because
  isotonic regression can output exactly 0 or 1.
- Reliability bins are equal-width on log10(score): fraud is rare, so equal-width
  bins on the score put almost every row in the first bin. Empty bins are left
  out. Intervals are Wilson 95% intervals.
- prior_shift moves calibrated probabilities to another fraud base rate with the
  standard odds correction. It is for "what if" tables only; nothing is
  adjusted for an unknown future base rate.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

EPS = 1e-7
METHODS = ("platt", "isotonic")


def _check(y, p):
    """Validate labels and probabilities; return them as float arrays."""
    y = np.asarray(y, dtype="float64")
    p = np.asarray(p, dtype="float64")
    if y.shape != p.shape or y.ndim != 1:
        raise ValueError("y and p must be 1-D arrays of the same length")
    if np.isnan(p).any() or ((p < 0) | (p > 1)).any():
        raise ValueError("probabilities must be in [0, 1] without missing values")
    if not np.isin(y, (0.0, 1.0)).all():
        raise ValueError("labels must be 0 or 1")
    return y, p


def logit(p):
    """Log-odds, with p clipped away from 0 and 1."""
    p = np.clip(np.asarray(p, dtype="float64"), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def _sigmoid(z):
    return 1 / (1 + np.exp(-z))


@dataclass(frozen=True)
class Calibrator:
    """A fitted calibrator: the method name and its plain parameters."""

    method: str
    params: dict

    def __post_init__(self):
        if self.method not in METHODS:
            raise ValueError(f"unknown method: {self.method}")

    def predict(self, p) -> np.ndarray:
        """Calibrated probabilities for raw model scores p."""
        p = np.asarray(p, dtype="float64")
        if np.isnan(p).any() or ((p < 0) | (p > 1)).any():
            raise ValueError("scores must be in [0, 1] without missing values")
        if self.method == "platt":
            return _sigmoid(self.params["a"] * logit(p) + self.params["b"])
        return np.interp(p, self.params["x"], self.params["y"])

    def to_dict(self) -> dict:
        return {"method": self.method, "params": self.params}

    @classmethod
    def from_dict(cls, d: dict) -> "Calibrator":
        return cls(method=d["method"], params=d["params"])


def fit_platt(y, p) -> Calibrator:
    """Platt scaling: logistic regression of the label on logit(p)."""
    y, p = _check(y, p)
    model = LogisticRegression(C=np.inf, max_iter=1000)  # C=inf: no penalty
    model.fit(logit(p).reshape(-1, 1), y.astype(int))
    return Calibrator("platt", {"a": float(model.coef_[0, 0]), "b": float(model.intercept_[0])})


def fit_isotonic(y, p) -> Calibrator:
    """Isotonic regression of the label on p, clipped outside the fitted range."""
    y, p = _check(y, p)
    model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(p, y)
    return Calibrator("isotonic", {"x": model.X_thresholds_.tolist(), "y": model.y_thresholds_.tolist()})


FITTERS = {"platt": fit_platt, "isotonic": fit_isotonic}


def brier(y, p) -> float:
    """Mean squared difference between probability and outcome."""
    y, p = _check(y, p)
    return float(np.mean((p - y) ** 2))


def log_loss_clipped(y, p, eps: float = EPS) -> float:
    """Log loss with p clipped to [eps, 1 - eps], so a 0 or 1 cannot make it infinite."""
    y, p = _check(y, p)
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def wilson_interval(k, n, z: float = 1.959964) -> tuple:
    """Wilson score interval for k successes out of n (arrays allowed)."""
    k = np.asarray(k, dtype="float64")
    n = np.asarray(n, dtype="float64")
    phat = np.where(n > 0, k / np.maximum(n, 1), np.nan)
    denom = 1 + z**2 / n
    centre = (phat + z**2 / (2 * n)) / denom
    half = z * np.sqrt(phat * (1 - phat) / n + z**2 / (4 * n**2)) / denom
    return centre - half, centre + half


def reliability_table(y, p, n_bins: int = 15, low: float = 1e-6) -> pd.DataFrame:
    """Observed vs predicted fraud rate in equal-width bins on log10(p).

    The first bin also holds every score below `low`. Empty bins are left out.
    """
    y, p = _check(y, p)
    edges = np.logspace(np.log10(low), 0, n_bins + 1)
    edges[0] = 0.0
    bins = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, n_bins - 1)
    df = pd.DataFrame({"bin": bins, "y": y, "p": p})
    out = df.groupby("bin").agg(n=("y", "size"), n_fraud=("y", "sum"), mean_pred=("p", "mean"))
    out["observed"] = out["n_fraud"] / out["n"]
    out["low"], out["high"] = wilson_interval(out["n_fraud"], out["n"])
    out.insert(0, "bin_low", edges[out.index])
    out.insert(1, "bin_high", edges[out.index + 1])
    out["n_fraud"] = out["n_fraud"].astype(int)
    return out.reset_index(drop=True)


def prior_shift(p, from_rate: float, to_rate: float) -> np.ndarray:
    """Move probabilities calibrated at base rate `from_rate` to base rate `to_rate`."""
    for r in (from_rate, to_rate):
        if not 0 < r < 1:
            raise ValueError("base rates must be strictly between 0 and 1")
    shift = np.log(to_rate / (1 - to_rate)) - np.log(from_rate / (1 - from_rate))
    return _sigmoid(logit(p) + shift)
