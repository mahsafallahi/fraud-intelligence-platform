"""Simulated step-up experiments on historical transactions (Phase 5).

Every effect here is a SIMULATED ASSUMPTION:
- block_rate (b): share of stepped-up fraud transactions that the step-up blocks;
- abandon_rate (a): share of stepped-up legitimate transactions the customer abandons;
- margin (m): share of an abandoned transaction's value counted as lost.

Only transactions with an F score at or above the threshold are stepped up, and only
on treated cards. The history itself does not react (no fraudster adaptation, no
novelty effect): those are stated limitations. A card is "triggered" when it has at
least one transaction at or above the threshold; F is frozen, so the trigger is known
in both arms from the logged scores.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.experiments.ab_design import (
    decide, harm_pvalue, holm, rate_above_pvalue, srm_pvalue, welch_analysis,
)


@dataclass(frozen=True)
class StepupData:
    """Per-card base outcomes plus the stepped-up transactions, for fast repeated simulation."""

    cards: np.ndarray
    fraud_loss: np.ndarray
    fraud_above: np.ndarray
    legit_above: np.ndarray
    legit_stepups: np.ndarray
    triggered: np.ndarray
    tx_card: np.ndarray
    tx_amt: np.ndarray
    tx_fraud: np.ndarray
    threshold: float


def prepare_stepup(tx: pd.DataFrame, threshold: float, cards) -> StepupData:
    """Per-card outcomes for the listed cards and the transactions a step-up would touch."""
    cards = np.asarray(cards)
    pos = pd.Series(np.arange(len(cards)), index=cards)
    tx = tx[tx["card_key"].isin(cards)]
    fraud = tx["is_fraud"].to_numpy().astype(bool)
    amt = tx["amt"].to_numpy(dtype="float64")
    code = pos.reindex(tx["card_key"].to_numpy()).to_numpy()
    above = (tx["score"] >= threshold).to_numpy()
    k = len(cards)
    return StepupData(
        cards=cards,
        fraud_loss=np.bincount(code, weights=amt * fraud, minlength=k),
        fraud_above=np.bincount(code, weights=amt * (fraud & above), minlength=k),
        legit_above=np.bincount(code, weights=amt * (~fraud & above), minlength=k),
        legit_stepups=np.bincount(code, weights=(~fraud & above), minlength=k),
        triggered=np.bincount(code, weights=above, minlength=k) > 0,
        tx_card=code[above], tx_amt=amt[above], tx_fraud=fraud[above], threshold=float(threshold),
    )


def assign_arms(n_cards: int, rng, p: float = 0.5) -> np.ndarray:
    """Independent Bernoulli assignment per card (True = treatment)."""
    return rng.random(n_cards) < p


def apply_stepup(data: StepupData, treat, block_rate: float, abandon_rate: float, margin: float, rng) -> pd.DataFrame:
    """Per-card outcomes after applying the simulated step-up to treated cards."""
    for name, v in (("block_rate", block_rate), ("abandon_rate", abandon_rate), ("margin", margin)):
        if not 0 <= v <= 1:
            raise ValueError(f"{name} must be in [0, 1]")
    treat = np.asarray(treat, dtype=bool)
    k = len(data.cards)
    u = rng.random(len(data.tx_amt))
    on = treat[data.tx_card]
    blocked = on & data.tx_fraud & (u < block_rate)
    abandoned = on & ~data.tx_fraud & (u < abandon_rate)
    abandoned_dollars = np.bincount(data.tx_card, weights=data.tx_amt * abandoned, minlength=k)
    fraud_loss = data.fraud_loss - np.bincount(data.tx_card, weights=data.tx_amt * blocked, minlength=k)
    return pd.DataFrame({
        "fraud_loss": fraud_loss,
        "abandoned_dollars": abandoned_dollars,
        "abandoned_n": np.bincount(data.tx_card, weights=abandoned, minlength=k),
        "legit_stepups": data.legit_stepups,
        "lost_revenue": margin * abandoned_dollars,
        "total_loss": fraud_loss + margin * abandoned_dollars,
    })


def true_effects(data: StepupData, block_rate: float, abandon_rate: float, margin: float, population) -> dict:
    """Expected per-card effects in the population (exact under the simulated assumptions)."""
    pop = np.asarray(population, dtype=bool)
    fraud_reduction = block_rate * data.fraud_above[pop].mean()
    lost = abandon_rate * margin * data.legit_above[pop].mean()
    return {"fraud_loss_reduction": float(fraud_reduction), "lost_revenue": float(lost),
            "total_loss_reduction": float(fraud_reduction - lost)}


def analyse(out: pd.DataFrame, treat, population, guardrail_margin_frac: float, abandon_tolerance: float,
            data: StepupData) -> dict:
    """Pre-registered analysis of one experiment on one population."""
    pop = np.asarray(population, dtype=bool)
    t = np.asarray(treat, dtype=bool)[pop]
    o = out[pop]
    primary = welch_analysis(o["fraud_loss"], t)
    total = welch_analysis(o["total_loss"], t)
    margin_g1 = guardrail_margin_frac * data.legit_above[pop][~t].mean()
    g1 = harm_pvalue(o["lost_revenue"], t, margin_g1)
    exposed = t & (o["legit_stepups"].to_numpy() > 0)
    g2 = (rate_above_pvalue(o["abandoned_n"].to_numpy()[exposed], o["legit_stepups"].to_numpy()[exposed], abandon_tolerance)
          if exposed.sum() >= 2 else 1.0)
    g_holm = holm([g1, g2])
    srm = srm_pvalue(int(t.sum()), int((~t).sum()))
    return {"primary_diff": primary["diff"], "primary_p": primary["p"], "primary_ci_low": primary["ci_low"],
            "total_diff": total["diff"], "total_ci_low": total["ci_low"], "total_ci_high": total["ci_high"],
            "g1_p_holm": g_holm[0], "g2_p_holm": g_holm[1], "srm_p": srm,
            "decision": decide(srm, total["ci_low"], g_holm), "n_cards": int(pop.sum())}


def simulate_experiments(data: StepupData, block_rate: float, abandon_rate: float, margin: float, n_sims: int,
                         seed: int, population: str = "triggered", guardrail_margin_frac: float = 0.05,
                         abandon_tolerance: float = 0.10) -> pd.DataFrame:
    """Repeat the whole experiment n_sims times: random assignment, simulated step-up, analysis."""
    if population not in ("triggered", "all"):
        raise ValueError("population must be 'triggered' or 'all'")
    pop = data.triggered if population == "triggered" else np.ones(len(data.cards), dtype=bool)
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_sims):
        treat = assign_arms(len(data.cards), rng)
        out = apply_stepup(data, treat, block_rate, abandon_rate, margin, rng)
        rows.append(analyse(out, treat, pop, guardrail_margin_frac, abandon_tolerance, data))
    return pd.DataFrame(rows)


def resample_population(values, n: int, rng, incidence: float = None) -> np.ndarray:
    """Draw n per-card values with replacement; optionally rescale the share of non-zero values."""
    values = np.asarray(values, dtype="float64")
    if incidence is None:
        return rng.choice(values, n)
    if not 0 < incidence < 1:
        raise ValueError("incidence must be strictly between 0 and 1")
    nonzero = values[values > 0]
    draw = rng.choice(nonzero, n)
    return np.where(rng.random(n) < incidence, draw, 0.0)
