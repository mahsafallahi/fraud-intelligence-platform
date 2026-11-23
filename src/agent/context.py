"""What the LLM may see about an alert (Phase 7, decisions D4 and D5).

The LLM never sees the card token (D4), the merchant coordinates or client_ref; the
values of the coarse profile features age_band and urban are hidden (D5). The model
below has extra="forbid", so a new field cannot slip through unnoticed.
"""

from datetime import datetime

from pydantic import BaseModel, ConfigDict

MASKED_FEATURES = {"age_band", "urban"}          # D5: shown by name, value hidden
FEATURE_MEANING = {
    "amt": "transaction amount (USD)",
    "hour": "hour of the day",
    "is_night": "night-time transaction (22:00-03:59)",
    "day_of_week": "day of the week",
    "category": "merchant category",
    "channel": "channel (online / in person)",
    "secs_since_prev": "seconds since the card's previous transaction",
    "amt_vs_card_median": "amount relative to the card's median amount",
    "card_txn_1h": "card's transactions in the last hour",
    "card_txn_24h": "card's transactions in the last 24 hours",
    "distance_km": "distance from the card holder's home area (km)",
    "is_public_holiday_in_state": "public holiday in the card holder's state",
    "is_nationwide_public_holiday": "nationwide public holiday",
    "is_bank_holiday": "bank holiday",
    "age_band": "card holder profile: age band (value hidden)",
    "urban": "card holder profile: urban area (value hidden)",
}


class LLMReason(BaseModel):
    model_config = ConfigDict(extra="forbid")
    feature: str
    meaning: str
    value: float | int | str | None
    contribution: float


class LLMTransaction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    amount: float
    category: str
    time: datetime


class LLMAlertContext(BaseModel):
    """The only alert data an LLM prompt or tool result may contain."""
    model_config = ConfigDict(extra="forbid")
    txn_ref: str
    alert: bool
    probability_raw: float
    threshold: float
    transaction: LLMTransaction
    reasons: list[LLMReason]


def masked_reasons(reasons: list[dict]) -> list[dict]:
    """The API's reasons in order, with D5 masking applied (used for the LLM and for the note)."""
    out = []
    for r in reasons:
        hidden = r["feature"] in MASKED_FEATURES
        out.append({"feature": r["feature"], "meaning": FEATURE_MEANING.get(r["feature"], r["feature"]),
                    "value": "hidden" if hidden else r["value"], "contribution": round(float(r["contribution"]), 4)})
    return out


def llm_context(txn_ref: str, request: dict, response: dict) -> LLMAlertContext:
    return LLMAlertContext(
        txn_ref=txn_ref, alert=response["alert"], probability_raw=round(response["probability_raw"], 4),
        threshold=round(response["threshold"], 4),
        transaction=LLMTransaction(amount=request["amt"], category=request["category"], time=request["trans_ts"]),
        reasons=[LLMReason(**r) for r in masked_reasons(response["reasons"])],
    )
