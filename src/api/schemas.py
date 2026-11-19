"""Request and response schemas of the scoring API (Phase 6).

The request holds only what is known when a card is authorised: a card token
(card_key, already pseudonymised upstream), the time, the amount, the merchant
category and the merchant's location. Everything else the model needs (card
profile, history, holidays, channel, distance) is resolved on the server from
the snapshot, so a client can never set a feature value directly.
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.models.features import CATEGORIES

Category = Literal[tuple(CATEGORIES)]


class ScoreRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    card_key: str = Field(pattern=r"^[0-9a-f]{32}$", description="pseudonymised card key (32 hex)")
    trans_ts: datetime = Field(description="local time without time zone, as in the data")
    amt: float = Field(gt=0, allow_inf_nan=False)
    category: Category
    merch_lat: float = Field(ge=-90, le=90, allow_inf_nan=False)
    merch_long: float = Field(ge=-180, le=180, allow_inf_nan=False)
    client_ref: str | None = Field(default=None, max_length=64,
                                   description="opaque id, echoed back, never logged")

    @field_validator("trans_ts")
    @classmethod
    def no_time_zone(cls, value: datetime) -> datetime:
        if value.tzinfo is not None:
            raise ValueError("trans_ts must be local time without a time zone, as in the data")
        return value


class Reason(BaseModel):
    feature: str
    value: float | int | str | None
    contribution: float = Field(description="SHAP contribution in log-odds (positive = towards fraud)")


class ScoreResponse(BaseModel):
    request_id: str
    client_ref: str | None
    model_version: str
    raw_score: float = Field(description="model F log-odds")
    probability_raw: float = Field(description="model F probability; the alert rule uses this")
    probability_calibrated: float = Field(description="Platt-calibrated probability")
    alert: bool
    threshold: float
    reasons: list[Reason]
