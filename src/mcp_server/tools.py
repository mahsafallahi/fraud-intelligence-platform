"""The two MCP tools as plain Python, with no MCP types (docs/rag_design.md §14).

- RegulationSearch: the frozen retrieval (src/rag/config.py, dense only) over the pinned
  index. No LLM call. Every label (document, section, pages, citation) is rendered by
  code from chunk metadata.
- Scorer: the Phase 6 scorer in process (src/api/scoring.py) with the API's own
  request model for validation and the API's error codes. No HTTP, no Docker.

Both raise ToolFailure(code, detail) for every anticipated failure; the server turns
it into a structured MCP tool error. Neither tool writes or sends anything.
"""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, ValidationError

from src.agent.context import MASKED_FEATURES
from src.api.history import DateOutsideCalendar, UnknownCard
from src.api.schemas import ScoreRequest
from src.api.scoring import score_transaction
from src.rag.answer import sanitize
from src.rag.chunk import citation, source_label
from src.rag.config import RETRIEVAL
from src.rag.index import IndexMismatch, check_embedder

MAX_K = RETRIEVAL["k"]           # 6, frozen
MAX_QUERY_CHARS = 500
HIDDEN = "hidden"                # D5: value of age_band / urban is not shown to an LLM


class ToolFailure(Exception):
    """An anticipated tool failure: a code from the API's vocabulary and a safe detail."""

    def __init__(self, code: str, detail=None):
        super().__init__(code)
        self.code, self.detail = code, detail


def validation_detail(exc: ValidationError) -> list[dict]:
    """loc / msg / type only, as in the API: pydantic's "input" would echo the submitted values."""
    return [{"loc": list(e["loc"]), "msg": e["msg"], "type": e["type"]} for e in exc.errors()]


def _object(arguments) -> dict:
    if not isinstance(arguments, dict):
        raise ToolFailure("validation_error",
                          [{"loc": [], "msg": "Input should be an object", "type": "dict_type"}])
    return arguments


# ---------------------------------------------------------------- score_transaction

class Reason(BaseModel):
    feature: str
    value: float | int | str | None = Field(description='feature value; "hidden" for age_band and urban')
    contribution: float = Field(description="SHAP contribution in log-odds (positive = towards fraud)")


class ScoreOutput(BaseModel):
    model_version: str
    raw_score: float = Field(description="model F log-odds")
    probability_raw: float = Field(description="model F probability; the alert rule uses this")
    probability_calibrated: float = Field(description="Platt-calibrated probability")
    alert: bool = Field(description="probability_raw >= threshold")
    threshold: float
    reasons: list[Reason] = Field(description="top 3 positive SHAP contributions")


def score_input_schema() -> dict:
    """The API's ScoreRequest schema without client_ref (an HTTP echo field, not needed here)."""
    schema = ScoreRequest.model_json_schema()
    schema["properties"] = {k: v for k, v in schema["properties"].items() if k != "client_ref"}
    schema["required"] = [k for k in schema.get("required", []) if k != "client_ref"]
    schema["additionalProperties"] = False
    schema["title"] = "score_transaction arguments"
    return schema


def parse_score_args(arguments) -> ScoreRequest:
    """Validate with the API's own ScoreRequest (same rules, same messages); no client_ref."""
    arguments = _object(arguments)
    errors = []
    if "client_ref" in arguments:
        errors.append({"loc": ["client_ref"], "msg": "Extra inputs are not permitted",
                       "type": "extra_forbidden"})
    try:
        request = ScoreRequest.model_validate({k: v for k, v in arguments.items() if k != "client_ref"})
    except ValidationError as exc:
        errors.extend(validation_detail(exc))
    if errors:
        raise ToolFailure("validation_error", errors)
    return request


def masked(reasons: list[dict]) -> list[dict]:
    return [{**r, "value": HIDDEN if r["feature"] in MASKED_FEATURES else r["value"]} for r in reasons]


class Scorer:
    """score_transaction over the verified bundle and the read-only serving snapshot."""

    def __init__(self, bundle, store):
        self.bundle, self.store = bundle, store

    def __call__(self, arguments) -> dict:
        req = parse_score_args(arguments)
        try:
            s = score_transaction(self.bundle, self.store, card_key=req.card_key, trans_ts=req.trans_ts,
                                  amt=req.amt, category=req.category,
                                  merch_lat=req.merch_lat, merch_long=req.merch_long)
        except UnknownCard:
            raise ToolFailure("unknown_card", "card_key is not in the serving snapshot") from None
        except DateOutsideCalendar as exc:     # the API returns the same text to its client
            raise ToolFailure("date_outside_calendar", str(exc)) from None
        return ScoreOutput(model_version=self.bundle.model_version, raw_score=s.raw_score,
                           probability_raw=s.probability_raw,
                           probability_calibrated=s.probability_calibrated, alert=s.alert,
                           threshold=self.bundle.threshold,
                           reasons=masked(s.reasons)).model_dump(mode="json")

    @staticmethod
    def log_fields(result: dict) -> dict:
        return {"model_version": result["model_version"], "alert": result["alert"],
                "probability_calibrated": round(result["probability_calibrated"], 3)}


# ---------------------------------------------------------------- search_regulations

Query = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_QUERY_CHARS)]


class SearchArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", title="search_regulations arguments")
    query: Query = Field(description="question or keywords about US AML / fraud regulation")
    k: int = Field(default=MAX_K, ge=1, le=MAX_K, description=f"number of excerpts (1..{MAX_K})")


class Excerpt(BaseModel):
    rank: int
    chunk_id: str
    doc_id: str
    source: str = Field(description="document and section, rendered from metadata")
    section: str
    section_title: str
    pdf_pages: list[int] = Field(description="[first, last] PDF page")
    printed_pages: list[str] | None = Field(description="[first, last] printed page label, if any")
    citation: str = Field(description="full citation, rendered from metadata")
    cosine: float = Field(description="dense cosine similarity to the query")
    text: str


class SearchOutput(BaseModel):
    retrieval: str = Field(description="frozen configuration: dense only (nomic-embed-text)")
    max_cosine: float = Field(description="best dense cosine over the whole index (no refusal threshold)")
    excerpts: list[Excerpt]


def _excerpt(hit, rec: dict) -> dict:
    printed = [str(rec["label_start"]), str(rec["label_end"])] if rec.get("label_start") else None
    return {"rank": hit.rank, "chunk_id": hit.chunk_id, "doc_id": rec["doc_id"],
            "source": source_label(rec), "section": str(rec["section"]),
            "section_title": str(rec["section_title"]),
            "pdf_pages": [int(rec["page_start"]), int(rec["page_end"])], "printed_pages": printed,
            "citation": citation(rec), "cosine": hit.cosine, "text": sanitize(rec["text"])}


class RegulationSearch:
    """The frozen dense retrieval; the embedder is checked against the index on first use."""

    def __init__(self, frame, retriever, manifest: dict, embedder):
        self.records = frame.set_index("chunk_id", drop=False).to_dict("index")
        self.retriever, self.manifest, self.embedder = retriever, manifest, embedder
        self._embedder_checked = False

    def _check_embedder(self) -> None:
        if self._embedder_checked:
            return
        try:
            check_embedder(self.manifest, self.embedder)
        except IndexMismatch:
            raise ToolFailure("index_mismatch",
                              "the query embedder is not the model the index was built with") from None
        except (OSError, RuntimeError, ValueError):
            # requests' errors subclass OSError; RuntimeError = model not in Ollama
            raise ToolFailure("embedder_unavailable",
                              "the local embedding model (Ollama, 127.0.0.1:11434) is not available") from None
        self._embedder_checked = True

    def __call__(self, arguments) -> dict:
        try:
            args = SearchArgs.model_validate(_object(arguments))
        except ValidationError as exc:
            raise ToolFailure("validation_error", validation_detail(exc)) from None
        self._check_embedder()
        try:
            result = self.retriever.search(args.query, k=args.k)
        except OSError:
            raise ToolFailure("embedder_unavailable",
                              "the local embedding model (Ollama, 127.0.0.1:11434) is not available") from None
        excerpts = [_excerpt(h, self.records[h.chunk_id]) for h in result.hits]
        return SearchOutput(retrieval=RETRIEVAL["mode"], max_cosine=result.max_cosine,
                            excerpts=excerpts).model_dump(mode="json")

    @staticmethod
    def log_fields(result: dict) -> dict:
        return {"n_hits": len(result["excerpts"])}
