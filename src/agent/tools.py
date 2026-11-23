"""The agent's two tools (Phase 7): score_transaction (Phase 6 API) and search_regulations (RAG index).

score_transaction takes only a txn_ref. Code looks the full request up in the local,
git-ignored case file and POSTs it, so the LLM never handles the card token or the
coordinates (D4). API errors become a structured result; nothing is retried with
altered input.
"""

import json
from dataclasses import dataclass, field

from src.agent.context import llm_context
from src.rag.answer import sanitize
from src.rag.chunk import source_label

API_URL = "http://127.0.0.1:8000"


@dataclass
class ScoreResult:
    ok: bool
    status: str                  # "scored" | "not_scored" | "api_unavailable"
    code: str = ""               # API error code (422) or "connection_error" / "http_503"
    response: dict | None = None # full API response, kept in code (never sent to the LLM as is)
    llm_view: dict | None = None # LLMAlertContext as a dict


class ScoreTool:
    def __init__(self, cases: dict[str, dict], api_url: str = API_URL, session=None, timeout: float = 30):
        import requests
        self.cases, self.api_url, self.timeout = cases, api_url.rstrip("/"), timeout
        self.session = session or requests.Session()

    def __call__(self, txn_ref: str) -> ScoreResult:
        import requests
        case = self.cases.get(txn_ref)
        if case is None:
            return ScoreResult(False, "not_scored", "unknown_txn_ref")
        try:
            r = self.session.post(f"{self.api_url}/score", json=case["request"], timeout=self.timeout)
        except requests.RequestException:
            return ScoreResult(False, "api_unavailable", "connection_error")
        if r.status_code == 422:
            try:
                code = r.json().get("code", "validation_error")
            except ValueError:
                code = "validation_error"
            return ScoreResult(False, "not_scored", code)
        if r.status_code >= 500:
            return ScoreResult(False, "api_unavailable", f"http_{r.status_code}")
        r.raise_for_status()
        response = r.json()
        view = llm_context(txn_ref, case["request"], response).model_dump(mode="json")
        return ScoreResult(True, "scored", "", response, view)


@dataclass
class ExcerptRegistry:
    """Numbers every chunk shown to the LLM: [1], [2], ... across all searches (same chunk, same number)."""
    records: dict[str, dict]
    ids: list[str] = field(default_factory=list)

    def add(self, chunk_id: str) -> int:
        if chunk_id not in self.ids:
            self.ids.append(chunk_id)
        return self.ids.index(chunk_id) + 1

    def view(self, chunk_id: str) -> dict:
        rec = self.records[chunk_id]
        return {"number": self.add(chunk_id), "source": sanitize(source_label(rec)), "text": sanitize(rec["text"])}


class SearchTool:
    def __init__(self, retriever, registry: ExcerptRegistry, k: int = 3):
        self.retriever, self.registry, self.k = retriever, registry, k

    def __call__(self, query: str, k: int | None = None) -> list[dict]:
        k = max(1, min(int(k or self.k), 6))
        hits = self.retriever.search(str(query)[:300], k=k).hits
        return [self.registry.view(h.chunk_id) for h in hits]


def tool_json(obj) -> str:
    return json.dumps(obj, ensure_ascii=False)
