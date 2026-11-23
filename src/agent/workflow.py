"""Deterministic baseline (Phase 7): score -> fixed queries from the reasons -> one writer call.

Code decides every step; the LLM only writes the two text sections. This is the
reference for the tool-calling agent (src/agent/loop.py) and runs with a FakeLLM in CI.
"""

import time
from dataclasses import dataclass, field

from src.agent.context import MASKED_FEATURES
from src.agent.note import AnalystNote, NoteReason
from src.agent.tools import ExcerptRegistry, ScoreResult, SearchTool
from src.agent.writer import RETRY, Written, check_written, excerpts_message, load_prompt

# one query per reason feature (masked profile features get none); plus the SAR query
FEATURE_QUERIES = {
    "amt": "large or unusual transaction amounts as potentially suspicious activity",
    "amt_vs_card_median": "transactions inconsistent with the customer's normal activity",
    "hour": "unusual transaction patterns that may indicate suspicious activity",
    "is_night": "unusual transaction patterns that may indicate suspicious activity",
    "day_of_week": "unusual transaction patterns that may indicate suspicious activity",
    "category": "transaction monitoring for unusual activity",
    "channel": "online payment activity red flags",
    "secs_since_prev": "many transactions in a short period of time",
    "card_txn_1h": "many transactions in a short period of time",
    "card_txn_24h": "many transactions in a short period of time",
    "distance_km": "transactions far from the customer's location or geographic risk",
    "is_public_holiday_in_state": "unusual transaction patterns that may indicate suspicious activity",
    "is_nationwide_public_holiday": "unusual transaction patterns that may indicate suspicious activity",
    "is_bank_holiday": "unusual transaction patterns that may indicate suspicious activity",
}
SAR_QUERY = "When must a bank file a suspicious activity report and how should it review alerts?"
MAX_EXCERPTS = 6
PER_QUERY = 2


@dataclass
class RunResult:
    note: AnalystNote
    llm_messages: list = field(default_factory=list)   # every message list sent to the LLM
    tool_calls: int = 0
    turns: int = 0
    seconds: float = 0.0
    written: Written | None = None


def note_from_score(txn_ref: str, mode: str, score: ScoreResult) -> AnalystNote:
    """The code-built part of the note: header and reasons (from the API, D5 masking applied)."""
    if score.status == "not_scored":
        return AnalystNote(txn_ref=txn_ref, mode=mode, status=f"NOT SCORED: {score.code}",
                           status_detail="the scoring API rejected the request (HTTP 422); "
                                         "no regulatory text was retrieved and no LLM text was written")
    if score.status == "api_unavailable":
        return AnalystNote(txn_ref=txn_ref, mode=mode, status="API UNAVAILABLE", status_detail=score.code)
    r = score.response
    return AnalystNote(txn_ref=txn_ref, mode=mode, status="ALERT" if r["alert"] else "BELOW THRESHOLD",
                       model_version=r["model_version"], probability_raw=r["probability_raw"],
                       threshold=r["threshold"],
                       reasons=[NoteReason(**x) for x in score.llm_view["reasons"]])


def queries_for(reasons: list[dict]) -> list[str]:
    out = []
    for r in reasons:
        q = FEATURE_QUERIES.get(r["feature"]) if r["feature"] not in MASKED_FEATURES else None
        if q and q not in out:
            out.append(q)
    return out + [SAR_QUERY]


def run_baseline(txn_ref: str, score_tool, retriever, llm, records: dict[str, dict]) -> RunResult:
    start = time.perf_counter()
    score = score_tool(txn_ref)
    note = note_from_score(txn_ref, "baseline", score)
    result = RunResult(note, tool_calls=1)
    if note.status != "ALERT":          # error or below threshold: no retrieval, no LLM
        result.seconds = time.perf_counter() - start
        return result

    registry = ExcerptRegistry(records)
    search = SearchTool(retriever, registry, k=PER_QUERY)
    for q in queries_for(score.response["reasons"]):
        if len(registry.ids) >= MAX_EXCERPTS:
            break
        search(q)
        result.tool_calls += 1
    ids = registry.ids[:MAX_EXCERPTS]
    excerpts = [registry.view(c) for c in ids]
    chunk_text = {c: f"{records[c]['heading']}\n{records[c]['text']}" for c in ids}
    messages = [{"role": "system", "content": load_prompt("writer_system.txt")},
                {"role": "user", "content": excerpts_message(score.llm_view, excerpts)}]
    for _ in (1, 2):
        result.llm_messages.append([dict(m) for m in messages])
        reply = llm.chat(messages).content.strip()
        result.turns += 1
        written = check_written(reply, ids, chunk_text)
        if written.ok:
            break
        messages += [{"role": "assistant", "content": reply},
                     {"role": "user", "content": RETRY.format(errors="\n".join(f"- {e}" for e in written.errors))}]
    result.written = written
    finish_note(note, written, ids)
    result.seconds = time.perf_counter() - start
    return result


def finish_note(note: AnalystNote, written: Written, excerpt_ids: list[str]) -> None:
    note.excerpt_ids = list(excerpt_ids)
    if written.ok:
        note.regulatory_context, note.next_steps, note.cited_chunks = written.context, written.steps, written.cited
    else:
        note.status = "INCOMPLETE"
        note.status_detail = "the LLM text failed validation twice; only the code-built part is shown"
