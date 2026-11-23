"""Mechanical checks of one analyst note (Phase 7; pass bar 10/10 cases per mode).

1. schema: the note is a valid AnalystNote with a known status (pydantic, on construction)
2. status: matches the case kind (alert -> ALERT, unknown card -> NOT SCORED: unknown_card,
   API stopped -> API UNAVAILABLE)
3. reasons: equal to the API's reasons, in order, with D5 masking
4. citations: every cited chunk is one of the excerpts shown; error cases have no LLM text
5. no PII to the LLM (D4): no message sent to the LLM contains the card token, the
   merchant coordinates or client_ref
6. limits: at most 4 tool calls and 6 model turns; the fixed disclaimer is present
"""

import json

from src.agent.context import masked_reasons
from src.agent.loop import MAX_TOOL_CALLS, MAX_TURNS
from src.agent.note import DISCLAIMER

EXPECTED_STATUS = {"alert": "ALERT", "unknown_card": "NOT SCORED: unknown_card", "api_down": "API UNAVAILABLE"}


def pii_values(request: dict) -> list[str]:
    values = [request["card_key"], str(request["merch_lat"]), str(request["merch_long"])]
    if request.get("client_ref"):
        values.append(request["client_ref"])
    return values


def check_note(result, case: dict, api_response: dict | None) -> dict[str, bool]:
    note = result.note
    checks = {"schema": True}            # construction would have raised otherwise
    checks["status"] = note.status == EXPECTED_STATUS[case["kind"]]
    if case["kind"] == "alert" and api_response is not None:
        expected = masked_reasons(api_response["reasons"])
        checks["reasons"] = [r.model_dump() for r in note.reasons] == expected
        checks["citations"] = (bool(note.cited_chunks) and set(note.cited_chunks) <= set(note.excerpt_ids)
                               and bool(note.regulatory_context))
    else:
        checks["reasons"] = note.reasons == []
        checks["citations"] = not (note.regulatory_context or note.next_steps or note.cited_chunks)
    sent = json.dumps(result.llm_messages, ensure_ascii=False)
    checks["no_pii_to_llm"] = not any(v and v in sent for v in pii_values(case["request"]))
    checks["limits"] = (result.tool_calls <= MAX_TOOL_CALLS and result.turns <= MAX_TURNS
                        and note.disclaimer == DISCLAIMER)
    return checks
