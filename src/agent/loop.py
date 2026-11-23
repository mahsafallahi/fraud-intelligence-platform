"""Tool-calling agent (Phase 7): the LLM decides the tool calls, code enforces the limits.

Tools: score_transaction(txn_ref) and search_regulations(query). At most 4 tool calls
and 6 model turns. If the API returns 422 or is unavailable, the loop stops at once:
no retrieval and no further LLM call (the note says NOT SCORED / API UNAVAILABLE).
The final answer is validated exactly like the baseline's (one retry, then INCOMPLETE).
"""

import json
import time

from src.agent.note import AnalystNote
from src.agent.tools import ExcerptRegistry, SearchTool, tool_json
from src.agent.workflow import RunResult, finish_note, note_from_score
from src.agent.writer import RETRY, check_written, load_prompt

MAX_TOOL_CALLS = 4
MAX_TURNS = 6
TOOLS = [
    {"type": "function", "function": {
        "name": "score_transaction",
        "description": "Score the transaction with the bank's fraud model; returns alert, probability and top reasons.",
        "parameters": {"type": "object", "properties": {"txn_ref": {"type": "string"}}, "required": ["txn_ref"]}}},
    {"type": "function", "function": {
        "name": "search_regulations",
        "description": "Search US AML regulation; returns numbered excerpts to cite as [n].",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
]


def _args(call: dict) -> dict:
    args = call.get("function", {}).get("arguments", {})
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except ValueError:
            args = {}
    return args if isinstance(args, dict) else {}


def run_agent(txn_ref: str, score_tool, retriever, llm, records: dict[str, dict]) -> RunResult:
    start = time.perf_counter()
    registry = ExcerptRegistry(records)
    search = SearchTool(retriever, registry, k=3)
    messages = [{"role": "system", "content": load_prompt("agent_system.txt")},
                {"role": "user", "content": f"Explain the alert for transaction {txn_ref}."}]
    result = RunResult(AnalystNote(txn_ref=txn_ref, mode="agent", status="INCOMPLETE",
                                   status_detail="the agent stopped before scoring the transaction"))
    score = None
    retried = False
    while result.turns < MAX_TURNS:
        result.llm_messages.append([dict(m) for m in messages])
        reply = llm.chat(messages, tools=TOOLS)
        result.turns += 1
        calls = reply.tool_calls or []
        if not calls:                                           # a final answer
            if score is None:
                break
            ids = list(registry.ids)
            chunk_text = {c: f"{records[c]['heading']}\n{records[c]['text']}" for c in ids}
            written = check_written(reply.content, ids, chunk_text)
            if written.ok or retried:
                result.written = written
                finish_note(result.note, written, ids)
                break
            retried = True
            messages += [{"role": "assistant", "content": reply.content},
                         {"role": "user", "content": RETRY.format(errors="\n".join(f"- {e}" for e in written.errors))}]
            continue
        messages.append({"role": "assistant", "content": reply.content or "", "tool_calls": calls})
        for call in calls:
            name = call.get("function", {}).get("name", "")
            if result.tool_calls >= MAX_TOOL_CALLS:
                out = {"error": "tool call limit reached; write the final answer now"}
            elif name == "score_transaction":
                result.tool_calls += 1
                if score is not None:
                    out = {"error": "already scored"}
                else:
                    score = score_tool(str(_args(call).get("txn_ref", txn_ref)))
                    result.note = note_from_score(txn_ref, "agent", score)
                    if result.note.status in ("API UNAVAILABLE",) or result.note.status.startswith("NOT SCORED"):
                        result.seconds = time.perf_counter() - start
                        return result                            # stop: no retrieval, no further LLM call
                    if result.note.status == "BELOW THRESHOLD":
                        result.seconds = time.perf_counter() - start
                        return result
                    out = score.llm_view
            elif name == "search_regulations":
                result.tool_calls += 1
                out = {"excerpts": search(str(_args(call).get("query", "")))}
            else:
                out = {"error": f"unknown tool {name!r}"}
            messages.append({"role": "tool", "content": tool_json(out), "tool_name": name})
    if result.written is None and score is not None and result.note.status == "ALERT":
        result.note.status = "INCOMPLETE"
        result.note.status_detail = f"no valid final answer within {MAX_TURNS} turns / {MAX_TOOL_CALLS} tool calls"
    result.seconds = time.perf_counter() - start
    return result
