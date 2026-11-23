"""Scripted tool-calling check for an Ollama model (Phase 7, rule C4 comparison). No API, no data.

    python scripts/agent_toolcall_check.py --model qwen2.5:7b-instruct

Sends the agent's real tools and system prompt with "Explain the alert for transaction A01."
and checks that the model answers with a structured call score_transaction(txn_ref="A01").
Then feeds a fixed, synthetic tool result and checks that the next turn is a structured
search_regulations call. Prints PASS/FAIL per step.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.agent.loop import TOOLS  # noqa: E402
from src.agent.writer import load_prompt  # noqa: E402
from src.rag.llm import OllamaChat  # noqa: E402

SYNTHETIC_RESULT = {"txn_ref": "A01", "alert": True, "probability_raw": 0.91, "threshold": 0.2941,
                    "transaction": {"amount": 912.4, "category": "shopping_net", "time": "2020-08-01T23:15:00"},
                    "reasons": [{"feature": "amt", "meaning": "transaction amount (USD)", "value": 912.4,
                                 "contribution": 2.3}]}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True)
    args = parser.parse_args(argv)
    llm = OllamaChat(model=args.model)
    messages = [{"role": "system", "content": load_prompt("agent_system.txt")},
                {"role": "user", "content": "Explain the alert for transaction A01."}]
    r1 = llm.chat(messages, tools=TOOLS)
    c1 = r1.tool_calls[0]["function"] if r1.tool_calls else {}
    ok1 = c1.get("name") == "score_transaction" and dict(c1.get("arguments") or {}).get("txn_ref") == "A01"
    print(f"step 1 score_transaction call: {'PASS' if ok1 else 'FAIL'} {json.dumps(r1.tool_calls)[:200]} "
          f"content={r1.content[:120]!r} ({r1.seconds:.1f}s)")
    if not ok1:
        return 1
    messages += [{"role": "assistant", "content": r1.content or "", "tool_calls": r1.tool_calls},
                 {"role": "tool", "content": json.dumps(SYNTHETIC_RESULT), "tool_name": "score_transaction"}]
    r2 = llm.chat(messages, tools=TOOLS)
    c2 = r2.tool_calls[0]["function"] if r2.tool_calls else {}
    ok2 = c2.get("name") == "search_regulations" and bool(dict(c2.get("arguments") or {}).get("query"))
    print(f"step 2 search_regulations call: {'PASS' if ok2 else 'FAIL'} {json.dumps(r2.tool_calls)[:200]} "
          f"content={r2.content[:120]!r} ({r2.seconds:.1f}s)")
    return 0 if ok2 else 1


if __name__ == "__main__":
    raise SystemExit(main())
