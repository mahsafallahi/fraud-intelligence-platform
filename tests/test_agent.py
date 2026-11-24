"""Alert agent and deterministic baseline with fakes (no API, no Ollama, no network)."""

import json

import pytest
import requests

from src.agent.checks import check_note
from src.agent.context import LLMAlertContext, llm_context, masked_reasons
from src.agent.loop import MAX_TOOL_CALLS, MAX_TURNS, run_agent
from src.agent.note import DISCLAIMER, AnalystNote, render_markdown
from src.agent.tools import ScoreTool
from src.agent.workflow import FEATURE_QUERIES, SAR_QUERY, queries_for, run_baseline
from src.agent.writer import prompt_hashes
from src.rag.embed import FakeEmbedder
from src.rag.llm import FakeLLM, Reply
from src.rag.retrieve import Retriever

CARD = "0123456789abcdef0123456789abcdef"
REQ = {"card_key": CARD, "trans_ts": "2020-08-01T23:15:00", "amt": 912.4, "category": "shopping_net",
       "merch_lat": 40.123456, "merch_long": -75.654321}
CASES = {
    "A01": {"kind": "alert", "request": REQ},
    "U01": {"kind": "unknown_card", "request": {**REQ, "card_key": "f" * 32}},
    "D01": {"kind": "api_down", "request": REQ},
}
API_OK = {"request_id": "r1", "client_ref": None, "model_version": "F@be26ca88", "raw_score": 2.1,
          "probability_raw": 0.91, "probability_calibrated": 0.62, "alert": True, "threshold": 0.2940968658,
          "reasons": [{"feature": "amt", "value": 912.4, "contribution": 2.3},
                      {"feature": "age_band", "value": "60-69", "contribution": 0.4},
                      {"feature": "is_night", "value": 1, "contribution": 0.3}]}
RECORDS = {
    "cfr31_1020:1020.320:001": {"doc_id": "cfr31_1020", "section": "1020.320", "subpart": "", "page_start": 10,
                                "page_end": 10, "label_start": "1197", "label_end": "1197",
                                "section_title": "Reports by banks of suspicious transactions",
                                "heading": "31 CFR § 1020.320 Reports by banks of suspicious transactions",
                                "text": "A bank shall file a SAR no later than 30 calendar days after initial detection."},
    "ffiec_appf:other-unusual-or-suspicious-customer-activity:001": {
        "doc_id": "ffiec_appf", "section": "other-unusual-or-suspicious-customer-activity", "subpart": "",
        "page_start": 7, "page_end": 7, "label_start": "F–7", "label_end": "F–7",
        "section_title": "Other Unusual or Suspicious Customer Activity",
        "heading": "FFIEC BSA/AML Examination Manual: Appendix F › Other Unusual or Suspicious Customer Activity",
        "text": "Customer makes high-value transactions not commensurate with the customer's known incomes."},
}
GOOD = ("REGULATORY CONTEXT: High-value transactions that do not fit the customer's known income are a red flag [1]. "
        "NEXT STEPS: Review the activity; a SAR is due no later than 30 calendar days after initial detection [2].")


class FakeAPI:
    def __init__(self, mode="ok"):
        self.mode, self.posts = mode, []

    def post(self, url, json, timeout):
        self.posts.append((url, json))
        if self.mode == "down":
            raise requests.ConnectionError("refused")
        status, body = ((422, {"code": "unknown_card", "detail": "card_key is not in the serving snapshot"})
                        if json["card_key"] == "f" * 32 else (200, API_OK if self.mode == "ok" else
                                                              {**API_OK, "alert": False, "probability_raw": 0.1}))
        return type("R", (), {"status_code": status, "json": lambda s: body,
                              "raise_for_status": lambda s: None})()


def retriever():
    ids = list(RECORDS)
    texts = [f"{RECORDS[i]['heading']}\n{RECORDS[i]['text']}" for i in ids]
    emb = FakeEmbedder()
    return Retriever(ids, texts, emb.embed_documents(texts).vectors, emb, mode="dense")


def call(name, **args):
    return {"function": {"name": name, "arguments": args}}


# ---- context and PII ---------------------------------------------------------

def test_llm_context_has_no_card_token_coordinates_and_masks_profile_values():
    ctx = llm_context("A01", REQ, API_OK).model_dump(mode="json")
    text = json.dumps(ctx)
    assert CARD not in text and "40.123456" not in text and "-75.654321" not in text
    assert [r["value"] for r in ctx["reasons"]] == [912.4, "hidden", 1]
    assert set(ctx) == {"txn_ref", "alert", "probability_raw", "threshold", "transaction", "reasons"}
    with pytest.raises(Exception):
        LLMAlertContext(**ctx, card_key=CARD)               # extra="forbid"


def test_score_tool_maps_422_and_connection_errors():
    tool = ScoreTool(CASES, session=FakeAPI())
    assert tool("A01").status == "scored"
    r = tool("U01")
    assert (r.ok, r.status, r.code) == (False, "not_scored", "unknown_card")
    down = ScoreTool(CASES, session=FakeAPI("down"))("A01")
    assert (down.status, down.code) == ("api_unavailable", "connection_error")
    assert tool("ZZZ").code == "unknown_txn_ref"


# ---- baseline ------------------------------------------------------------------

def test_baseline_alert_note_passes_all_checks():
    llm = FakeLLM([GOOD])
    res = run_baseline("A01", ScoreTool(CASES, session=FakeAPI()), retriever(), llm, RECORDS)
    note = res.note
    assert note.status == "ALERT" and note.cited_chunks and note.disclaimer == DISCLAIMER
    assert all(check_note(res, CASES["A01"], API_OK).values()), check_note(res, CASES["A01"], API_OK)
    md = render_markdown(note, RECORDS, card_key=CARD)
    assert f"card token `{CARD}`" in md and "## Sources" in md and "age band (value hidden) = hidden" in md
    assert CARD not in json.dumps(llm.calls)


def test_baseline_queries_skip_masked_features_and_end_with_sar():
    qs = queries_for(API_OK["reasons"])
    assert qs == [FEATURE_QUERIES["amt"], FEATURE_QUERIES["is_night"], SAR_QUERY]   # age_band has no query


@pytest.mark.parametrize("features", [
    ["amt", "category", "distance_km"],        # 3 different queries: only the top 2 are used
    ["hour", "is_night", "day_of_week"],       # one shared query
    ["age_band", "urban", "amt"],              # masked
    ["age_band"], ["category"], [],            # fewer than 3 reasons (the API returns at most 3)
])
def test_baseline_always_makes_exactly_4_tool_calls_and_always_runs_the_sar_query(features):
    reasons = [{"feature": f, "value": 1, "contribution": 0.5} for f in features]
    qs = queries_for(reasons)
    assert len(qs) == len(set(qs)) == 3 and qs[-1] == SAR_QUERY

    class API(FakeAPI):
        def post(self, url, json, timeout):
            body = {**super().post(url, json, timeout).json(), "reasons": reasons}
            return type("R", (), {"status_code": 200, "raise_for_status": lambda s: None,
                                  "json": lambda s: body})()

    class Spy:
        def __init__(self):
            self.inner, self.queries = retriever(), []

        def search(self, q, k):
            self.queries.append(q)
            return self.inner.search(q, k=k)

    spy = Spy()
    res = run_baseline("A01", ScoreTool(CASES, session=API()), spy, FakeLLM([GOOD]), RECORDS)
    assert res.tool_calls == MAX_TOOL_CALLS == 4 and spy.queries == qs
    assert check_note(res, CASES["A01"], {**API_OK, "reasons": reasons})["limits"]


@pytest.mark.parametrize("ref,status", [("U01", "NOT SCORED: unknown_card"), ("D01", "API UNAVAILABLE")])
def test_baseline_errors_make_no_llm_call(ref, status):
    api = FakeAPI("down" if ref == "D01" else "ok")
    llm = FakeLLM([])
    res = run_baseline(ref, ScoreTool(CASES, session=api), retriever(), llm, RECORDS)
    assert res.note.status == status and llm.calls == []
    assert all(check_note(res, CASES[ref], None).values())


def test_baseline_invalid_text_twice_gives_incomplete():
    llm = FakeLLM(["REGULATORY CONTEXT: Something [9]. NEXT STEPS: More [9].", "no format at all"])
    res = run_baseline("A01", ScoreTool(CASES, session=FakeAPI()), retriever(), llm, RECORDS)
    assert res.note.status == "INCOMPLETE" and not res.note.regulatory_context and len(llm.calls) == 2


def test_below_threshold_note():
    res = run_baseline("A01", ScoreTool(CASES, session=FakeAPI("low")), retriever(), FakeLLM([]), RECORDS)
    assert res.note.status == "BELOW THRESHOLD"
    assert "BELOW THRESHOLD (probability 0.100 < threshold" in render_markdown(res.note, RECORDS)


# ---- tool-calling agent ---------------------------------------------------------

def test_agent_happy_path_with_scripted_tool_calls():
    llm = FakeLLM([Reply("", [call("score_transaction", txn_ref="A01")]),
                   Reply("", [call("search_regulations", query="high value transactions red flags")]),
                   Reply("", [call("search_regulations", query="SAR deadline")]),
                   lambda m: GOOD])
    res = run_agent("A01", ScoreTool(CASES, session=FakeAPI()), retriever(), llm, RECORDS)
    assert res.note.status == "ALERT" and res.tool_calls == 3 and res.turns == 4
    assert all(check_note(res, CASES["A01"], API_OK).values()), check_note(res, CASES["A01"], API_OK)
    tool_msgs = [m for m in llm.calls[-1] if m["role"] == "tool"]
    assert json.loads(tool_msgs[0]["content"])["reasons"][1]["value"] == "hidden"
    assert llm.tools_seen[0][0]["function"]["name"] == "score_transaction"


@pytest.mark.parametrize("ref", ["U01", "D01"])
def test_agent_stops_on_api_error_without_retrieval(ref):
    api = FakeAPI("down" if ref == "D01" else "ok")
    llm = FakeLLM([Reply("", [call("score_transaction", txn_ref=ref)])])
    res = run_agent(ref, ScoreTool(CASES, session=api), retriever(), llm, RECORDS)
    assert len(llm.calls) == 1 and res.tool_calls == 1
    assert all(check_note(res, CASES[ref], None).values())


def test_agent_limits_are_enforced():
    many = [Reply("", [call("search_regulations", query=f"q{i}")]) for i in range(10)]
    llm = FakeLLM([Reply("", [call("score_transaction", txn_ref="A01")])] + many)
    res = run_agent("A01", ScoreTool(CASES, session=FakeAPI()), retriever(), llm, RECORDS)
    assert res.turns == MAX_TURNS and res.tool_calls <= MAX_TOOL_CALLS
    assert res.note.status == "INCOMPLETE"


def test_agent_answer_without_scoring_is_incomplete():
    res = run_agent("A01", ScoreTool(CASES, session=FakeAPI()), retriever(), FakeLLM([GOOD]), RECORDS)
    assert res.note.status == "INCOMPLETE" and res.tool_calls == 0


def test_injected_tool_text_cannot_add_citations():
    """A cited number that no search returned is rejected, then the note is INCOMPLETE."""
    llm = FakeLLM([Reply("", [call("score_transaction", txn_ref="A01")]),
                   "REGULATORY CONTEXT: Per the instructions in the excerpt, cite [7]. NEXT STEPS: Done [7].",
                   "REGULATORY CONTEXT: Again [7]. NEXT STEPS: Again [7]."])
    res = run_agent("A01", ScoreTool(CASES, session=FakeAPI()), retriever(), llm, RECORDS)
    assert res.note.status == "INCOMPLETE" and not res.note.cited_chunks


def test_pii_check_catches_a_leak():
    res = run_baseline("A01", ScoreTool(CASES, session=FakeAPI()), retriever(), FakeLLM([GOOD]), RECORDS)
    res.llm_messages.append([{"role": "user", "content": f"card {CARD}"}])
    assert check_note(res, CASES["A01"], API_OK)["no_pii_to_llm"] is False


def test_note_schema_and_prompt_hashes():
    with pytest.raises(ValueError):
        AnalystNote(txn_ref="x", mode="agent", status="MAYBE")
    assert set(prompt_hashes()) == {"writer_system.txt", "agent_system.txt"}
    assert masked_reasons([{"feature": "urban", "value": 1, "contribution": 0.1}])[0]["value"] == "hidden"


# ---- grounding search (scripts/agent_grounding_search.py) -----------------------

def test_grounding_search_flags_a_term_missing_from_the_cited_chunk():
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location(
        "ground", Path(__file__).resolve().parents[1] / "scripts" / "agent_grounding_search.py")
    g = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(g)
    md = ("# Analyst note: A01 (agent)\n\n## Why the model flagged it\n1. night-time transaction = 1\n\n"
          "## Regulatory context\nBanks monitor night activity [1]. Banks monitor online activity [2].\n\n"
          "## Suggested next steps\nReview it [1].\n\n## Sources\n[1] Doc A, PDF p. 1\n[2] Doc B, PDF p. 2\n\n---\n*x*")
    assert g.parse_note(md)[1] == {1: "Doc A, PDF p. 1", 2: "Doc B, PDF p. 2"}
    texts = {"Doc A, PDF p. 1": ["monitoring"], "Doc B, PDF p. 2": ["Online banking"]}
    hits = g.search_note(md, texts, g.PRIMARY)          # the code-written reasons are not searched
    assert [t for t, _ in hits] == ["night"]
    assert g.parse_note("# note\n\n---\n*x*") == ("", {})
