"""MCP server (Phase 7, docs/rag_design.md §14).

CI-safe (no corpus, no Gold, no model): the synthetic serving snapshot, the committed
bundle and a tiny index built with FakeEmbedder. Covers the tool listing and schemas,
validation and error codes, the logging rule, scorer equality on synthetic rows and an
end-to-end run of scripts/mcp_server.py with the SDK's own stdio client.

Local only:
- corpus + ollama: search over the real pinned index returns valid chunk ids.
- gold: score_transaction equals the in-process API (TestClient) exactly on real
  test-period rows, picked by score only (labels are never read).
- gold + corpus + ollama: the real server end to end over stdio.
"""

import json
import logging
import sys

import anyio
import duckdb
import numpy as np
import pytest
from mcp import Client, StdioServerParameters
from mcp.shared.exceptions import MCPError

from src.api.scoring import score_transaction
from src.mcp_server.server import (DEFAULT_INDEX_DIR, build_server, load_tools, pinned_index_sha256)
from src.mcp_server.tools import HIDDEN, MAX_K
from src.rag.chunk import chunk_sections, citation
from src.rag.corpus import load_manifest
from src.rag.embed import FakeEmbedder
from src.rag.index import build_index
from src.rag.parse import Line, Section
from tests.conftest import BUNDLE_DIR, GOLD_DEV_DB, PROJECT_ROOT, SERVING_DB

SCRIPT = PROJECT_ROOT / "scripts" / "mcp_server.py"
SCORE_FIELDS = {"card_key", "trans_ts", "amt", "category", "merch_lat", "merch_long"}
OUTPUT_FIELDS = {"model_version", "raw_score", "probability_raw", "probability_calibrated",
                 "alert", "threshold", "reasons"}
QUERY = "secret query about structuring wires"

TEXTS = [
    ("cfr31_1020", "1020.320", "Reports by banks of suspicious transactions",
     "A bank shall file a SAR no later than 30 calendar days after initial detection.", 10, "1197"),
    ("cfr31_1010", "1010.311", "Filing obligations for reports of transactions in currency",
     "Each financial institution shall file a report of each transaction in currency of more "
     "than $10,000.", 40, "260"),
    ("cfr31_1010", "1010.410", "Records to be made and retained by financial institutions",
     "Records of funds transfers of $3,000 or more must be retained for five years.", 55, "275"),
]


# ------------------------------------------------------------------ fixtures and helpers

@pytest.fixture(scope="session")
def fake_index(tmp_path_factory):
    """A 3-chunk index built with FakeEmbedder over the REAL corpus hashes (so load_index accepts it)."""
    out = tmp_path_factory.mktemp("index")
    sections = [Section(doc, sec, title, "", [Line(text, page, label)])
                for doc, sec, title, text, page, label in TEXTS]
    hashes = {d["doc_id"]: d["sha256"] for d in load_manifest()["documents"]}
    build_index(chunk_sections(sections), FakeEmbedder(), out, hashes)
    return out


@pytest.fixture(scope="module")
def tools(synthetic, fake_index):
    scorer, search, store = load_tools(BUNDLE_DIR, synthetic[0], fake_index, FakeEmbedder(), None)
    yield scorer, search
    store.close()


@pytest.fixture
def server(tools):
    return build_server(*tools)


def run(server, *calls):
    """Open one in-process client session and run each (tool, arguments); return the results."""
    async def go():
        async with Client(server) as client:
            return [await client.call_tool(name, args) for name, args in calls]
    return anyio.run(go)


def listed(server):
    async def go():
        async with Client(server) as client:
            return (await client.list_tools()).tools
    return anyio.run(go)


@pytest.fixture
def txn(synthetic):
    t = synthetic[1].iloc[7]
    return {"card_key": t.card_key, "trans_ts": t.trans_ts.isoformat(), "amt": float(t.amt),
            "category": t.category, "merch_lat": float(t.merch_lat), "merch_long": float(t.merch_long)}


def log_lines(caplog):
    return [r.getMessage() for r in caplog.records if r.name == "fraud_mcp"]


def assert_no_request_data(text: str, args: dict) -> None:
    for key in ("card_key", "trans_ts", "amt", "merch_lat", "merch_long", "query"):
        if key in args:
            value = str(args[key])
            assert value not in text, key
    if "trans_ts" in args:
        assert str(args["trans_ts"])[:10] not in text   # not even the date


# ------------------------------------------------------------------ listing and schemas

def test_tools_are_listed_with_strict_schemas_and_read_only_annotations(server):
    tools = {t.name: t for t in listed(server)}
    assert set(tools) == {"search_regulations", "score_transaction"}
    for t in tools.values():
        a = t.annotations
        assert a.read_only_hint is True and a.destructive_hint is False
        assert a.idempotent_hint is True and a.open_world_hint is False
        assert t.input_schema["additionalProperties"] is False
        assert t.output_schema["type"] == "object" and t.description

    score = tools["score_transaction"]
    assert set(score.input_schema["properties"]) == SCORE_FIELDS     # no client_ref
    assert set(score.input_schema["required"]) == SCORE_FIELDS
    assert score.input_schema["properties"]["card_key"]["pattern"] == "^[0-9a-f]{32}$"
    assert set(score.output_schema["properties"]) == OUTPUT_FIELDS

    search = tools["search_regulations"]
    assert set(search.input_schema["properties"]) == {"query", "k"}
    assert search.input_schema["required"] == ["query"]
    assert search.input_schema["properties"]["k"]["maximum"] == MAX_K == 6
    assert set(search.output_schema["properties"]) == {"retrieval", "max_cosine", "excerpts"}


# ------------------------------------------------------------------ results

def test_score_equals_the_api_scorer_on_synthetic_rows(server, tools, synthetic):
    scorer, _ = tools
    rows = synthetic[1].iloc[::6]
    calls = [("score_transaction", {"card_key": t.card_key, "trans_ts": t.trans_ts.isoformat(),
                                    "amt": float(t.amt), "category": t.category,
                                    "merch_lat": float(t.merch_lat), "merch_long": float(t.merch_long)})
             for t in rows.itertuples()]
    results = run(server, *calls)
    for t, r in zip(rows.itertuples(), results):
        assert not r.is_error
        out = r.structured_content
        assert json.loads(r.content[0].text) == out
        s = score_transaction(scorer.bundle, scorer.store, card_key=t.card_key,
                              trans_ts=t.trans_ts.to_pydatetime(), amt=t.amt, category=t.category,
                              merch_lat=t.merch_lat, merch_long=t.merch_long)
        assert set(out) == OUTPUT_FIELDS
        assert (out["raw_score"], out["probability_raw"], out["probability_calibrated"], out["alert"]) == \
               (s.raw_score, s.probability_raw, s.probability_calibrated, s.alert)
        assert out["threshold"] == scorer.bundle.threshold
        assert out["model_version"] == scorer.bundle.model_version
        assert [x["feature"] for x in out["reasons"]] == [x["feature"] for x in s.reasons]
        assert [x["contribution"] for x in out["reasons"]] == [x["contribution"] for x in s.reasons]
        for mine, api in zip(out["reasons"], s.reasons):
            want = HIDDEN if api["feature"] in {"age_band", "urban"} else api["value"]
            assert mine["value"] == want


def test_search_returns_chunk_ids_of_the_index_with_code_rendered_labels(server, tools):
    _, search = tools
    (r,) = run(server, ("search_regulations", {"query": "When must a bank file a SAR?", "k": 2}))
    assert not r.is_error
    out = r.structured_content
    assert out["retrieval"] == "dense" and len(out["excerpts"]) == 2
    assert out["excerpts"][0]["chunk_id"].startswith("cfr31_1020:1020.320")
    for i, e in enumerate(out["excerpts"], start=1):
        rec = search.records[e["chunk_id"]]
        assert e["rank"] == i
        assert e["citation"] == citation(rec)
        assert e["pdf_pages"] == [rec["page_start"], rec["page_end"]]
        assert e["section"] == rec["section"] and e["doc_id"] == rec["doc_id"]
    (r,) = run(server, ("search_regulations", {"query": "currency report"}))   # k defaults to 6
    assert len(r.structured_content["excerpts"]) == 3                        # only 3 chunks exist


# ------------------------------------------------------------------ errors

@pytest.mark.parametrize("change, code, loc", [
    ({"card_key": "XYZ"}, "validation_error", ["card_key"]),
    ({"amt": -1}, "validation_error", ["amt"]),
    ({"amt": float("nan")}, "validation_error", ["amt"]),
    ({"category": "groceries"}, "validation_error", ["category"]),
    ({"merch_lat": 91}, "validation_error", ["merch_lat"]),
    ({"trans_ts": "2020-12-01T10:00:00+01:00"}, "validation_error", ["trans_ts"]),
    ({"trans_ts": "not a time"}, "validation_error", ["trans_ts"]),
    ({"client_ref": "x"}, "validation_error", ["client_ref"]),
    ({"extra": 1}, "validation_error", ["extra"]),
    ({"card_key": "e" * 32}, "unknown_card", None),
    ({"trans_ts": "2021-06-01T00:00:00"}, "date_outside_calendar", None),
])
def test_score_errors_use_the_api_codes(server, txn, change, code, loc):
    (r,) = run(server, ("score_transaction", {**txn, **change}))
    assert r.is_error
    body = r.structured_content
    assert body["code"] == code
    assert json.loads(r.content[0].text) == body
    if loc is not None:
        assert loc in [e["loc"] for e in body["detail"]]
        assert all(set(e) == {"loc", "msg", "type"} for e in body["detail"])   # no "input" echo


def test_missing_fields_are_validation_errors(server):
    (r,) = run(server, ("score_transaction", {}))
    assert r.is_error and r.structured_content["code"] == "validation_error"
    assert {tuple(e["loc"]) for e in r.structured_content["detail"]} == {(f,) for f in SCORE_FIELDS}


@pytest.mark.parametrize("args", [{"query": ""}, {"query": "   "}, {"query": "x" * 501},
                                  {"query": "sar", "k": 0}, {"query": "sar", "k": 7}, {}, {"q": "sar"}])
def test_search_validation_errors(server, args):
    (r,) = run(server, ("search_regulations", args))
    assert r.is_error and r.structured_content["code"] == "validation_error"


def test_embedder_down_and_mismatch_are_structured_errors(tools, monkeypatch):
    _, search = tools

    def down(*a, **k):
        raise ConnectionError("refused")       # requests' errors are OSError subclasses

    monkeypatch.setattr(search, "_embedder_checked", False)
    monkeypatch.setattr(search.embedder, "describe", down)
    (r,) = run(build_server(tools[0], search), ("search_regulations", {"query": "sar"}))
    assert r.is_error and r.structured_content["code"] == "embedder_unavailable"

    monkeypatch.setattr(search, "embedder", FakeEmbedder(dim=128))
    (r,) = run(build_server(tools[0], search), ("search_regulations", {"query": "sar"}))
    assert r.is_error and r.structured_content["code"] == "index_mismatch"


def test_unexpected_error_returns_only_internal_error(tools, txn, monkeypatch, caplog):
    import src.mcp_server.tools as mod

    def boom(*args, **kwargs):
        raise RuntimeError(f"secret detail {txn['card_key']}")

    monkeypatch.setattr(mod, "score_transaction", boom)
    with caplog.at_level(logging.DEBUG):
        (r,) = run(build_server(*tools), ("score_transaction", txn))
    assert r.is_error and r.structured_content == {"code": "internal_error"}
    assert r.content[0].text == '{"code": "internal_error"}'
    lines = log_lines(caplog)
    assert json.loads(lines[-1])["error"] == "internal_error:RuntimeError"
    everything = "\n".join(rec.getMessage() for rec in caplog.records)
    assert "secret detail" not in everything and "Traceback" not in caplog.text
    assert_no_request_data(caplog.text, txn)


# ------------------------------------------------------------------ logging rule

def test_logs_are_one_json_line_per_call_without_request_data(server, txn, caplog):
    calls = [("score_transaction", txn),
             ("score_transaction", {**txn, "card_key": "e" * 32}),
             ("score_transaction", {**txn, "amt": -5.25}),
             ("search_regulations", {"query": QUERY, "k": 2})]
    with caplog.at_level(logging.DEBUG):    # every logger, every level (the SDK's included)
        results = run(server, *calls)
    lines = log_lines(caplog)
    events = [json.loads(line) for line in lines]
    assert len(events) == 4
    assert [e.get("error") for e in events] == [None, "unknown_card", "validation_error", None]
    assert set(events[0]) == {"event", "request_id", "tool", "status", "latency_ms",
                              "model_version", "alert", "probability_calibrated"}
    assert events[0]["probability_calibrated"] == round(results[0].structured_content["probability_calibrated"], 3)
    assert set(events[1]) == {"event", "request_id", "tool", "status", "latency_ms", "error"}
    assert set(events[3]) == {"event", "request_id", "tool", "status", "latency_ms", "n_hits"}
    everything = "\n".join(rec.getMessage() for rec in caplog.records)
    for args in (txn, {"card_key": "e" * 32}, {"amt": -5.25}, {"query": QUERY}):
        assert_no_request_data(everything, args)
    assert "structuring" not in everything and "SAR no later" not in everything   # query, excerpt


# ------------------------------------------------------------------ end to end over stdio

def stdio_session(args: list[str], errlog_path, *calls):
    params = StdioServerParameters(command=sys.executable, args=[str(SCRIPT), *args], cwd=str(PROJECT_ROOT))

    async def go():
        with open(errlog_path, "w", encoding="utf-8") as errlog:
            from mcp.client.stdio import stdio_client
            from mcp import ClientSession
            async with stdio_client(params, errlog=errlog) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    names = sorted(t.name for t in (await session.list_tools()).tools)
                    results = []
                    for name, arguments in calls:
                        try:
                            results.append(await session.call_tool(name, arguments))
                        except MCPError as exc:   # a JSON-RPC error, e.g. an unknown tool
                            results.append(exc)
                    return names, results
    return anyio.run(go)


def test_end_to_end_over_stdio_with_the_sdk_client(synthetic, fake_index, txn, tmp_path):
    errlog = tmp_path / "stderr.txt"
    names, (score, bad, search, unknown) = stdio_session(
        ["--snapshot", str(synthetic[0]), "--index-dir", str(fake_index), "--embedder", "fake"], errlog,
        ("score_transaction", txn), ("score_transaction", {**txn, "card_key": "e" * 32}),
        ("search_regulations", {"query": QUERY, "k": 1}), ("delete_everything", {}))
    assert names == ["score_transaction", "search_regulations"]
    assert not score.is_error and set(score.structured_content) == OUTPUT_FIELDS
    assert bad.is_error and bad.structured_content["code"] == "unknown_card"
    assert not search.is_error and len(search.structured_content["excerpts"]) == 1
    assert isinstance(unknown, MCPError) and unknown.error.code == -32602    # protocol error
    assert "Traceback" not in unknown.error.message
    text = errlog.read_text(encoding="utf-8")
    events = [json.loads(line) for line in text.splitlines()]      # every stderr line is our JSON
    assert [e["event"] for e in events] == ["startup"] + ["tool_call"] * 4
    assert events[-1]["error"] == "unknown_tool"
    assert_no_request_data(text, {**txn, "query": QUERY})


def test_server_refuses_to_start_on_a_bad_index(tmp_path):
    errlog = tmp_path / "stderr.txt"
    with pytest.raises(Exception):
        stdio_session(["--index-dir", str(tmp_path / "missing"), "--embedder", "fake"], errlog)
    event = json.loads(errlog.read_text(encoding="utf-8").splitlines()[-1])
    assert event["event"] == "startup_failed"


# ------------------------------------------------------------------ local: real index, real data

@pytest.fixture(scope="module")
def real_search():
    from src.rag.embed import OllamaEmbedder
    scorer, search, store = load_tools(BUNDLE_DIR, SERVING_DB, DEFAULT_INDEX_DIR, OllamaEmbedder(),
                                       pinned_index_sha256())
    yield scorer, search
    store.close()


@pytest.mark.gold
@pytest.mark.corpus
@pytest.mark.ollama
def test_search_over_the_pinned_index_returns_valid_chunk_ids(real_search):
    server = build_server(*real_search)
    _, search = real_search
    assert len(search.records) == 344
    queries = ["When must a bank file a suspicious activity report?",
               "What records must be kept for funds transfers?",
               "customer due diligence for legal entity customers"]
    results = run(server, *[("search_regulations", {"query": q}) for q in queries])
    for r in results:
        assert not r.is_error and len(r.structured_content["excerpts"]) == 6
        for e in r.structured_content["excerpts"]:
            assert e["chunk_id"] in search.records
            assert e["citation"] == citation(search.records[e["chunk_id"]])


@pytest.mark.gold
def test_score_equals_the_in_process_api_on_real_test_period_rows():
    """Rows are picked by score only: 1,500 random test-period rows, then up to 5 alerts + 15 others."""
    from fastapi.testclient import TestClient

    from src.api.bundle import load_bundle
    from src.api.history import SnapshotStore
    from src.api.main import create_app
    from src.mcp_server.tools import Scorer

    with duckdb.connect(str(GOLD_DEV_DB), read_only=True) as con:   # no is_fraud column is read
        rows = con.execute("""
            select f.card_key, f.trans_ts, f.amt, m.category, f.merch_lat, f.merch_long
            from fact_transactions f join dim_merchant m using (merchant_key)
            where f.trans_ts >= '2020-10-01' order by f.trans_num""").df()
    rng = np.random.default_rng(7)
    rows = rows.iloc[np.sort(rng.choice(len(rows), size=1500, replace=False))]
    bodies = [{"card_key": r.card_key, "trans_ts": r.trans_ts.isoformat(), "amt": float(r.amt),
               "category": r.category, "merch_lat": float(r.merch_lat), "merch_long": float(r.merch_long)}
              for r in rows.itertuples()]

    app = create_app(BUNDLE_DIR, SERVING_DB)
    try:
        client = TestClient(app)
        api = [client.post("/score", json=b).json() for b in bodies]
    finally:
        app.state.store.close()
    alerts = [i for i, a in enumerate(api) if a["alert"]][:5]
    others = [i for i, a in enumerate(api) if not a["alert"]][:15]
    pick = alerts + others

    store = SnapshotStore(SERVING_DB)
    try:
        server = build_server(Scorer(load_bundle(BUNDLE_DIR), store), lambda a: {})
        results = run(server, *[("score_transaction", bodies[i]) for i in pick])
    finally:
        store.close()
    print(f"\n[mcp vs api] rows compared: {len(pick)} ({len(alerts)} alerts)")
    for i, r in zip(pick, results):
        assert not r.is_error
        out, ref = r.structured_content, api[i]
        for k in ("model_version", "raw_score", "probability_raw", "probability_calibrated", "alert", "threshold"):
            assert out[k] == ref[k], (k, out[k], ref[k])
        want = [{**x, "value": HIDDEN if x["feature"] in {"age_band", "urban"} else x["value"]}
                for x in ref["reasons"]]
        assert out["reasons"] == want


@pytest.mark.gold
@pytest.mark.corpus
@pytest.mark.ollama
def test_real_server_end_to_end_over_stdio(real_search, tmp_path):
    """Default paths: committed bundle, real snapshot, pinned index, Ollama embedder."""
    con = real_search[0].store._con.cursor()   # the open snapshot (DuckDB allows one config per file)
    card, ts = con.execute("""select card_key, max(trans_ts) from card_history
                              group by card_key order by card_key limit 1""").fetchone()
    txn = {"card_key": card, "trans_ts": ts.isoformat(), "amt": 25.0, "category": "grocery_pos",
           "merch_lat": 40.0, "merch_long": -75.0}
    errlog = tmp_path / "stderr.txt"
    names, (score, search) = stdio_session(
        [], errlog, ("score_transaction", txn),
        ("search_regulations", {"query": "How long must a bank keep SAR records?", "k": 3}))
    assert names == ["score_transaction", "search_regulations"]
    assert not score.is_error and set(score.structured_content) == OUTPUT_FIELDS
    assert not search.is_error and len(search.structured_content["excerpts"]) == 3
    text = errlog.read_text(encoding="utf-8")
    assert json.loads(text.splitlines()[0])["n_chunks"] == 344
    assert_no_request_data(text, txn)
