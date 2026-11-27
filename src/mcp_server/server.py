"""MCP server over stdio with two read-only tools (docs/rag_design.md §14).

Run (normally the MCP client starts it; see docs/rag_design.md §14):

    .venv/Scripts/python.exe scripts/mcp_server.py

Built on the official SDK's low-level Server, so the schemas and the error payloads
are this project's own. stdout is the protocol channel; logs go to stderr only.

Logging (mirrors the API): one JSON line per tool call with request_id, tool, status,
latency and error code; for score_transaction also model_version, alert and the
calibrated probability rounded to 3 decimals; for search_regulations the number of
hits. Never logged: card_key, amount, time, coordinates, query text, excerpt text.
Unexpected errors log only the exception type, and the client gets "internal_error".
"""

import argparse
import json
import logging
import os
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import anyio
import mcp_types as types
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server
from mcp.shared.exceptions import MCPError

from src.mcp_server.tools import (MAX_K, RegulationSearch, Scorer, ScoreOutput, SearchArgs,
                                  SearchOutput, ToolFailure, score_input_schema)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BUNDLE_DIR = PROJECT_ROOT / "models" / "F"
DEFAULT_SNAPSHOT = PROJECT_ROOT / "data" / "serving" / "serving.duckdb"
DEFAULT_INDEX_DIR = PROJECT_ROOT / "data" / "rag" / "index"
PREREGISTRATION = PROJECT_ROOT / "data" / "rag" / "eval" / "preregistration.json"
SERVER_NAME = "fraud-intelligence"

logger = logging.getLogger("fraud_mcp")

READ_ONLY = types.ToolAnnotations(read_only_hint=True, destructive_hint=False,
                                  idempotent_hint=True, open_world_hint=False)

SEARCH_SPEC = types.Tool(
    name="search_regulations",
    title="Search US AML / fraud regulation",
    description=(
        "Dense retrieval over a fixed, hash-pinned corpus of US AML regulation: 31 CFR Parts 1010 "
        "and 1020 (edition of July 1, 2025), FinCEN SAR FAQs (Oct. 2025) and six FFIEC BSA/AML "
        f"Examination Manual sections. Returns up to {MAX_K} excerpts with document, section, "
        "pages and a citation rendered from metadata. No LLM is involved; there is no refusal "
        "threshold, so check that the excerpts actually answer the question."),
    input_schema=SearchArgs.model_json_schema(),
    output_schema=SearchOutput.model_json_schema(),
    annotations=READ_ONLY,
)

SCORE_SPEC = types.Tool(
    name="score_transaction",
    title="Score one card transaction (model F)",
    description=(
        "Scores one transaction with the Phase 6 fraud model F (LightGBM), in process, against a "
        "read-only replay snapshot (2019-01-01 to 2020-12-31, known cards only). Returns the raw "
        "and calibrated probability, the alert flag (probability_raw >= threshold), the "
        "threshold, the model version and the top 3 SHAP reasons (values of the card-holder "
        "profile features age_band and urban are hidden). Decision support only."),
    input_schema=score_input_schema(),
    output_schema=ScoreOutput.model_json_schema(),
    annotations=READ_ONLY,
)


@dataclass(frozen=True)
class ToolEntry:
    spec: types.Tool
    run: Callable[[dict], dict]
    log_fields: Callable[[dict], dict]


def setup_logging(stream=None) -> None:
    """Our JSON lines on stderr; the SDK's own loggers only from WARNING up."""
    if not any(getattr(h, "_fraud_mcp", False) for h in logger.handlers):
        handler = logging.StreamHandler(stream or sys.stderr)
        handler.setFormatter(logging.Formatter("%(message)s"))
        handler._fraud_mcp = True
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logging.getLogger("mcp").setLevel(logging.WARNING)


def _log(event: dict) -> None:
    logger.info(json.dumps(event, separators=(",", ":")))


def _result(payload: dict, is_error: bool = False) -> types.CallToolResult:
    text = json.dumps(payload, ensure_ascii=False)
    return types.CallToolResult(content=[types.TextContent(type="text", text=text)],
                                structured_content=payload, is_error=is_error)


def error_result(code: str, detail=None) -> types.CallToolResult:
    body = {"code": code}
    if detail is not None:
        body["detail"] = detail
    return _result(body, is_error=True)


def build_server(scorer, search) -> Server:
    """The MCP server for the given tool callables (real or test doubles)."""
    setup_logging()
    tools = {e.spec.name: e for e in (
        ToolEntry(SEARCH_SPEC, search, RegulationSearch.log_fields),
        ToolEntry(SCORE_SPEC, scorer, Scorer.log_fields),
    )}
    limiter = anyio.CapacityLimiter(1)   # one call at a time, off the event loop

    async def list_tools(ctx, params) -> types.ListToolsResult:
        return types.ListToolsResult(tools=[e.spec for e in tools.values()])

    async def call_tool(ctx, params: types.CallToolRequestParams) -> types.CallToolResult:
        entry = tools.get(params.name)
        if entry is None:
            _log({"event": "tool_call", "tool": None, "status": "error", "error": "unknown_tool"})
            raise MCPError(code=types.INVALID_PARAMS, message="Unknown tool")
        event = {"event": "tool_call", "request_id": str(uuid.uuid4()), "tool": entry.spec.name}
        start = time.perf_counter()
        try:
            payload = await anyio.to_thread.run_sync(entry.run, params.arguments or {}, limiter=limiter)
            event.update(status="ok", **entry.log_fields(payload))
            return _result(payload)
        except ToolFailure as failure:
            event.update(status="error", error=failure.code)
            return error_result(failure.code, failure.detail)
        except Exception as exc:  # unexpected: no message, no traceback, only the type in the log
            event.update(status="error", error=f"internal_error:{type(exc).__name__}")
            return error_result("internal_error")
        finally:
            event["latency_ms"] = round(1000 * (time.perf_counter() - start), 2)
            _log(event)

    return Server(SERVER_NAME, version="1", on_list_tools=list_tools, on_call_tool=call_tool,
                  instructions=("Two read-only tools: search_regulations (US AML regulation excerpts "
                                "with citations) and score_transaction (fraud model F on a replay "
                                "snapshot). Neither calls an LLM."))


def pinned_index_sha256(path: Path = PREREGISTRATION) -> str:
    return json.loads(path.read_text(encoding="utf-8"))["fingerprint"]["index_manifest_sha256"]


def load_tools(bundle_dir, snapshot_path, index_dir, embedder, pinned_sha256: str | None):
    """Verify and load everything local; raise on any mismatch (fail fast, as the API)."""
    from src.api.bundle import load_bundle
    from src.api.history import SnapshotStore
    from src.rag.config import build_retriever
    from src.rag.corpus import load_manifest
    from src.rag.index import MANIFEST_FILE, IndexMismatch, load_index
    from src.rag.runmeta import file_sha256

    index_dir = Path(index_dir)
    if pinned_sha256 is not None and file_sha256(index_dir / MANIFEST_FILE) != pinned_sha256:
        raise IndexMismatch("index manifest differs from the frozen index (preregistration.json)")
    hashes = {d["doc_id"]: d["sha256"] for d in load_manifest()["documents"]}
    frame, embeddings, manifest = load_index(index_dir, hashes)
    search = RegulationSearch(frame, build_retriever(frame, embeddings, embedder), manifest, embedder)
    bundle = load_bundle(bundle_dir)
    store = SnapshotStore(snapshot_path)
    return Scorer(bundle, store), search, store


async def serve_stdio(server: Server) -> None:
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Fraud-intelligence MCP server (stdio).")
    parser.add_argument("--bundle-dir", default=os.environ.get("BUNDLE_DIR", DEFAULT_BUNDLE_DIR))
    parser.add_argument("--snapshot", default=os.environ.get("SNAPSHOT_PATH", DEFAULT_SNAPSHOT))
    parser.add_argument("--index-dir", default=os.environ.get("RAG_INDEX_DIR", DEFAULT_INDEX_DIR))
    parser.add_argument("--embedder", choices=["ollama", "fake"], default="ollama",
                        help="fake = hashed bag of words, for tests with a fake index only")
    args = parser.parse_args(argv)
    setup_logging()

    if args.embedder == "ollama":
        from src.rag.embed import OllamaEmbedder
        embedder, pinned = OllamaEmbedder(timeout=30), pinned_index_sha256()
    else:   # cannot serve the real index: check_embedder rejects a fake embedder on it
        from src.rag.embed import FakeEmbedder
        embedder, pinned = FakeEmbedder(), None
    try:
        scorer, search, store = load_tools(args.bundle_dir, args.snapshot, args.index_dir, embedder, pinned)
    except Exception as exc:  # local files only: names and hashes, never request data
        _log({"event": "startup_failed", "error": type(exc).__name__, "detail": str(exc)})
        return 1
    _log({"event": "startup", "model_version": scorer.bundle.model_version,
          "snapshot_built_at": store.meta.get("built_at"), "n_chunks": len(search.records)})
    try:
        anyio.run(serve_stdio, build_server(scorer, search))
    finally:
        store.close()
    return 0
