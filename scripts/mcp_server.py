"""Start the local MCP server over stdio (Phase 7; docs/rag_design.md §14).

    .venv/Scripts/python.exe scripts/mcp_server.py [--snapshot PATH] [--index-dir PATH]

Normally an MCP client (e.g. Claude Desktop) starts this process and talks to it on
stdin/stdout. Logs are JSON lines on stderr.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.mcp_server.server import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
