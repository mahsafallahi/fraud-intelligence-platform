"""The API's pins must agree with the pins the model was built and tested with."""

import re

from tests.conftest import PROJECT_ROOT

PIN = re.compile(r"^([A-Za-z0-9_.\-]+)==([^\s\\]+)")


def pins(name: str) -> dict:
    out = {}
    for line in (PROJECT_ROOT / name).read_text(encoding="utf-8").splitlines():
        m = PIN.match(line.strip())
        if m:
            out[m.group(1).lower().replace("_", "-")] = m.group(2)
    return out


def test_api_pins_match_dev_and_runtime_pins():
    api = pins("requirements-api.txt")
    for other in ("requirements.txt", "requirements-dev.txt", "requirements-dbt.txt"):
        shared = set(api) & set(pins(other))
        diff = {p: (api[p], pins(other)[p]) for p in shared if api[p] != pins(other)[p]}
        assert not diff, f"requirements-api.txt and {other} disagree: {diff}"


def test_rag_pins_match_every_other_requirements_file():
    rag = pins("requirements-rag.txt")
    assert rag == {"numpy": "2.5.3", "pypdf": "7.1.1"}
    for other in ("requirements.txt", "requirements-dev.txt", "requirements-api.txt", "requirements-dbt.txt"):
        shared = set(rag) & set(pins(other))
        diff = {p: (rag[p], pins(other)[p]) for p in shared if rag[p] != pins(other)[p]}
        assert not diff, f"requirements-rag.txt and {other} disagree: {diff}"
    lines = (PROJECT_ROOT / "requirements-rag.txt").read_text(encoding="utf-8").splitlines()
    assert "-r requirements.txt" in lines


def test_lock_contains_every_api_pin_with_the_same_version():
    api, lock = pins("requirements-api.txt"), pins("requirements-api.lock")
    assert {p: lock.get(p) for p in api} == api


def test_every_lock_entry_is_pin_continuation_then_hash():
    # pip format: "name==version \" on one line, "    --hash=sha256:<64 hex>" on the next
    lines = (PROJECT_ROOT / "requirements-api.lock").read_text(encoding="utf-8").splitlines()
    entries = [i for i, line in enumerate(lines) if PIN.match(line)]
    assert entries
    for i in entries:
        assert re.fullmatch(r"[a-z0-9.\-]+==\S+ \\", lines[i]), lines[i]
        assert re.fullmatch(r"    --hash=sha256:[0-9a-f]{64}", lines[i + 1]), lines[i + 1]
    assert sum("--hash=" in line for line in lines) == len(entries)


def test_mcp_pins_match_every_other_requirements_file():
    mcp = pins("requirements-mcp.txt")
    assert mcp == {"lightgbm": "4.7.0", "scipy": "1.18.1", "pydantic": "2.14.1", "mcp": "2.3.0"}
    for other in ("requirements.txt", "requirements-dev.txt", "requirements-api.txt",
                  "requirements-rag.txt", "requirements-dbt.txt"):
        shared = set(mcp) & set(pins(other))
        diff = {p: (mcp[p], pins(other)[p]) for p in shared if mcp[p] != pins(other)[p]}
        assert not diff, f"requirements-mcp.txt and {other} disagree: {diff}"
    lines = (PROJECT_ROOT / "requirements-mcp.txt").read_text(encoding="utf-8").splitlines()
    assert "-r requirements-rag.txt" in lines
