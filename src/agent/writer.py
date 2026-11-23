"""Validate the LLM-written part of an analyst note (shared by the baseline and the agent).

The text must be "REGULATORY CONTEXT: ... NEXT STEPS: ...". Each section is checked
with the RAG citation validator against the excerpts shown ([1]..[k]); "NO_SUPPORT"
is an allowed, empty section. One retry with the errors, then the note is INCOMPLETE.
"""

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

from src.agent.note import parse_sections
from src.rag.citations import mechanical_claim_check, validate

PROMPT_DIR = Path(__file__).resolve().parent / "prompts"
PROMPT_FILES = ("writer_system.txt", "agent_system.txt")
NO_SUPPORT = "NO_SUPPORT"
RETRY = ("Your previous answer failed these checks:\n{errors}\n\nWrite it again in exactly the format "
         "REGULATORY CONTEXT: ... NEXT STEPS: ..., end every sentence with the number(s) of the excerpt(s) "
         "that support it, and use only excerpt numbers you were given (or write NO_SUPPORT).")


def load_prompt(name: str) -> str:
    return (PROMPT_DIR / name).read_text(encoding="utf-8").replace("\r\n", "\n")


def prompt_hashes() -> dict[str, str]:
    return {n: hashlib.sha256(load_prompt(n).encode("utf-8")).hexdigest() for n in PROMPT_FILES}


@dataclass
class Written:
    ok: bool
    context: str = ""
    steps: str = ""
    cited: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    claim_checks: list[dict] = field(default_factory=list)


def check_written(text: str, excerpt_ids: list[str], chunk_text: dict[str, str]) -> Written:
    parts = parse_sections(text)
    if parts is None:
        return Written(False, errors=["the answer is not in the format REGULATORY CONTEXT: ... NEXT STEPS: ..."])
    cited, errors, claims = [], [], []
    for name, section in zip(("REGULATORY CONTEXT", "NEXT STEPS"), parts):
        if section.strip(" .").upper() == NO_SUPPORT:
            continue
        v = validate(section, excerpt_ids, chunk_text)
        errors += [f"{name}: {e}" for e in v.errors]
        cited += [c for c in v.cited if c not in cited]
        claims += mechanical_claim_check(v.sentences, chunk_text)
    ctx, steps = ("" if p.strip(" .").upper() == NO_SUPPORT else p for p in parts)
    return Written(not errors, ctx, steps, cited, errors, claims)


def excerpts_message(context: dict, excerpts: list[dict]) -> str:
    """User message for the baseline writer: the alert facts (LLMAlertContext) and numbered excerpts."""
    blocks = [f"[{e['number']}] {e['source']}\n<excerpt number=\"{e['number']}\">\n{e['text']}\n</excerpt>"
              for e in excerpts]
    return ("Alert facts (JSON):\n" + json.dumps(context, ensure_ascii=False) + "\n\nExcerpts:\n"
            + "\n\n".join(blocks) + "\n\nWrite the two sections now.")
