"""The analyst note (Phase 7): a pydantic model, rendered to Markdown by code.

The header and the reasons are filled in by code from the API response; the LLM
writes only "Regulatory context" and "Suggested next steps", with [n] citations
that code maps to chunks and renders from metadata. The card token is added to the
header by code after generation (D4). The disclaimer is fixed.
"""

import re

from pydantic import BaseModel, ConfigDict, Field

from src.rag.chunk import citation
from src.rag.citations import NUM_CITE, numbers_in

DISCLAIMER = ("Decision support only; this is not a SAR determination. Regulatory text: 31 CFR Parts 1010 and "
              "1020 (edition of July 1, 2025), FinCEN SAR FAQs (October 9, 2025) and FFIEC BSA/AML "
              "Examination Manual sections; later amendments are not included.")
STATUSES = ("ALERT", "BELOW THRESHOLD", "NOT SCORED", "API UNAVAILABLE", "INCOMPLETE")
SECTION_RE = re.compile(r"REGULATORY CONTEXT:\s*(?P<ctx>.*?)\s*NEXT STEPS:\s*(?P<steps>.*)\Z", re.S | re.I)


class NoteReason(BaseModel):
    model_config = ConfigDict(extra="forbid")
    feature: str
    meaning: str
    value: float | int | str | None
    contribution: float


class AnalystNote(BaseModel):
    model_config = ConfigDict(extra="forbid")
    txn_ref: str
    mode: str                                   # "baseline" | "agent"
    status: str
    status_detail: str = ""                     # e.g. the API error code
    model_version: str | None = None
    probability_raw: float | None = None
    threshold: float | None = None
    reasons: list[NoteReason] = Field(default_factory=list)
    regulatory_context: str = ""                # LLM, with [n] citations
    next_steps: str = ""                        # LLM, with [n] citations
    cited_chunks: list[str] = Field(default_factory=list)   # in order of first citation
    excerpt_ids: list[str] = Field(default_factory=list)    # what [1]..[k] meant for this note
    disclaimer: str = DISCLAIMER

    def model_post_init(self, _):
        if not any(self.status == s or self.status.startswith(s + ":") for s in STATUSES):
            raise ValueError(f"unknown status {self.status!r}")


def parse_sections(text: str) -> tuple[str, str] | None:
    m = SECTION_RE.search(text.strip())
    return (m.group("ctx").strip(), m.group("steps").strip()) if m else None


def _renumber(text: str, excerpt_ids: list[str], display: dict[str, int]) -> str:
    def sub(m):
        out = []
        for n in numbers_in(m.group(0)):
            d = display.get(excerpt_ids[n - 1]) if 1 <= n <= len(excerpt_ids) else None
            if d is not None and d not in out:
                out.append(d)
        return "".join(f"[{d}]" for d in out)
    return NUM_CITE.sub(sub, text)


def render_markdown(note: AnalystNote, records: dict[str, dict], card_key: str | None = None) -> str:
    """Markdown for the analyst. `card_key` is added here, by code, never by the LLM (D4)."""
    display = {cid: n for n, cid in enumerate(note.cited_chunks, start=1)}
    head = [f"# Analyst note: {note.txn_ref} ({note.mode})", ""]
    meta = f"**Status:** {note.status}"
    if note.status == "ALERT":
        meta += f" (probability {note.probability_raw:.3f} >= threshold {note.threshold:.4f})"
    elif note.status == "BELOW THRESHOLD":
        meta += f" (probability {note.probability_raw:.3f} < threshold {note.threshold:.4f})"
    if note.model_version:
        meta += f" | model {note.model_version}"
    if card_key:
        meta += f" | card token `{card_key}`"
    head.append(meta)
    if note.status_detail:
        head.append(f"**Detail:** {note.status_detail}")
    if note.reasons:
        head += ["", "## Why the model flagged it (top SHAP reasons, from the API)"]
        for i, r in enumerate(note.reasons, start=1):
            head.append(f"{i}. {r.meaning} = {r.value} (SHAP {r.contribution:+.3f} log-odds)")
    if note.regulatory_context:
        head += ["", "## Regulatory context", _renumber(note.regulatory_context, note.excerpt_ids, display)]
    if note.next_steps:
        head += ["", "## Suggested next steps", _renumber(note.next_steps, note.excerpt_ids, display)]
    if note.cited_chunks:
        head += ["", "## Sources"] + [f"[{n}] {citation(records[c])}" for c, n in display.items()]
    head += ["", "---", f"*{note.disclaimer}*"]
    return "\n".join(head)
