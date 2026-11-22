"""Sections -> chunks with citation metadata (Phase 7).

A chunk never crosses a section. Inside a section, chunks are about TARGET_WORDS
long, end at a sentence end when one falls between MIN_CUT_WORDS and TARGET_WORDS,
and overlap by OVERLAP_WORDS. A remainder of at most MAX_WORDS stays in one chunk
(no tiny tails). Words keep their PDF page, so page_start/page_end are exact.

The model will only ever cite a chunk_id ("[C:<chunk_id>]"); section names and
pages are rendered from this metadata by code (citation()).
"""

import hashlib
import re
from dataclasses import asdict, dataclass

from src.rag.parse import DOC_KINDS, Section

TARGET_WORDS = 350
MAX_WORDS = 450
MIN_CUT_WORDS = 280
OVERLAP_WORDS = 50
CHUNKER_VERSION = "1"

FAQ_NAME = "FinCEN SAR FAQs (Oct. 9, 2025)"
FFIEC_NAME = "FFIEC BSA/AML Examination Manual"


def ffiec_heading(doc_id: str, subpart: str, title: str) -> str:
    """'FFIEC BSA/AML Examination Manual: <section> › <parent headings> › <heading>' (no repeats)."""
    _, doc_title = DOC_KINDS[doc_id]
    parts = [doc_title]
    for part in [p.strip() for p in subpart.split("›") if p.strip()] + [title]:
        if all(part.casefold() != seen.casefold() for seen in parts):
            parts.append(part)
    return f"{FFIEC_NAME}: " + " › ".join(parts)


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    section: str
    section_title: str
    subpart: str
    seq: int
    page_start: int
    page_end: int
    label_start: str
    label_end: str
    n_words: int
    text: str
    text_sha256: str

    @property
    def heading(self) -> str:
        kind, _ = DOC_KINDS[self.doc_id]
        if kind == "cfr":
            return f"31 CFR § {self.section} {self.section_title}"
        if kind == "ffiec":
            return ffiec_heading(self.doc_id, self.subpart, self.section_title)
        return f"{FAQ_NAME}, {self.section_title}"

    @property
    def embed_text(self) -> str:
        """Text that is embedded and BM25-indexed: the heading gives every chunk its section context."""
        return f"{self.heading}\n{self.text}"

    def to_record(self) -> dict:
        return {**asdict(self), "heading": self.heading}


def chunker_params(paragraphs: bool = False) -> dict:
    params = {"version": CHUNKER_VERSION, "target_words": TARGET_WORDS, "max_words": MAX_WORDS,
              "min_cut_words": MIN_CUT_WORDS, "overlap_words": OVERLAP_WORDS}
    if paragraphs:      # round-4 retrieval candidate: chunks also never cross a CFR paragraph (a), (b), ...
        params["paragraphs"] = True
    return params


PARAGRAPH = re.compile(r"^\(([a-z])\)\s")


def paragraph_groups(lines: list) -> list[list]:
    """Split a section's lines at top-level paragraph markers "(a) ", "(b) ", ... at a line start.

    A marker counts only if it is the next letter in sequence, so "(i)" (a roman numeral inside
    a paragraph) does not split unless the paragraph before it was "(h)".
    """
    groups, expected = [[]], "a"
    for line in lines:
        m = PARAGRAPH.match(line.text)
        if m and m.group(1) == expected:
            if groups[-1]:
                groups.append([])
            expected = chr(ord(expected) + 1)
        groups[-1].append(line)
    return [g for g in groups if g]


def _ends_sentence(word: str) -> bool:
    return word.endswith((".", ";", ":")) and not word.endswith("..")


def split_spans(ends: list[bool], target=TARGET_WORDS, max_words=MAX_WORDS,
                min_cut=MIN_CUT_WORDS, overlap=OVERLAP_WORDS) -> list[tuple[int, int]]:
    """[start, end) word spans; `ends[i]` says whether word i ends a sentence."""
    if not 0 <= overlap < min_cut <= target <= max_words:
        raise ValueError("need 0 <= overlap < min_cut <= target <= max_words")
    n, start, spans = len(ends), 0, []
    while n - start > max_words:
        cut = next((j for j in range(start + target, start + min_cut - 1, -1) if ends[j - 1]),
                   start + target)
        spans.append((start, cut))
        start = cut - overlap
    if n > start:
        spans.append((start, n))
    return spans


def chunk_section(section: Section, paragraphs: bool = False, **params) -> list[Chunk]:
    groups = paragraph_groups(section.lines) if paragraphs else [section.lines]
    word_groups = [[(w, line.pdf_page, line.page_label) for line in g for w in line.text.split()] for g in groups]
    pieces = []          # (words of one paragraph group, span)
    for words in word_groups:
        if words:
            pieces += [(words, span) for span in split_spans([_ends_sentence(w) for w, _, _ in words], **params)]
    chunks = []
    for seq, (words, (a, b)) in enumerate(pieces, start=1):
        part = words[a:b]
        text = " ".join(w for w, _, _ in part)
        chunks.append(Chunk(
            chunk_id=f"{section.doc_id}:{section.section}:{seq:03d}",
            doc_id=section.doc_id, section=section.section, section_title=section.title,
            subpart=section.subpart, seq=seq,
            page_start=part[0][1], page_end=part[-1][1],
            label_start=part[0][2], label_end=part[-1][2],
            n_words=len(part), text=text,
            text_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        ))
    return chunks


def chunk_sections(sections: list[Section], paragraphs: bool = False, **params) -> list[Chunk]:
    chunks = [c for s in sections
              for c in chunk_section(s, paragraphs=paragraphs and DOC_KINDS[s.doc_id][0] == "cfr", **params)]
    ids = [c.chunk_id for c in chunks]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate chunk_id; a section number appears twice")
    return chunks


def _pages(a, b, prefix: str) -> str:
    if a == b:
        return f"{prefix} {a}"
    joiner = " to " if any(ch in f"{a}{b}" for ch in "–-") else "–"   # "F–2 to F–3", not "F–2–F–3"
    return f"{prefix} {a}{joiner}{b}"


def citation(record: dict) -> str:
    """Human-readable citation, built only from chunk metadata (never from model output)."""
    kind, _ = DOC_KINDS[record["doc_id"]]
    pdf = _pages(record["page_start"], record["page_end"], "PDF p.")
    printed = (f", printed p. {_pages(record['label_start'], record['label_end'], '').strip()}"
               if record.get("label_start") else "")
    if kind == "cfr":
        return f"31 CFR § {record['section']} ({record['section_title']}), {pdf}{printed}"
    if kind == "ffiec":
        _, doc_title = DOC_KINDS[record["doc_id"]]
        return f"{FFIEC_NAME}, {doc_title}, “{record['section_title']}”, {pdf}{printed}"
    return f"{FAQ_NAME}, {record['section_title']}, {pdf}{printed}"


def source_label(record: dict) -> str:
    """'<document> — <section title>', shown with each numbered excerpt so a rule is read in its context.

    e.g. '31 CFR § 1010.330 — Reports relating to currency in excess of $10,000 received in a trade
    or business' (round 4, 2026-11-23; built from metadata, like citation()).
    """
    kind, doc_title = DOC_KINDS[record["doc_id"]]
    if kind == "cfr":
        return f"31 CFR § {record['section']} — {record['section_title']}"
    if kind == "ffiec":
        path = ffiec_heading(record["doc_id"], record.get("subpart", ""), record["section_title"])
        _, _, rest = path.partition(": ")
        doc, _, sec = rest.partition(" › ")
        return f"{FFIEC_NAME}, {doc} — {sec or record['section_title']}"
    return f"{FAQ_NAME} — {record['section_title']}"
