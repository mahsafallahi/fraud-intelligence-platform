"""PDF text -> cleaned lines -> sections, keeping the PDF page of every line (Phase 7).

What is stripped (found by inspecting the real PDFs, 2026-11-20):
- CFR (GovInfo annual edition): line 1 = printed page number (kept as the page
  label), line 2 = running header ("31 CFR Ch. X (7-1-25 Edition) § ..." or
  "Financial Crimes Enforc. Network, Treas. § ..."; absent on a part's first
  page), the printer's slug ("VerDate ... with CFR", split over two lines on
  image pages) and image placeholders ("ER28SE17.004</GPH>"; Part 1010 PDF
  pages 16-18 are the CDD certification form printed as images, so their text
  is not in the corpus). "Subpart X—..." lines (titles wrap over up to 4 lines)
  are moved into metadata.
  Text outside the document's own part (the tail of § 1010.980 at the start of
  the Part 1020 file, Part 1020 at the end of the Part 1010 file), the part's
  table of contents / authority notes before the first section, and [Reserved]
  sections are dropped.
- FinCEN FAQ: blank lines and the page number at the top of each page.
Words split by a hyphen at a line break are rejoined (see dehyphenate).
"""

import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

# doc_id -> (kind, CFR part). FFIEC documents get their own kind when they arrive.
DOC_KINDS = {
    "cfr31_1010": ("cfr", "1010"),
    "cfr31_1020": ("cfr", "1020"),
    "fincen_sar_faq_2025": ("faq", None),
}

PAGE_NUMBER = re.compile(r"^\d{1,5}$")
CFR_HEADER = re.compile(r"^(31 CFR Ch\. X \(.*Edition\)|Financial Crimes Enforc\. Network, Treas\.)")
CFR_SLUG = re.compile(r"^VerDate ")
CFR_SLUG_END = re.compile(r"PROD with CFR$")       # second half of a slug split over two lines
CFR_GRAPHIC = re.compile(r"^(\S+</GPH>\s*)+$")       # image placeholder (forms printed as images)
CFR_PART = re.compile(r"^PART (\d{4})\b")
CFR_SUBPART = re.compile(r"^Subpart ([A-Z])\s*[—–-]\s*(.*)$")
CFR_SECTION = re.compile(r"^§\s?(\d{4})\.(\d+)\s+(.*)$")
CFR_MULTI_SECTION = re.compile(r"^§§\s?\d{4}\.\d+")
CFR_TOC_ENTRY = re.compile(r"^(\d{4})\.(\d+)\s+\S")
FAQ_QUESTION = re.compile(r"^Question (\d+):\s*(.*)$")
WORD = re.compile(r"[A-Za-z]+(?:-[A-Za-z]+)*")


@dataclass
class Line:
    text: str
    pdf_page: int        # 1-based page in the downloaded PDF
    page_label: str      # printed page number ("" if none)


@dataclass
class PageClean:
    pdf_page: int
    page_label: str
    lines: list[str]
    stripped: dict = field(default_factory=dict)  # what was removed, for the report


@dataclass
class Section:
    doc_id: str
    section: str          # "1020.320", "Q3", "intro"
    title: str
    subpart: str
    lines: list[Line]


def read_pdf_pages(path: Path) -> list[str]:
    from pypdf import PdfReader  # lazy: CI has no pypdf
    return [page.extract_text() or "" for page in PdfReader(str(path)).pages]


def _strip_blank(lines: list[str]) -> list[str]:
    return [line.rstrip() for line in lines if line.strip()]


def clean_cfr_page(text: str, pdf_page: int) -> PageClean:
    lines = _strip_blank(text.splitlines())
    stripped, label = {}, ""
    if lines and PAGE_NUMBER.match(lines[0].strip()):
        label = lines.pop(0).strip()
        stripped["page_number"] = label
    if lines and CFR_HEADER.match(lines[0].strip()):
        stripped["running_header"] = lines.pop(0).strip()
    kept = []
    for line in (line.strip() for line in lines):
        if CFR_SLUG.match(line) or CFR_SLUG_END.search(line):
            stripped["slug"] = stripped.get("slug", 0) + 1
        elif CFR_GRAPHIC.match(line):
            stripped["graphic_placeholder"] = stripped.get("graphic_placeholder", 0) + 1
        else:
            kept.append(line)
    return PageClean(pdf_page, label, kept, stripped)


def clean_faq_page(text: str, pdf_page: int) -> PageClean:
    lines = _strip_blank(text.splitlines())
    stripped, label = {}, ""
    if lines and PAGE_NUMBER.match(lines[0].strip()):
        label = lines.pop(0).strip()
        stripped["page_number"] = label
    return PageClean(pdf_page, label, [line.strip() for line in lines], stripped)


def dehyphenate(lines: list[Line]) -> list[Line]:
    """Rejoin 'peri-' + 'odic' split over a line (or page) break.

    The joined form wins when it occurs elsewhere in the document; the hyphenated
    form wins when only it occurs elsewhere (e.g. 'anti-money'); with no evidence
    the parts are joined, since line-break hyphenation is far more common in CFR.
    """
    vocab = Counter(w.lower() for line in lines for w in WORD.findall(line.text))
    out = [Line(line.text, line.pdf_page, line.page_label) for line in lines]
    for i in range(len(out) - 1):
        cur, nxt = out[i], out[i + 1]
        m = re.search(r"([A-Za-z]+)-$", cur.text)
        if not m or not nxt.text[:1].islower():
            continue
        head, _, rest = nxt.text.partition(" ")
        tail = re.match(r"[a-z]+", head).group(0)
        trailing = head[len(tail):]
        joined, hyphened = m.group(1) + tail, m.group(1) + "-" + tail
        word = hyphened if (vocab[hyphened.lower()] and not vocab[joined.lower()]) else joined
        cur.text = cur.text[: m.start()] + word + trailing
        nxt.text = rest
    return [line for line in out if line.text]


def _lines(pages: list[PageClean]) -> list[Line]:
    return [Line(t, p.pdf_page, p.page_label) for p in pages for t in p.lines]


def _is_heading(match, part: str, last_num: int) -> bool:
    """A real section heading: own part, numbers only increase, title starts with a capital or '['.

    This rejects cross-references that happen to start a line, such as
    '§ 1010.100 of this chapter' (lower case) or a reference back to an earlier section.
    """
    if not match or match.group(1) != part or int(match.group(2)) <= last_num:
        return False
    return match.group(3)[:1].isupper() or match.group(3).startswith("[")


def split_cfr_sections(doc_id: str, part: str, pages: list[PageClean]) -> tuple[list[Section], list[str]]:
    """Sections of one CFR part, plus the section numbers listed in the part's table of contents."""
    lines = dehyphenate(_lines(pages))
    sections, toc = [], []
    started, in_toc, subpart = False, False, ""
    current, last_num = None, -1
    i = 0
    while i < len(lines):
        text = lines[i].text
        part_m = CFR_PART.match(text)
        if part_m:
            if part_m.group(1) == part:
                started, in_toc = True, True
            elif started:
                break  # the next part begins: end of this document
            i += 1
            continue
        if not started:
            i += 1
            continue
        sub_m = CFR_SUBPART.match(text)
        if sub_m:
            title = sub_m.group(2).strip()
            # a subpart title can wrap over up to 3 more lines; it ends before "§", "Sec." or a TOC entry
            for _ in range(3):
                nxt = lines[i + 1].text if i + 1 < len(lines) else ""
                if not nxt or nxt.startswith(("§", "Sec.")) or CFR_TOC_ENTRY.match(nxt) \
                        or CFR_SUBPART.match(nxt) or CFR_PART.match(nxt):
                    break
                title = f"{title} {nxt}"
                i += 1
            subpart = f"Subpart {sub_m.group(1)}—{title}"
            i += 1
            continue
        if CFR_MULTI_SECTION.match(text):
            current = None  # "§§ x-y [Reserved]"
            i += 1
            continue
        sec_m = CFR_SECTION.match(text)
        if _is_heading(sec_m, part, last_num):
            in_toc = False
            last_num = int(sec_m.group(2))
            title = sec_m.group(3).strip()
            while not title.endswith((".", "]")) and i + 1 < len(lines) and len(title) < 200:
                i += 1
                title = f"{title} {lines[i].text}".strip()
            if title.startswith("[Reserved]"):
                current = None
            else:
                current = Section(doc_id, f"{part}.{sec_m.group(2)}", title.rstrip("."), subpart, [])
                sections.append(current)
            i += 1
            continue
        if in_toc:
            toc_m = CFR_TOC_ENTRY.match(text)
            if toc_m and toc_m.group(1) == part and "[Reserved]" not in text:
                toc.append(f"{part}.{toc_m.group(2)}")
        elif current is not None:
            current.lines.append(lines[i])
        i += 1
    return sections, toc


def split_faq_sections(doc_id: str, pages: list[PageClean]) -> list[Section]:
    lines = dehyphenate(_lines(pages))
    current = Section(doc_id, "intro", "Introduction", "", [])
    sections = [current]
    for line in lines:
        m = FAQ_QUESTION.match(line.text)
        if m:
            current = Section(doc_id, f"Q{m.group(1)}", f"Question {m.group(1)}: {m.group(2).strip()}", "", [])
            sections.append(current)
        else:
            current.lines.append(line)
    return [s for s in sections if s.lines]


def parse_document(doc_id: str, page_texts: list[str]) -> tuple[list[Section], list[PageClean], list[str]]:
    """(sections, cleaned pages, TOC entries) for one document of DOC_KINDS."""
    kind, part = DOC_KINDS[doc_id]
    if kind == "cfr":
        pages = [clean_cfr_page(t, n) for n, t in enumerate(page_texts, start=1)]
        sections, toc = split_cfr_sections(doc_id, part, pages)
        return sections, pages, toc
    if kind == "faq":
        pages = [clean_faq_page(t, n) for n, t in enumerate(page_texts, start=1)]
        return split_faq_sections(doc_id, pages), pages, []
    raise ValueError(f"unknown kind {kind!r} for {doc_id}")
