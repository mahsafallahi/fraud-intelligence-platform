"""Keyword grounding search over written analyst notes (Phase 7, docs/rag_design.md §6e).

    python scripts/agent_grounding_search.py data/agent/runs/<stamp>

For every ALERT note of both modes: which sentences use a transaction-detail term (night,
online, ...) that occurs in none of the chunks the sentence cites. Read-only, no model.
The cited chunks are found from the note's own source list; two chunks with the same
rendered citation are both accepted (lenient, reported).
"""

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.rag.chunk import citation  # noqa: E402
from src.rag.citations import numbers_in, split_sentences  # noqa: E402

PRIMARY = {"night": r"\bnight", "online": r"\bonline\b", "merchant": r"\bmerchants?\b"}
EXTENDED = {"large": r"\blarge\b", "high-value": r"\bhigh[- ]value\b", "high-risk": r"\bhigh[- ]risk\b",
            "hour": r"\bhours?\b", "holiday": r"\bholidays?\b", "distance": r"\bdistance\b",
            "location": r"\blocations?\b", "category": r"\bcategor(?:y|ies)\b", "in person": r"\bin[- ]person\b"}
SOURCE_LINE = re.compile(r"^\[(\d+)\] (.+)$", re.M)


def parse_note(md: str) -> tuple[str, dict[int, str]]:
    """(LLM-written text, {display number: rendered citation}) of one Markdown note."""
    body, _, sources = md.partition("\n## Sources\n")
    parts = re.findall(r"## (?:Regulatory context|Suggested next steps)\n(.*?)(?=\n## |\Z)", body, re.S)
    sources = sources.partition("\n---")[0]
    return "\n".join(p.strip() for p in parts), {int(n): c.strip() for n, c in SOURCE_LINE.findall(sources)}


def search_note(md: str, text_of: dict[str, list[str]], terms: dict[str, str]) -> list[tuple[str, str]]:
    """[(term, sentence)] for every term that none of the sentence's cited chunks contains."""
    text, sources = parse_note(md)
    out = []
    for sentence in split_sentences(text):
        cited = " ".join(t for n in numbers_in(sentence) for t in text_of.get(sources.get(n, ""), []))
        for term, pattern in terms.items():
            if re.search(pattern, sentence, re.I) and not re.search(pattern, cited, re.I):
                out.append((term, sentence))
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args(argv)
    from src.rag.index import INDEX_DIR, load_index
    frame, _, _ = load_index(INDEX_DIR)
    text_of: dict[str, list[str]] = {}
    for r in frame.to_dict("records"):
        text_of.setdefault(citation(r), []).append(f"{r['heading']}\n{r['text']}")
    for mode in ("baseline", "agent"):
        notes = {f.stem.split("_")[0]: f.read_text(encoding="utf-8") for f in sorted(args.run_dir.glob(f"A*_{mode}.md"))}
        written = {k: v for k, v in notes.items() if parse_note(v)[0]}
        ambiguous = sum(len(text_of.get(c, [])) > 1 for v in written.values() for c in parse_note(v)[1].values())
        unmatched = sum(c not in text_of for v in written.values() for c in parse_note(v)[1].values())
        n_sent = sum(len(split_sentences(parse_note(v)[0])) for v in written.values())
        print(f"\n{mode}: {len(notes)} alert notes, {len(written)} with regulatory text, {n_sent} sentences "
              f"({ambiguous} source lines match more than one chunk, {unmatched} match none)")
        for name, terms in (("primary", PRIMARY), ("extended", EXTENDED)):
            hits = {k: h for k, v in written.items() if (h := search_note(v, text_of, terms))}
            flagged = len({(k, s) for k, h in hits.items() for _, s in h})
            print(f"  {name}: {len(hits)}/{len(written)} notes flagged, {flagged}/{n_sent} sentences")
            for k, h in hits.items():
                for term, sentence in h:
                    print(f"    {k} [{term}] {sentence[:170]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
