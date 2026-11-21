"""Citation validator and mechanical claim check (Phase 7, design §5 and option B).

The model sees the retrieved excerpts numbered [1]..[k] and may cite only those
numbers ("[2]" or "[1, 3]"). Code maps each number back to its chunk id (round 3,
2026-11-22; before, the model copied chunk-id tags and changed them). An answer
passes when:
  1. every citation is well-formed and every number is within 1..k (so it maps to a
     chunk that exists in the index and was retrieved for this question);
  2. every regulation section number in a sentence ("§ 1010.311", "31 CFR 1020.320(b)")
     occurs in the heading or text of a chunk cited in that same sentence;
  3. every sentence carries at least one citation.
Pages and section names are never taken from the model: they are rendered from
chunk metadata (chunk.citation).
"""

import re
from dataclasses import dataclass, field

NUM_CITE = re.compile(r"\[(\d{1,3}(?:\s*,\s*\d{1,3})*)\]")
# anything bracketed that looks like an attempt to cite: digits, "C:", "excerpt", ranges like [1-3]
CITE_LIKE = re.compile(r"\[\s*(?:C\s*:|excerpt|source|\d)[^\]]*\]", re.I)
SECTION_REF = re.compile(r"(?<![\d.])(10\d\d\.\d{2,4})(?![\d])")
ABBREVIATIONS = ("U.S.C.", "C.F.R.", "U.S.", "e.g.", "i.e.", "No.", "Sec.", "Inc.", "Co.", "vs.", "etc.")
LEADER = "․"   # one-dot leader: stands in for "." inside abbreviations while splitting
# option B: numbers, dollar amounts, day counts and section references must be in a cited chunk
DOLLARS = re.compile(r"\$\s?\d[\d,]*(?:\.\d+)?")
NUMBER = re.compile(r"(?<![\w.$])\d[\d,]*(?:\.\d+)?(?![\w])")


def contains_token(text: str, token: str) -> bool:
    """`token` occurs in `text` as a whole number / amount / section ("30" does not match "300")."""
    return re.search(rf"(?<![\d,.]){re.escape(token)}(?![\d]|,\d)", text) is not None


@dataclass
class Validation:
    ok: bool
    errors: list[str] = field(default_factory=list)
    cited: list[str] = field(default_factory=list)       # distinct chunk ids, in order of first use
    sentences: list[tuple[str, list[str]]] = field(default_factory=list)   # (sentence, chunk ids)


def _protect(text: str) -> str:
    for abbr in ABBREVIATIONS:
        text = text.replace(abbr, abbr.replace(".", LEADER))
    return text


def split_sentences(answer: str) -> list[str]:
    """Sentences, with citations that follow the full stop moved before it ("x. [1]" -> "x [1].")."""
    text = _protect(answer.strip())
    text = re.sub(r"([.!?])\s*((?:\[[^\]]*\]\s*)+)", lambda m: f" {m.group(2).strip()}{m.group(1)} ", text)
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    out = []
    for part in parts:
        part = part.strip().replace(LEADER, ".")
        if len(re.findall(r"[A-Za-z]{2,}", CITE_LIKE.sub("", part))) >= 2:   # skip stray tags / fragments
            out.append(part)
    return out


def numbers_in(sentence: str) -> list[int]:
    return [int(n) for m in NUM_CITE.finditer(sentence) for n in re.split(r"\s*,\s*", m.group(1))]


def validate(answer: str, excerpt_ids: list[str], chunk_text: dict[str, str]) -> Validation:
    """`excerpt_ids[n-1]` is the chunk shown to the model as [n]; `chunk_text` maps id -> heading + text."""
    k = len(excerpt_ids)
    errors, cited, sentences = [], [], []
    for raw in CITE_LIKE.findall(answer):
        if not NUM_CITE.fullmatch(raw):
            errors.append(f"malformed citation {raw}; cite excerpt numbers like [1] or [1, 2]")
            continue
        for n in numbers_in(raw):
            if not 1 <= n <= k:
                errors.append(f"citation [{n}] is not one of the excerpts [1]..[{k}]")
            elif excerpt_ids[n - 1] not in cited:
                cited.append(excerpt_ids[n - 1])
    for sentence in split_sentences(answer):
        ids = [excerpt_ids[n - 1] for n in numbers_in(sentence) if 1 <= n <= k]
        sentences.append((sentence, ids))
        if not numbers_in(sentence):
            errors.append(f"sentence without citation: {sentence[:80]!r}")
            continue
        texts = " ".join(chunk_text.get(i, "") for i in ids)
        for ref in SECTION_REF.findall(NUM_CITE.sub("", sentence)):
            if not contains_token(texts, ref):
                errors.append(f"section {ref} is not in the excerpts cited in: {sentence[:80]!r}")
    if not sentences:
        errors.append("no sentence")
    return Validation(not errors, errors, cited, sentences)


def claim_tokens(sentence: str) -> list[str]:
    """Numbers, dollar amounts, day counts and section references in a sentence (citations removed)."""
    text = NUM_CITE.sub(" ", sentence)
    tokens = [d.replace(" ", "") for d in DOLLARS.findall(text)]
    text = DOLLARS.sub(" ", text)
    tokens += SECTION_REF.findall(text)
    text = SECTION_REF.sub(" ", text)
    tokens += NUMBER.findall(text)
    return tokens


def mechanical_claim_check(sentences: list[tuple[str, list[str]]], chunk_text: dict[str, str]) -> list[dict]:
    """Option B: for each sentence with claim tokens, whether every token occurs in a cited chunk."""
    results = []
    for sentence, ids in sentences:
        tokens = claim_tokens(sentence)
        if not tokens:
            continue
        texts = " ".join(chunk_text.get(i, "") for i in ids)
        missing = [t for t in tokens if not contains_token(texts, t)]
        results.append({"sentence": sentence, "tokens": tokens, "missing": missing, "ok": not missing})
    return results
