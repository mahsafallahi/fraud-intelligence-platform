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


# round 4 (2026-11-23): number words and amount qualifiers are claims too
NUMBER_WORDS = {
    "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90, "hundred": 100, "thousand": 1000, "million": 1000000,
}   # "one" is left out: it is mostly a pronoun ("one or more", "no one")
NUMBER_WORD = re.compile(r"\b(" + "|".join(NUMBER_WORDS) + r")\b", re.I)
QUALIFIER_CLASS = {
    "more than": "gt", "over": "gt", "in excess of": "gt", "exceeding": "gt", "above": "gt", "greater than": "gt",
    "at least": "ge", "not less than": "ge", "or more": "ge", "or greater": "ge",
    "less than": "lt", "under": "lt", "below": "lt",
    "up to": "le", "no more than": "le", "not more than": "le", "not exceeding": "le", "or less": "le",
}
_PRE = "|".join(sorted((q for q in QUALIFIER_CLASS if not q.startswith("or ")), key=len, reverse=True))
_POST = "|".join(q for q in QUALIFIER_CLASS if q.startswith("or "))
QUALIFIED_AMOUNT = re.compile(rf"(?:\b({_PRE})\s+)?(\$\s?\d[\d,]*(?:\.\d+)?)(?:\s+({_POST})\b)?", re.I)


def qualified_amounts(text: str) -> list[tuple[str, str]]:
    """[(amount, class)] for every dollar amount; class is gt/ge/lt/le, or "bare" with no qualifier."""
    out = []
    for m in QUALIFIED_AMOUNT.finditer(text):
        q = (m.group(1) or m.group(3) or "").lower()
        out.append((m.group(2).replace(" ", ""), QUALIFIER_CLASS.get(q, "bare")))
    return out


def claim_tokens(sentence: str) -> list[str]:
    """Claims to check in a sentence (citations removed): dollar amounts with their qualifier class
    ("q:gt:$10,000", "q:bare:$10,000"), section references, digits, and number words ("w:five")."""
    text = NUM_CITE.sub(" ", sentence)
    tokens = [f"q:{cls}:{amt}" for amt, cls in qualified_amounts(text)]
    text = DOLLARS.sub(" ", text)
    tokens += SECTION_REF.findall(text)
    text = SECTION_REF.sub(" ", text)
    tokens += NUMBER.findall(text)
    tokens += [f"w:{w.lower()}" for w in NUMBER_WORD.findall(text)]
    return tokens


def token_supported(token: str, text: str) -> bool:
    """Whether a claim token is backed by `text` (the cited chunks)."""
    if token.startswith("q:"):
        _, cls, amount = token.split(":", 2)
        found = [c for a, c in qualified_amounts(text) if a == amount]
        if not found:
            return False
        if cls == "bare":     # a bare amount is wrong when the source always qualifies it
            return "bare" in found
        return cls in found
    if token.startswith("w:"):
        word = token[2:]
        return (re.search(rf"\b{word}\b", text, re.I) is not None
                or contains_token(text, str(NUMBER_WORDS[word])))
    return contains_token(text, token)


def mechanical_claim_check(sentences: list[tuple[str, list[str]]], chunk_text: dict[str, str]) -> list[dict]:
    """Option B: for each sentence with claim tokens, whether every token is backed by a cited chunk."""
    results = []
    for sentence, ids in sentences:
        tokens = claim_tokens(sentence)
        if not tokens:
            continue
        texts = " ".join(chunk_text.get(i, "") for i in ids)
        missing = [t for t in tokens if not token_supported(t, texts)]
        results.append({"sentence": sentence, "tokens": tokens, "missing": missing, "ok": not missing})
    return results
