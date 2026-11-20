"""Citation validator and mechanical claim check (Phase 7, design §5 and option B).

The model may only cite chunks as [C:<chunk_id>]. An answer passes when:
  1. every tag is well-formed, exists in the index AND was retrieved for this question;
  2. every regulation section number in a sentence ("§ 1010.311", "31 CFR 1020.320(b)")
     occurs in the heading or text of a chunk cited in that same sentence;
  3. every sentence carries at least one citation.
Pages and section names are never taken from the model: they are rendered from
chunk metadata (chunk.citation).
"""

import re
from dataclasses import dataclass, field

TAG = re.compile(r"\[C:([a-z0-9_]+:[A-Za-z0-9.\-]+:\d{3})\]")
ANY_TAG = re.compile(r"\[\s*C\s*:[^\]]*\]")
SECTION_REF = re.compile(r"(?<![\d.])(10\d\d\.\d{2,4})(?![\d])")
ABBREVIATIONS = ("U.S.C.", "C.F.R.", "U.S.", "e.g.", "i.e.", "No.", "Sec.", "Inc.", "Co.", "vs.", "etc.")
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
    sentences: list[tuple[str, list[str]]] = field(default_factory=list)


def _protect(text: str) -> str:
    for abbr in ABBREVIATIONS:
        text = text.replace(abbr, abbr.replace(".", "․"))   # one-dot leader: not a sentence end
    return text


def split_sentences(answer: str) -> list[str]:
    """Sentences, with citations that follow the full stop moved before it ("x. [C:a]" -> "x [C:a].")."""
    text = _protect(answer.strip())
    text = re.sub(r"([.!?])\s*((?:\[C:[^\]]*\]\s*)+)", lambda m: f" {m.group(2).strip()}{m.group(1)} ", text)
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    out = []
    for part in parts:
        part = part.strip().replace("․", ".")
        if len(re.findall(r"[A-Za-z]{2,}", ANY_TAG.sub("", part))) >= 2:   # skip stray tags / fragments
            out.append(part)
    return out


def validate(answer: str, retrieved: set[str], index_ids: set[str], chunk_text: dict[str, str]) -> Validation:
    """`chunk_text` maps chunk_id -> heading + text (embed_text), used for check 2."""
    errors, cited, sentences = [], [], []
    for raw in ANY_TAG.findall(answer):
        m = TAG.fullmatch(raw)
        if not m:
            errors.append(f"malformed citation tag {raw}")
            continue
        cid = m.group(1)
        if cid not in index_ids:
            errors.append(f"citation {raw} does not exist")
        elif cid not in retrieved:
            errors.append(f"citation {raw} was not among the excerpts given for this question")
        elif cid not in cited:
            cited.append(cid)
    for sentence in split_sentences(answer):
        ids = [m.group(1) for m in TAG.finditer(sentence)]
        sentences.append((sentence, ids))
        if not ids:
            errors.append(f"sentence without citation: {sentence[:80]!r}")
            continue
        texts = " ".join(chunk_text.get(i, "") for i in ids)
        for ref in SECTION_REF.findall(TAG.sub("", sentence)):
            if not contains_token(texts, ref):
                errors.append(f"section {ref} is not in the excerpts cited in: {sentence[:80]!r}")
    if not sentences:
        errors.append("no sentence")
    return Validation(not errors, errors, cited, sentences)


def claim_tokens(sentence: str) -> list[str]:
    """Numbers, dollar amounts, day counts and section references in a sentence (tags removed)."""
    text = TAG.sub("", sentence)
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
