"""Answer a question from the retrieved excerpts, with validated citations or a fixed refusal (Phase 7).

Flow: retrieve top 6 -> (refuse if tau is set and max_cosine < tau, decision C1;
tau stays None until it is set on the dev set) -> prompt the model with the
excerpts as delimited data -> "NO_ANSWER" means refuse -> validate citations ->
one retry with the validator's errors -> otherwise the fixed refusal.
"""

import hashlib
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from src.rag.chunk import citation
from src.rag.citations import TAG, Validation, validate

PROMPT_DIR = Path(__file__).resolve().parent / "prompts"
PROMPT_FILES = ("answer_system.txt", "answer_user.txt", "answer_retry.txt")
NO_ANSWER = "NO_ANSWER"
REFUSAL = ("I can't answer this from the documents I have (31 CFR Parts 1010 and 1020, the FinCEN SAR "
           "FAQs of October 2025 and six FFIEC BSA/AML Examination Manual sections).")
CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def load_prompt(name: str) -> str:
    return (PROMPT_DIR / name).read_text(encoding="utf-8").replace("\r\n", "\n")


def prompt_hashes() -> dict[str, str]:
    """SHA-256 of each prompt file with line endings normalised to LF (pre-registration C7)."""
    return {name: hashlib.sha256(load_prompt(name).encode("utf-8")).hexdigest() for name in PROMPT_FILES}


def sanitize(text: str) -> str:
    """Excerpt text is data: drop control characters and anything that could close the excerpt tag."""
    text = CONTROL.sub(" ", text)
    return re.sub(r"</?\s*excerpt[^>]*>", "[removed tag]", text, flags=re.I)


def format_excerpts(chunk_ids: list[str], records: dict[str, dict]) -> str:
    blocks = []
    for cid in chunk_ids:
        rec = records[cid]
        blocks.append(f'<excerpt tag="[C:{cid}]" source="{sanitize(rec["heading"])}">\n'
                      f"{sanitize(rec['text'])}\n</excerpt>")
    return "\n\n".join(blocks)


@dataclass
class Answer:
    question: str
    status: str                     # "answered" | "refused"
    reason: str                     # "" | "low_similarity" | "model_no_answer" | "validation_failed"
    text: str                       # the model's answer with [C:...] tags, or the fixed refusal
    rendered: str                   # tags replaced by [1], [2], ... plus a source list
    cited: list[str] = field(default_factory=list)
    retrieved: list[str] = field(default_factory=list)
    max_cosine: float = 0.0
    validations: list[Validation] = field(default_factory=list)
    attempts: int = 0
    seconds: float = 0.0
    llm_seconds: list[float] = field(default_factory=list)


def render(text: str, cited: list[str], records: dict[str, dict]) -> str:
    numbers = {cid: n for n, cid in enumerate(cited, start=1)}
    body = TAG.sub(lambda m: f"[{numbers[m.group(1)]}]" if m.group(1) in numbers else m.group(0), text)
    sources = "\n".join(f"[{n}] {citation(records[cid])}" for cid, n in numbers.items())
    return f"{body}\n\nSources:\n{sources}" if sources else body


class Answerer:
    def __init__(self, retriever, llm, records: dict[str, dict], tau: float | None = None, k: int = 6):
        self.retriever, self.llm, self.records, self.tau, self.k = retriever, llm, records, tau, k
        self.index_ids = set(records)
        self.chunk_text = {cid: f"{r['heading']}\n{r['text']}" for cid, r in records.items()}
        self.system, self.user_tpl, self.retry_tpl = (load_prompt(n) for n in PROMPT_FILES)

    def _refuse(self, question, reason, start, **kw) -> Answer:
        return Answer(question, "refused", reason, REFUSAL, REFUSAL, seconds=time.perf_counter() - start, **kw)

    def answer(self, question: str) -> Answer:
        start = time.perf_counter()
        result = self.retriever.search(question, k=self.k)
        ids = [h.chunk_id for h in result.hits]
        base = {"retrieved": ids, "max_cosine": result.max_cosine}
        if self.tau is not None and result.max_cosine < self.tau:
            return self._refuse(question, "low_similarity", start, **base)

        messages = [{"role": "system", "content": self.system},
                    {"role": "user", "content": self.user_tpl.format(
                        question=question, excerpts=format_excerpts(ids, self.records))}]
        validations, llm_seconds = [], []
        for attempt in (1, 2):
            t0 = time.perf_counter()
            reply = self.llm.chat(messages)
            llm_seconds.append(time.perf_counter() - t0)
            content = reply.content.strip()
            if content.strip(" .\"'`").upper() == NO_ANSWER:
                return self._refuse(question, "model_no_answer", start, validations=validations,
                                    attempts=attempt, llm_seconds=llm_seconds, **base)
            check = validate(content, set(ids), self.index_ids, self.chunk_text)
            validations.append(check)
            if check.ok:
                return Answer(question, "answered", "", content, render(content, check.cited, self.records),
                              check.cited, validations=validations, attempts=attempt,
                              seconds=time.perf_counter() - start, llm_seconds=llm_seconds, **base)
            messages += [{"role": "assistant", "content": content},
                         {"role": "user", "content": self.retry_tpl.format(
                             errors="\n".join(f"- {e}" for e in check.errors))}]
        return self._refuse(question, "validation_failed", start, validations=validations, attempts=2,
                            llm_seconds=llm_seconds, **base)
