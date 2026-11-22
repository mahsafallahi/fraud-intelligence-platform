"""Generic AML query glossary (Phase 7, round-4 retrieval candidate "hybrid + glossary").

Written on 2026-11-23 BEFORE any selection run, and listed in docs/rag_design.md.
Only common AML jargon and abbreviations, mapped to the wording the regulations use.
It is not tuned to any question. Caveat: the same author wrote the questions (see the doc).
A trigger matches as a whole word or phrase, ignoring case; its expansion is appended to the query.
"""

import re

GLOSSARY = {
    "SAR": "suspicious activity report",
    "CTR": "currency transaction report, report of transactions in currency",
    "travel rule": "transmittal of funds, transmittal order, information included in a transmittal order",
    "wire": "funds transfer, transmittal of funds",
    "wire transfer": "funds transfer, transmittal of funds",
    "cash": "currency",
    "structuring": "structure transactions to evade reporting requirements",
    "KYC": "customer identification program, customer due diligence",
    "know your customer": "customer identification program, customer due diligence",
    "CIP": "customer identification program",
    "CDD": "customer due diligence",
    "beneficial owner": "legal entity customer, equity interests, control",
    "FBAR": "report of foreign bank and financial accounts",
    "CMIR": "report of international transportation of currency or monetary instruments",
    "Form 8300": "currency received in a trade or business",
    "314(a)": "information sharing between government agencies and financial institutions",
    "314(b)": "voluntary information sharing among financial institutions",
    "red flag": "potentially suspicious activity",
    "PEP": "senior foreign political figure",
    "politically exposed person": "senior foreign political figure",
}


def _pattern(term: str) -> re.Pattern:
    return re.compile(rf"(?<![\w(]){re.escape(term)}(?![\w)])", re.I)


_PATTERNS = [(_pattern(t), e) for t, e in GLOSSARY.items()]


def expand_query(query: str) -> str:
    """The query plus the expansion of every glossary term it contains (each expansion once)."""
    extra = []
    for pattern, expansion in _PATTERNS:
        if pattern.search(query) and expansion not in extra:
            extra.append(expansion)
    return f"{query} ({'; '.join(extra)})" if extra else query
