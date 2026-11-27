# Claim labels: second rater (post hoc, not pre-registered)

Rater 2 is an LLM: the same chat assistant that translated the sentences and excerpts into Persian for
the human annotator. It labelled the same 26 sentences of `claim_labels.md`, AFTER the human audit, under
the same rule: everything the sentence claims must be in its cited excerpt, and a sentence that is only
partly backed is `not supported`. **Rater 2 was not blind:** it had seen the human labels during the
labelling session. The official, pre-registered verdict stays the human annotator's labels in
`claim_labels.md`; these labels are only used to measure agreement (`scripts/rag_claim_audit.py`).

| # | Rater-2 label | Rater-2 reason |
|---|---|---|
| T02-1 | supported | |
| T02-2 | not supported | Excerpt says "develop" policies indicating when to escalate; "review" is not in the excerpt. |
| T07-1 | supported | |
| T07-2 | not supported | Second half (TIN for U.S. persons, several other ID types for non-U.S. persons) is in neither excerpt; [6] is cut off at "(i) For a U.S.", [3] only mentions the applied-for-TIN exception. |
| T15-1 | supported | |
| T15-2 | not supported | Excerpt [2] is 31 CFR 1010.955 on beneficial ownership information and defines foreign national security activity; it does not mention NSLs, counterintelligence or counterterrorism. |
| T15-3 | supported | Borderline: excerpt says questions "should be directed to" the local FBI field office; "responsible for handling" is a paraphrase. |
| T08-1 | supported | |
| T08-2 | supported | |
| T20-1 | supported | |
| T20-2 | supported | |
| T20-3 | supported | |
| T04-1 | supported | Borderline: omits the knowledge condition, which T04-2 addresses; the amounts come from the question. |
| T04-2 | not supported | The excerpts state knowledge as a condition ("if the financial institution has knowledge"); the sentence asserts the bank has knowledge, which neither the excerpts nor the question say. |
| T04-3 | supported | |
| T22-1 | not supported | Excerpt is about currency received in a trade or business (Form 8300), not carrying cash across the border; the first condition ("if the entire transaction occurs in the United States") is an inversion of the excerpt's exception, not stated by it. |
| T22-2 | supported | Borderline: drops "generally" and "entire transaction". |
| T22-3 | supported | |
| T10-1 | not supported | Excerpt is about retaining documentation of a reporting company's consent under 1010.955; it does not mention section 314(b) notices or their validity. |
| T10-2 | supported | |
| T13-1 | supported | |
| T13-2 | supported | |
| T19-1 | supported | Borderline: "obtained to identify and verify" merges two record types; the excerpt keeps verification descriptions for five years after the record is made. |
| T19-2 | supported | |
| T19-3 | supported | |
| T19-4 | supported | |
