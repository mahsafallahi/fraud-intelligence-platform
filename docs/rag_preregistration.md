# Phase 7 RAG: pre-registration (frozen before the test run)

Written by `scripts/rag_freeze.py` from the chosen dev run. It is committed and pushed before
the test run (decision C7); the push gives a public timestamp. The machine-readable copy is
`data/rag/eval/preregistration.json`. **If any value below changes before the test run, the dev
evaluation is re-run and this file is regenerated.** Details and reasons: `docs/rag_design.md`.

## Frozen configuration

| Item | Value |
|---|---|
| Chat model | `llama3.1:latest` |
| Chat model digest | `46e0c10c039e019119339687c3c1757cc81b9da49709a3b3924863ba87ca666e` |
| Chat options | `{"temperature": 0, "seed": 42, "num_ctx": 8192, "num_predict": 400}` |
| Embedding model | `nomic-embed-text` (task prefixes `search_document: ` / `search_query: `) |
| Embedding digest | `0a109f422b47e3a30ba2b10eca18548e944e8a23073ee3f3e947efcf3c45e59f` |
| Ollama version | `0.40.17` |
| Retrieval | `{"mode": "dense", "glossary": false, "paragraphs": false, "k": 6, "rrf_k": 60}` (dense only, chosen on dev) |
| tau (refusal threshold) | `None`: disabled; dev did not separate. Refusal = NO_ANSWER + validator |
| Index manifest SHA-256 | `722cfa220aa6c9f4ad6aa98fcbd6afea134ccf56a1f626ce73677fcf6b4e1467` |
| questions.jsonl SHA-256 | `fd5abebf730d9b5f2700607a63e4b5402d27e516fc9b955d46f6bcc2f9e96a0f` (CRLF normalised to LF) |
| Prompt `answer_system.txt` SHA-256 | `b728bbe4e45c59b93575fe5f23bf7de67e244ac0ee6ca573a41924f673223049` |
| Prompt `answer_user.txt` SHA-256 | `a551854bf2f0c862d2082977f991a59b05a373a18f7b446f7cf16c0829d95bd4` |
| Prompt `answer_retry.txt` SHA-256 | `bc981d1475780994eabfc0842f82f81af6bb911a156db77b994cca6ecee8fb0b` |
| Agent prompt `writer_system.txt` SHA-256 | `d5dad52f624b4d51d426d14005e22971b83efb0d05090e0f18f9056b2d2d2828` |
| Agent prompt `agent_system.txt` SHA-256 | `9cbfb8a6822ddfbfc016bbf020f1c844e8953bba583ded4a7f9a93dc7327e86e` |

Dev run: `data/rag/runs/eval_dev_20261122T210528Z.json`.

## Model choice (rule C4, dev set)

| Dev metric | llama3.1:latest | qwen2.5:7b-instruct |
|---|---|---|
| recall_at_6 | 7/8 | 7/8 |
| answered_answerable | 8/8 | 7/8 |
| false_refusals | 0/8 | 1/8 |
| correct_refusals | 4/4 | 4/4 |
| citation_correctness | 7/8 | 7/7 |
| raw_citation_validity | 33/33 = 1.000 | 15/15 = 1.000 |
| mechanical_check | 9/10 = 0.900 | 6/7 = 0.857 |
| C4 passed | False | False |

Chosen: `llama3.1:latest`. Reason: both models fail C4 on dev and llama3.1 has the higher mechanical-check rate.

## Known limitation: rule C4 is not met on dev

Both models fail C4 on dev. The chosen model `llama3.1:latest` fails `mechanical_check_ok`; its mechanical check is 9/10 against the required 0.95.
The other model `qwen2.5:7b-instruct`: 6/7. No threshold is
changed because of this: the test run is judged by the thresholds below, and a fail is reported as a fail.
Failing dev sentences of the chosen model:

- `D03`: {"sentence": "The US travel rule requires financial institutions to report funds transfers of $3,000 or more to or from a person outside the United States, unless the transfer is made on the books of a domestic financial institution involving the account of a customer whose address is within approximately 50 miles of the location of the institution, or who is known to be temporarily outside the United States [3].", "tokens": ["q:ge:$3,000", "50"], "missing": ["q:ge:$3,000"], "ok": false}

## Human claim-labelling sample (option B)

Order drawn with `random.Random(20261121).shuffle` over the sorted ids of the 22 answerable test
questions. The sample is the first 10 that the system answers; a refused one is replaced by the next.

First 10: `T05`, `T02`, `T07`, `T15`, `T08`, `T20`, `T04`, `T22`, `T10`, `T13`.
Replacements in order: `T19`, `T21`, `T14`, `T16`, `T01`, `T18`, `T11`, `T03`, `T12`, `T09`, `T06`, `T17`.

## Pass thresholds (test set: 22 answerable + 8 not answerable)

| Metric | Definition | Pass |
|---|---|---|
| Retrieval recall@6 | answerable test questions with at least one expected section in the top 6 | **>= 19/22** |
| Citation validity, raw | cited excerpt numbers within 1..k, before the validator | **>= 0.95** |
| Citation validity, final | after the validator (true by construction) | **= 1.00** |
| Citation correctness | answered answerable questions with at least one citation in an expected section | **>= 0.80** |
| Claim support, human | sentences of the 10-question human sample labelled supported / unsupported / contradicted by their cited chunks | **>= 0.90 supported and 0 contradicted** |
| Claim support, mechanical | on all answered test questions: sentences with a number, number word, dollar amount (with qualifier), day count or section reference whose tokens all occur in a cited chunk | **>= 0.95** |
| Correct refusal | unanswerable test questions refused | **>= 7/8** |
| False refusal | answerable test questions refused | **<= 2/22** |
| Agent, mechanical | 10 fixed alert cases: schema, status, reasons, citations, no PII to the LLM, limits | **10/10** |

Reporting: counts with Wilson 95% intervals. The test set is run once with the frozen configuration.
There are no prompt or configuration changes after the test run, and a failed threshold is reported as a fail.
