# Phase 7 RAG: test results (one-time run)

The 30 test questions were run **once** with the frozen configuration of `docs/rag_preregistration.md`, started 2026-11-24T20:14:33+00:00 at commit `ddd10a4`. Every fingerprint field was checked
against the freeze before the first question. Nothing was changed after the run. Raw record:
`data/rag/runs/test_run.jsonl` (git-ignored). Written by `scripts/rag_test_report.py`.

Model `llama3.1:latest` (`46e0c10c039e`), embeddings `nomic-embed-text` (`0a109f422b47`), Ollama 0.40.17, retrieval `{"mode": "dense", "glossary": false, "paragraphs": false, "k": 6, "rrf_k": 60}`, tau `None`.

## Pre-registered metrics

| Metric | Result | Rate | Wilson 95% interval | Threshold | Verdict |
|---|---|---|---|---|---|
| Retrieval recall@6 | 19/22 | 0.864 | 0.667 to 0.953 | >= 19/22 | **pass** |
| Citation validity, raw | 75/75 | 1.000 | 0.951 to 1.000 | >= 0.95 | **pass** |
| Citation validity, final | 72/72 | 1.000 | 0.949 to 1.000 | = 1.00 | **pass** |
| Citation correctness | 16/21 | 0.762 | 0.549 to 0.894 | >= 0.80 | **FAIL** |
| Claim support, mechanical | 13/16 | 0.812 | 0.570 to 0.934 | >= 0.95 | **FAIL** |
| Correct refusal | 7/8 | 0.875 | 0.529 to 0.978 | >= 7/8 | **pass** |
| False refusal (lower is better) | 1/22 | 0.045 | 0.008 to 0.218 | <= 2/22 | **pass** |
| Claim support, human | pending | | | >= 0.90 supported and 0 contradicted | pending the user's labels |
| Agent, mechanical: baseline (recommended mode) | 10/10 | 1.000 | 0.722 to 1.000 | 10/10 | **pass** |
| Agent, mechanical: tool-calling agent | 6/10 | 0.600 | 0.313 to 0.832 | 10/10 | **FAIL** |

Failed in this run: Citation correctness, Claim support, mechanical (the tool-calling agent failed before it, see `docs/rag_design.md` §6e).
The verdict compares the observed rate with the threshold. The intervals are wide because the set is small.

## Human claim labels (option B)

Sample: T02, T07, T15, T08, T20, T04, T22, T10, T13, T19. Refused and replaced: T05. 26 sentences to label in `data/rag/eval/claim_labels.md`.

## Per question

| Id | Answerable | Retrieved expected section | Status | Reason | Attempts | Cited expected section | Mechanical | Seconds |
|---|---|---|---|---|---|---|---|---|
| T01 | yes | yes | answered | - | 1 | yes | 0/0 | 2.42 |
| T02 | yes | yes | answered | - | 1 | yes | 0/0 | 2.19 |
| T03 | yes | yes | answered | - | 1 | yes | 1/1 | 1.88 |
| T04 | yes | yes | answered | - | 1 | yes | 0/1 | 2.89 |
| T05 | yes | **no** | **refused** | model_no_answer | 2 | - | - | 6.65 |
| T06 | yes | yes | answered | - | 1 | yes | 0/0 | 3.33 |
| T07 | yes | yes | answered | - | 1 | yes | 0/0 | 3.05 |
| T08 | yes | yes | answered | - | 1 | yes | 2/2 | 2.33 |
| T09 | yes | yes | answered | - | 1 | yes | 0/0 | 4.13 |
| T10 | yes | yes | answered | - | 1 | **no** | 1/2 | 2.21 |
| T11 | yes | **no** | answered | - | 1 | **no** | 0/0 | 3.47 |
| T12 | yes | yes | answered | - | 1 | yes | 1/1 | 2.95 |
| T13 | yes | yes | answered | - | 1 | yes | 1/1 | 2.56 |
| T14 | yes | yes | answered | - | 1 | yes | 0/0 | 2.83 |
| T15 | yes | yes | answered | - | 1 | yes | 0/0 | 2.75 |
| T16 | yes | yes | answered | - | 1 | **no** | 0/0 | 2.72 |
| T17 | yes | yes | answered | - | 1 | yes | 0/0 | 2.54 |
| T18 | yes | yes | answered | - | 1 | yes | 0/1 | 2.61 |
| T19 | yes | **no** | answered | - | 1 | **no** | 3/3 | 3.72 |
| T20 | yes | yes | answered | - | 1 | yes | 0/0 | 2.99 |
| T21 | yes | yes | answered | - | 1 | yes | 1/1 | 3.24 |
| T22 | yes | yes | answered | - | 1 | **no** | 2/2 | 3.55 |
| T23 | no | - | refused | model_no_answer | 1 | - | - | 2.0 |
| T24 | no | - | refused | model_no_answer | 1 | - | - | 2.36 |
| T25 | no | - | refused | model_no_answer | 1 | - | - | 1.8 |
| T26 | no | - | **answered** | - | 1 | - | 1/1 | 1.52 |
| T27 | no | - | refused | model_no_answer | 2 | - | - | 1.62 |
| T28 | no | - | refused | model_no_answer | 1 | - | - | 1.27 |
| T29 | no | - | refused | model_no_answer | 2 | - | - | 1.2 |
| T30 | no | - | refused | model_no_answer | 1 | - | - | 0.6 |

Bold marks a wrong outcome: a refused answerable question, an answered unanswerable one, or a missed section.

## Sentences that fail the mechanical check

- `T04`: The bank must look at the three cash deposits together for reporting purposes because they are multiple currency transactions that result in cash in totaling more than $10,000 during any one business day [1, 2]. Missing from the cited text: `w:three`.
- `T10`: A notice to share information under section 314(b) is effective for 5 years after it is last relied upon in connection with a request for information [1]. Missing from the cited text: `314`.
- `T18`: The two steps of a BSA/AML risk assessment are identifying specific risk categories and analyzing the information obtained from these categories to assess ML/TF and other illicit financial activity risks [2, 3]. Missing from the cited text: `w:two`.

## Unanswerable questions that were answered

- `T26` How many days does a cardholder have to raise a Visa or Mastercard chargeback? → A cardholder has 60 days to raise a Visa or Mastercard chargeback [3].

## Known limitation

Both models failed rule C4 on dev, and llama3.1 was kept by the pre-registered rule (dev mechanical check 9/10 = 0.900; see `docs/rag_preregistration.md`).
