# Phase 7 – RAG assistant and alert-explanation agent: design

Status: design approved 2026-11-20 (decisions D1–D9, corrections C1–C7). Round 1
(corpus, parsing, chunking, retrieval; **no LLM answering yet**) is built.
The thresholds in §8 are pre-registered: they may not change after any test-set run.

## 1. Goal

1. **RAG assistant**: answers questions about US AML / fraud regulation with
   citations to a document, section and page, and refuses when the corpus does
   not contain the answer. Evaluated for faithfulness on a hand-labelled set.
2. **Alert agent**: for one transaction, calls our Phase 6 API (`POST /score`) as a
   tool, takes the top SHAP reasons, retrieves the relevant regulation and writes a
   short analyst note with citations.

Constraints: free, no bank card, works from Iran, laptop (Windows 11, 32 GB RAM),
Python 3.14.6 in `.venv`, **no LLM calls in CI**.

## 2. Decisions (summary)

| # | Decision | Rejected alternatives (reason) |
|---|---|---|
| Corpus | US public-domain only: 31 CFR Parts 1010 + 1020 (GovInfo annual edition, July 1, 2025), FinCEN SAR FAQs (Oct. 2025), six FFIEC BSA/AML Manual sections (manual download, later) | FATF Recommendations (terms page showed a bot check from this network, so its terms **could not be verified**; out of scope); eCFR XML as primary (no page numbers, changes daily); FinCEN SAR XML User Guide (filing schema, not regulation); 2003 SAR narrative guidance (old URL 404, no current URL found) |
| Storage of documents | Not committed. `data/rag/corpus_manifest.json` (URL, SHA-256, bytes, pages) is committed; `scripts/rag_download_corpus.py` fetches into git-ignored `data/rag/raw/` | Committing PDFs (legal, but the hash manifest proves the same with a smaller repo) |
| LLM | `llama3.1:8b` (Q4_K_M, 4.9 GB) in local Ollama 0.40.16, already on disk; GPU (RTX 5060 Laptop, 8 GB) for development and runs, CPU timing reported too (D1). README gets a "Built with Llama" credit (D3) | qwen2.5-coder 7B (no structured tool calls: it wrote the call as text); gemma2 9B (no tool support in Ollama); larger models (≈5 tok/s on CPU); cloud free tiers (accounts, Iran access / sanctions terms, data leaves the laptop) |
| Fallback LLM | `qwen2.5:7b-instruct` (Apache-2.0), only under rule C4 and after its own approval | — |
| Embeddings | `nomic-embed-text` v1.5 in Ollama (137M, F16, 274 MB, Apache-2.0, 768-d, context 2048), with task prefixes (C3) | sentence-transformers (19 packages incl. torch 2.15; Hugging Face downloads); fastembed (14 packages incl. onnxruntime) |
| Vector store | numpy exact cosine search over unit vectors; files `chunks.parquet` + `embeddings.npy` + `index_manifest.json` | DuckDB `vss` (extension download, HNSW pointless for ~250–500 rows); Chroma (46 packages, kubernetes, would bump pydantic 2.14.1→2.14.4 against the API pin); LanceDB (not needed) |
| Lexical retrieval | Our own BM25 (numpy, ~40 lines) | rank-bm25 (unmaintained; trivial to own and test) |
| Retrieval | Hybrid: dense + BM25, reciprocal-rank fusion (k=60), top **6** | Dense only (reported as an ablation, does not change the choice) |
| PDF parser | `pypdf==7.1.1` (BSD, 1 package, pure Python) (D8) | PyMuPDF 2.1 (AGPL, pulls onnxruntime and 6 more) |
| Agent | Tool-calling agent **and** a deterministic workflow baseline (D7) | Agent only (no reference for its behaviour) |
| PII | The LLM never sees `card_key` (D4), lat/long or `client_ref`; `age_band`/`urban` reason values are hidden (D5) | Passing the card token (not needed to write the note) |

## 3. Corpus (round 1)

Pinned on first download (trust on first use) with `--pin`, 2026-11-20; every later
run verifies the SHA-256 and stops on a mismatch without overwriting anything.

| doc_id | Source | Bytes | PDF pages | Sections | Chunks |
|---|---|---|---|---|---|
| `cfr31_1010` | GovInfo `CFR-2025-title31-vol3-part1010.pdf` | 705,617 | 84 | 75 | 190 |
| `cfr31_1020` | GovInfo `CFR-2025-title31-vol3-part1020.pdf` | 244,497 | 17 | 22 | 48 |
| `fincen_sar_faq_2025` | fincen.gov `SAR-FAQs-October-2025.pdf` | 284,868 | 4 | 5 | 7 |
| `ffiec_sar` | FFIEC manual, Suspicious Activity Reporting — Overview (manual download) | 485,588 | 17 | 20 | 29 |
| `ffiec_ctr` | FFIEC manual, Currency Transaction Reporting (Feb. 2021) | 193,520 | 5 | 9 | 10 |
| `ffiec_cdd` | FFIEC manual, Customer Due Diligence — Overview (05/05/2018) | 147,059 | 7 | 5 | 10 |
| `ffiec_cip` | FFIEC manual, Customer Identification Program (Feb. 2021) | 247,283 | 9 | 18 | 20 |
| `ffiec_risk` | FFIEC manual, BSA/AML Risk Assessment (March 2020) | 37,916 | 4 | 8 | 8 |
| `ffiec_appf` | FFIEC manual, Appendix F: ML/TF "Red Flags" (2/27/2015.V2) | 166,184 | 10 | 21 | 22 |
| **Total** | | | **157** | **183** | **344** |

The FFIEC files were downloaded by the user from bsaaml.ffiec.gov, which returns a
Cloudflare 403 to automated requests from this network. They were pinned with
`--pin` on 2026-11-21.

Licences: GovInfo "Public Domain & Copyright Notice" (17 U.S.C. §105: no copyright for
US Government works); fincen.gov site policy: public information, credit requested.

**FFIEC (D2)**: SAR, CTR, CDD, CIP, BSA/AML Risk Assessment, Appendix F (red flags).
The user downloads them manually into `data/rag/raw/` under the filenames in the
manifest; `--pin` then records their hashes. Adding them changes the index, so the
question set, τ and the pre-registration (C7) are done **after** the corpus is final.

## 4. Parsing and chunking

What is stripped (inspected on the real PDFs):

| Document | Removed | Count (round 1) |
|---|---|---|
| CFR | printed page number, line 1 (kept as `page_label`) | 101 |
| CFR | running header "31 CFR Ch. X (7–1–25 Edition) § …" / "Financial Crimes Enforc. Network, Treas. § …" (absent on a part's first page) | 100 |
| CFR | printer's slug "VerDate … PROD with CFR" (split over 2 lines on image pages) | 104 lines |
| CFR | image placeholders "ER28SE17.00x</GPH>"; Part 1010 PDF pp. 16–18 are the CDD certification form printed as **images: their text is not in the corpus** | 3 |
| CFR | text of another part (tail of § 1010.980 at the start of the 1020 file; Part 1020 at the end of the 1010 file; Part 1021 at the end of the 1020 file), each part's table of contents and authority notes, `[Reserved]` sections | — |
| CFR | "Subpart X—…" lines (wrapping over up to 4 lines) → `subpart` metadata | — |
| FAQ | page number at the top, blank lines | 4 |
| FFIEC | running title, line 1 of every page (the SAR file repeats it as the page-1 heading, which is kept) | 52 |
| FFIEC | "FFIEC BSA/AML Examination Manual <printed page> <edition date>" (kept as `page_label`, e.g. `3`, `F–2`); **absent in the SAR section, which has no printed page numbers** | 35 |
| FFIEC | "Return to Contents" link (Risk Assessment) | 1 |

**FFIEC headings (round 2, 2026-11-21)** come from **font size**, not from text patterns:
- A line is a heading when its pypdf text runs are larger than the document's body size. The body size is the most common size, weighted by characters.
- Consecutive heading lines of the same size are one wrapped heading. A heading printed twice (Risk Assessment) is kept once.
- A larger heading, or a bold one of the same size, becomes the parent of the smaller headings after it. The parent path is stored in `subpart` and goes into the embedded heading. For example, Appendix F's two "Funds Transfers" sections sit under "…Money Laundering" and "…Terrorist Financing".
- The section id is a slug of the heading: at most 7 words, with `-2` added to a repeat. Example: `ffiec_sar:timing-of-a-sar-filing:001`.
- Inline bold labels at body size, such as "Objective:", are not headings.
- Footnotes stay in the text, because they hold the CFR references.

**pypdf log messages (decision 2026-11-21).**
- pypdf's logger is captured in the parser, so its messages are visible to tests instead of only printed.
- **Any message from pypdf stops parsing (`PdfLogError`), with one allow-listed exception.** The message "Ignoring wrong pointing object" (19 times) is accepted for exactly one file, `ffiec_sar.pdf`, and only with SHA-256 `bd42e263…a3eee67e`.
- The message comes from FFIEC's own file. The user saved it with Chrome "Save link as", which keeps the server's bytes unchanged, and its producer is macOS Quartz. The file's cross-reference table has wrong offsets, and pypdf recovers by scanning.
- Tests:
  - extraction is complete (17 pages, no empty page, more than 7,000 words)
  - the other five FFIEC files log nothing
  - an unexpected message, or the allowed message from any other hash, fails
- Rejected:
  - silencing pypdf logging globally (it would hide future problems)
  - repairing the PDF with another tool (a new install, and it changes the pinned source)

Words hyphenated at a line or page break are rejoined. The joined form wins when it
occurs elsewhere in the document, and the hyphenated form wins when only that form
occurs ("anti-money"); with no evidence either way, the parts are joined.

Check: every section in each CFR part's table of contents was found as a heading.
`§ 1010.663` is the one heading with no table-of-contents line in the PDF text.

- **Sections**: a CFR section (`1020.320`), or for the FAQ `intro`, `Q1`…`Q4`. A chunk never crosses a section.
- **Size**: target 350 words, cut at a sentence end between 280 and 350 words, and 50 words of overlap. A remainder of at most 450 words stays whole.
- **Measured size** (nomic tokens, min/median/max):

  | Document | Tokens |
  |---|---|
  | 1010 | 28 / 461.5 / 662 |
  | 1020 | 34 / 422 / 574 |
  | FAQ | 253 / 397 / 533 |
  | ffiec_sar | 104 / 409 / 574 |
  | ffiec_ctr | 38 / 380 / 615 |
  | ffiec_cdd | 277 / 427 / 497 |
  | ffiec_cip | 43 / 318.5 / 660 |
  | ffiec_risk | 142 / 269 / 523 |
  | ffiec_appf | 103 / 243 / 420 |
- **Deviation from the plan (accepted 2026-11-20, kept as is)**: the plan said "max ~600 tokens", but the limits are implemented in **words** (max 450). Legal text runs at about 1.45 tokens per word, so the largest chunk is **662 tokens**. That is still far below the model's context of 2048, and `truncate=False` makes any overflow an error. **Any change to chunking is chosen on the dev set only.**
- **Short chunks** (10–60 words) are real short sections, such as "Refer to § 1010.311 of this chapter". They are valid citation targets, so they are kept.
- **Chunk id**: `{doc_id}:{section}:{seq:03d}`, e.g. `cfr31_1020:1020.320:002`. Each record carries `doc_id`, `section`, `section_title`, `subpart`, `page_start`/`page_end` (PDF pages), `label_start`/`label_end` (printed pages), `n_words`, `n_tokens`, `text` and `text_sha256`.
- **Embedded and BM25 text**: the heading plus the text, e.g. "31 CFR § 1020.320 Reports by banks of suspicious transactions\n…".

## 5. Citations

- The model may only write `[C:<chunk_id>]`. **Pages and section names are rendered by
  code** from metadata, e.g. `31 CFR § 1020.320 (Reports by banks of suspicious
  transactions), PDF p. 10–11, printed p. 1197–1198`.
- Validator (round 2): (1) every tag matches the regex, exists in the index and is in
  this question's retrieved set; (2) every "§ 10xx.xxx" in the prose occurs in the
  text of a cited chunk; (3) every answer sentence has ≥ 1 citation. On failure: one
  retry with the error, then the fixed refusal.

## 6. Retrieval and refusal (C1)

- Retrieval returns the top 6 by RRF, plus **`max_cosine`**: the dense cosine of the best chunk in the whole corpus.
- **The refusal signal is `max_cosine`, never the fused RRF score.** RRF is rank-based, so the top chunk scores high even when nothing is relevant. `max_bm25` is an optional second condition.
- **τ is set on the dev set only, by this rule:**
  - If the dev max_cosine values separate (every answerable question is above every unanswerable one), τ = the midpoint between the lowest answerable and the highest unanswerable value.
  - Otherwise, τ is the value that maximises correct decisions on dev (refusing the unanswerable ones and answering the answerable ones). A tie goes to the lower τ.
  - The BM25 condition (refuse only if `max_cosine < τ_cos` **and** `max_bm25 < τ_bm25`) is adopted only if it raises dev accuracy.
  - Both choices are fixed and pushed before the test run (C7).
- Further refusals: the model answers `NO_ANSWER`, or the validator fails twice.
- **Round-1 evidence** (ad-hoc sanity queries, not the dev set):
  - A question that is not in the corpus (the GDPR breach-notification deadline) scored max_cosine **0.696**.
  - Answerable questions scored 0.714 (travel rule), 0.784 (CTR), 0.811 and 0.829 (SAR).
  - The margin is narrow, so the retrieval gate alone will be weak, and the model-level `NO_ANSWER` matters.

## 7. Question set

- **42 questions** (C5):
  - **Dev: 12** (8 answerable, 4 not). Used for prompts, τ and the fallback rule C4.
  - **Test: 30** (22 answerable, 8 not). Locked and run once.
- File `data/rag/eval/questions.jsonl`. Fields:
  - `id`, `split`, `question`, `answerable`
  - `expected_sources`: a list of `(doc_id, section)`, not chunk ids, so labels survive re-chunking
  - `key_facts`: literal strings, each found in at least one expected section
  - `author`: `user` or `claude`
  - `wording`: `paraphrase` or `source_terms`, plus `avoided_terms` for a paraphrase
  - `seen`: true for the 5 sanity queries
  - `absence_terms`: for unanswerable questions, the search terms; the checker shows they have 0 hits
- **Original plan, C6 (superseded, kept for the record):** at least **10 of the 22 answerable test questions** written by the user in their own words; Claude drafts the rest, and the user reviews and corrects **every** label.
- **Documented change, option B (user decision 2026-11-21, made BEFORE any retrieval or LLM run on the questions):**
  - **C6 is dropped.** Claude drafts all 42 questions (`author: "claude"`), and the user reviews every label.
  - **Human claim labelling** is done on a random sample of **10 answered test questions**, not all of them.
  - **How the sample is drawn:** with Python's `random.Random(20261121).shuffle` over the sorted ids of the 22 answerable test questions. The sample is the **first 10 in this order that the system answers**; a refused question is replaced by the next one. The order is fixed now, before any run (`src/rag/questions.py`, `labelling_order`):
    `T05, T02, T07, T15, T08, T20, T04, T22, T10, T13`, then `T19, T21, T14, T16, T01, T18, T11, T03, T12, T09, T06, T17`.
  - **Mechanical claim check on ALL answered test questions:** every number, dollar amount, day count and section reference in an answer sentence must occur in the text of a chunk that sentence cites.
  - The thresholds for both are in §8. All other thresholds are unchanged.
- **Wording rule:** at least half of the answerable questions avoid the exact words of their source. A question counts as `paraphrase` when it contains none of its `avoided_terms`: the section's defining words, such as "vacation" for the employee red flag or "structur" for structuring. `scripts/rag_check_questions.py` checks this.
- **Draft of 2026-11-21:** 16 of the 30 answerable questions are paraphrased. All 12 unanswerable topics have **0 hits** for all their terms. `scripts/rag_check_questions.py` reports 0 problems. No retrieval and no LLM have been run on the 37 new questions.
- **C2**: before an "unanswerable" label is accepted, a keyword search over the full corpus text must show the topic is absent. The search covers synonyms and, if they exist, section numbers, and its terms are recorded. Example: the US "travel rule" **is** in the corpus (31 CFR 1010.410(f)), so it is answerable.
- The 5 round-1 sanity queries (§13) are seen data: dev set only, never verbatim in the test set.

## 8. Pre-registered thresholds (test set)

| Metric | Definition | Pass |
|---|---|---|
| Retrieval recall@6 | Answerable questions with at least one expected section in the top 6 | **≥ 19/22** |
| Citation validity, raw | Cited ids that exist and were retrieved, before the validator | **≥ 0.95** |
| Citation validity, final | After the validator (true by construction; checks the code) | **= 1.00** |
| Citation correctness | Answered answerable questions with at least one citation in an expected section | **≥ 0.80** |
| Claim support, human (option B) | Answer sentences in the **10-question human sample** (§7), labelled by the user as supported / unsupported / contradicted by their cited chunks. *(Original plan: all answered test questions.)* | **≥ 0.90 supported and 0 contradicted** |
| Claim support, mechanical (option B) | On **all** answered test questions: sentences that contain a number, dollar amount, day count or section reference, where every such token occurs in the text of a chunk that sentence cites | **≥ 0.95 of such sentences pass** |
| Correct refusal | Unanswerable questions refused | **≥ 7/8** |
| False refusal | Answerable questions refused | **≤ 2/22** |
| Agent (mechanical) | Valid note schema, every citation valid, reasons equal to the API's, 422 and 503 cases handled | **10/10** |

- **How results are reported**: as counts with Wilson 95% intervals. There is no LLM judge, because an 8B model judging itself is not credible.
- **Run settings**: temperature 0, fixed seed, and the model digests are recorded.
- **After the test run**: there are no prompt changes. A failed threshold is reported as a fail.
- **C4, the model fallback rule (decided on dev, before the test run):** switch to `qwen2.5:7b-instruct` only if, on dev, any of these holds:
  - raw citation validity < **0.95**
  - claim support < **0.90**
  - the agent passes < **9/10** mechanical checks

  The download needs its own approval. *Open point:* option B does not say who labels claim support on the **dev** set for this rule. Proposal: the user labels all answered dev questions, which is at most 8.
- **C7, pre-registration:** before the test run, these are committed and **pushed**, which gives a public timestamp:
  - this document's thresholds
  - `questions.jsonl` with its labels
  - a SHA-256 of every prompt file

## 9. Agent (round 3)

**Tools:**
- `score_transaction(txn_ref)`: code looks up the full request in a local alert fixture and POSTs it to the API, so the model never sees `card_key`, lat/long or `client_ref`.
- `search_regulations(query, k≤6)`.

**Limits:** at most 4 tool calls, 6 model turns and 10 minutes. Beyond that, code writes an "INCOMPLETE" note with the reasons only.

**Note** (a pydantic model rendered to Markdown, about 200 words or less):

| Part | Written by |
|---|---|
| Header: txn_ref, model_version, probability_raw, threshold, alert (and card_key, added after generation) | code |
| Top-3 SHAP reasons, copied word for word from the API (values of `age_band`/`urban` hidden) | code |
| "Regulatory context": 2–4 cited sentences | LLM |
| "Suggested next steps": cited | LLM |
| Fixed disclaimer: decision support only, not a SAR determination, CFR edition of July 1, 2025 | code |

**Errors:**
- An API **422** (`validation_error`, `unknown_card`, `date_outside_calendar`) gives `NOT SCORED: <code>`, with no retrieval and no LLM call.
- A 503 or connection error gives `API UNAVAILABLE`.
- `alert=false` gives a short "below threshold" note.

**Baseline:** a deterministic workflow (score, then fixed queries from a feature-to-query map, then write) is built alongside the agent. It is the CI path and the comparison point.

## 10. Guardrails

- **No made-up citations**: the validator in §5.
- **Refusal when evidence is missing**: §6.
- **No PII to the LLM**: an `LLMAlertContext` model with `extra="forbid"` allows only amount, category, time and SHAP reasons, with the `age_band`/`urban` values masked. A test checks that nothing else gets through.
- **Prompt injection from documents**:
  - The sources are hash-pinned government PDFs.
  - Chunk text is still treated as data: it is wrapped in delimiters, the system prompt says to ignore instructions inside it, and control characters are stripped.
  - The tools are read-only, the API URL is fixed, and no tool can send or write.
  - A test plants an injected chunk and checks that the validator rejects any citation it asks for.

## 11. Repository layout, requirements and CI

```
src/rag/      corpus.py parse.py chunk.py embed.py bm25.py retrieve.py index.py   (round 1)
              llm.py answer.py citations.py                                        (round 2)
src/agent/    context.py tools.py loop.py workflow.py note.py                      (round 3)
scripts/      rag_download_corpus.py rag_build_index.py rag_query.py (round 1), rag_eval.py, agent_explain.py
data/rag/     corpus_manifest.json, eval/questions.jsonl (committed); raw/ index/ runs/ (git-ignored)
tests/        test_rag_corpus.py test_rag_parse.py test_rag_chunk.py test_rag_embed.py test_rag_retrieve.py
```

- **`requirements-rag.txt`**: `-r requirements.txt`, `numpy==2.5.3`, `pypdf==7.1.1`. pypdf is the only new package. `tests/test_requirements_pins.py` checks that its pins agree with every other requirements file.
- **Markers**:
  - `corpus`: needs `data/rag/raw/` and pypdf
  - `ollama`: needs the local server

  `tests/conftest.py` skips them with a reason when their prerequisites are missing.
- **CI**:
  - It uses FakeEmbedder (a hashed bag of words), the fake HTTP sessions and, later, a scripted fake LLM. There is no network and no model.
  - `ci.yml` (approved 2026-11-20): the test job also installs `requirements-rag.txt` and runs `-m "not gold and not mlflow and not container and not corpus and not ollama"`, so the tests that need local documents or a model are deselected (not silently skipped).

## 12. Gates (user approval each)

| Gate | Status |
|---|---|
| `pip install pypdf==7.1.1` (.venv) | approved, done |
| `ollama pull nomic-embed-text` (274 MB, `~/.ollama`) | approved, done |
| GovInfo + FinCEN download (1.23 MB) | approved, done |
| FFIEC six sections | user, manual |
| `ollama pull qwen2.5:7b-instruct` | not approved (only under C4) |
| `ci.yml` change (install requirements-rag.txt, new marker selection) and push | approved 2026-11-20 |

## 13. Round-1 findings and risks

**Retrieval misses on ad-hoc sanity queries** (to be measured properly on the dev set):
- *Travel rule.* "What does the US travel rule require for funds transfers of $3,000 or more?" does not reach § 1010.410 in the top 6. Its best chunk is #1 on BM25 but #66 on dense, and #17 after fusion. The regulation never uses the words "travel rule".
- *CTR threshold.* "What is the CTR threshold for banks?" ranks § 1010.311 #8. § 1010.330 (cash over $10,000 received in a trade or business, Form 8300) fills the top 5.
- *Possible fixes*: a small glossary that expands queries ("travel rule" → "transmittal of funds … pass on"), paragraph-aware chunking at "(a)", "(b)", …, or changes to fusion. **Any retrieval fix is chosen on the dev set only** (decision 2026-11-20).
- **The 5 sanity queries have been seen** (their retrieval results are known). They, or their verbatim wording, may appear only in the **dev** set, never in the test set:
  1. "What is the deadline for a bank to file a suspicious activity report?"
  2. "What is the currency transaction report threshold for banks?"
  3. "What does the US travel rule require for funds transfers of $3,000 or more?"
  4. "What is the GDPR time limit to report a personal data breach to the supervisory authority?"
  5. "How long must a bank keep a copy of a SAR and its supporting documentation?"

**Other risks:**
- The refusal margin is narrow (§6).
- The FFIEC sections are not in yet.
- The CDD form pages are images, so their text is missing.
- The CFR edition is frozen at July 2025, so later amendments are not in the corpus.
- The samples are small, so the intervals are wide.
- The Llama licence is custom.
