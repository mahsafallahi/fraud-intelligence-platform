# Phase 7 question set: review sheet

Generated from `data/rag/eval/questions.jsonl` (42 questions: dev 8 answerable + 4 not, test 22 + 8).
All drafted by Claude (`author: "claude"`, option B); **every label is for the user to review**.
`scripts/rag_check_questions.py` reports 0 problems:
- every expected section exists and contains at least one key fact
- every not-answerable topic has 0 hits for all its terms
- 16 of the 30 answerable questions are paraphrased
- the 5 seen sanity queries are dev only

No retrieval and no LLM have been run on the 37 questions that are not seen queries.

*(seen)* = one of the 5 round-1 sanity queries (dev only). "paraphrased" = the question avoids its
`avoided_terms` (the source section's defining words; listed in the JSONL).

| id | split | question | expected sources | key facts | paraphrased |
|---|---|---|---|---|---|
| D01 | dev | What is the deadline for a bank to file a suspicious activity report? *(seen)* | `cfr31_1020:1020.320`<br>`ffiec_sar:timing-of-a-sar-filing`<br>`fincen_sar_faq_2025:Q3` | “30 calendar days”<br>“60 calendar days” | no |
| D02 | dev | What is the currency transaction report threshold for banks? *(seen)* | `cfr31_1010:1010.311`<br>`ffiec_ctr:filing-obligations`<br>`ffiec_ctr:aggregation-of-currency-transactions`<br>`cfr31_1010:1010.313` | “more than $10,000” | no |
| D03 | dev | What does the US travel rule require for funds transfers of $3,000 or more? *(seen)* | `cfr31_1010:1010.410` | “$3,000 or more”<br>“transmittal order” | no |
| D04 | dev | What is the GDPR time limit to report a personal data breach to the supervisory authority? *(seen)* | — *(not answerable)* | absence terms, 0 hits each: “GDPR”, “General Data Protection”, “personal data breach”, “72 hours” | — |
| D05 | dev | How long must a bank keep a copy of a SAR and its supporting documentation? *(seen)* | `cfr31_1020:1020.320`<br>`ffiec_sar:record-retention-and-supporting-documentation` | “five years” | no |
| D06 | dev | Does a bank have to write down why it decided not to file a SAR? | `fincen_sar_faq_2025:Q4` | “no requirement or expectation” | no |
| D07 | dev | Above what ownership share is an individual treated as a beneficial owner of a legal entity customer? | `cfr31_1010:1010.230` | “25 percent or more” | no |
| D08 | dev | If a bank employee keeps refusing to take time off, could that be a warning sign? | `ffiec_appf:employees` | “reluctant to take a vacation” | yes |
| D09 | dev | When the same suspicious behaviour keeps going on after a first report, how often may a bank file follow-up reports? | `ffiec_sar:sar-filing-on-continuing-activity`<br>`fincen_sar_faq_2025:Q2`<br>`fincen_sar_faq_2025:Q3` | “90” | yes |
| D10 | dev | How does a firm in the United Kingdom request a defence against money laundering (DAML) from the National Crime Agency? | — *(not answerable)* | absence terms, 0 hits each: “DAML”, “Defence against money laundering”, “Defense against money laundering”, “National Crime Agency”, “United Kingdom” | — |
| D11 | dev | Which PCI DSS requirements apply to storing cardholder data? | — *(not answerable)* | absence terms, 0 hits each: “PCI”, “Payment Card Industry”, “cardholder data” | — |
| D12 | dev | Who bears the loss for counterfeit card fraud after the EMV chip liability shift? | — *(not answerable)* | absence terms, 0 hits each: “EMV”, “liability shift”, “chip card” | — |
| T01 | test | Is a bank required to file a SAR for a robbery that it has already reported to the police? | `cfr31_1020:1020.320` | “robbery or burglary”<br>“reported to appropriate law enforcement authorities” | no |
| T02 | test | A teller sees what looks like an ongoing money laundering scheme. Apart from the written report, what should the bank do straight away? | `cfr31_1020:1020.320` | “by telephone” | yes |
| T03 | test | Within how many days after the transaction must a CTR be filed? | `cfr31_1010:1010.306`<br>`ffiec_ctr:filing-and-record-retention` | “15 days”<br>“15 calendar days” | no |
| T04 | test | A customer makes three cash deposits of $4,000 each at different branches on the same day. Must the bank look at them together for reporting purposes? | `cfr31_1010:1010.313`<br>`ffiec_ctr:aggregation-of-currency-transactions` | “single transaction”<br>“more than $10,000” | yes |
| T05 | test | What is it called when someone splits a large sum of cash into smaller deposits to avoid a government report, and is it allowed? | `cfr31_1010:1010.100`<br>`cfr31_1010:1010.314`<br>`fincen_sar_faq_2025:Q1`<br>`ffiec_ctr:structured-transactions-ctr-requirements` | “structuring” | yes |
| T06 | test | Someone wants to buy a cashier's check with $5,000 in cash. What must the bank record? | `cfr31_1010:1010.415` | “$3,000 or more in currency” | yes |
| T07 | test | What information must a bank obtain from an individual before opening an account under its customer identification program? | `cfr31_1020:1020.220`<br>`ffiec_cip:customer-information-required` | “Date of birth” | no |
| T08 | test | For how long must a bank keep a customer's identifying details after the account is closed? | `cfr31_1020:1020.220`<br>`ffiec_cip:recordkeeping-and-retention-requirements` | “five years after the date the account is closed”<br>“five years after the account is closed” | yes |
| T09 | test | How must a bank let customers know that it is checking who they are? | `ffiec_cip:adequate-customer-notice`<br>`cfr31_1020:1020.220` | “adequate notice” | yes |
| T10 | test | For how long is a notice to share information under section 314(b) effective? | `cfr31_1010:1010.540` | “one year period” | no |
| T11 | test | Can a customer successfully sue a bank for passing a report about their transactions to the authorities? | `ffiec_sar:safe-harbor-for-banks-from-civil-liability` | “civil liability” | yes |
| T12 | test | May a bank tell a customer that it has reported them to FinCEN? | `ffiec_sar:prohibition-of-sar-disclosure`<br>`cfr31_1020:1020.320` | “any information that would reveal the existence of a SAR” | yes |
| T13 | test | What phone number should a bank call right away if a transaction may be linked to terrorism against the United States? | `ffiec_sar:identifying-underlying-crime`<br>`ffiec_sar:sar-quality` | “(866) 556-3974” | yes |
| T14 | test | Who at the top of the bank must be told that suspicious activity reports have been filed? | `ffiec_sar:notifying-board-of-directors-of-sar-filings` | “board of directors” | yes |
| T15 | test | What is a National Security Letter and who can issue one? | `ffiec_sar:national-security-letters` | “Federal Bureau of Investigation” | no |
| T16 | test | What patterns in a customer's wire activity could point to money laundering? | `ffiec_appf:funds-transfers` | “round dollar” | yes |
| T17 | test | What red flags suggest that a customer may be using a shell company? | `ffiec_appf:shell-company-activity` | “registered agent”<br>“Purpose of the shell company is unknown or unclear” | no |
| T18 | test | What are the two steps of a BSA/AML risk assessment? | `ffiec_risk:analysis-of-specific-risk-categories`<br>`ffiec_risk:identification-of-specific-risk-categories`<br>`ffiec_risk:assessing-the-bank-s-bsa-aml-risk` | “two-step”<br>“first step”<br>“second step” | no |
| T19 | test | Does a bank have to keep what it knows about a customer current for as long as the relationship lasts? | `ffiec_cdd:ongoing-monitoring-of-the-customer-relationship`<br>`ffiec_cdd:customer-due-diligence` | “maintain and update customer information” | yes |
| T20 | test | What should a bank do when it discovers that it failed to file some cash transaction reports in the past? | `ffiec_ctr:ctr-backfiling-and-amendment` | “backfile” | yes |
| T21 | test | Is a cash transaction just at or near the CTR threshold, by itself, a reason to file a SAR? | `fincen_sar_faq_2025:Q1` | “not information sufficient to require the filing” | no |
| T22 | test | When does someone carrying cash across the US border have to report it? | `cfr31_1010:1010.340` | “exceeding $10,000” | yes |
| T23 | test | What is the threshold for AUSTRAC threshold transaction reports in Australia? | — *(not answerable)* | absence terms, 0 hits each: “AUSTRAC”, “Australia”, “threshold transaction report” | — |
| T24 | test | When must a Canadian bank file a large cash transaction report with FINTRAC? | — *(not answerable)* | absence terms, 0 hits each: “FINTRAC”, “Canada”, “large cash transaction” | — |
| T25 | test | Which powers does the new EU Anti-Money Laundering Authority (AMLA) have over banks? | — *(not answerable)* | absence terms, 0 hits each: “Anti-Money Laundering Authority”, “AMLA”, “European Union”, “AMLD” | — |
| T26 | test | How many days does a cardholder have to raise a Visa or Mastercard chargeback? | — *(not answerable)* | absence terms, 0 hits each: “chargeback”, “Visa”, “Mastercard” | — |
| T27 | test | Under PSD2, how much of an unauthorised payment can a payer be made to bear? | — *(not answerable)* | absence terms, 0 hits each: “PSD2”, “Payment Services Directive”, “unauthorised payment”, “unauthorized payment”, “strong customer authentication” | — |
| T28 | test | What is the minimum Common Equity Tier 1 capital ratio that a bank must maintain? | — *(not answerable)* | absence terms, 0 hits each: “Common Equity Tier 1”, “Tier 1”, “capital ratio”, “risk-weighted” | — |
| T29 | test | What is the statute of limitations for prosecuting a money laundering offence? | — *(not answerable)* | absence terms, 0 hits each: “statute of limitations”, “limitations period”, “3282” | — |
| T30 | test | Which credit score cut-off separates prime from subprime mortgage borrowers? | — *(not answerable)* | absence terms, 0 hits each: “credit score”, “FICO”, “subprime”, “loan-to-value” | — |

## Changes in round 3 (2026-11-22, before any run on these questions)

**Not-answerable replacements.** The old topics sat next to rules that are in the corpus:

| Id | Old topic | Why it was replaced | New topic | New terms (0 hits each) |
|---|---|---|---|---|
| T28 | travel rule for crypto/virtual currency | the $3,000 funds-transfer travel rule is in the corpus (§ 1010.410), so a model could reasonably answer from it | the Common Equity Tier 1 capital ratio | "Common Equity Tier 1", "Tier 1", "capital ratio", "risk-weighted" |
| T30 | Regulation E liability for unauthorized EFTs | the Electronic Fund Transfer Act appears in a § 1010.100 definition | the credit-score cut-off between prime and subprime borrowers | "credit score", "FICO", "subprime", "loan-to-value" |

The word "Basel" occurs once in the corpus, so it is not used in the new T28 question.

**Stronger key facts:**
- **T12:** "confidential" → "any information that would reveal the existence of a SAR". The phrase is in both expected sources (§ 1020.320(e) and the FFIEC prohibition section).
- **T17:** "shell" → "registered agent" and "Purpose of the shell company is unknown or unclear". Both are from the FFIEC shell-company red flags.

**Extra sources added in round 2, kept.** The checker shows each one contains the key fact:
- T05: `ffiec_ctr:structured-transactions-ctr-requirements`
- T13: `ffiec_sar:sar-quality`
- T19: `ffiec_cdd:customer-due-diligence`

**New rule: every expected source must contain at least one key fact.** It found five of Claude's round-2 labels where a source did not:

| Id | Fix | Reason |
|---|---|---|
| T02 | removed `ffiec_sar:identifying-underlying-crime` | it has no "notify by telephone" instruction |
| T03 | added the fact "15 calendar days" | this is the FFIEC CTR wording |
| T08 | added the fact "five years after the account is closed" | this is the FFIEC CIP wording |
| T16 | removed `ffiec_appf:funds-transfers-2` | that is the terrorist-financing list; the question asks about money laundering |
| T18 | added the facts "first step" and "second step"; removed `ffiec_risk:bsa-aml-risk-assessment-process` | that section does not describe the two steps |
