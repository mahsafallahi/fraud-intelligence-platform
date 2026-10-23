# Phase 3 — EDA and hypothesis tests: findings

Source: Gold star schema (`data/gold/fraud_dev.duckdb`, read-only), notebook `notebooks/03_eda.ipynb`.
Data: 1,852,394 transactions, 9,651 fraud (base rate 0.521%), 999 cards, 2019-01-01 to 2020-12-31.
All rates below come from the notebook; confidence intervals are Wilson 95% unless stated.

## 1. Amount — strongest raw signal, not monotonic

- Median amount: fraud $390.00 vs legitimate $47.24. Maximum: fraud $1,376.04 vs legitimate $28,948.90.
- Top decile (above $136.33): 7,349 fraud, rate 3.967% (7.6× base) — 76% of all fraud.
- Second cluster at $7.75–32.08 (deciles 3–4): rate ≈ 0.48%; 2,009 fraud (21%) lie below $32.08.
- $32.09–136.33 (deciles 5–9): only 293 fraud. Below $4.10 (decile 1): 22 fraud.
- Consequence: a linear term on raw amount misses the small-amount cluster; prefer tree-based models or binned features.

## 2. Time of day and day of week

- Hours 22:00–03:59 hold 23.5% of transactions (435,021) but 84.6% of fraud (8,169). Hour 22: 2.601%. Hours 04–21: ≈ 0.1%.
- Hour is the local transaction time recorded by the simulator; there is no time zone.
- Day of week: Monday 0.402% to Friday 0.640%. Fraud counts per weekday are nearly flat (1,125–1,590); the rate differences come mostly from changing legitimate volume.

## 3. Night × amount — independent, roughly multiplicative

| Amount band | Day rate | Night rate | Night / day |
|---|---|---|---|
| ≤ $32.08 | 0.054% | 1.081% | ≈ 20× |
| $32.08–136.33 | 0.007% | 0.106% | ≈ 15× |
| > $136.33 | 0.820% | 12.713% | ≈ 15× |

- Night AND above $136.33: 49,021 transactions (2.6%), 6,232 fraud (64.6%), rate 12.713% (24× base).

## 4. Category and channel

- Category lift ranges from 0.29 (home, health_fitness) to 3.06 (shopping_net).
- Overall, online 1.213% vs in-person 0.789%, but the direction depends on the MCC:
  - 5311 department store: online 1.593% vs in-person 0.634%
  - 5999 miscellaneous: online 1.304% vs in-person 0.282%
  - 5411 grocery: online 0.270% vs in-person 1.265% (reversed)
- This is effect modification (interaction), not Simpson's paradox: the direction flips in only one of three groups.
- Category stays a strong, independent signal inside the night + high-amount slice: shopping_net 35.3%, misc_net 20.75%, six categories 0 fraud.
- `home` is the lowest-risk category overall (0.151%) but 10.6% inside that slice: 225 of its 265 fraud are night + high-amount.

## 5. Demographics (card-level attributes, 999 cards)

- Age (approximate: transaction year − birth year): lowest 36–45 (0.398%), highest 76+ (0.769%).
- Gender: male 0.567% vs female 0.483%.

## 6. Location and Census enrichment — no useful signal

- Distance customer → merchant: every decile within 0.96–1.06× base.
- City population quintiles: 0.96–1.07× base. Median household income quintiles: 0.95–1.08× base.
- Missing income: 0.434% (57 cards) vs 0.527% (942 cards) — too few cards to rely on.
- Distance is a classic real-world signal; its absence is a simulator limitation (merchant locations are random around the customer).

## 7. Holidays — no evidence of an effect

- Public holiday (nationwide or customer's state): 0.471% (247 fraud, 22 days) vs 0.522%.
- Nationwide bank holiday: 0.468% (20 days) vs 0.523%.

## 8. Per-card behaviour (past transactions only)

- Time since the card's previous transaction: < 1 min 2.233%, 1–10 min 1.870%, 10–60 min 1.178%, over one hour below base.
- Amount vs the median of the card's past amounts (at least 5 past transactions):
  - > 10×: 23,001 transactions, 4,355 fraud (45%), rate 18.934% (36× base)
  - 5–10×: 3.705%; 1–2×: 0.058% (safest); < 0.5×: 0.249% (possible card testing)
- A behavioural feature beats raw amount: "> 10× the card's usual amount" has higher precision (18.9%) than "night and above $136" (12.7%).

## 9. Card lifecycle and fraud episodes

- 906 cards appear in the first week of data: 1 of their first transactions is fraud.
- 93 cards appear later: 90 of their first transactions are fraud (≈ 97%), mostly 00:00–03:59, ≈ $300 or ≈ $10.
- 976 of 999 cards have fraud. Each has one episode of 2–19 fraudulent transactions (median 10), lasting at most 2.89 days (median 1.89).
- After the last fraud, cards stay active for a median of 353 days (real cards would be reissued).

## 10. Hypothesis tests

Single-predictor logistic regression, cluster-robust standard errors (cluster = card, or day for holidays), Holm correction across the 7 tests.

| # | Test | Clusters | Odds ratio | 95% CI | Holm-significant |
|---|---|---|---|---|---|
| 1 | Public holiday vs not | 730 days | 0.90 | 0.64–1.27 | No |
| 2 | Night (22–03) vs day | 999 cards | 18.3 | 17.3–19.4 | Yes (p < 1e-300) |
| 3 | Online vs in-person, 5311 | 996 | 2.54 | 2.35–2.74 | Yes |
| 4 | Online vs in-person, 5411 | 993 | 0.211 | 0.178–0.25 | Yes (reversed) |
| 5 | Online vs in-person, 5999 | 985 | 4.67 | 4.06–5.37 | Yes |
| 6 | Male vs female | 999 | 1.18 | 1.08–1.29 | Yes (small effect) |
| 7 | Distance per 10 km | 999 | 1.00 | 0.995–1.01 | No |

- Distance: tight interval around 1 → evidence of no meaningful effect.
- Holidays: wide interval from 22 days → no evidence either way.
- Caveat: these hypotheses were found in the same data, so the tests are not independent confirmation; the real check is the Phase 4 model on an untouched test period.

## 11. Baseline rule the model must beat

- "Night (22–03) AND amount above $136.33": flags 2.6% of transactions (≈ 67 per day), recall 64.6%, precision 12.7%.
- For comparison, "amount above $136.33" alone: recall 76%, precision 4.0%.

## 12. Leakage rules for Phase 4

1. Time-based split into train / validation / test with a 7-day gap between periods (fraud episodes last at most 2.89 days).
2. No random split by transaction: one episode would land in both train and test.
3. Per-card features use only earlier transactions of the same card.
4. No feature built from past labels: chargebacks arrive weeks after the transaction.
5. Imputation and scaling are fitted inside the training pipeline on the training period only.
6. Validation period: model selection, probability calibration, threshold choice. Test period: used once at the end.

## 13. Feature list for Phase 4

Use:
- amount (raw), hour and night flag, day of week
- category, channel
- seconds since the card's previous transaction; amount vs the card's past median
- counts of the card's transactions in the last 1 hour and 24 hours (past only)
- age (subject to a fairness check)
- city population, median income + `has_income`, distance, holiday flags — expected to add little; kept to measure their importance honestly

Run twice, with and without:
- card-history length (number of past transactions / first-transaction flag), because of the simulator artifact in section 9

Do not use:
- gender (kept only to audit whether errors fall equally on both groups)
- trans_num, card_key, merchant name, merchant coordinates
- any feature derived from past fraud labels

## 14. Limitations (simulator artifacts)

- An almost empty amount band ($32–136) and a sharp night pattern are cleaner than real data.
- 98% of cards have a fraud episode, and cards stay active after fraud.
- Mid-period cards start with fraud 97% of the time.
- Distance carries no signal.
- Strong model results will partly reflect these artifacts; this is stated, not hidden.