# Fraud model: results

Phase 4 of the Fraud Intelligence Platform. Every number below comes from `notebooks/04_modelling.ipynb`; "cell N" refers to that notebook. Sections 1–11 were written and published **before** the test period was opened. Section 12 is filled in only after the one-time test evaluation.

Intervals are 95% card-bootstrap intervals: whole cards are resampled (500 draws, seed 42), because the transactions of one fraud episode are not independent.

## 1. Data

- Simulated card transactions (the Sparkov generator), loaded through the Bronze, Silver and Gold layers of this project into a DuckDB star schema (cell 5).
- Before the test period: 1,570,873 transactions, 8,715 of them fraud (cell 0); 993 cards (cell 19), of which 883 have fraud (cell 4).
- The test period holds 281,521 transactions with 936 fraud. These two counts were seen once, on 2026-10-23, to size the split; no model was trained, tuned or scored on test rows before step H (cell 2).
- The monthly fraud rate varies between 0.374% and 1.037% (cell 1).

## 2. Split

A time-based split, so that no fraud episode is spread over two periods (cell 3):

| Period | Dates | Transactions | Fraud | Fraud cards | Fraud rate |
|---|---|---|---|---|---|
| Train | 2019-01-01 to 2020-06-23 | 1,306,680 | 7,528 | 762 | 0.576% |
| Gap (7 days) | between train and validation | 36,006 | 208 | 23 | 0.578% |
| Validation | 2020-07-01 to 2020-09-23 (85 days) | 228,187 | 979 | 102 | 0.429% |
| Test | 2020-10-01 to 2020-12-31 (92 days) | 281,521 | 936 | — | — |

The train fraud rate is a third higher than the validation rate (0.576% vs 0.429%); the effect of a further shift is discussed in section 6.

## 3. Features

The final model F uses 16 features (cell 23), all computed from the transaction and from that card's **strictly earlier** transactions:

- the transaction: amount, hour, night flag (22–03h), weekday, merchant category, channel;
- the card's past behaviour: time since its previous transaction, amount divided by its past median, number of its transactions in the past 1 hour and 24 hours;
- distance to the merchant, three holiday flags;
- a coarse customer profile: 10-year age band, and an urban flag (city population of at least 50,000).

Features are built on the full timeline and only then split, so card history continues across period boundaries. A leakage test checks that removing later rows never changes an earlier feature value.

Left out on purpose: gender (used only to audit fairness), the card's transaction count (a simulator artefact), and the exact age, city population and income. The exact profile was unique to every one of the 993 cards, so it could identify a card; the coarse profile has only 18 groups (cell 19).

## 4. Models

All models are trained on the training period only and logged to a local MLflow store with their git commit.

| Model | What it is |
|---|---|
| Baseline rule | Flag a transaction at night (22–03h) with an amount above $136.33 |
| Logistic regression | Log of skewed features, median imputation, scaling and one-hot categories, fitted on training rows only |
| LightGBM A–D | Four variants: class weights on/off × card-history feature on/off; hyperparameters chosen on an inner time split inside the training period (cell 11) |
| LightGBM E | A without any customer feature (cell 18) |
| **LightGBM F (final)** | A with the coarse profile instead of the exact one: no class weights, no card history (cell 19) |
| XGBoost | A robustness check with F's 16 features (section 9) |

## 5. Validation results

**Baseline and logistic regression** (cells 8–10). At the rule's own workload of 5,832 alerts (68.6 per day), the rule catches 629 fraud cases (recall 0.6425, precision 0.1079, dollar recall 0.8382, PR-AUC 0.0708). The logistic regression at the same workload catches 672 (recall 0.6864, dollar recall 0.9063, PR-AUC 0.4285); its recall is 0.044 higher (interval 0.024 to 0.063).

**LightGBM variants** (cells 12–14, 18–20). At 1,000 alerts (11.8 per day):

| Model | PR-AUC | Recall at 1,000 alerts | Dollar recall at 1,000 alerts |
|---|---|---|---|
| A | 0.9203 | 0.8550 | 0.9463 |
| B | 0.9193 | 0.8529 | 0.9395 |
| C | 0.9162 | 0.8478 | 0.9453 |
| D | 0.9132 | 0.8458 | 0.9401 |
| E (no customer features) | 0.8738 | 0.8069 | 0.8929 |
| **F (final)** | **0.9200** | **0.8478** | **0.9382** |
| Logistic regression | 0.4285 | 0.4607 | 0.6916 |

- A, B, C and D are statistically indistinguishable: every paired interval includes 0 (cell 14).
- Removing all customer features costs real performance: A minus E is 0.0466 in PR-AUC (interval 0.0343 to 0.0607) (cell 18).
- The coarse profile recovers it: A minus F is 0.0003 in PR-AUC (interval −0.0057 to 0.0068) and 0.0072 in recall (interval −0.0128 to 0.0191), not distinguishable (cell 19).

**Why F is the final model.** Equal performance to A, no card-fingerprint risk, and less personal data. At 1,000 alerts on validation, F catches 830 of 979 fraud cases (recall 0.8478), $484,351 of fraud dollars (dollar recall 0.9382), with precision 0.830 (cell 29). All eight logged runs reproduce their logged metrics exactly when the notebook is rerun (74 values, worst difference 0; cell 20).

## 6. Calibration

(cell 28)

- **Raw F is already well calibrated:** mean predicted fraud probability 0.435% vs an observed rate of 0.429% on validation; Brier score 0.000939, log loss 0.003717.
- **Method choice on a time split inside validation.** Fitted on July–August (174,607 rows, 736 fraud, 76 fraud cards), evaluated on September (53,580 rows, 243 fraud, 26 fraud cards); no fraud card appears in both parts.

| September | Log loss | Brier |
|---|---|---|
| Raw F | 0.003451 | 0.000933 |
| Platt | **0.003449** | 0.000929 |
| Isotonic | 0.003702 | **0.000919** |

- The pre-registered criterion (log loss) chose **Platt**; the Brier score would have chosen isotonic, and the two disagree.
- The final Platt calibrator, refit on the whole validation period, is close to the identity (a = 0.995, b = −0.086). PR-AUC (0.9200) and recall at 1,000 alerts (0.8478) are unchanged.
- **Base-rate shift.** Train 0.576%, validation 0.429%. If the fraud rate moved to 0.30%, a calibrated probability of 0.5 would correspond to 0.41; at 0.80%, to 0.65. Ranking by probability is unaffected. Nothing is adjusted for an unknown future rate.

## 7. Ranking decision

(cells 29–31)

| At 1,000 alerts | Fraud caught | Recall | Dollars caught | Dollar recall | Precision |
|---|---|---|---|---|---|
| By probability | 830 | 0.8478 | $484,351 | 0.9382 | 0.830 |
| By expected loss (calibrated probability × amount) | 725 | 0.7406 | $498,991 | 0.9665 | 0.725 |

- The pre-registered rule chose expected loss, because its dollar-recall gain had an interval above 0 (+0.0284, interval 0.0085 to 0.0513).
- **Decision (2026-11-09, before the test period): rank by probability**, a documented deviation from the rule (cell 30). Reasons:
  1. The gain was not confirmed out of sample. In September, with the July–August calibrator, at 271 alerts: +0.0482 dollar recall, interval −0.0018 to 0.1051.
  2. Expected loss catches 105 fewer fraud cases (recall −0.1073, interval −0.1425 to −0.0662) and loses 10.5 points of precision, for $14,640 more fraud dollars.
  3. It widens the gender gap in recall (section 8).
  4. The rule weighed only dollars.
- On the test period both rankings are reported; probability ranking is the primary claim.

**Alert threshold.** Capacity-based: about 12 alerts per day (1,000 alerts over the 85 validation days). A cost-based threshold is not possible without investigation-cost data.

## 8. Fairness

Descriptive, at 1,000 alerts, probability ranking (cell 32). Gender is not a model feature; age band is.

| Group | Fraud cards | Recall | False-alarm rate |
|---|---|---|---|
| Women | 55 | 0.797 (0.717 to 0.869) | 0.100% (0.042% to 0.166%) |
| Men | 47 | 0.909 (0.858 to 0.951) | 0.045% (0.017% to 0.089%) |

- **Women's fraud is caught less and women get more false alarms.** Because gender is not a feature, the gap comes through other features. Under the expected-loss ranking the gap widens: recall 0.677 for women vs 0.819 for men, with intervals that no longer overlap.
- **By age band**, recall is about 0.79 for ages 20–49 and 0.90 for ages 50–59; false alarms are concentrated in ages 20–49 (0.083% to 0.130%). Ages 10–19 (no fraud), 60–69 (8 fraud cards), 80–89 (3) and 90–99 (4) have fewer than 10 fraud cards and are too few to judge.
- No causal claim is made.

## 9. XGBoost check

(cell 33) Same 16 features, same inner time split, native categorical support, no class weights, grid `max_depth` {4, 6} × `min_child_weight` {0.25, 2.5}. Best: depth 6, `min_child_weight` 0.25, 609 trees, inner average precision 0.9462.

| Validation | PR-AUC | Recall at 1,000 alerts |
|---|---|---|
| F (LightGBM) | 0.9200 | 0.8478 |
| XGBoost | 0.9194 | 0.8539 |
| XGBoost minus F | −0.0006 (−0.0045 to 0.0033) | 0.0061 (−0.0077 to 0.0149) |

Not distinguishable: the result does not depend on the choice of gradient-boosting library. F stays the final model.

## 10. Explanations (SHAP)

Exact TreeSHAP contributions from LightGBM, on all 228,187 validation rows, for the F model as logged in MLflow; contributions add up to the raw score within 1.9e-13 (cell 23).

**Most important features of F** (mean absolute contribution, share of the total): amount (27.9%), merchant category (21.1%), amount relative to the card's median (9.5%), hour (7.3%), time since the card's previous transaction (6.7%), night flag (6.2%), card transactions in the past 24 hours (5.9%), age band (4.4%).

**Coarse vs exact profile** (cell 24). Compared as groups, F's profile (age band + urban) carries 76% of the weight of A's exact age and city population, and 67% of A's full profile including income. Most of the remaining signal moves to amount, merchant category, hour and time since the previous transaction.

**Hour and night overlap** (cell 31). Together they carry 0.475, more than either alone (0.332 and 0.285) but 77% of their sum: each separate share understates their joint effect, and adding the two overstates it.

**Four local cases**, chosen by rules fixed before looking (cells 25–26):

| Case | Amount, time | Rank (of 228,187) | What drove the score |
|---|---|---|---|
| a. Caught, highest amount | $1,319.97, 02:00 | 834 | amount, 38.7× the card's median, night |
| b. Caught, lowest amount | $5.60, 01:00 | 828 | category `grocery_net`, night; no burst of card transactions, so not card testing |
| c. Missed, highest amount | $1,117.27, 14:00 | 1,198 | amount pushed it up; daytime pushed it down, just outside 1,000 alerts |
| d. Best-ranked false alarm | $284.77, 03:00 | 538 | amount, category `grocery_pos`, night; the 537 alerts above it were all fraud |

## 11. Limitations

1. **Daytime fraud is caught less.** At 1,000 alerts, recall is 0.651 for daytime fraud vs 0.883 at night, a difference of 0.232 (interval 0.137 to 0.313) (cell 31). 830 of the 979 validation fraud cases happen at night, so the model has learned the night pattern well; daytime fraud such as local case c is ranked lower.
2. **Gender gap.** Women's fraud is caught less and women get more false alarms (section 8), although gender is not a model feature.
3. **Simulated data.** The transactions come from a generator; patterns such as the strong night effect may be cleaner than in real card data, and results may not transfer.
4. **No cost data.** Without the cost of investigating an alert, the threshold is capacity-based (about 12 alerts per day), not cost-based; dollar amounts are the transaction amounts, not losses after recovery.
5. **Small number of fraud episodes.** Validation has 102 fraud cards, so intervals are wide and small groups cannot be judged.
6. **In-sample calibrator.** The final calibrator was fitted on the same validation rows it is evaluated on; the test period is the honest check.

## 12. Test results
