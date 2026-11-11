# Step-up verification: experiment design and causal methods (Phase 5)

> **Every treatment effect in this document is a SIMULATED ASSUMPTION, not a measurement.** No step-up was ever shown to a customer. The value of this phase is the design and the analysis pipeline, not a finding about real customers.

Every number below comes from `notebooks/05_causal.ipynb` ("cell N" refers to that notebook). The reusable logic is in `src/experiments/` (`ab_design.py`, `simulation.py`, `observational.py`), tested in `tests/test_experiments_*.py`.

## 1. The question

The fraud model F (Phase 4) scores every transaction. Instead of only sending the riskiest transactions to analysts (about 12 per day, capacity-bound), the bank could ask the customer for an extra verification step ("step-up", for example a one-time code) whenever F's score is above a threshold. Step-up is automated, so its threshold is not limited by analyst capacity.

The trade-off: a step-up blocks some fraud, but some legitimate customers give up (abandon the purchase). The experiment measures both.

## 2. Setting

- **The model is frozen.** F (LightGBM, 16 features) is loaded exactly as logged in MLflow, read only. It is not retrained or changed, no MLflow run is created or modified, and the SHA-256 of `mlflow.db` is identical before and after the notebook (cell 20).
- **Periods** (cell 3):

| Period | Dates | Transactions | Fraud transactions | Active cards | Fraud cards |
|---|---|---|---|---|---|
| Pre-period | 2020-07-01 to 2020-09-30 (92 days) | 244,140 | 1,076 | 917 | 111 |
| Experiment period | 2020-10-01 to 2020-12-31 (92 days) | 281,521 | 936 | 915 | 94 |

- **Why these periods are safe.** F was trained up to 2020-06-23, so its scores are out of sample in both periods. The experiment period is the Phase 4 test period, which was opened once and reported; Phase 5 only uses it as material for simulation, and no decision about F is taken here.
- **The calibrator.** The pre-period overlaps the validation period on which the Platt calibrator was fitted. This notebook uses F's raw score, and that calibrator is close to the identity (a = 0.995, b = −0.086), so the effect is negligible, but it is stated.

## 3. The design

| Item | Decision | Rejected alternative, and why |
|---|---|---|
| Randomisation unit | **The card**, by an independent coin flip (probability 0.5) | The transaction: one fraud episode spans several transactions of a card, which would end up in both arms (contamination), and a card's transactions are not independent |
| Analysis unit | The card: per-card metrics are plain means over cards, so **no delta method is needed**. Only the pooled abandonment rate (a ratio of sums) uses the delta method | — |
| Trigger and threshold | Step-up when F's score is at least **0.02**; other thresholds as sensitivity only | 0.1: the break-even table (section 7) already answers it (step-up pays even if every stepped-up customer abandons); a three-arm design: 1.5× the cards and a multi-arm correction, for a question the break-even table answers |
| Population | **Triggered cards**: a card enters at its first transaction with F score ≥ 0.02, computed identically in both arms from the logged scores (the control arm logs the score without showing a step-up). Secondary: all enrolled cards (intention to treat) | All enrolled only: dilutes the effect with cards that are never stepped up |
| Primary metric | **Fraud loss per card** (the single confirmatory test), two-sided Welch test, α = 0.05 | Mann–Whitney: with about 90% zeros it tests a shift in the distribution, not mean dollars |
| Decision metric | **Total loss per card** = fraud loss + m × abandoned legitimate dollars, m = 1.0 (the full transaction value counts as lost; m = 0.25 and 0.02 as sensitivity) | — |
| Guardrails | One-sided harm tests, Holm-adjusted over two, α = 0.05. **G1**: lost legitimate revenue per card, breached if the increase exceeds 5% of the control arm's mean legitimate dollars stepped up per card. **G2**: abandonment rate of stepped-up legitimate transactions (pooled, delta method), breached if above 10% | Bonferroni across all metrics: needlessly weakens the primary |
| Validity check | Sample-ratio mismatch (SRM): chi-square on cards per arm; p < 0.001 makes the experiment invalid | — |
| Horizon | **Fixed 92 days**, no interim looks | Repeated looks: inflate false positives (section 6) |

**Decision rule (pre-registered in cell 2, before any simulated result).** **Ship** only if there is no SRM (p ≥ 0.001), the 95% interval of (control − treatment) total loss per card has a lower bound above 0, and no Holm-adjusted guardrail is breached. **Don't ship** otherwise. **Extend** only if enrolment fell short of the plan with no SRM and no breach.

The guardrail margins (5% for G1, 10% for G2) were set by the analyst when the rule was written; they are design choices, not derived from data.

## 4. Thresholds

Experiment period, 92 days, 915 active cards (cell 6):

| F score threshold | Step-ups per day | Fraud tx stepped up | Share of fraud $ | Legit tx per day | Legit $ stepped up | Share of legit $ | Cards touched |
|---|---|---|---|---|---|---|---|
| 0.5 | 9.1 | 759 | 89.7% | 0.9 | $63,116 | 0.33% | 155 |
| 0.294 (analyst threshold) | 10.5 | 800 | 93.6% | 1.8 | $112,407 | 0.59% | 213 |
| 0.1 | 14.6 | 860 | 96.9% | 5.3 | $230,354 | 1.21% | 360 |
| **0.02 (primary)** | **27.8** | **899** | **99.4%** | **18.1** | **$640,485** | **3.37%** | **590** |
| 0.005 | 52.3 | 914 | 99.8% | 42.3 | $1,207,784 | 6.35% | 768 |

Every fraud card is triggered even at 0.5: F separates fraud very well on this simulated data.

## 5. Sample size

Per-card fraud loss is zero-inflated and heavy-tailed: over the 915 active cards, mean $535.40, standard deviation $1,722.67, only 10.3% non-zero (cell 4). Cards per arm for a two-sided Welch test, α = 0.05, power 0.8 (cell 7):

| Population | Cards | Mean $ | MDE with these cards | 5% effect | 10% | 20% | 50% |
|---|---|---|---|---|---|---|---|
| All enrolled | 915 | 535.40 | 60% | 65,004 | 16,251 | 4,063 | 650 |
| Triggered (0.02) | 590 | 830.33 | 58% | 39,706 | 9,927 | 2,482 | 397 |
| All enrolled, rescaled to 1% incidence | — | 52.12 | — | 725,300 | 181,325 | 45,331 | 7,253 |
| All enrolled, rescaled to 0.5% incidence | — | 26.06 | — | 1,456,879 | 364,220 | 91,055 | 14,569 |

- **What 915 cards can and cannot detect.** Only a reduction of about 60% of mean fraud loss. That is enough here, because a step-up at 0.02 touches 99.4% of fraud dollars, so a simulated block rate of 50% means a loss reduction of about half.
- **Bank scale.** This simulator gives 10.3% of cards a fraud episode per quarter, far more than a real bank. With the same loss sizes but 1% incidence, a 10% effect needs 181,325 cards per arm; at 0.5%, 364,220.
- **Triggering at 0.02 saves little.** 64% of cards trigger, so 9,927 triggered cards per arm correspond to about 15,400 enrolled cards, against 16,251. Triggering pays more at higher thresholds.
- **The formula is slightly conservative.** Resampling the real per-card values at the formula's sample size gave power 0.87 (as-is) and 0.85 (1% incidence) instead of 0.80, because the simulated effect also shrinks the treatment arm's variance.

## 6. Checks of the pipeline

- **Reconciliation** (cells 4–5, stop on any mismatch). Deterministic values matched exactly: F's test recall 0.8846 and dollar recall 0.9494; the threshold rows; the per-card mean, SD and non-zero share; 16,251 cards per arm; CUPED 1.9% and 15.1%. Simulated rates agreed within 3 Monte Carlo standard errors (the design script used different random draws): A/A 0.052 vs 0.045; peeking 0.191 vs 0.211; fixed horizon 0.048 vs 0.056.
- **CUPED** (cell 8). Variance reduction is the squared correlation between a pre-period covariate and the experiment metric:

| Population | Metric | Pre-period covariate | Correlation | Variance reduction |
|---|---|---|---|---|
| All enrolled | Fraud loss (primary) | max F score | −0.138 | 1.9% |
| Triggered (0.02) | Fraud loss (primary) | max F score | −0.227 | 5.2% |
| Triggered (0.02) | Legit $ stepped up (guardrail base) | the same metric | +0.383 | 14.7% |
| Triggered (0.02) | Legit spend (secondary) | the same metric | +0.931 | 86.8% |

  The primary gains little, and the correlation is *negative*: each card has a single fraud episode, so a card with fraud in the pre-period almost never has fraud again. CUPED is used for guardrail and secondary metrics only.
- **Full-pipeline A/A** (cell 9): 1,000 simulated experiments with no effect, through the real assignment, step-up and analysis code. False-positive rate 0.050 on triggered cards (required 0.035–0.065) and 0.059 on all enrolled cards; SRM never fired; the designated assignment split the cards 458 vs 457 (SRM p = 0.974).
- **Peeking** (cells 5 and 10). Stopping at the first p < 0.05 over 13 weekly looks gave a false-positive rate of 0.191, against 0.048 at the fixed horizon. As an illustration only, O'Brien–Fleming boundaries for 3 looks are z = 3.46, 2.45, 2.00.
- **Novelty effect** (cell 15). Abandonment by week since a card's first step-up is flat, **by construction**: the simulator applies the same rate every week. This is part of the design (in a real experiment: compare early and later weeks, and exclude each card's first 14 days as a sensitivity analysis), not a finding.

## 7. Break-even abandonment rate

The abandonment rate at which blocked fraud dollars equal lost legitimate revenue: b × fraud $ stepped up ÷ (m × legitimate $ stepped up), where b (share of stepped-up fraud blocked) and m (share of an abandoned transaction counted as lost) are assumptions (cell 11).

| Threshold | b = 50%, m = 100% | b = 65%, m = 100% | b = 80%, m = 100% | b = 65%, m = 25% | b = 65%, m = 2% |
|---|---|---|---|---|---|
| 0.5 | 348% | 453% | 557% | 1,810% | 22,630% |
| 0.294 | 204% | 265% | 326% | 1,060% | 13,254% |
| 0.1 | 103% | 134% | 165% | 536% | 6,695% |
| **0.02** | **38%** | **49%** | **61%** | 198% | 2,470% |
| 0.005 | 20% | 26% | 32% | 105% | 1,316% |

Above 100% means step-up pays even if every stepped-up legitimate customer abandons. Only at thresholds of 0.02 and below does the trade-off become real, which is why 0.02 is the primary threshold.

## 8. Simulated experiments (SIMULATED effects)

Threshold 0.02, triggered cards (590 per experiment), m = 1.0, 500 simulated experiments per scenario; dollars per triggered card (cell 13):

| b (fraud blocked) | a (legit abandoned) | True fraud reduction | Mean estimate | Primary power | True total-loss reduction | G1 breach | G2 breach | Ship rate | Decision |
|---|---|---|---|---|---|---|---|---|---|
| 50% | 1% | 412.50 | 419.18 | 0.88 | 401.65 | 0.00 | 0.00 | 0.86 | ship |
| 50% | 5% | 412.50 | 405.86 | 0.89 | 358.22 | 0.00 | 0.00 | 0.71 | ship |
| 50% | 10% | 412.50 | 407.52 | 0.87 | 303.95 | 0.68 | 0.04 | 0.19 | don't ship |
| 50% | 20% | 412.50 | 399.99 | 0.85 | 195.39 | 1.00 | 1.00 | 0.00 | don't ship |
| 65% | 1% | 536.25 | 531.49 | 0.99 | 525.40 | 0.00 | 0.00 | 0.99 | ship |
| 65% | 5% | 536.25 | 538.11 | 0.99 | 481.97 | 0.00 | 0.00 | 0.97 | ship |
| 65% | 10% | 536.25 | 533.08 | 0.99 | 427.70 | 0.65 | 0.04 | 0.33 | don't ship |
| 65% | 20% | 536.25 | 533.28 | 1.00 | 319.14 | 1.00 | 1.00 | 0.00 | don't ship |
| 80% | 1% | 660.00 | 660.45 | 1.00 | 649.15 | 0.00 | 0.00 | 1.00 | ship |
| 80% | 5% | 660.00 | 664.61 | 1.00 | 605.72 | 0.00 | 0.00 | 1.00 | ship |
| 80% | 10% | 660.00 | 661.99 | 1.00 | 551.45 | 0.71 | 0.04 | 0.29 | don't ship |
| 80% | 20% | 660.00 | 661.44 | 1.00 | 442.89 | 1.00 | 1.00 | 0.00 | don't ship |

- **The estimates are unbiased**: the largest relative difference from the truth is 3.0%, within Monte Carlo error.
- **The guardrails decide**, not the primary metric: total loss falls in every scenario, but at 10% abandonment G1 is usually breached and at 20% both are.
- **All enrolled cards (intention to treat)** gives the same decisions, with similar power (for example 0.86 vs 0.88 at b = 50%, a = 1%) (cell 14).
- **m sensitivity** (b = 65%): with m = 0.25 or 0.02, 10% abandonment ships (G1 no longer breached), while 20% does not (G2, a rate, does not depend on m).
- **Other thresholds** (b = 65%, a = 5%): every threshold ships; the true fraud reduction per triggered card is larger at higher thresholds because fewer, riskier cards are triggered.

## 9. Observational methods against a known truth (SIMULATED)

**Why semi-synthetic.** With the real outcomes, pre-period risk is *negatively* related to fraud loss in the experiment period (correlation −0.129 on the 909 cards active in both periods), and only 88 of them have fraud. A rollout that targets riskier cards picks cards with *lower* loss: in 500 simulated rollouts on the real outcomes, treated cards' own loss without treatment averaged $312 against $582 for control cards, so the naive comparison claimed $471 of benefit against a true $321 (ATE) and $200 (ATT); inverse probability weighting on the targeting variable gave $319 (cell 17). The main analysis therefore uses real per-card covariates with a simulated outcome and a simulated rollout targeting riskier cards, so that both potential outcomes, and every estimand, are known exactly.

**Rejected alternative.** A "natural" causal question inside the simulated data (for example, does night cause fraud) has no ground truth, so the methods could not be checked.

**Overlap and balance** (one rollout of 20,000 cards, cell 18): 29.9% treated; propensity scores from 0.014 to 0.975 among treated and 0.009 to 0.940 among controls; 0.9% of treated and 6.9% of controls outside the 0.05–0.95 trimming range. Standardised mean differences before adjustment reach 0.893 (max F score); after weighting and after matching all are below 0.011. Gender, used by neither the rollout nor the outcome, is 0.010 before and 0.026 after weighting. Matching left 16 of 5,978 treated cards unmatched (caliper 0.255 on the logit scale = 0.2 SD of the logit of the propensity score).

**Bias, RMSE and coverage**, 200 rollouts of 20,000 cards per scenario, each estimator against its own estimand (cell 19; in dollars per card, positive = treatment reduces loss):

| Method | Estimand | Measured confounders: bias / RMSE / coverage | Hidden confounder: bias / RMSE / coverage | Randomised: bias / RMSE / coverage |
|---|---|---|---|---|
| Naive difference | ATE | −217.0 / 217.4 / 0.00 | −217.0 / 217.4 / 0.00 | 0.8 / 13.9 / 0.99 |
| Matching (caliper) | ATT, matched treated | 2.6 / 106.3 / — | −15.6 / 102.9 / — | 2.5 / 58.8 / — |
| IPW, trimmed | ATE, trimmed | 0.8 / 20.3 / 0.99 | −55.1 / 57.5 / 0.15 | 0.7 / 13.9 / 1.00 |
| IPW, ATT weights, trimmed | ATT, trimmed | 1.3 / 54.7 / 0.92 | −176.3 / 181.4 / 0.00 | 1.8 / 26.4 / 0.73 |
| Regression adjustment | ATE | −31.8 / 35.7 / — | −57.3 / 59.4 / — | 0.7 / 13.9 / — |
| Doubly robust | ATE | 0.2 / 20.8 / 0.99 | −61.0 / 63.4 / 0.08 | 0.7 / 13.9 / 1.00 |

- **Each estimator has its own truth.** The true ATE is $358 on average, the true ATT about $610: the effect is larger on the cards the policy targets. Trimming changes the estimand too ($371 on the trimmed population vs $358 overall).
- **With all confounders measured**, weighting and doubly robust estimation recover the truth; the naive difference understates the benefit by $217 (the rollout picks riskier cards); linear regression adjustment is biased by −$32, because a linear model does not fit a zero-inflated outcome with a heterogeneous effect.
- **With one confounder hidden** (the card's night-time share, which drives both the rollout and the outcome), every adjusted method is biased again, the doubly robust estimate by −$61, and the intervals no longer cover the truth (8% for doubly robust). No adjustment can fix a confounder it cannot see. This is the reason randomised experiments are preferred: randomisation balances unmeasured variables too.
- **Matching** is reported with bias and RMSE only. A naive bootstrap is invalid for matching estimators (Abadie & Imbens 2008), and the Abadie–Imbens standard error for an estimated propensity score is out of scope.
- **The IPW standard error with ATT weights under-covers** (0.73 under randomisation, 0.92 with measured confounders). It treats the propensity score as known, which is conservative for the ATE but not for the ATT. Its intervals should not be used.
- **Coverage above 95%** for the naive difference under randomisation is expected: the interval targets the population effect, while the truth used here is the effect in the simulated sample.

## 10. Limitations

1. **Every effect is simulated**: the block rate, the abandonment rate and the share of an abandoned purchase counted as lost are assumptions.
2. **Friction is not priced**: legitimate customers who complete the step-up also pay a cost (time, annoyance), which is not in the total loss.
3. **Interference and fraud rings are not identifiable**: a fraudster blocked on a treatment card may move to control cards, which would overstate the effect; the data has no link between cards.
4. **Simulator properties**: the fraud incidence (10.3% of cards per quarter), F's separation and the single fraud episode per card are properties of the data generator; real data would put more legitimate transactions above the threshold and make the guardrails bind more often.
5. **No reaction to the treatment**: the history is replayed unchanged (no novelty effect, no fraudster adaptation, no change in later transactions).
6. **The guardrail margins** (5% and 10%) are design choices.
7. **Bank-scale sizes** come from resampling 915 cards; the rescaled incidences (1%, 0.5%) assume real loss sizes stay the same.

## 11. Reproducibility

- Master seed 42; every simulation uses a fixed seed derived from it. Re-running the notebook gives identical results.
- Run with `.venv/Scripts/python.exe -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 notebooks/05_causal.ipynb`; the notebook prints the interpreter path, the git commit and the SHA-256 of `mlflow.db` before and after.
- The executed notebook ran in 0.7 minutes. Tests: `tests/test_experiments_ab_design.py`, `test_experiments_simulation.py` and `test_experiments_observational.py`, with positive controls (a known simulated effect is recovered, an A/A is not significant, a hidden confounder leaves bias).
- Nothing is logged to MLflow: the results are simulations with fixed seeds, kept in the notebook and in this document.
