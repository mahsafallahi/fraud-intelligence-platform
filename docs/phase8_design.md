# Phase 8 – Analyst layer and live system: design

Status: **design only** (2026-11-27). Nothing in this document is built yet. No package
was installed, Docker was not started, and no implementation code was written.
**Revised the same day** with the user's decisions on the eight open questions and five
review fixes (a–e); see "Decisions of 2026-11-27" at the end. Rejected alternatives stay
visible in each item.

Numbers marked **[measured]** were taken on 2026-11-27 with throwaway read-only scripts
(DuckDB `read_only=True` on `data/gold/fraud_dev.duckdb` and `data/serving/serving.duckdb`,
the committed bundle `models/F`). Those scripts **never loaded a test-period row**
(`trans_ts >= 2020-10-01`), except one label-free row count. Round 1 re-measures every
one of these numbers with committed code and must reproduce it exactly (stop
condition S1, §15).

**Deliberately not measured now:** anything computed on test-quarter scores or labels
(time-to-detection on test, test PSI, red-team results). Measuring them before the
thresholds and rules are pushed would be choosing thresholds after seeing the outcome.

Sanity check of the measurement path: scoring the validation period with the bundle
gives exactly **1,000 alerts** and **979 fraud rows** on **228,187** rows, and **102 fraud
cards** (55 women, 47 men). These are the published Phase 4 numbers.

---

## 0. Ground rules for the whole phase

| # | Rule | Reason |
|---|---|---|
| G1 | **Model F, its threshold (raw probability ≥ 0.2940968658100669), features and bundle `be26ca88…` do not change.** Any new model is a challenger in its own bundle. | The Phase 4 and Phase 6 claims stay valid. |
| G2 | **The test quarter (2020-10-01..2020-12-31) has been opened once** (Phase 4, aggregate metrics). Phase 8 reuses it as an *operations simulation*, not as a new unbiased estimate of F. Any number reported on it says so. | The user calls it "untouched": its *timeline* is untouched by the replay (no shifted timestamps), but its labels were used once. |
| G3 | **Every threshold, rule and seed in §14 is committed and pushed before the first replay, red-team or test-TTD run.** Scripts refuse to run when the pre-registration file's SHA-256 differs from the pushed value. | Same pattern as Phase 4 (`33212a1`) and Phase 7 (C7). |
| G4 | **Docker Desktop is never started without the user's approval.** The primary design needs no Docker. | Docker Desktop can restart the unrelated `influencer-main` containers. |
| G5 | `fraud.duckdb` (Airflow's file) is never opened. Phase 8 reads `fraud_dev.duckdb` and the serving snapshot, read-only. | Existing two-file rule. |
| G6 | One writer per file: the replay engine is the only writer of `data/replay/replay.duckdb`; the dashboard opens it read-only; analyst feedback goes to a separate append-only file. | DuckDB allows one writer per file. |
| G7 | No coordinates, `card_hash`, `card_key`, amounts or times in logs; the API's logging rule extends to every new process. | Existing privacy rule. |

---

## 8a. Analyst layer

### 1. Hand-written SQL exercises + automatic checker

**What it shows:** window functions and set logic on the Gold star schema, written by the
user without AI; a checker proves each answer correct against an independent Python
reference.

**Authorship rule:** the SQL answers in `sql/answers/q01.sql`..`q08.sql` are **written only
by the user**. Claude never creates, edits or suggests text for files in `sql/answers/`.
Claude writes the question specs, the Python reference answers and the checker.

**Inputs:** the five mart tables of `fraud_dev.duckdb` only (staging views use relative
paths and resolve only from `src/dbt`). All periods (2019-01-01..2020-12-31): this is
retrospective analytics, not model evaluation.

**Proposed exercises** (each spec fixes output columns, order and tie rules, so the answer
is unique):

| # | Exercise | SQL technique | Known control value |
|---|---|---|---|
| Q1 | **Fraud episodes per card (gaps-and-islands):** a card's fraud transactions in time order; a new episode starts when the gap to the previous fraud transaction is **more than 24 hours**. Output `card_key, episode_no, start_ts, end_ts, n_txn, fraud_usd`. | `LAG`, running `SUM` of a "new island" flag | Pre-test: **910** episodes with 24 h, **883** with 3, 7 or 30 days (= one episode per fraud card) **[measured]**. The 24 h gap is proposed because a 7-day gap makes the exercise trivial. |
| Q2 | **Time since the card's previous transaction** (seconds) for every transaction, ordered by `trans_ts, trans_num`; then the median by `is_fraud`. | `LAG` with a deterministic order | Same-second pairs must give 0 s, not NULL (spec). |
| Q3 | **Running fraud loss within an episode** (Q1's episodes): cumulative fraud dollars per transaction. | `SUM() OVER (PARTITION BY card, episode ORDER BY ts ROWS UNBOUNDED PRECEDING)` | Last row of each episode = Q1's `fraud_usd`. |
| Q4 | **Top-3 risky categories per month** by fraud rate; ties by fraud dollars, then category name. | `ROW_NUMBER` / `QUALIFY` | 24 months × 3 = 72 rows. |
| Q5 | **Rolling 24-hour transaction count per card**, strictly earlier transactions only, same-second rows excluded. | `RANGE BETWEEN INTERVAL 24 HOURS PRECEDING` + tie handling | Must equal the model feature `card_txn_24h` from `build_features` **exactly** on every row: the reference is the production feature code itself. |
| Q6 | **Hour × amount band fraud rate** (bands `<20, 20–50, 50–100, 100–136.33, 136.33–300, 300–500, 500–1000, ≥1000`), plus the precision and recall of the baseline rule (night and amount > $136.33). | `CASE`, conditional aggregation, `PIVOT` | Baseline rule recall/precision on validation: 0.6425 / 0.1079 (`docs/model_results.md` §5). |
| Q7 | **Merchant names in more than one category**, with both categories and their fraud rates. | `GROUP BY … HAVING`, self-join | **7** names (status doc §8). |
| Q8 | **State risk with a Wilson 95% interval**, computed in SQL (fraud transactions / transactions), flagged "too few" when the state has fewer than 10 fraud cards. | Arithmetic in SQL, `sqrt` | Must equal the Python Wilson function to 1e-12 (used again by Power BI, item 3). |

**Checker (built by Claude):**

| File | Role |
|---|---|
| `sql/exercises/README.md` | The 8 specs: question, exact output columns and types, ordering, tie rules |
| `sql/answers/` | **User's files only** (empty `.gitkeep` from Claude) |
| `src/analyst/reference.py` | Pure pandas reference answers, one function per exercise; Q5 calls `build_features` |
| `scripts/sql_check.py` | Runs each user query read-only, compares with the reference, prints PASS / FAIL / NOT ATTEMPTED and the first differing row |
| `tests/test_analyst_reference.py` | References on tiny hand-made frames with hand-computed answers (gaps, same-second ties, empty months) |
| `tests/test_sql_check.py` | The comparer: column names, row order, int/string exact, float relative tolerance 1e-9; a deliberately wrong query must FAIL (positive control) |

Comparison rule: same column names and order; rows compared after sorting by the spec's
key columns (or in the spec's order when order is part of the answer); integers and
strings exact; floats with relative tolerance 1e-9 (floating-point sums in DuckDB and
pandas can differ in the last bits).

**Decisions:**

| Decision | Rejected alternative (reason) |
|---|---|
| Reference answers in pandas, not SQL | A second SQL file could share the user's mistake pattern and would be AI-written SQL next to the user's |
| Q5 checked against `build_features` | A separate reference would test SQL against another copy, not against the code that feeds the model |
| Checker runs on `fraud_dev.duckdb` mart tables | Staging views break outside `src/dbt` |

**Who:** specs, references, checker, tests: Claude. SQL answers: **user**.

### 2. Time-to-detection

**What it shows:** how many fraud transactions and dollars an episode costs before the
first alert, with F at the frozen threshold.

**Definitions (pre-registered, §14):**
- Episode = Q1's rule with a **7-day** gap (one episode per fraud card on this data
  **[measured]**; max duration 69.32 h = 2.89 days, 2–19 fraud transactions, median 10).
- An episode belongs to a period when its **first fraud transaction** falls in it.
- **First alert** = the first transaction of that card **at or after the episode's first
  fraud transaction** with raw probability ≥ threshold (any label: an alert on a legit
  transaction of the card would also start an investigation).
- **Detected** = the first alert is no later than the episode's last fraud transaction.
- **Lost before detection** = fraud transactions and dollars with `trans_ts` strictly
  before the first alert. The alerted transaction itself is reported separately (an alert
  does not decline the payment; it starts a review).
- **Hours to detection** = first alert time − first fraud time.
- Undetected episodes: all their fraud dollars count as lost.
- Reported: share detected, median / p90 / max of fraud transactions before the alert,
  total and median dollars before the alert, hours median / p90; card-bootstrap 95%
  intervals (500 draws, seed 42).
- Detection means "alert raised", not "card blocked": analyst capacity and queue delay
  are item 9's what-if, not part of this metric.

**Reconciliation: 102 fraud cards but 101 episodes in validation (review fix a) [measured].**
Phase 3's finding holds: every fraud card has exactly **one** episode (883 episodes = 883
fraud cards before the test period, with a 3-, 7- or 30-day gap). The difference is a
**boundary case, not a second episode**: one card's episode starts on **2020-06-30 at
00:03:56**, in the 7-day gap before validation, and ends on **2020-07-01 at 15:21:14**. Of its
11 fraud rows, 9 are in the gap and 2 in validation. So 102 cards have fraud rows in
validation, but only 101 episodes *start* there. The gap still does its job: no episode
touches both train (ends 2020-06-23) and validation, because episodes last at most 2.89
days and the gap is 7 days. Its two validation rows are counted in Phase 4's 979 fraud
rows. Three validation episodes also run past 2020-09-23 into the September gap (12, 12
and 15 fraud rows in total). By the membership rule above (the first fraud transaction
decides), the June boundary episode belongs to the gap and is left out of the
validation TTD; the three September ones stay in, with their gap rows included.
Rejected: assigning an episode to every period it touches (one episode would be counted
twice).

**Validation [measured]:** **98 of 101 detected.** 84.7% of detected episodes are caught on their
**first** fraud transaction; median 0, p90 1, max 3 fraud transactions before the alert.
**$4,069.90** lost before the first alert across all detected episodes, plus **$3,076.96**
in the 3 undetected episodes, out of $527,034.25 of fraud in these episodes. Hours to
detection: median 0.00, p90 1.27. Every first alert was on a fraud transaction.

**Test:** computed **once**, in round 2, after the definitions are pushed (G3), with the
reconciliation check that F gives exactly 969 test alerts (stop condition S4).

**Files:** `src/monitoring/ttd.py` (pure function on a scored frame:
`card_key, trans_ts, amt, is_fraud, alert`), `scripts/ttd_report.py`,
`tests/test_monitoring_ttd.py` (hand-made episodes: detected on first txn, detected late,
undetected, alert on a legit txn of the card before and inside the episode, same-second
rows).

**Decisions:**

| Decision | Rejected alternative (reason) |
|---|---|
| Alert on any transaction of the card counts | Only alerts on fraud transactions: an analyst reviewing an alerted legit payment would still find the card compromised; the validation result is the same (every first alert was fraud) |
| "Strictly before" as the primary loss | Including the alerted transaction: depends on whether the bank blocks at alert time, which this project does not model; reported as a secondary column |
| 7-day episode gap | 24 h: cuts some cards' single episode into pieces (910 islands vs 883 fraud cards pre-test) and would count re-detections |

**Who:** Claude.

### 3. Power BI report on the star schema

**What it shows:** the analyst-facing view of two years of transactions and of the model's
results, for a non-technical audience.

**Gate:** Power BI Desktop is installed by the user. Everything below except building the
pages can be prepared without it.

**Export (built by Claude):** `scripts/export_powerbi.py` writes Parquet (zstd) to
git-ignored `data/powerbi/`:

| File | Content | Size |
|---|---|---|
| `fact_transactions.parquet` | All 1,852,394 rows; **drops** `merch_lat`, `merch_long` | **83.7 MB** with the coordinates **[measured]**; smaller without them (re-measured in round 1) |
| `dim_card.parquet` | 999 rows; **drops** `card_hash` and `job`; `birth_year` replaced by `age_band` | 61 KB with all columns [measured] |
| `dim_merchant.parquet`, `dim_date.parquet` | As in Gold | 22 KB, 7 KB [measured] |
| `dim_location.parquet` | 985 ZIPs, all columns (ZIP centroids are public Census Gazetteer data, not customer coordinates) | 62 KB [measured] |
| `fact_scores.parquet` | Validation + test rows: `trans_num`, F raw probability, alert flag, baseline rule flag | new |
| `state_risk.parquet` | Per state: transactions, fraud, cards, fraud cards, fraud rate, **card-cluster bootstrap** 95% bounds, `too_few` flag (and the Wilson bounds as a labelled comparison column) | new |
| `reference_numbers.json` | One expected number per visual, computed in Python (for the user to reconcile the report) | new |

**Pages:**

| Page | Visuals |
|---|---|
| 1. Monthly trends | Transactions, fraud rate, fraud dollars by month (24 months), the validation and test periods shaded |
| 2. Risky categories | Fraud rate and fraud dollars by category and channel; top-3 per month (= SQL Q4) |
| 3. Hour × amount | Heat map of fraud rate, hour (0–23) × amount band (= SQL Q6 bands); the baseline rule's cell outlined |
| 4. Rule vs model | Validation and test: alerts per day, recall, dollar recall, precision for the baseline rule and F (numbers equal `docs/model_results.md` §5 and §12) |
| 5. State risk map | **Shape Map** visual with its built-in US states map (works offline); fraud rate per state, card-cluster bootstrap 95% interval in the tooltip; states with fewer than 10 fraud cards hatched as "too few to judge" |

**Test-quarter labels are included** (user decision 2026-11-27), clearly marked: every
page shows the period as a column/legend (`train`, `gap`, `validation`, `test`), the test
quarter is shaded, and page 4 carries the note "test quarter: opened once in Phase 4; shown
here retrospectively, not a new evaluation".

**State-level facts [measured, pre-test rows]:** 51 states (incl. DC); **16 states have
fewer than 10 fraud cards**, 6 states fewer than 5 cards; cards per state min 1, median 17,
max 73. So most state differences are noise, and the map must say so.

**Decisions:**

| Decision | Rejected alternative (reason) |
|---|---|
| **Card-cluster bootstrap** interval for the state fraud rate (review fix b): resample whole cards **within each state** (a card lives in one state: its address never changes, PROJECT_STATUS §7), recompute fraud transactions / transactions, 500 draws, seed 42, percentile interval; plus the "fewer than 10 fraud cards" flag (Phase 4's rule) | *Original proposal, rejected in review:* Wilson interval on transactions, because it treats a card's transactions as independent and is too narrow. Card-level fraud rate: 883 of 993 pre-test cards have fraud, so it is ~0.89 everywhere. Known limit kept visible: with 1–4 cards a state's bootstrap is close to degenerate, which is why the `too_few` flag stays and those states show no interval |
| Intervals precomputed in Python and exported | DAX: harder to test; one tested implementation is reused. SQL Q8 keeps the Wilson formula as an SQL exercise, reported next to the bootstrap as a comparison, not as the map's interval |
| Parquet import, coordinates and `card_hash` dropped at export | DirectQuery on DuckDB: needs an ODBC driver install; full columns: personal-data rule (coordinates never leave Silver/serving) |
| Map: **Shape Map visual with its built-in US states map** (user decision; works offline) | Azure Maps / Filled Map / ArcGIS: online map services, not allowed (user decision); a custom TopoJSON: unnecessary when the built-in US map exists |
| Report saved as a **Power BI project (PBIP, text files)** if the installed version supports it; only the definition is committed, never the data cache | `.pbix`: binary with the imported data inside (≈ the 84 MB fact table) |

**Can be prepared without Power BI:** export script and tests, Parquet files, data
dictionary, measure definitions (DAX text in `docs/powerbi_report.md`), page specs,
`reference_numbers.json`, the state-name column in the form the Shape Map's US map keys on
(checked in Desktop). **Needs Power BI:** building the pages, PBIP save, screenshots.

**Tests:** `tests/test_export_powerbi.py`: dropped columns absent (positive control: the
check fails on a frame that has them), row counts equal Gold, Wilson against known values
(e.g. 0 of 10, 5 of 10, 1 of 1,000), card-cluster bootstrap on a hand-made state (all cards
identical → zero-width interval; one card → flagged, no interval; same seed → same bounds),
`too_few` boundary at 9/10 fraud cards, every exported row carries its period label.

**Who:** export, specs, reference numbers: Claude. Install, pages, PBIP: **user**.

### 4. One-page decision memo without jargon

**Audience (user decision): a fraud-operations manager.** Rejected: a product manager (the
Phase 5 memo's reader; this memo is about running the alert queue, not a product change).

**What it shows:** for that manager: what the system catches, how fast, what an
adaptive fraudster can do to it, what analyst capacity buys, and a recommendation.

**File:** **`docs/decision_memo_phase8.md`** (new). `docs/decision_memo.md` is the Phase 5
memo (extra verification step) and stays unchanged.

**Content (one page):** the question; three facts with numbers (catch rate and fraud
dollars at ~12 alerts/day from `model_results.md` §12; time to detection; the red-team
drop); what more analysts would buy (item 9 what-if); risks (daytime fraud and women's
fraud caught less, simulated data, no cost data); recommendation and what would change it.
Rules: no metric names without a plain sentence ("of every 100 fraud payments, the system
flags 88"), every number traceable to a results file, simulated data stated in the first
paragraph.

**Decision:** written **last** (round 3), from final numbers only. Rejected: drafting it now
(it would contain guessed numbers).

**Who:** Claude drafts; **user** edits and approves.

---

## 8b. Live system

### 5. Replay of the test quarter on a simulated clock

**What it shows:** the model operating day by day as if in production: transactions arrive
in time order, labels arrive later, monitoring and retraining run on a schedule.

**Inputs:** test-quarter transactions from the serving snapshot's source (Gold
`fraud_dev.duckdb`, read-only): 281,521 rows **[measured, label-free count]**, scored through
the Phase 6 scorer in process (`load_bundle`, `SnapshotStore`, `score_transaction`), exactly
as the MCP server does.

**Timestamps are not shifted.** The scorer receives the original 2020 `trans_ts`. The
simulated clock decides only *when* an event is processed and when a label becomes
visible.

**Alert rule in the replay (review fix d): the frozen threshold, raw probability ≥
0.2940968658100669** (`models/F/threshold.json`, rule `>=`, applied to `probability_raw`),
decided per transaction at the moment it is scored. **No top-N budget is used anywhere in
the replay**: a real-time scorer cannot rank a quarter it has not seen yet (Phase 6
decision; 1,082 was an offline comparison budget only). The capacity of 12 alerts per day
is applied only *after* alerting, as the analysts' FIFO review rate (backlog), never to
decide which transactions alert. A challenger alerts at the threshold frozen in its own
bundle (item 7). Test: the replayed champion's alert set equals `probability_raw >=
threshold` row by row and totals exactly 969.

**Event types, processed in simulated-time order:**

| Event | Time | Effect |
|---|---|---|
| Transaction | its `trans_ts` | Scored by the current champion (and, in shadow, by the challenger); result row written; an alert enters the analyst queue |
| Label release | `trans_ts` + **14 days** (primary; 30 days in the sensitivity run) | `is_fraud` becomes visible to monitoring and retraining |
| Daily monitoring | 00:05 the next simulated day | PSI, alerts vs capacity, calibration on labels released so far, drift alarms (item 6) |
| Retraining | 2020-10-15 00:00 and 2020-11-15 00:00 | Challenger trained on labels released so far (item 7) |
| Promotion decision | 2020-11-15 00:00 and 2020-12-15 00:00 (primary); 2020-12-01 00:00 and 2020-12-31 23:59:59 (30-day sensitivity) | Pre-registered rule (item 7) |

**Label delay (user decision 2026-11-27): 14 days is the PRIMARY run; 30 days is a
pre-registered SENSITIVITY run, reported alongside.** With 14 days, two complete cycles
(train → shadow → decide) fit inside the replay, each decision with 16–17 days of labelled
shadow data. With 30 days the same retraining dates leave later decisions (2020-12-01 and
2020-12-31, the second one at the very end of the replay, so its promotion would serve no
replay day); this run shows how a slower label feed delays every label-based signal.
Real chargebacks can take longer than either value; the delay is an assumption, stated in
every result. Both runs use the same transactions, scores of F and rules; only the delay
and the decision dates differ (§14). Rejected: 0 days (labels at transaction time do not
exist in practice); 30 days as the only run (only one late decision); analyst feedback as
labels (see item 9: selection bias).

**Speeds:**

| Speed | Pace | Use |
|---|---|---|
| `live` | 1 simulated second per real second, for a chosen window (e.g. one evening) | Showing a stream arriving in the dashboard |
| `demo` | As fast as the per-transaction scorer allows, optional pacing of N real seconds per simulated day | Full-quarter run through the real serving path. Expected ≈ 3.1 h for the quarter (281,521 × 39.3 ms p50 in process, `api_design.md` §3), about 1–3 min per simulated day (1,465–4,144 transactions per day in validation [measured], median 2,330) |
| `batch` | One `build_features` call per simulated day over history + that day | Tests, red-team, shadow scoring of challengers. Pre-test, `build_features` takes 3.5 s and bundle prediction 6.2 s for 1.57 M rows [measured] |

`batch` and `demo` must give **identical** result rows (raw score, probabilities, alert,
reasons) on three fixed days (stop condition S3). This is the Phase 6 parity argument
applied to the replay.

**Plain-Python design (primary, no Docker):**

| File | Role |
|---|---|
| `src/replay/clock.py` | `SimClock(start, speed)`: simulated now, sleep until the next event (or not, in `batch`) |
| `src/replay/events.py` | Builds the ordered event stream (transactions, label releases, scheduled jobs); pure, testable |
| `src/replay/engine.py` | Processes events, calls the scorer and the jobs, writes results; checkpoint per simulated day so a stopped run resumes at the last finished day |
| `src/replay/store.py` | `data/replay/replay.duckdb` (git-ignored): `scores`, `labels_released`, `monitoring_daily`, `alarms`, `models`, `promotions`, `replay_state`; the engine is the only writer |
| `scripts/replay.py` | CLI: `--speed live|demo|batch --start --end --resume`; refuses to start unless the pre-registration hash matches (G3) |

Scheduler: a simple loop over the event stream with `time.sleep` (standard library).
Rejected: APScheduler (a dependency for what a sorted event list does; wall-clock
schedulers do not run on a simulated clock); cron/Task Scheduler (no simulated time).

**Airflow variant (user decision: design only in Phase 8, built in Phase 9; starting
Docker needs the user's approval):** DAG `replay_daily` with a daily data interval
from 2020-10-01 to 2020-12-31 and catch-up (one run per simulated day, the same pattern as
`bronze_transactions_daily`), whose tasks call the same CLI entry points
(`python -m src.replay.cli run-day --date …`), plus `retrain_monthly` on the 15th. It needs:
Docker started (gate), and an image change, because the Airflow image (Python 3.12) has no
LightGBM; the scorer's environment is Python 3.14 in `.venv` and the API image. Options
then: a dedicated venv in the image (as `/opt/dbt-venv`) or `ExternalPythonOperator`; in
either case Phase 6 parity is re-proved in that environment. Rejected: building it in
Phase 8 (Docker gate, image change, no new evidence beyond the plain run).

**Service down during the replay:** `scripts/replay.py --scorer http` calls a locally
started uvicorn API (no Docker) instead of the in-process scorer. When the API does not
answer: **fail open** (the payment is not blocked), the transaction is written as
`unscored`, and it is scored when the service returns (backlog). The delay is recorded,
and time-to-detection is reported with and without the outage. Rejected: fail closed
(declining all card payments during a model outage is worse than delayed review).

**Tests:** `tests/test_replay_events.py` (order, label release exactly at +14 days, no label
visible before its release, scheduled jobs at fixed times), `tests/test_replay_engine.py`
(synthetic snapshot; resume after a stop gives the same rows; `batch` = per-transaction;
fail-open path with a fake API that goes down and comes back), local `gold` marker: F's
replay over the quarter gives exactly **969** alerts.

**Who:** Claude. Running the full replay: Claude, after the pre-registration push (user
approval).

### 6. Monitoring and automatic drift alerts

**What it shows:** the signals an operations team would watch, and alarms that fire against
thresholds fixed before the replay.

**Signals (computed by the daily monitoring job, `src/monitoring/`):**

| Signal | Definition | Needs labels |
|---|---|---|
| Alerts per day vs capacity | Count of alerts; 7-day rolling mean; FIFO backlog at capacity 12/day | No |
| Score drift | PSI of the champion's raw probability vs its reference (F: validation), 10 quantile bins of the reference, bin shares floored at 1e-4; daily and 7-day rolling | No |
| Feature drift | PSI of 12 of the 16 model features, exact list and rule per variable in the table "PSI variables" below | No |
| Delayed calibration | Observed / expected fraud count over the trailing 28 days of **released** labels, exact Poisson 95% interval | Yes (14-day delay) |
| Delayed performance | Recall and dollar recall of released labels, trailing 28 days, card-bootstrap interval | Yes |

**Pre-test evidence for the thresholds [measured]:**

| Measurement (raw F probability) | PSI |
|---|---|
| Each validation day vs the whole validation period (85 days) | median 0.0081, p95 0.0277, max 0.0371 |
| Each September day vs July–August (out of reference) | median 0.0066, p95 0.0325, max 0.0363; September as a whole 0.0003 |
| Each of the 14 gap days vs validation (out of sample for F) | max 0.0388 |
| 7-day blocks of validation vs validation (12 blocks) | median 0.0004, max 0.0012 |
| Training months vs validation (in sample for F, reference only) | max 0.0114 (2019-12), next 0.0086 (2019-01), others ≤ 0.0013 |

**Synthetic positive controls on validation [measured]** (applied to validation rows only,
features rebuilt, labels not used):

| Shift | Daily PSI median / max | Whole-period PSI | Alerts per day |
|---|---|---|---|
| All amounts × 1.2 | 0.016 / 0.072 | 0.0079 | 15.3 |
| All amounts × 1.5 | 0.068 / 0.175 | 0.0559 | 47.5 |
| All times + 3 hours | 0.034 / 0.129 | 0.0277 | 14.5 |

Daily PSI is dominated by sampling noise (~2,300 transactions per day): a daily alarm
would either fire on noise or miss a 20% amount shift. The 7-day window is ~20 times less
noisy.

**PSI variables and the rule per variable (review fix c).** Reference = the validation
period (F's features as built by `build_features`, the champion's raw probability).
Numeric variables: 10 quantile bins of the reference **plus one bin for missing values**
(`secs_since_prev` is missing on a card's first transaction, `amt_vs_card_median` with
fewer than 5 earlier transactions). Categorical and small-integer variables: one bin per
value. All shares floored at 1e-4. Windows: the 7-day rolling window ending on the
monitored day; daily values are shown, never alarmed, except A2.

Pre-test evidence per variable [measured; validation 7-day blocks vs validation /
September vs July–August / largest training month vs validation / largest single
validation day]:

| Variable | 7-day blocks max | Sep vs Jul–Aug | Largest train month | Largest day | Rule (7-day rolling PSI) |
|---|---|---|---|---|---|
| **Model score** (raw probability) | 0.0012 | 0.0003 | 0.0114 (2019-12) | 0.0371 | **A1 alarm ≥ 0.02, A1w warning ≥ 0.01; A2 daily ≥ 0.10 informational** |
| `amt` | 0.0008 | 0.0002 | 0.0008 | 0.0090 | F1 warning ≥ 0.02 |
| `hour` | 0.0022 | 0.0007 | 0.0007 | 0.0254 | F1 warning ≥ 0.02 |
| `is_night` | 0.0001 | 0.0000 | 0.0001 | 0.0032 | F1 warning ≥ 0.02 |
| `category` | 0.0009 | 0.0005 | 0.0005 | 0.0133 | F1 warning ≥ 0.02 |
| `channel` | 0.0005 | 0.0001 | 0.0001 | 0.0056 | F1 warning ≥ 0.02 |
| `distance_km` | 0.0009 | 0.0002 | 0.0005 | 0.0105 | F1 warning ≥ 0.02 |
| `age_band` | 0.0006 | 0.0001 | 0.0058 | 0.0146 | F1 warning ≥ 0.02 |
| `urban` | 0.0002 | 0.0000 | 0.0001 | 0.0040 | F1 warning ≥ 0.02 |
| `secs_since_prev` | 0.0200 | 0.0206 | 0.1990 (2019-01) | 0.1855 | F2 warning ≥ 0.10 |
| `amt_vs_card_median` | 0.0018 | 0.0006 | 0.5703 (2019-01) | 0.0226 | F2 warning ≥ 0.10 |
| `card_txn_1h` | 0.0075 | 0.0066 | 0.0536 (2019-12) | 0.0742 | F2 warning ≥ 0.10 |
| `card_txn_24h` | 0.0629 | 0.0513 | 0.5135 (2019-12) | 0.8404 | F2 warning ≥ 0.10 |
| `day_of_week`, `is_public_holiday_in_state`, `is_nationwide_public_holiday`, `is_bank_holiday` | 0.0013 / 0.4151 | 0.0351 / 0.0760 | 0.1713 / 0.1332 | 8.71 / 12.59 | **Not monitored by PSI**: set by the calendar, not by the population (one weekday per day; a holiday week moves the flags by design) |

Reading the table: transaction and profile attributes (F1) are very stable, so 0.02 is
about 9 times their largest pre-test 7-day value. Card-behaviour features (F2) move with
the season and with the start of the data (2019-01: cards without history; 2019-12:
holiday spending raises transaction counts), so they get a looser 0.10 (above every
validation 7-day value, 0.063). **A December F2 warning on `card_txn_24h` is plausible**
(2019-12 was 0.51) and is stated here before the replay. **Feature rules are warnings
only:** they never page; they are shown next to A1 to localise a score shift. Only the
score (A1), workload (A3), calibration (A4) and delayed recall (A5) raise alarms.
Rejected: alarming on every feature (16 tests a day, false alarms from calendar and
season); score PSI only (an alarm then gives no hint where to look).

**Alerts per day [measured, validation, frozen threshold]:** median 11, p10 3.4, p90 21.2,
max 31; 35 of 85 days above 12, 9 days above 20, 2 days with 0. 7-day rolling mean: min
7.1, median 11.4, max 18.9. FIFO backlog at 12/day: max 69 alerts, 28 at the end.

**Calibration [measured, validation, mean predicted vs observed, %]:** July 0.394 vs 0.374,
August 0.454 vs 0.468, September 0.467 vs 0.454 (O/E 0.95, 1.03, 0.97).

**Pre-registered alarms (§14):**

| Alarm | Fires when | Level | Why this value |
|---|---|---|---|
| A1 Score drift | 7-day rolling PSI ≥ **0.02** | alarm | Above every pre-test 7-day value (max 0.0012) and every training month (max 0.0114); fires on the +3 h and × 1.5 controls (0.028, 0.056), not on × 1.2 (0.008) |
| A1w Score drift | 7-day rolling PSI ≥ **0.01** | warning | Catches the 2019-12-like seasonal shift level |
| A2 Daily PSI | ≥ **0.10** | informational | The conventional "moderate shift" ruler; above every pre-test day (max 0.039) |
| A3 Workload | 7-day rolling mean alerts > **21** per day, or backlog > **100** | alarm | 1.75 × capacity 12; above the validation maxima (18.9 and 69) |
| A4 Calibration | O/E 95% interval entirely outside **[0.67, 1.5]** | alarm | Validation months were 0.95–1.03 |
| A5 Delayed recall | Trailing-28-day recall upper bound < **0.70** | alarm | F: 0.885 on the test period at the 1,082 budget, 0.855 at the frozen threshold (`model_results.md` §12) |
| F1 Feature drift (stable attributes) | 7-day PSI ≥ **0.02** on any F1 variable | warning | Table "PSI variables" |
| F2 Feature drift (card behaviour) | 7-day PSI ≥ **0.10** on any F2 variable | warning | Table "PSI variables" |

Each alarm writes a row to `alarms` with the measured value and the threshold, and points
to its runbook section (item 10). **Positive control before the replay:** each alarm must
fire on a synthetic input built to cross it (unit tests); A1 must fire on the validation
"+3 hours" control above.

**Expectation stated now, tested by item 8:** an adaptive fraudster changes only the fraud
rows (0.33% of test traffic), so **score PSI is not expected to detect it**; only delayed
recall (A5) can. If true, this is a documented limit of unlabelled drift monitoring.

**Files:** `src/monitoring/psi.py`, `capacity.py`, `calibration.py`, `alarms.py`,
`src/monitoring/reference/F_validation_psi.json` (bin edges and shares, committed, hashed);
tests `tests/test_monitoring_*.py` (PSI against a hand-computed example, identical
distributions → 0, floor behaviour, Poisson interval against known values, every alarm
fires on its control and stays silent just below its threshold).

**Who:** Claude.

### 7. Scheduled retraining, champion/challenger in shadow, promotion rule

**What it shows:** safe model replacement: a new model is trained on labels available so
far, scored in parallel without affecting alerts, and promoted only by a rule fixed in
advance.

**Schedule:** retrain at R1 = 2020-10-15 and R2 = 2020-11-15 (simulated); decisions at
2020-11-15 (challenger C1) and 2020-12-15 (challenger C2).

**Challenger recipe (pre-registered, nothing tuned):**
- Labelled data L = transactions with `trans_ts` < R − label delay. Primary (14 days): R1
  up to 2020-09-30, i.e. F's training data plus the validation quarter; R2 up to
  2020-10-31. Sensitivity (30 days): R1 up to 2020-09-14, R2 up to 2020-10-15.
- Window W = the last 28 days of L; a 7-day gap before W; training = everything before
  the gap (the Phase 4 split logic).
- Same 16 features (`FINAL_COLUMNS`), same `build_features`, LightGBM with **F's
  parameters as stored in `models/F/model.txt`**: 1,012 trees, learning rate 0.05,
  `num_leaves` 15, `min_data_in_leaf` 500, bagging 0.8 every iteration, feature fraction
  0.8, seed 42. No hyperparameter search.
- Calibrator: Platt on W (F's method).
- Threshold: the k-th highest raw probability on W, k = round(1000 / 85 × days in W) =
  round(11.765 × 28) = 329: the same alerts per day as F.
- Exported as a bundle with the existing format (`manifest.json`, SHA-256 per file,
  `bundle_id`) to git-ignored `data/replay/bundles/<bundle_id>/`. `models/F` is never
  touched; a promotion inside the replay changes only the replay's registry.

**Shadow:** from R, every transaction is scored by champion and challenger; only the
champion's alerts enter the queue.

**Promotion gate, in order (any failure = not promoted):**
1. `load_bundle` accepts the challenger bundle (hashes, columns, category mapping).
2. **Exact parity** on the Phase 6 parity sample design (3,000 rows of the shadow window,
   chosen without labels): in-process scorer vs offline batch scoring of the same bundle:
   features exactly equal, raw log-odds max difference 0, flags and reasons identical.
3. The threshold reproduces exactly k = 329 alerts on W.
4. Performance on the **labelled shadow window** (R → decision date − label delay; 16–17
   days in the primary run). **Paired comparison** (user decision 2026-11-27): champion
   and challenger are compared on **exactly the same shadow days and transactions**, each
   at the threshold frozen in its own bundle, with a **card-level bootstrap** (whole cards
   resampled, the same resample applied to both models; 500 draws, seed 42):
   - recall: the 95% lower bound of (challenger − champion) ≥ **−0.03** (non-inferiority;
     accepted), **and**
   - dollar-recall point estimate ≥ the champion's, **and**
   - **workload:** the challenger's mean alerts per day over the shadow window, at its
     frozen threshold, **≤ capacity (12)** (user decision), **and**
   - **fairness guardrail:** the challenger's gender recall gap (men − women) is at most
     **0.05** larger than the champion's on the same window (accepted).

   *Superseded workload rule (first draft, rejected in review):* alerts per day within
   [0.75, 1.25] × 11.765 = [8.82, 14.71]. It allowed a challenger above capacity. The mean
   over the window is used, not every single day: on validation F itself had 35 of 85 days
   above 12 (max 31), so a per-day rule would reject every model, F included.

**"Not promoted" is a normal outcome** (user decision) and is reported in the same format
as a promotion: the gate values, the intervals, and which condition failed. The shadow
windows hold only ~16–17 days of labels (about a fifth of a quarter), so the intervals
will be wide and "not promoted" is likely.

**Promotion = the replay's registry switches the champion** (`models` and `promotions`
tables, with the previous champion kept for rollback). After the replay the user decides
whether any challenger bundle is ever copied into `models/` (a separate commit, outside
the replay).

**Decisions:**

| Decision | Rejected alternative (reason) |
|---|---|
| F's hyperparameters fixed | Re-tuning at each retrain: more run time, more freedom to fit noise in 17 days of evidence |
| Non-inferiority + no dollar loss + workload + fairness guardrail | Superiority with an interval above 0: almost impossible with ~17 days of labels, so nothing could ever be replaced; point estimates only: promotes on noise |
| Capacity-matched threshold on W | F's threshold value reused: a new model's probabilities sit on a different scale, so workload would drift |
| Retraining on delayed ground truth only | Analyst feedback as labels: only alerted transactions get feedback, so the model would learn only what it already catches |
| Bundles of promoted challengers stay in `data/replay/` | Committing them automatically: a simulated promotion must not change the real serving model |

**Files:** `src/models/retrain.py` (recipe, uses `src/models/train.py` functions),
`src/replay/shadow.py`, `src/replay/promotion.py`, `scripts/export_challenger.py` (reuses
`export_bundle.py` logic); tests: window arithmetic for R1 and R2 (no label released after
R used), threshold gives exactly k on a synthetic W, the gate rejects a bundle with one
altered byte and a bundle whose scores differ by one ulp from offline, the promotion rule
on synthetic paired results at each boundary.

**"Worse new model" drill:** a deliberately weakened challenger (trained on the last 30
days of L only, pre-registered) must be **rejected** by the rule. If it is promoted, the
rule is too weak: stop condition S8.

**Who:** Claude.

### 8. Red-teaming the fraud model

**What it shows:** how much F's recall falls when fraudsters adapt to what it has learned
(night, large amounts, risky categories), at the frozen threshold, with rules fixed before
any run.

**Population (user decision 2026-11-27): the test quarter's fraud transactions** (936 rows,
94 cards; labels already opened once, G2). Legit rows are never changed. Rejected:
validation (it set the threshold, so its baseline recall is tuned to it).

**Perturb raw transactions only, never feature columns (user decision).** An attack edits
only the fields a fraudster controls, which are the raw request fields of the Phase 6 API
(`trans_ts`, `amt`, `category`, `merch_lat`, `merch_long`; `card_key` fixed). Every feature
is then rebuilt, past-only, **through the serving feature path**:
1. A red-team copy of the serving snapshot is built in the scratch area
   (`data/redteam/<attack>/serving.duckdb`, git-ignored): the card's original fraud rows are
   removed from `card_history` and the perturbed raw rows inserted, with their derived
   columns (`distance_km`, holiday flags, channel) computed by the **same resolver SQL** the
   API uses (`src/api/resolve.py`), never by new code.
2. Each perturbed transaction is scored with `score_transaction` (in process) against that
   copy, so its history contains the attacker's own earlier perturbed transactions and not
   the originals, and `build_features` runs exactly as in serving.

Identity test before any attack: with zero perturbations the red-team copy's
`card_history` equals the serving snapshot's row for row, and the scores equal the
champion replay's exactly. Rejected: editing feature columns directly (for example
setting `is_night` = 0 while `hour` stays 2); it produces rows no real transaction can
produce and skips the derived and history features.

**Attacks (pre-registered):**

| # | Attack | Transformation | Seed |
|---|---|---|---|
| R1 | Shift to daytime | Each fraud transaction moves to an hour drawn uniformly from 10..17 on the **same date**; minutes and seconds kept | 42 |
| R2a | Split under the baseline cut | Each fraud transaction above $136.33 becomes n = ceil(amt / 136.33) equal pieces, 120 seconds apart, same merchant and category | — |
| R2b | Split under $50 | Same with $50 | — |
| R3 | Low-risk categories | Category replaced by one of the 3 categories with the **lowest training-period fraud rate** (chosen by code from training rows), uniformly; channel follows the category | 42 |
| R4 | Combined | R1 + R2a + R3 | 42 |

Validation context [measured]: 830 of 979 fraud rows are at night; fraud amounts p10 $15.53,
median $379.30, p90 $1,025.02; 474 fraud rows above $500.

**Metrics:** recall per original fraud transaction (for splits: caught if any piece is
alerted, plus per-piece recall), dollar recall (alerted dollars / fraud dollars), episodes
detected and time-to-detection (item 2), alerts per day; each as a paired drop vs the
unmodified quarter with a card-bootstrap 95% interval (500 draws, seed 42).

**Pre-registered reading:** an attack is a **material vulnerability** when the recall or
dollar-recall drop is ≥ 0.10 with the interval's lower bound > 0. Material
vulnerabilities go into the runbook, the memo and the Phase 9 model card. Nothing is
retrained on the attacks (adversarial training is out of scope). Also reported: whether
any drift alarm (item 6) fires on the attacked stream (expected: no).

**Expected direction, stated before any run (pushed with §14):**

| Attack | Expected effect on recall and dollar recall | Reason |
|---|---|---|
| R1 daytime | **Down** | Night carries much of the signal (hour + night group SHAP 0.475); daytime fraud recall was 0.699 vs 0.922 at night on test |
| R2a split < $136.33 | **Dollar recall down; per-original-transaction recall uncertain** | Amount is the top feature (27.9% of SHAP), but the pieces come in bursts 120 s apart, which raise `card_txn_1h` / `card_txn_24h` and can *increase* detection |
| R2b split < $50 | **Down, more than R2a**, with the same burst caveat | Smaller pieces remove more of the amount signal |
| R3 low-risk categories | **Down** | Category is the second feature (21.1% of SHAP) |
| R4 combined | **Down, the largest drop** of all attacks | Removes night, amount and category signal together |
| Drift alarms | **A1 does not fire** on any attack | Only fraud rows change, 0.33% of traffic |

A result in the opposite direction is reported as found, never explained away, and is not
a reason to change a rule. That R2 may raise detection is stated here so it cannot later
read as tuning.

**Files:** `src/redteam/attacks.py` (pure transforms on raw request fields),
`src/redteam/snapshot.py` (the red-team snapshot copy through the resolver SQL),
`scripts/redteam_run.py`; tests: the identity test above, transforms touch only raw
request fields (a transform that writes a feature column fails), legit rows
byte-identical after every attack (S9), R1 keeps the date and lands in 10..17,
R2 pieces sum to the original amount (to the cent) and each is below the cut, R3 never
picks a category outside the pre-registered rule, determinism with the seed.

**Who:** Claude, after the pre-registration push.

### 9. Streamlit dashboard

**Pages** (`dashboard/app.py` + `dashboard/pages/`), all reading `replay.duckdb` read-only:

| Page | Content |
|---|---|
| Alert queue | Champion alerts in simulated-time order, FIFO at the chosen capacity: time, amount, category, probability, top-3 SHAP reasons (values of `age_band` and `urban` shown, as for the analyst; hidden only for an LLM, D5); optional **Phase 7 note** (user decision: optional and **live locally**, generated on demand by the deterministic baseline, `src/agent/workflow.py`, which needs the local API and Ollama, otherwise "note unavailable"; the **hosted app shows pre-generated notes for a fixed set of alerts**, labelled "pre-generated on <date> with llama3.1, not live", see item 11); buttons **Confirm fraud / Not fraud / Unsure** |
| What-if | Capacity slider 5–50 alerts/day: threshold re-derived from validation for that capacity (online-feasible), then recall, dollar recall, precision and backlog on the labelled window; investigation cost per alert slider and a recovery share, giving net dollars. Costs are **user assumptions**, labelled as such (no cost data, Phase 4 limitation) |
| Fairness | Recall and false-alarm rate by gender, age band, day/night, category, on released labels and on validation for reference; card-bootstrap 95% intervals precomputed by the monitoring job; "too few to judge" under 10 fraud cards |
| Replay monitoring | Alerts per day vs capacity and backlog, daily and 7-day PSI with thresholds, calibration O/E with interval, alarm log with runbook links, champion/challenger status and promotion decisions |

**Measured for the what-if (validation, frozen F threshold, FIFO) [measured]:** maximum
backlog 579 alerts at 5/day, 169 at 10, 69 at 12, 31 at 15, 15 at 20, 1 at 30, 0 at 50.

**Human in the loop:** each click appends one JSON line to `data/replay/feedback.jsonl`
(`alert_id`, decision, simulated and wall time, `model_version`; no amounts, times of the
transaction or card keys beyond the alert id). Feedback is shown as analyst agreement with
the model and is **not** used for retraining (item 7).

**Decisions:**

| Decision | Rejected alternative (reason) |
|---|---|
| Streamlit | Dash/Panel: more code for the same pages; Power BI covers the analyst report already |
| Built-in Streamlit/Altair charts (Altair comes with Streamlit) | Plotly: one more dependency tree |
| Bootstrap intervals precomputed by the monitoring job | Computing them in the page: 500 draws per click |
| Feedback in a separate append-only file | Writing into `replay.duckdb`: a second writer |

**Tests:** `tests/test_dashboard_data.py` (page data functions, not the UI): the what-if
curve at capacity 12 reproduces the validation numbers, the queue order, feedback lines
carry only allowed keys, the dashboard opens `replay.duckdb` read-only (a write attempt
fails). A Streamlit `AppTest` smoke test per page (Streamlit's own testing API) on a
synthetic replay file.

**Dependency:** `streamlit==1.67.0` (§13). **Who:** Claude.

### 10. Operations runbook

**File:** `docs/runbook.md`. Each scenario: signal, first five minutes, diagnosis, action,
verification, who decides. Commands are for the plain-Python setup (no Docker); Docker
commands are listed separately and marked "needs approval".

| Scenario | Signal | Action | Drill (round 2) |
|---|---|---|---|
| Drift alarm (A1–A5) | `alarms` row | Check data first (row counts, category list, snapshot `source_max_trans_ts`), then feature PSI to localise; drift with stable recall = watch; with falling recall = retrain early (challenger path, never a direct swap) | A1 on the "+3 hours" control |
| Service down | Health check fails, `unscored` rows grow | Fail open (item 5); restart the API; check startup log (`startup_failed` reasons: hash, snapshot schema); score the backlog; report detection delay | Kill the local uvicorn mid-replay, restart, verify backlog scored and nothing lost |
| Worse new model | Promotion gate rejects, or A5 fires after a promotion | Do not promote / roll back; keep the challenger bundle for analysis | The weakened challenger of item 7 |
| Rollback to the previous bundle | Any of the above after a promotion | Registry switches champion to the previous `bundle_id`; serving restarts with `BUNDLE_DIR` = previous bundle; `load_bundle` verifies hashes (fail fast); verify `/model-info` shows the old `bundle_id` and parity on 100 rows | Promote a challenger in a test copy of the registry, roll back, check scores equal F's exactly |

**Who:** Claude drafts and runs the drills; **user** reviews.

### 11. Free hosting of the model and dashboard

**May the serving snapshot be published? No, recommended.** It holds `home_lat` /
`home_long` for all 999 cards plus every card's full history [measured: tables `cards`
(with home coordinates), `card_history`, `calendar`, `state_holidays`, `categories`,
`meta`; 25.4 MB]. The data is simulated (Sparkov, CC0 licence), so publishing is legally
possible, but the project's own rule is that customer coordinates never leave Silver and
the serving snapshot (decision table, PROJECT_STATUS §7). Publishing them would contradict
the privacy story the project tells.

**What to publish (proposed):**

| Published | Content |
|---|---|
| Replay extract | Test-quarter rows (281,521): a **new random card id** (not `card_key`, which is `md5(card_hash)`), `trans_ts`, amount, category, channel, `age_band`, state, `distance_km`, F's scores, alert, reasons, label and its release time; plus the monitoring, alarm and promotion tables |
| Model bundle | Already public in git (`models/F`) |
| Free-form scoring | In process inside the app, on the **synthetic snapshot** (`scripts/make_synthetic_snapshot.py`, already used in CI), clearly labelled "synthetic cards" |
| Pre-generated analyst notes | Phase 7 baseline notes for a **fixed set of alerts chosen by a label-free rule before generation**: the first champion alert of each simulated week of the replay, weeks counted from 2020-10-01 (at most 14 alerts; the 14th week is the single day 2020-12-31). Generated locally once (Ollama), stored with the model digest and prompt hashes, `card_key` removed from the header; each shown with the label "pre-generated, not live". Rejected: live notes in the hosted app (no Ollama there; a hosted LLM would need an account and send data out); hand-picking alerts (selection by outcome) |
| Not published | `serving.duckdb`, `card_hash`, `card_key`, home and merchant coordinates, ZIP, city, job, `birth_year` |

**Hosting: undecided (user decision 2026-11-27, not blocking).** The user checks the terms
before round 3. Design: one Streamlit app (dashboard + in-process scorer, like the MCP
server: no separate API host). Candidates: Streamlit Community Cloud, Hugging Face Spaces.
Their resource limits, Python 3.14 support and terms for users in Iran are not verified
here (as with the FATF terms in Phase 7). Account creation is the user's. Everything in
rounds 1–2 and the local dashboard is independent of this choice.

**Files:** `scripts/build_public_extract.py` with a check that no forbidden column exists
(positive control: the check fails on a frame that contains one), `tests/test_public_extract.py`.

**Who:** extract, app config: Claude. Platform choice, terms check, account: **user**.

---

## 12. Build rounds (3)

| Round | Content | New dependencies | Ends with (gates) |
|---|---|---|---|
| **1. Foundations, no runs on test** | SQL specs + references + checker (1); TTD module (2) with validation numbers; Power BI export + reference numbers (3); replay events/clock/engine/store (5) tested on the synthetic snapshot and validation; monitoring + alarms + PSI reference file (6); retraining recipe, shadow, promotion gate (7); red-team transforms on raw fields + red-team snapshot copy with the identity test (8); state card-cluster bootstrap (3); feature-PSI reference (6); pre-registration file (§14) | none | Re-measured numbers equal this document (S1); **commit + push the pre-registration (user approval)**; CI green |
| **2. Runs** | Test TTD once (2); champion replay in `batch`, then one `demo` run (5): 969 alerts; **primary run (14-day delay)** and the **30-day sensitivity run**, each with retraining, shadow and both promotion decisions (7); red-team runs (8); runbook drills (10); results in `docs/phase8_results.md` | none | Results reviewed by the user before any push |
| **3. Surfaces** | Throwaway-venv dry run for streamlit on Python 3.14, then install after approval (§13); Streamlit dashboard (9); pre-generated notes for the hosted app; public extract and hosting config once the user has chosen a host (11); decision memo (4); Power BI pages (**user**, after install) | `streamlit==1.67.0` (after the wheel check and install approval) | User approves memo; hosting by the user |

The user's SQL answers (1) and Power BI install (3) can happen at any time; the checker is
ready after round 1.

## 13. Dependencies

| Package | Version | Where | Status / wheel check plan |
|---|---|---|---|
| (rounds 1–2) | — | — | None new. PSI, Wilson, Poisson intervals with numpy/scipy 1.18.1 (already pinned); scheduler from the standard library |
| `streamlit` | **1.67.0** | new `requirements-dashboard.txt` (`-r requirements.txt` + the API serving pins the scorer imports + streamlit), never in `requirements.txt` (Airflow image) or `requirements-api.txt` | Latest on PyPI on 2026-11-27; uploaded 2026-11-05 (3 weeks old); pure-Python wheel `py3-none-any`; requires Python ≥ 3.11. Declared requirements accept the existing pins: `starlette<2` (1.9.0), `uvicorn<1` (0.54.0), `pyarrow!=25.0.0,<27` (25.0.1), `pandas<4` (3.0.5), `numpy<3` (2.5.3), `python-multipart` (already present via `mcp`). New transitive packages expected: altair, pydeck, protobuf, pillow, websockets, watchdog, click, itsdangerous, and their own dependencies. **Plan:** `pip install --dry-run --ignore-installed --only-binary=:all: --report` against `.venv`'s pins → stop if any installed version would change (S10); list every new package and version in this table; check every wheel's `WHEEL` generator (lesson from dbt-core 1.12); binary packages (pillow, protobuf, watchdog) must have `cp314` / `abi3` Windows wheels; repeat for Linux before hosting. **Round 3, review fix e:** additionally, in a **throwaway venv** (new Python 3.14.6 venv in the scratch area, deleted afterwards; `.venv` untouched), `python -m pip install --dry-run --only-binary=:all: --report report.json -r requirements-dashboard.txt` must resolve the **full dependency tree** on Python 3.14 without building anything from source; the report's package list is recorded in this table. Install into `.venv` only after the user's approval. **Nothing is installed now** |
| Power BI Desktop | the version the user installs | user's machine | Version recorded in `docs/powerbi_report.md` |
| Airflow variant | — | — | Not built in Phase 8 |

## 14. Pre-registration (pushed before any replay, red-team or test-TTD run)

One file, `src/replay/preregistration.json` (committed; loaded by every Phase 8 script, which
records its SHA-256 and refuses to run when it differs from the pushed value), plus its
human-readable copy in `docs/phase8_preregistration.md`.

| Item | Value |
|---|---|
| Replay window | 2020-10-01 00:00:00 .. 2020-12-31 23:59:59, original timestamps (not shifted) |
| **Primary run** | **label delay 14 days**; all headline Phase 8 results come from this run |
| Sensitivity run | label delay 30 days, reported alongside the primary run, never replacing it; same transactions, champion, rules and seeds; only the delay and the decision dates differ |
| Capacity | 12 alerts per day (Phase 4); FIFO review rate only, never an alert budget |
| Champion | bundle `be26ca8810f6215e0b24875f80ed063287062e039c2222eda09be08bc426dae9` |
| Alert rule | `probability_raw >= 0.2940968658100669` per transaction, frozen; no top-N budget anywhere in the replay |
| Replay reconciliation | F's quarter: exactly 969 alerts, row-by-row equal to the threshold rule |
| PSI variables and bins | model score (raw probability) + 12 features: F1 = `amt`, `hour`, `is_night`, `category`, `channel`, `distance_km`, `age_band`, `urban`; F2 = `secs_since_prev`, `amt_vs_card_median`, `card_txn_1h`, `card_txn_24h`; not monitored: `day_of_week` and the 3 holiday flags. Reference = validation; numeric: 10 reference-quantile bins + a missing-value bin; categorical: one bin per value; floor 1e-4; bin edges in `F_validation_psi.json` (hash recorded) |
| Alarms (page) | A1 score 7-day PSI ≥ 0.02; A3 7-day mean alerts > 21/day or backlog > 100; A4 O/E 95% interval entirely outside [0.67, 1.5], trailing 28 labelled days; A5 trailing-28-day recall 95% upper bound < 0.70 |
| Warnings / info (no page) | A1w score 7-day PSI ≥ 0.01; A2 score daily PSI ≥ 0.10 (info); F1 any F1 feature 7-day PSI ≥ 0.02; F2 any F2 feature 7-day PSI ≥ 0.10; stated expectation: an F2 warning on `card_txn_24h` in December is plausible |
| Retraining | R1 2020-10-15, R2 2020-11-15 (both runs); L = `trans_ts` < R − delay; W = last 28 days of L, 7-day gap before W, training before the gap; F's parameters from `models/F/model.txt` (1,012 trees, lr 0.05, leaves 15, min leaf 500, bagging 0.8 every iteration, feature fraction 0.8, seed 42); no tuning; Platt on W; threshold = k-th highest raw probability on W, k = 329 |
| Decision dates | primary: 2020-11-15 (C1), 2020-12-15 (C2); sensitivity: 2020-12-01 (C1), 2020-12-31 23:59:59 (C2) |
| Promotion gates | 1 `load_bundle` accepts; 2 exact parity on 3,000 label-free shadow rows (features equal, raw log-odds difference 0, flags and reasons identical); 3 threshold gives exactly k = 329 alerts on W |
| Promotion rule | paired on the same shadow days and transactions, each model at its own frozen threshold, card-level bootstrap with the same resample for both (500 draws, seed 42): recall difference 95% lower bound ≥ −0.03; dollar recall (point) ≥ champion; challenger mean alerts/day over the shadow window ≤ 12; gender recall gap (men − women) ≤ champion's + 0.05. "Not promoted" is a normal outcome, reported in the same format |
| Worse-model drill | challenger trained on the last 30 days of L only; must be rejected (else S8) |
| TTD | item 2 definitions; 7-day episode gap; episode belongs to the period of its first fraud transaction; first alert = first alerted transaction of the card at or after the episode start; loss = fraud rows strictly before it; bootstrap 500, seed 42; test computed once, must use F's 969 alerts |
| Red team | test-quarter fraud rows only; raw request fields only, features rebuilt past-only through the serving path (red-team snapshot copy + resolver SQL + `score_transaction`); identity test first; R1 daytime hour U{10..17} same date (seed 42); R2a split ceil(amt/136.33) pieces, R2b ceil(amt/50), 120 s apart; R3 one of the 3 lowest-train-fraud-rate categories (seed 42); R4 = R1+R2a+R3; frozen threshold; material = recall or dollar-recall drop ≥ 0.10 with lower bound > 0; expected directions as in item 8 |
| Fairness | gender, age band, day/night (22–03), category; recall and false-alarm rate; card bootstrap 500, seed 42; "too few to judge" < 10 fraud cards |
| State map (Power BI) | fraud transactions / transactions per state; card-cluster bootstrap within state, 500 draws, seed 42, percentile 95%; "too few" < 10 fraud cards; test-quarter rows included and marked |
| Publication | the published / not-published lists of item 11; pre-generated notes = first champion alert of each replay week |

**When it is pushed:** at the end of round 1, as its own commit, after the user's
approval and before any replay, red-team or test-TTD run (G3). Every round-2 script
checks the file's SHA-256 against the pushed value (S2).

## 15. Stop conditions

Each stops the current round; the finding is reported, never fixed by tuning on the test
quarter.

| # | Condition (measurement) | Scope |
|---|---|---|
| S1 | A re-measured pre-test number (validation alerts 1,000, TTD, PSI table, backlog table, state counts) differs from this document | Round 1 |
| S2 | The pre-registration SHA-256 differs from the pushed value, or a run starts before the push | Every round-2 script |
| S3 | `batch` and `demo` differ in any result column on the 3 fixed equality days (exact for features, raw log-odds and flags; 0 tolerance) | Replay |
| S4 | F's replayed quarter ≠ 969 alerts, or test TTD uses a different alert set | Replay, TTD |
| S5 | Any computation reads a label whose release time is after the simulated clock (asserted in the store, tested) | Monitoring, retraining, dashboard |
| S6 | Challenger parity: raw log-odds difference > 0, any feature, flag or reason mismatch, a bundle hash failure, or threshold ≠ k alerts on W | Each challenger |
| S7 | Replay process memory above 4 GB (peak working set) or a `batch` quarter above 30 minutes | Replay |
| S8 | The weakened challenger passes the promotion rule | Promotion |
| S9 | A red-team transform changes any legit row (byte comparison) | Red team |
| S10 | A new dependency would change an installed version, or a wheel's generator is not a standard build tool | Round 3 install |
| S11 | Any log or feedback file contains a card key, amount, transaction time or coordinate (count must be 0, with a positive control that finds a planted value) | All new processes |
| S12 | A step needs Docker | All rounds: stop and ask |

---

## Decisions of 2026-11-27 (user, on the eight open questions)

| # | Question | Decision | Where |
|---|---|---|---|
| 1 | Label delay | 14 days = primary run; 30 days = pre-registered sensitivity run reported alongside | items 5, 7; §14 |
| 2 | Promotion rule | accepted (recall lower bound ≥ champion − 0.03; gender gap growth ≤ 0.05), plus: paired on the same shadow days with a card-level bootstrap; alerts/day at the frozen threshold not above capacity; "not promoted" is a normal outcome | item 7; §14 |
| 3 | Red-team population | test quarter; perturb raw transactions, rebuild past-only features through the serving path, never edit feature columns; rules and expected directions pushed before running | item 8; §14 |
| 4 | Power BI | test-quarter labels included and clearly marked; Shape Map with the built-in US states map; no online map services | item 3 |
| 5 | Hosting | undecided; the user checks terms before round 3; not blocking | item 11 |
| 6 | Memo audience | fraud-operations manager | item 4 |
| 7 | Airflow variant | design only in Phase 8; built in Phase 9 (Docker = user approval) | item 5 |
| 8 | Phase 7 note | optional and live locally; the hosted app shows pre-generated notes for a fixed set of alerts, clearly labelled | items 9, 11 |

**Review fixes:** (a) 102 fraud cards vs 101 episodes reconciled in item 2 (one episode
starts in the June gap and ends in validation; still one episode per card); (b) state
intervals by card-cluster bootstrap, item 3; (c) PSI variables and rule per variable,
item 6; (d) frozen threshold, no top-N budget, item 5; (e) throwaway-venv dry run for
streamlit on Python 3.14 in round 3, §13.

*The original open questions are kept below for the record.*

### Original open questions (answered above)

1. **Label delay 14 days** (two promotion cycles fit) or 30 days (more realistic, one late decision)?
2. **Promotion rule:** non-inferiority margin −0.03 and the gender guardrail +0.05 acceptable, knowing "not promoted" is the likely outcome?
3. **Red-team population:** the test quarter (proposed; frozen model, validation set the threshold) or validation?
4. **Power BI:** include the test quarter's labels in the retrospective report (proposed: yes, shaded), and which map visual works on your installed version without online map services?
5. **Hosting:** Streamlit Community Cloud or Hugging Face Spaces, after you check their terms for your location?
6. **Decision memo audience:** a fraud-operations manager (proposed) or a product manager like the Phase 5 memo?
7. **Airflow variant:** design only in Phase 8 (proposed), or build it in Phase 9 with the Docker and image gates?
8. **Phase 7 note in the alert queue:** keep it optional (needs local API + Ollama; absent in the hosted app), or drop it?
