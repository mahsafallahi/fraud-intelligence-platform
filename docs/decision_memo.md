# Decision memo: should we add an extra verification step to risky card payments?

**For:** product manager · **Date:** 2026-11-12 · **Status:** design and rehearsal only

> **All effects in this memo are SIMULATED.** We have not shown the extra step to any customer. The numbers below come from replaying three months of past transactions under assumed customer behaviour. They show how the experiment would work and what it could detect, not what will happen.

## The idea

Our fraud model gives every card payment a risk score. Today only the riskiest payments go to the fraud team, about 12 per day, because the team has limited time. An extra verification step (for example a code sent to the customer's phone) is automatic, so we could ask for it on many more payments: about 28 per day at the threshold we propose, out of roughly 3,060 payments a day in our data.

## What we would gain and risk (SIMULATED)

- **Gain:** the payments that get the extra step include 99.4% of all fraud dollars. If the step stopped half of the fraud it sees (assumed), fraud losses would fall by about $413 per affected card over three months (SIMULATED).
- **Risk:** some honest customers will give up when asked to verify. The step would touch about 18 honest payments a day, 3.4% of honest spending.
- **Break-even:** if the step stops half of the fraud it sees, it pays for itself as long as fewer than 38% of the honest customers it asks give up, even if we count each lost purchase at its full value (SIMULATED).

## How we would test it

- Half of the cards, chosen at random, get the extra step; the other half do not. We compare the two halves after three months, and do not peek earlier (peeking more than triples the chance of a false alarm: 19% instead of 5% in our rehearsal).
- We decided in advance what counts as success: **launch only if** total losses (fraud plus lost purchases) are clearly lower with the step, **and** honest customers do not give up more than we can accept: no more than 10% of the honest customers asked, and lost purchases no more than 5% of the honest spending that gets the step.
- We rehearsed the whole test on past data with no effect switched on: it raised a false alarm 5.0% of the time, as designed.

## What the rehearsal shows (SIMULATED)

| If the step stops this much fraud | ...and this many asked honest customers give up | The test would say |
|---|---|---|
| 50–80% | 1% or 5% | **Launch** (in 71–100% of rehearsals) |
| 50–80% | 10% or 20% | **Don't launch**: too many honest customers lost |

The deciding factor is how many honest customers give up, not how much fraud is stopped: that is what the real test must measure.

## A trade-off for you to decide

If about 1 in 10 asked honest customers gave up, the money would still work out: total losses would still fall, because the step pays for itself unless 38–61% give up. Our success rule would still say **don't launch**, because it also limits how many honest customers we lose. That limit is there because the money count leaves out the annoyance, the lost trust and the customers who stop using their card. Whether the fraud savings are worth losing those customers is **your decision**. We wrote the limit down before the test so that this choice is visible, not hidden inside one number.

## Size of a real test

Our data has 915 active cards, enough only to detect very large effects (about a 60% drop in fraud losses). A real bank, where far fewer cards see fraud than in our simulated data, would need about **180,000 cards in each half** to detect a 10% drop in fraud losses with 1% of cards seeing fraud per quarter.

## What we have not covered

The time and annoyance for honest customers who complete the step; fraudsters moving to unprotected cards; customers getting used to the step over time. And our data is simulated: fraud is more frequent and easier to spot than in real life.

## Recommendation

Run a real randomised test at the proposed threshold, with the success rule above, for a fixed three months, on enough cards (sized from the bank's real fraud rate). Do not decide from past data or from a non-random rollout: our rehearsal shows that comparing the cards a policy chose with the cards it skipped can be badly wrong, and that adjusting for what we can see does not fix what we cannot see.
