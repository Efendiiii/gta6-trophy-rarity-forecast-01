# gta6-trophy-rarity-forecast-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Efendiiii/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Efendiiii/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts every GTA VI (PS5) trophy's earn rate on Exophase at a case's target time, as five quantiles (5/25/50/75/95%) per trophy. A single-file Python entry (stdlib only, deterministic) that compresses the page history to its distinct refreshes, then fits a damped logit trend for multi-snapshot histories or models the next refresh explicitly for flat histories.

## Run it

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=4fxfm1jz&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>

## How it works

A rate is earners over players tracked, `rate = 100 * E / P`, and the
page is only as fresh as Exophase's last refresh. The program first
compresses the history to its **distinct snapshots** (consecutive
identical reads are one stale page), then takes one of two paths:

**A. History with >= 2 distinct snapshots (the final case).** Per
trophy, a least-squares line of `logit(rate)` on `ln(1 + hours since
release)` over the last 48 h of snapshots, slope shrunk toward the
cross-trophy median slope (weight `n/(n+4)`), anchored at the last
snapshot and read at the target minus half the median inter-refresh gap
(the answer is the page as of the last refresh before the target).
The extrapolated rise is damped by 0.9 (the curve saturates). The
median is floored at `100 * E_last / P_target` — earners cannot fall —
with `P_target` from the same log-log fit on `ln P` (smaller of the
all-history and recent slopes). Spread combines the pooled residual SD,
slope uncertainty over the extrapolation distance, a refresh-lag term
and a misspecification term growing with `sqrt(days of horizon)`.

**B. History with one distinct snapshot (the practice cases).** No
trend exists, so a refresh is forecast explicitly: a Weibull hazard
(mean interval 4 h, shape 2 — the page was unchanged 23:48–04:00Z,
then refreshed by 04:30Z) gives the refresh probability; given a
refresh, players tracked jump by `G` (lognormal, median `1 + staleness
in hours` at the target, capped at 8 — the documented first refresh
went 638 -> 3,868 after ~4.7 h stale) and the new cohort earns the
intro trophy (the one at 100% stays 100%) plus `lambda = 0.032` extra
trophies per new player. That `lambda` comes from the documented
refresh aggregates (4,276 - 943 = 3,333 new trophies over 3,230 new
players, 3,230 of them the intro trophy, so 103 spread over the other
54 trophies); the 103 are split by an **early-earnability** score read
off each trophy's own description and tier (chapter 1 before chapter
4, a fishing challenge before 250 stunts). The forecast is the mixture
of the no-refresh atom and the refresh distribution (deterministic grid
over `G` and a `lambda` scale), and its quantiles are the mixture's
quantiles on the logit scale.

Rates of 0/100 are handled in earner space with `(E + 0.5)/(P + 1)`
smoothing. Snapshots with a null players-tracked count are dropped
before any fit (a rate without a player count cannot become earners).
A case with an empty history still gets a valid output: a flat 1.0%
forecast with a small spread. Any other failure on a case falls back
to the house baseline's persistence forecast, so a case is never
left invalid.

## Validation (local, `../eval/`)

The practice answers are not published, so the kit's `grade-local`
cannot score them (it refuses program/v1 kits: "graded on the server
only"). Instead `eval/` implements the contest's exact pinball scorer
and two synthetic testbeds:

* **Practice refresh scenarios** built from the documented refresh
  aggregates (P 638 -> 3,868, earned 943 -> 4,276) with six different
  per-trophy splits of the 103 new non-intro trophies. The starter's
  loss on the central scenario is 1.43, close to the 1.51 RULES.txt
  documents for it — the scenario family is realistic. Model:
  **0.41–0.73 vs starter 1.09–1.69** across all 12 case/scenarios.
* **Synthetic final cases** (cut 21 Nov 11:30Z, target 26 Nov 00:00Z)
  from a cohort-progress model — players join over time, a player of
  age `a` has played ~15% of `a` hours, time-to-earn lognormal per
  trophy class — under three P-growth shapes (power-law, logistic,
  exponential-then-flat) x two progression speeds. Model:
  **0.21–0.59 vs starter 0.49–1.15**, 90% coverage 0.80–0.96
  (honest target 0.90). The exponential-then-flat shape is the
  hardest: player growth stops dead at the cut, an event no
  history-based model can foresee; the model still beats the starter
  there by 0.44–0.56.
* **Constant sweep** (fit window x damp x spread, `eval/sweep.py`):
  window 48 h is a smooth optimum (24 h: 0.411, 36 h: 0.370,
  48 h: 0.353, 72 h: 0.472 mean loss); damp 0.9, spread
  0.30/sqrt(day) chosen for best mean loss with 0.90 coverage.

## Biggest assumptions / risks

1. The flat-history prior's scale (`lambda = 0.032`, `G` median) is
   anchored on the single documented refresh; a different refresh
   (e.g. a much larger or smaller jump) moves the practice medians.
   The final case does not use this branch.
2. The trend branch assumes the log-time slope of the last 48 h,
   damped, extrapolates 4.5 days. A regime break at the cut — player
   growth stopping (rates accelerate) or the weekend wave (rates
   decelerate) — is the largest final-case risk; the spread term
   (0.30/sqrt(day)) is the hedge, not a cure.
3. The synthetic finals share one generator family; the real week's
   shape may differ (notably the true Exophase refresh cadence and
   the true player-growth curve, both unobservable before the final).
