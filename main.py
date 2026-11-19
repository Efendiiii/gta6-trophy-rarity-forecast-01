#!/usr/bin/env python3
"""Forecast GTA VI (PS5) trophy earn rates on Exophase at a case's target.

Method (see README.md for the full justification and validation numbers)
------------------------------------------------------------------------------
The case gives a history of reads of Exophase's trophy page.  A read is a
snapshot of the page: every trophy's earn rate (percent of the players
Exophase tracks who have earned it) and the players-tracked count.  Two
facts about that page drive the whole model:

1. The page is only as fresh as Exophase's last refresh.  Between refreshes
   the page is byte-identical: the house saw 638 players tracked and every
   rate unchanged from 2026-11-18T23:48Z through its 04:00Z read on launch
   night, then a refresh by 04:30Z: 3,868 players tracked, 4,276 trophies
   earned, every rate but the one at 100% lower.  So consecutive identical
   reads carry no new information; the model first compresses the history
   to its *distinct snapshots* (one per refresh) and works from those.

2. A rate is earners over players tracked, rate = 100 * E / P.  Earners E
   only grow (a trophy once earned stays earned), while P grows as new
   players appear.  So a rate falls when dilution beats progression and
   rises when it does not -- and E's last value, divided by the forecast P,
   is a hard floor on the target rate.

Two regimes, chosen by what the history shows:

A. History with >= 2 distinct snapshots (the final case: reads every
   10 minutes, refreshes hours apart, real growth over ~2.5 days).
   Per trophy, fit a least-squares line of logit(rate) on ln(1 + hours
   since release) over the distinct snapshots of the last 48 hours --
   the curve saturates, so the earliest climb is not the current trend,
   and the first ~12 h of the final's history are the flat launch-night
   page -- with the slope shrunk toward the cross-trophy median slope (weight n/(n+4) on the trophy's own slope, n = its
   snapshot count in the window).  Anchor the line at the last snapshot
   and read it at the target, minus half the median inter-refresh gap
   -- the answer is the page as of Exophase's last refresh before the
   target, not the state at the target itself -- with the extrapolated
   rise damped by 0.9, because even the recent slope overstates a
   saturating curve 4.5 days out.  The median is floored at
   100 * E_last / P_target, where P_target comes from the same log-log
   fit on ln P (the smaller of the all-history and recent slopes).  Spread: the pooled residual SD, the slope's standard error
   times the extrapolation distance, a refresh-lag term and a
   misspecification term growing like sqrt(days of horizon), combined in
   quadrature.

B. History with a single distinct snapshot (the practice cases: the page
   flat all launch night).  There is no trend to fit, so forecast a
   refresh explicitly.  A refresh before the target has probability w from
   a Weibull hazard (mean interval 4 h, shape 2) on the staleness at the
   cut plus the horizon.  Given a refresh, P jumps by G (lognormal, median
   1 + staleness-in-hours at the target, capped at 8: the documented first
   refresh jumped 638 -> 3,868 after ~4.7 h stale) and the new cohort
   earns the intro trophy (the one at 100% stays 100%) plus, in total,
   lambda extra trophies per new player (lambda = 0.032 at launch, from
   the documented refresh aggregates: 4,276 - 943 = 3,333 new trophies
   over 3,230 new players, 3,230 of them the intro trophy, so 103 spread
   over the other 54 trophies; lambda grows slowly with hours since
   release).  The 103 are split across trophies by an early-earnability
   score read off each trophy's own description and tier -- chapter 1
   before chapter 4, a fishing challenge before 250 stunts -- never by
   guessing an answer.  The forecast is the mixture of the no-refresh
   atom (weight 1-w) and the refresh distribution (a small deterministic
   grid over G and a lambda scale), and its quantiles are the mixture's
   quantiles on the logit scale.

Rates of 0 or 100 are handled in earner space with a +0.5 smoothing
(E + 0.5)/(P + 1) so logits stay finite; a trophy at 100% in the last
read is forecast tight around 100 (every tracked player holds it, and
the documented refresh left it at 100%).

No network, no randomness, no model calls: two runs on the same case
write byte-identical output.  Standard library only.  Any failure on a
case falls back to the house baseline's persistence forecast for that
case, so a case is never left invalid.  Two edge cases are handled
explicitly, not by that fallback: snapshots whose players_tracked is
null (or <= 0) carry no usable earner information -- a rate without a
player count cannot be turned into earners -- so distinct_snapshots
drops them before any fit, which keeps every (x, ln P) pair aligned;
and a case with an empty history still gets a valid output file, a
flat 1.0% forecast (the same neutral rate the persistence fallback
uses for a trophy with no known rate) with a small spread, because
with no history there is no evidence to spread wider.
"""

import json
import math
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import NormalDist

# The grader's -P leaves /bot off sys.path: put it back first, so a module
# shipped beside this file would import; this entry is one file.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817,
     1.6448536269514722]

# --- regime A (trend) constants -------------------------------------------
# Shrinkage: a trophy's own slope gets weight n / (n + SLOPE_SHRINK).
SLOPE_SHRINK = 4.0
# Slopes are fit on the last FIT_WINDOW_HOURS of snapshots: a rate's
# logit-vs-ln-time curve saturates, so the earliest climb is not the
# current trend -- but the window must stay long enough to see the trend
# at all.  48 h is the sweet spot in the eval (24 h under-uses the
# history, 72 h is the whole launch transient again); it also drops the
# first ~12 h of the final's history, when the page was flat and the
# tracked base was reviewer-heavy.
FIT_WINDOW_HOURS = 48.0
# Even the recent slope overstates a saturating curve 4.5 days out, so the
# extrapolated rise is damped (see README: sensitivity in the eval).
EXTRAP_DAMP = 0.9
# Spread terms, in logit units (see README for where each comes from).
REFRESH_LAG_SIGMA = 0.15   # the answer lags the trend by up to a refresh gap
MODEL_SIGMA_PER_SQRT_DAY = 0.30  # trend misspecification over the horizon
MIN_RESID_SIGMA = 0.08     # floor on the residual spread
MAX_LOGIT = 9.21           # logit(99.99): the scoring clamp

# --- regime B (flat history / refresh prior) constants --------------------
# Exophase refresh interval, hours: mean 4 (the page was unchanged
# 23:48Z-04:00Z, then refreshed by 04:30Z), Weibull shape 2 (refreshes
# are fairly regular, not memoryless).
REFRESH_MEAN_HOURS = 4.0
REFRESH_SHAPE = 2.0
# Extra trophies per new player beyond the intro trophy, at launch:
# (4276 - 943 - 3230) / 3230 = 0.0319, from the documented refresh
# aggregates (players 638 -> 3868, trophies earned 943 -> 4276).
LAMBDA_LAUNCH = 0.032
LAMBDA_GROWTH_PER_DAY = 1.0 / 48.0  # lambda doubles ~every 2 days of age
LAMBDA_MAX = 0.30
# Lognormal spread of the players-tracked jump G at a refresh.
LN_G_SIGMA = 0.6
# Grid over (G quantile, lambda scale) for the refresh distribution.
G_GRID = [0.10, 0.30, 0.50, 0.70, 0.90]
LAMBDA_GRID = [(0.5, 0.25), (1.0, 0.50), (2.0, 0.25)]


def parse_time(text):
    """Parse an ISO-8601 UTC timestamp into an aware datetime."""
    if not text:
        return None
    try:
        return datetime.fromisoformat(str(text).replace("Z", "+00:00"))
    except ValueError:
        return None


def clamp_pct(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(lg):
    lg = max(-MAX_LOGIT, min(MAX_LOGIT, lg))
    return 100.0 / (1.0 + math.exp(-lg))


def round4(x):
    """Round half-up to 4 decimals, as JavaScript Math.round(x*1e4)/1e4."""
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if float(value).is_integer() else value


def median(values):
    vals = sorted(values)
    n = len(vals)
    if n == 0:
        return 0.0
    mid = n // 2
    if n % 2:
        return vals[mid]
    return 0.5 * (vals[mid - 1] + vals[mid])


def ols(xs, ys):
    """Least squares y ~ a + b x. Returns (a, b, resid_sd, se_b) or None."""
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = my - b * mx
    if n > 2:
        resid = [y - (a + b * x) for x, y in zip(xs, ys)]
        dof = n - 2
        var = sum(r * r for r in resid) / dof
        se_b = math.sqrt(var / sxx) if var > 0 else 0.0
        resid_sd = math.sqrt(var)
    else:
        resid_sd = 0.0
        se_b = float("inf")
    return (a, b, resid_sd, se_b)


def distinct_snapshots(case):
    """Compress history to distinct snapshots (one per Exophase refresh).

    Consecutive reads with identical players_tracked and rates are the
    same stale page; only the first-seen time of each distinct state is
    kept.  Reads whose players_tracked is null or <= 0 are dropped here:
    without a player count a rate cannot be turned into earners, so
    such a snapshot is unusable for every fit downstream -- dropping it
    in this one place keeps the (time, ln P) pairs in forecast_trend
    aligned and keeps a null inside the fit window from poisoning the
    whole trend fit.  Returns [(slot_datetime, players, [rates]], ...]
    oldest first.
    """
    snaps = []
    prev_key = None
    for read in case.get("history") or []:
        players = read.get("players_tracked")
        if players is None or players <= 0:
            continue
        rates = read.get("rates") or []
        key = (players, tuple(rates))
        if key == prev_key:
            continue
        prev_key = key
        when = parse_time(read.get("slot")) or parse_time(read.get("at"))
        if when is None:
            continue
        snaps.append((when, read.get("players_tracked"), list(rates)))
    return snaps


def earners(rate, players):
    """Earners implied by a printed rate (rates print to 2 decimals, so
    E is recovered to a fraction of a player at launch-night sizes)."""
    if rate is None or players is None or players <= 0:
        return None
    return rate * players / 100.0


def smoothed_logit(rate, players):
    """logit of (E + 0.5) / (P + 1): finite at rates 0 and 100."""
    if rate is None or players is None or players <= 0:
        return None
    e = rate * players / 100.0
    p = (e + 0.5) / (players + 1.0)
    p = min(0.9999, max(0.0001, p))
    return math.log(p / (1.0 - p))


# ---------------------------------------------------------------------------
# Early-earnability: how much of a new cohort earns a trophy in its first
# hours of play, read off the trophy's own description and tier.
# ---------------------------------------------------------------------------

_CHAPTER_RE = re.compile(r"chapter\s+(\d+)", re.IGNORECASE)
_FINAL_RE = re.compile(r"final mission", re.IGNORECASE)


def early_earnability(trophy):
    """Score in (0, 1]: a new player's chance of earning this trophy in
    their first hours, relative to the easiest non-intro trophy.

    Chapter trophies fall with chapter number; named story missions sit
    below chapter 1; quick open-world activities above grinds; the
    platinum and 100%-completion sit at the bottom.  Keyword order is
    first-match-wins, most specific first.
    """
    desc = (trophy.get("description") or "").lower()
    name = (trophy.get("name") or "").lower()
    text = desc + " " + name
    tier = (trophy.get("tier") or "").lower()
    if tier == "platinum":
        return 0.005
    m = _CHAPTER_RE.search(desc)
    if m:
        k = int(m.group(1))
        return 0.55 * (0.55 ** (k - 1))
    if _FINAL_RE.search(desc):
        return 0.06
    if "complete" in desc and ("mission" in desc or "chapter" in desc):
        return 0.12
    # Quick open-world activities a fresh player can hit in hour one.
    if "fishing" in text:
        return 0.30
    if "arcade" in text:
        return 0.20
    if "try each type of activity" in desc:
        return 0.15
    if "bowling" in text or "cocktail" in text or "showroom" in text:
        return 0.15
    if "wanted level" in text:
        return 0.12
    if "tank" in text or "rhino" in text:
        return 0.12
    if "discover" in desc and "vehicle" in text:
        return 0.10
    if "partner" in text:
        return 0.10
    if "rob " in text or text.startswith("rob"):
        return 0.08
    if "race" in text:
        return 0.05
    if "hunt" in text or "combat focus" in text:
        return 0.05
    # Grinds, collections and completionism: ~nothing in the first hours.
    if "100% completion" in text:
        return 0.005
    if "gold medal" in text or "stunt" in text or "waink" in text:
        return 0.01
    if "meditate" in text or "explore all" in text:
        return 0.02
    if "collectible card" in text or "stranger" in text:
        return 0.02
    if "challenge" in text:
        return 0.03
    if "earn $" in text or "spend $" in text or "purchase all" in text \
            or "register" in text or "bank vault" in text:
        return 0.03
    if tier == "gold":
        return 0.03
    if tier == "silver":
        return 0.04
    return 0.06


# ---------------------------------------------------------------------------
# Regime A: trend over distinct snapshots
# ---------------------------------------------------------------------------

def forecast_trend(case, snaps, release, target, horizon_days):
    trophies = case["trophies"]
    n_tr = len(trophies)
    # Time in ln(1 + hours since release) for each snapshot.
    xs = []
    for when, _, _ in snaps:
        hours = max(0.0, (when - release).total_seconds() / 3600.0)
        xs.append(math.log1p(hours))
    # Median gap between distinct snapshots -> refresh lag at the target.
    gaps = [(snaps[i + 1][0] - snaps[i][0]).total_seconds() / 3600.0
            for i in range(len(snaps) - 1)]
    lag_hours = 0.5 * median(gaps) if gaps else 0.0
    lag_hours = min(lag_hours, 4.0)
    target_eff = target.timestamp() - lag_hours * 3600.0
    t_star = math.log1p(max(0.0, (target_eff - release.timestamp()) / 3600.0))
    x_last = xs[-1]

    # --- P(t): log-log linear fit on ln P ---------------------------------
    # P growth decelerates, so take the smaller of the all-history slope
    # and the recent (last-24h) slope, anchored at the last snapshot.
    # The (x, ln P) pairs are built per snapshot in one zip, so a
    # snapshot can never shift the alignment (distinct_snapshots
    # already drops null-P snapshots, and the `if p` here is the
    # belt-and-braces guard for the same invariant).
    p_last = snaps[-1][1] or 1
    paired_all = [(x, math.log(max(1, p)))
                  for x, (_, p, _) in zip(xs, snaps) if p]
    fit_all = (ols([x for x, _ in paired_all],
                   [y for _, y in paired_all])
               if len(paired_all) >= 2 else None)
    win_idx = [j for j, (when, _, _) in enumerate(snaps)
               if (snaps[-1][0] - when).total_seconds() / 3600.0
               <= FIT_WINDOW_HOURS]
    fit_win = None
    if len(win_idx) >= 2:
        paired_win = [(xs[j], math.log(max(1, snaps[j][1])))
                      for j in win_idx if snaps[j][1]]
        fit_win = (ols([x for x, _ in paired_win],
                       [y for _, y in paired_win])
                   if len(paired_win) >= 2 else None)
    slopes_p = [f[1] for f in (fit_all, fit_win) if f is not None]
    b_p = max(0.0, min(slopes_p)) if slopes_p else 0.0
    p_target = max(float(p_last),
                   float(p_last) * math.exp(b_p * (t_star - x_last)))

    # --- per-trophy logit fits (last-24h window; all snaps if too few) ---
    slopes = []
    fits = []  # (xs, ys, ols result or None)
    for i in range(n_tr):
        tx, ty = [], []
        for j in win_idx:
            _, p, rates = snaps[j]
            if i < len(rates) and rates[i] is not None and p:
                lg = smoothed_logit(rates[i], p)
                if lg is not None:
                    tx.append(xs[j])
                    ty.append(lg)
        if len(tx) < 2:
            tx, ty = [], []
            for j, (_, p, rates) in enumerate(snaps):
                if i < len(rates) and rates[i] is not None and p:
                    lg = smoothed_logit(rates[i], p)
                    if lg is not None:
                        tx.append(xs[j])
                        ty.append(lg)
        fit = ols(tx, ty) if len(tx) >= 2 else None
        fits.append((tx, ty, fit))
        if fit is not None:
            slopes.append(fit[1])
    global_slope = median(slopes) if slopes else 0.0
    # Pooled residual SD across trophies.
    sq, dof = 0.0, 0
    for tx, ty, fit in fits:
        if fit is not None and len(tx) > 2:
            a, b = fit[0], fit[1]
            sq += sum((y - (a + b * x)) ** 2 for x, y in zip(tx, ty))
            dof += len(tx) - 2
    pooled_sd = math.sqrt(sq / dof) if dof > 0 else MIN_RESID_SIGMA
    pooled_sd = max(pooled_sd, MIN_RESID_SIGMA)

    rows = []
    for i, trophy in enumerate(trophies):
        tx, ty, fit = fits[i]
        last_rate = snaps[-1][2][i] if i < len(snaps[-1][2]) else None
        last_p = snaps[-1][1] or 1
        # Trophy at 100%: every tracked player holds it; forecast ~100.
        if last_rate is not None and last_rate >= 99.99:
            rows.append({"id": trophy["id"],
                         "q": [99.9, 99.97, 100, 100, 100]})
            continue
        if fit is None or not tx:
            # No usable history for this trophy: persist the last rate.
            base = last_rate if last_rate is not None else 1.0
            lg = logit(base)
            sigma = 0.5 * math.sqrt(horizon_days) + 0.2
            rows.append({"id": trophy["id"], "q": [
                round4(inv_logit(lg + z * sigma)) for z in Z]})
            continue
        n_i = len(tx)
        w = n_i / (n_i + SLOPE_SHRINK)
        b_star = w * fit[1] + (1.0 - w) * global_slope
        # Slope uncertainty, shrunk the same way (2-point fits: se inf ->
        # fall back to the cross-trophy spread of slopes).
        if math.isfinite(fit[3]) and fit[3] > 0:
            se_b = math.sqrt(w * w * fit[3] ** 2 +
                             (1.0 - w) ** 2 * 0.04)
        else:
            spread = (median([abs(s - global_slope) for s in slopes])
                      if slopes else 0.3)
            se_b = math.sqrt(w * w * (spread + 0.2) ** 2 +
                             (1.0 - w) ** 2 * 0.04)
        anchor = ty[-1]
        # Damped extrapolation: the recent slope overstates a saturating
        # curve over a 4.5-day horizon.
        med_logit = anchor + EXTRAP_DAMP * b_star * (t_star - x_last)
        # Hard floor: earners cannot fall, so rate >= 100 * E_last / P_target.
        e_last = earners(last_rate, last_p) if last_rate is not None else 0.0
        if e_last is not None and e_last > 0:
            floor_logit = logit(100.0 * e_last / p_target)
            med_logit = max(med_logit, floor_logit)
        resid = fit[2] if len(tx) > 2 else pooled_sd
        resid = max(resid, MIN_RESID_SIGMA)
        sigma = math.sqrt(
            resid ** 2
            + (se_b * abs(t_star - x_last)) ** 2
            + REFRESH_LAG_SIGMA ** 2
            + (MODEL_SIGMA_PER_SQRT_DAY * math.sqrt(horizon_days)) ** 2
        )
        qs = [round4(inv_logit(med_logit + z * sigma)) for z in Z]
        qs = sorted(qs)
        # Keep the 5% quantile from implying earners below the floor by
        # more than the P forecast's own uncertainty allows.
        if e_last is not None and e_last > 0:
            soft_floor = 100.0 * e_last / (p_target * 2.0)
            qs[0] = max(qs[0], round4(soft_floor))
            qs = sorted(qs)
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


# ---------------------------------------------------------------------------
# Regime B: flat history -- a refresh is the forecast
# ---------------------------------------------------------------------------

def refresh_probability(stale_hours, horizon_hours):
    """P(a refresh lands in the next horizon_hours | already stale for
    stale_hours), Weibull renewal with mean REFRESH_MEAN_HOURS."""
    lam = REFRESH_MEAN_HOURS / math.gamma(1.0 + 1.0 / REFRESH_SHAPE)
    k = REFRESH_SHAPE
    surv = math.exp(-((stale_hours + horizon_hours) / lam) ** k
                    + (stale_hours / lam) ** k)
    return min(0.95, max(0.05, 1.0 - surv))


def forecast_flat(case, snaps, release, target, horizon_days):
    trophies = case["trophies"]
    when0, p_last, rates_last = snaps[0]
    at = parse_time(case.get("at")) or when0
    stale_at_cut = max(0.0, (at - when0).total_seconds() / 3600.0)
    horizon_hours = horizon_days * 24.0
    stale_at_target = stale_at_cut + horizon_hours
    w_refresh = refresh_probability(stale_at_cut, horizon_hours)

    # Jump in players tracked at a refresh: median 1 + staleness (hours),
    # capped at 8 -- the documented first refresh was 638 -> 3,868 after
    # ~4.7 h stale (x6.06).
    g_med = min(8.0, max(2.0, 1.0 + stale_at_target))
    ln_g_med = math.log(g_med)
    nd = NormalDist()
    g_values = [math.exp(ln_g_med + LN_G_SIGMA * nd.inv_cdf(q))
                for q in G_GRID]
    g_weights = [0.2, 0.2, 0.2, 0.2, 0.2]

    # Extra trophies per new player beyond the intro trophy.
    age_hours = max(0.0, (target - release).total_seconds() / 3600.0)
    lam = min(LAMBDA_MAX,
              LAMBDA_LAUNCH * (1.0 + age_hours * LAMBDA_GROWTH_PER_DAY))

    # Early-earnability split (the 100% trophy is not split: it stays 100%).
    scores = []
    for i, trophy in enumerate(trophies):
        r = rates_last[i] if i < len(rates_last) else None
        if r is not None and r >= 99.99:
            scores.append(0.0)
        else:
            scores.append(early_earnability(trophy))
    total_score = sum(scores) or 1.0

    rows = []
    for i, trophy in enumerate(trophies):
        rate_last = rates_last[i] if i < len(rates_last) else None
        if rate_last is not None and rate_last >= 99.99:
            rows.append({"id": trophy["id"],
                         "q": [99.9, 99.97, 100, 100, 100]})
            continue
        e_last = earners(rate_last, p_last) if rate_last is not None else 0.0
        e_last = e_last or 0.0
        share = scores[i] / total_score
        # Mixture points on the logit scale: (logit, weight).
        points = []
        # No-refresh atom: the page stays exactly as read.
        points.append((logit(rate_last if rate_last is not None else 1.0),
                       (1.0 - w_refresh)))
        for g, gw in zip(g_values, g_weights):
            p_new = p_last * g
            d_p = p_new - p_last
            for lam_scale, lw in LAMBDA_GRID:
                g_i = lam * lam_scale * share
                e_new = e_last + g_i * d_p
                rate_new = 100.0 * e_new / p_new
                points.append((logit(rate_new), w_refresh * gw * lw))
        # Weighted quantiles of the mixture on the logit scale.
        points.sort(key=lambda pt: pt[0])
        total_w = sum(w for _, w in points)
        qs = []
        for tau in LEVELS:
            acc = 0.0
            value = points[-1][0]
            for lg, wgt in points:
                acc += wgt
                if acc >= tau * total_w:
                    value = lg
                    break
            qs.append(round4(inv_logit(value)))
        qs = sorted(qs)
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


# ---------------------------------------------------------------------------
# Fallback: house baseline persistence (never leave a case invalid)
# ---------------------------------------------------------------------------

def forecast_empty(case):
    """Honest fallback for a case with no usable history at all.

    There is no evidence about any trophy, so every trophy gets the
    neutral 1.0% rate (the same value the persistence fallback uses
    for a trophy with no known rate) with a deliberately small spread
    (+/-0.1 pp across the five quantiles): the spread says "no
    information" rather than pretending a wide, evidence-backed
    distribution, and the file is still a valid, complete forecast
    so the case is never silently skipped.
    """
    rows = []
    for trophy in case.get("trophies") or []:
        rows.append({"id": trophy["id"], "q": [0.9, 0.95, 1.0, 1.05, 1.1]})
    return {"trophies": rows}


def forecast_persistence(case):
    last = case["history"][-1]
    spread = 0.5 * math.sqrt(case["horizon_minutes"] / 1440.0)
    rows = []
    for i, trophy in enumerate(case["trophies"]):
        rate = last["rates"][i] if i < len(last["rates"]) else None
        lg = logit(rate if rate is not None else 1.0)
        rows.append({"id": trophy["id"], "q": [
            round4(inv_logit(lg + z * spread)) for z in Z]})
    return {"trophies": rows}


def forecast_case(case, name):
    if case.get("levels") != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}")
    if not case.get("history"):
        return forecast_empty(case)
    release_text = (case.get("game") or {}).get("release_date") or "2026-11-19"
    release = parse_time(release_text + "T00:00:00Z") or parse_time(
        case["history"][0].get("slot"))
    target = parse_time(case.get("target")) or parse_time(case.get("at"))
    horizon_days = case["horizon_minutes"] / 1440.0
    snaps = distinct_snapshots(case)
    if not snaps:
        return forecast_empty(case)
    if len(snaps) >= 2:
        return forecast_trend(case, snaps, release, target, horizon_days)
    return forecast_flat(case, snaps, release, target, horizon_days)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = None
        try:
            case = json.loads(path.read_text(encoding="utf-8"))
            forecast = forecast_case(case, path.name)
        except Exception:
            # Never leave a case invalid: fall back to persistence,
            # and to the empty-history forecast if even that is
            # impossible (e.g. no history at all).
            try:
                case = json.loads(path.read_text(encoding="utf-8"))
                forecast = forecast_persistence(case)
            except Exception:
                if case is None:
                    continue
                try:
                    forecast = forecast_empty(case)
                except Exception:
                    continue
        out = OUTPUT_DIR / f"{path.stem}.out"
        out.write_text(json.dumps(forecast, separators=(",", ":")) + "\n",
                       encoding="utf-8")


if __name__ == "__main__":
    main()
