"""Per-PA outcome rate estimation.

Turns raw StatsAPI game logs into a shrunk, recency-weighted probability
vector over the eight PA outcomes.

Two ideas do the heavy lifting:

1. **Recency weighting.** Recent games get exponentially more weight than old
   ones, so a hitter on a genuine hot streak (or a pitcher whose velocity is
   down) moves the estimate. The half-life is deliberately long relative to how
   bettors think about "hot" -- most short-run streaks are noise, and a short
   half-life would chase it.

2. **Empirical-Bayes shrinkage.** A weighted rate off 40 PA is mostly noise. We
   shrink every outcome toward the league mean using that outcome's known
   stabilization point as the prior strength, so fast-stabilizing skills (K%,
   BB%) keep their signal and noisy ones (HR/PA for pitchers, triples) collapse
   toward league average. This is what stops the model from paying up for a
   1-for-30 slump or a 4-HR week.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .constants import (
    LEAGUE_GB_RATE,
    LEAGUE_RATE_VECTOR,
    N_OUTCOMES,
    O_1B,
    O_2B,
    O_3B,
    O_BB,
    O_HBP,
    O_HR,
    O_K,
    O_OUT,
    OUTCOMES,
    STABILIZATION_BF,
    STABILIZATION_PA,
)

# Half-life, in games, for the recency decay. ~30 games is roughly a month of
# playing time: long enough to be mostly signal, short enough that a real
# mid-season change in true talent shows up.
DEFAULT_HALF_LIFE_GAMES = 30.0

# Weight applied to prior-season counts relative to a current-season game.
# Early in the year this is what keeps estimates sane; by August the
# current-season sample dominates on volume alone.
PRIOR_SEASON_WEIGHT = 0.45


@dataclass
class RateEstimate:
    """A shrunk per-PA outcome distribution plus its provenance."""

    probs: np.ndarray                  # shape (8,), sums to 1
    effective_pa: float                # weighted sample size behind the estimate
    raw_pa: int                        # unweighted PA actually observed
    gb_rate: float = LEAGUE_GB_RATE    # ground-ball share of in-play outs
    pitches_per_pa: float = 0.0        # pitchers only; 0 when unknown
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.probs.shape != (N_OUTCOMES,):
            raise ValueError(f"probs must have shape ({N_OUTCOMES},)")
        total = float(self.probs.sum())
        if not math.isclose(total, 1.0, abs_tol=1e-6):
            raise ValueError(f"probs must sum to 1, got {total:.6f}")

    def as_dict(self) -> dict[str, float]:
        return {name: float(p) for name, p in zip(OUTCOMES, self.probs)}

    def __repr__(self) -> str:
        parts = " ".join(f"{n}={p:.3f}" for n, p in zip(OUTCOMES, self.probs))
        return f"<RateEstimate ePA={self.effective_pa:.0f} {parts}>"


# ---------------------------------------------------------------------------
# Parsing raw stat lines into outcome counts
# ---------------------------------------------------------------------------

def _num(stat: dict, key: str) -> float:
    """Read a numeric field, tolerating strings and missing keys."""
    v = stat.get(key, 0)
    if isinstance(v, (int, float)):
        return float(v)
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def counts_from_hitting(stat: dict) -> tuple[np.ndarray, float]:
    """Convert a hitting stat line into (counts vector, PA).

    ``OUT`` is computed as the residual so the eight categories are exhaustive
    by construction -- it absorbs in-play outs, sacrifices, and reached-on-error
    without any of them being double counted.
    """
    pa = _num(stat, "plateAppearances")
    if pa <= 0:
        # Older or partial lines may omit PA; reconstruct it.
        pa = (_num(stat, "atBats") + _num(stat, "baseOnBalls")
              + _num(stat, "hitByPitch") + _num(stat, "sacFlies")
              + _num(stat, "sacBunts") + _num(stat, "catchersInterference"))
    if pa <= 0:
        return np.zeros(N_OUTCOMES), 0.0

    hits = _num(stat, "hits")
    doubles = _num(stat, "doubles")
    triples = _num(stat, "triples")
    hr = _num(stat, "homeRuns")
    singles = max(0.0, hits - doubles - triples - hr)

    c = np.zeros(N_OUTCOMES)
    c[O_K] = _num(stat, "strikeOuts")
    c[O_BB] = _num(stat, "baseOnBalls")
    c[O_HBP] = _num(stat, "hitByPitch")
    c[O_HR] = hr
    c[O_3B] = triples
    c[O_2B] = doubles
    c[O_1B] = singles
    c[O_OUT] = max(0.0, pa - c.sum())
    return c, pa


def counts_from_pitching(stat: dict) -> tuple[np.ndarray, float]:
    """Convert a pitching stat line into (counts vector, batters faced)."""
    bf = _num(stat, "battersFaced")
    if bf <= 0:
        return np.zeros(N_OUTCOMES), 0.0

    hits = _num(stat, "hits")
    doubles = _num(stat, "doubles")
    triples = _num(stat, "triples")
    hr = _num(stat, "homeRuns")
    singles = max(0.0, hits - doubles - triples - hr)

    c = np.zeros(N_OUTCOMES)
    c[O_K] = _num(stat, "strikeOuts")
    c[O_BB] = _num(stat, "baseOnBalls")
    # Pitching lines use hitBatsmen; hitByPitch is present too but can be 0.
    c[O_HBP] = max(_num(stat, "hitBatsmen"), _num(stat, "hitByPitch"))
    c[O_HR] = hr
    c[O_3B] = triples
    c[O_2B] = doubles
    c[O_1B] = singles
    c[O_OUT] = max(0.0, bf - c.sum())
    return c, bf


def ground_ball_rate(stat_lines: list[dict]) -> float:
    """Ground-ball share of in-play outs, shrunk toward the league rate."""
    gb = sum(_num(s, "groundOuts") for s in stat_lines)
    air = sum(_num(s, "airOuts") for s in stat_lines)
    total = gb + air
    if total <= 0:
        return LEAGUE_GB_RATE
    prior = 120.0  # ~120 batted-ball outs to reach half weight
    return float((gb + prior * LEAGUE_GB_RATE) / (total + prior))


# ---------------------------------------------------------------------------
# Weighting and shrinkage
# ---------------------------------------------------------------------------

def neutralize(counts: np.ndarray, park: dict | None,
               is_home: bool | None) -> np.ndarray:
    """Strip a player's home-park effect out of one game's counts.

    Raw StatsAPI stat lines are *not* park neutral. A Rockies hitter has half
    his games at Coors, so his observed home-run and doubles rates already
    contain that park -- and the matchup model then multiplies by the game's
    park factor a second time. Left uncorrected, this double-counts the largest
    park effect in the sport in both directions: Colorado hitters get
    over-projected at home and, worse, stay over-projected on the road where
    the inflation should have been removed entirely.

    Only home games are adjusted. A player's road schedule averages out close
    to neutral across a season, so dividing road counts by anything would add
    noise without removing bias.

    Plate appearances are preserved: whatever the adjustment adds or removes
    from the event categories is absorbed by in-play outs.
    """
    if not park or not is_home:
        return counts
    adjusted = counts.astype(np.float64).copy()
    for idx, key in ((O_HR, "pf_hr"), (O_3B, "pf_3b"), (O_2B, "pf_2b"),
                     (O_1B, "pf_1b"), (O_K, "pf_so")):
        factor = float(park.get(key, 1.0) or 1.0)
        if factor > 0:
            adjusted[idx] /= factor
    # Rebalance so the PA total is unchanged.
    adjusted[O_OUT] += counts.sum() - adjusted.sum()
    return np.maximum(adjusted, 0.0)


def half_season_park(park: dict | None) -> dict | None:
    """A half-strength park record, for season-aggregate stat lines.

    A full-season total is roughly half home games and half road, so only about
    half of the park's effect is present in it. Averaging each factor toward
    1.0 applies the right fraction of the correction.
    """
    if not park:
        return None
    out = dict(park)
    for key in ("pf_hr", "pf_3b", "pf_2b", "pf_1b", "pf_so"):
        out[key] = (float(park.get(key, 1.0) or 1.0) + 1.0) / 2.0
    return out


def recency_weights(n_games: int,
                    half_life: float = DEFAULT_HALF_LIFE_GAMES) -> np.ndarray:
    """Exponential decay weights for game logs ordered oldest -> newest.

    The most recent game has weight 1.0; a game ``half_life`` games back has
    weight 0.5.
    """
    if n_games <= 0:
        return np.zeros(0)
    age = np.arange(n_games - 1, -1, -1, dtype=np.float64)
    return np.exp(-math.log(2.0) * age / max(half_life, 1e-9))


def weighted_counts(
    stat_lines: list[dict],
    parser,
    half_life: float = DEFAULT_HALF_LIFE_GAMES,
    home_park: dict | None = None,
) -> tuple[np.ndarray, float]:
    """Recency-weighted, park-neutralized outcome counts across game logs."""
    if not stat_lines:
        return np.zeros(N_OUTCOMES), 0.0
    w = recency_weights(len(stat_lines), half_life)
    total_c = np.zeros(N_OUTCOMES)
    total_pa = 0.0
    for weight, line in zip(w, stat_lines):
        c, pa = parser(line)
        if home_park is not None:
            c = neutralize(c, home_park, line.get("_home"))
        total_c += weight * c
        total_pa += weight * pa
    return total_c, total_pa


def shrink(
    counts: np.ndarray,
    n: float,
    league: np.ndarray | None = None,
    *,
    stabilization: dict[str, float] | None = None,
) -> np.ndarray:
    """Empirical-Bayes shrink observed counts toward the league distribution.

    Each outcome is shrunk independently with its own prior strength, then the
    vector is renormalized. Independent shrinkage followed by renormalization
    is an approximation to a full Dirichlet posterior, but it is the right
    approximation here: it lets K% (stabilizes at ~60 PA) stay sharp while
    pitcher HR/PA (~1300 BF) is pulled almost entirely to league average, which
    a single shared Dirichlet concentration could not do.
    """
    league = LEAGUE_RATE_VECTOR if league is None else league
    stabilization = stabilization or STABILIZATION_PA

    n = max(0.0, float(n))
    out = np.empty(N_OUTCOMES, dtype=np.float64)
    for i, name in enumerate(OUTCOMES):
        k = float(stabilization.get(name, 200.0))
        # Posterior mean of a Beta(k*p_lg, k*(1-p_lg)) prior with Binomial data.
        out[i] = (counts[i] + k * league[i]) / (n + k)

    total = out.sum()
    if total <= 0:
        return league.copy()
    return out / total


# ---------------------------------------------------------------------------
# Top-level estimators
# ---------------------------------------------------------------------------

def estimate_batter(
    current_logs: list[dict],
    prior_season_stat: dict | None = None,
    *,
    league: np.ndarray | None = None,
    half_life: float = DEFAULT_HALF_LIFE_GAMES,
    home_park: dict | None = None,
) -> RateEstimate:
    """Shrunk per-PA outcome distribution for a hitter."""
    c, n = weighted_counts(current_logs, counts_from_hitting, half_life,
                           home_park)
    raw_pa = int(sum(counts_from_hitting(s)[1] for s in current_logs))

    if prior_season_stat:
        pc, pn = counts_from_hitting(prior_season_stat)
        pc = neutralize(pc, half_season_park(home_park), True)
        c = c + PRIOR_SEASON_WEIGHT * pc
        n = n + PRIOR_SEASON_WEIGHT * pn

    probs = shrink(c, n, league, stabilization=STABILIZATION_PA)
    return RateEstimate(
        probs=probs,
        effective_pa=n,
        raw_pa=raw_pa,
        gb_rate=ground_ball_rate(current_logs),
        meta={"games": len(current_logs), "half_life": half_life},
    )


def estimate_pitcher(
    current_logs: list[dict],
    prior_season_stat: dict | None = None,
    *,
    league: np.ndarray | None = None,
    half_life: float = DEFAULT_HALF_LIFE_GAMES,
    home_park: dict | None = None,
) -> RateEstimate:
    """Shrunk per-BF outcome distribution for a pitcher (rates allowed)."""
    c, n = weighted_counts(current_logs, counts_from_pitching, half_life,
                           home_park)
    raw_bf = int(sum(counts_from_pitching(s)[1] for s in current_logs))

    if prior_season_stat:
        pc, pn = counts_from_pitching(prior_season_stat)
        pc = neutralize(pc, half_season_park(home_park), True)
        c = c + PRIOR_SEASON_WEIGHT * pc
        n = n + PRIOR_SEASON_WEIGHT * pn

    probs = shrink(c, n, league, stabilization=STABILIZATION_BF)

    # Pitches per PA drives the hook model, so estimate it from starts where
    # the pitcher actually threw a meaningful number of pitches.
    starts = [s for s in current_logs if _num(s, "numberOfPitches") > 0]
    ppa = 0.0
    if starts:
        pitches = sum(_num(s, "numberOfPitches") for s in starts)
        faced = sum(_num(s, "battersFaced") for s in starts)
        if faced > 0:
            ppa = pitches / faced

    return RateEstimate(
        probs=probs,
        effective_pa=n,
        raw_pa=raw_bf,
        gb_rate=ground_ball_rate(current_logs),
        pitches_per_pa=ppa,
        meta={"games": len(current_logs), "half_life": half_life},
    )


def recent_form(stat_lines: list[dict], games: int = 10) -> dict:
    """Raw totals over a player's last N games, for display.

    This is deliberately *not* what drives the projection -- the model already
    weights recent games more heavily via exponential decay over the whole
    season, which is a better estimator than a hard 10-game window. This exists
    so a reader can see the streak the model is declining to chase.
    """
    recent = [s for s in stat_lines if _num(s, "plateAppearances") > 0][-games:]
    if not recent:
        return {}
    counts = np.zeros(N_OUTCOMES)
    pa = 0.0
    for line in recent:
        c, n = counts_from_hitting(line)
        counts += c
        pa += n
    hits = counts[O_1B] + counts[O_2B] + counts[O_3B] + counts[O_HR]
    ab = pa - counts[O_BB] - counts[O_HBP]
    tb = counts[O_1B] + 2 * counts[O_2B] + 3 * counts[O_3B] + 4 * counts[O_HR]
    return {
        "games": len(recent),
        "pa": int(pa),
        "hits": int(hits),
        "hr": int(counts[O_HR]),
        "tb": int(tb),
        "bb": int(counts[O_BB]),
        "k": int(counts[O_K]),
        "avg": round(float(hits / ab), 3) if ab > 0 else None,
        "hits_per_game": round(float(hits / len(recent)), 2),
    }


# Internal market name -> how to read that market's value out of a single
# StatsAPI game-log line. Parsing game logs is this module's job; markets.py
# owns the *simulated* distributions, which are a different thing that happens
# to share these names.
#
# `singles` and `hits_runs_rbis` are not StatsAPI fields and are derived the
# same way the simulator derives them, so the historical column and the
# projected column mean the same thing.
MARKET_LOG_VALUE = {
    "hits": lambda s: _num(s, "hits"),
    "total_bases": lambda s: _num(s, "totalBases"),
    "home_runs": lambda s: _num(s, "homeRuns"),
    "rbi": lambda s: _num(s, "rbi"),
    "runs": lambda s: _num(s, "runs"),
    "singles": lambda s: max(0.0, _num(s, "hits") - _num(s, "doubles")
                             - _num(s, "triples") - _num(s, "homeRuns")),
    "doubles": lambda s: _num(s, "doubles"),
    "triples": lambda s: _num(s, "triples"),
    "walks": lambda s: _num(s, "baseOnBalls"),
    "batter_strikeouts": lambda s: _num(s, "strikeOuts"),
    "hits_runs_rbis": lambda s: _num(s, "hits") + _num(s, "runs") + _num(s, "rbi"),
    "pitcher_strikeouts": lambda s: _num(s, "strikeOuts"),
    "pitcher_outs": lambda s: _num(s, "outs"),
    "pitcher_earned_runs": lambda s: _num(s, "earnedRuns"),
    "pitcher_hits_allowed": lambda s: _num(s, "hits"),
    "pitcher_walks": lambda s: _num(s, "baseOnBalls"),
}


def recent_appearances(stat_lines: list[dict], games: int,
                       starts_only: bool = False) -> list[dict]:
    """The last N game-log lines a player actually appeared in.

    Hitters are filtered on plate appearances, so a defensive replacement who
    never batted does not consume one of the ten slots. Pitchers are filtered on
    ``gamesStarted``, which StatsAPI sets per line -- a starter's relief cameo or
    an opener appearance would otherwise be counted as a "start" and drag a
    strikeout hit rate down for a reason that has nothing to do with his form.
    """
    if starts_only:
        recent = [s for s in stat_lines if _num(s, "gamesStarted") > 0]
    else:
        recent = [s for s in stat_lines if _num(s, "plateAppearances") > 0]
    return recent[-games:]


def hit_rates(stat_lines: list[dict], lines_by_market: dict[str, list[float]],
              *, games: int, starts_only: bool = False) -> dict:
    """How often a player cleared each prop line in his last N appearances.

    This is a *descriptive* counter, not an estimator, and it is deliberately
    kept away from anything that feeds the projection. A 7-of-10 hit rate is the
    single most quoted number in prop betting and it is also one of the most
    misleading: it ignores the line's price, the opponent, the park, and the
    batting order the player happened to occupy, and at n=10 (or n=5 for
    pitchers) its standard error is enormous. It is shown so a reader can see
    the streak *and* see what the model makes of it -- when the model's
    probability and the hit rate disagree, that gap is the interesting part.

    Counts are exposed alongside the rate (``7`` of ``10``) rather than only a
    percentage, because "70%" and "7 of 10" invite very different confidence and
    only one of them is honest about the sample.

    Returns ``{}`` when the player has no qualifying games, so the caller can
    distinguish "no history" from "never cleared it".
    """
    recent = recent_appearances(stat_lines, games, starts_only=starts_only)
    if not recent:
        return {}

    out: dict[str, dict] = {}
    for market, lines in lines_by_market.items():
        getter = MARKET_LOG_VALUE.get(market)
        if getter is None or not lines:
            continue
        values = [getter(s) for s in recent]
        out[market] = {
            str(ln): {
                "hit": sum(1 for v in values if v > ln),
                "of": len(values),
                "rate": round(sum(1 for v in values if v > ln) / len(values), 3),
            }
            for ln in lines
        }

    if not out:
        return {}
    dates = [s.get("_date") for s in recent if s.get("_date")]
    return {
        "games": len(recent),
        "basis": "starts" if starts_only else "games",
        "from": dates[0] if dates else None,
        "to": dates[-1] if dates else None,
        "markets": out,
    }


def league_baseline(league_totals: dict) -> np.ndarray:
    """Build the league PA-outcome vector from StatsAPI league hitting totals.

    Falls back to the compiled-in constant if the payload is unusable, so a
    network hiccup degrades the model slightly rather than crashing it.
    """
    if not league_totals:
        return LEAGUE_RATE_VECTOR.copy()
    counts, pa = counts_from_hitting(league_totals)
    if pa <= 0 or counts.sum() <= 0:
        return LEAGUE_RATE_VECTOR.copy()
    vec = counts / pa
    total = vec.sum()
    if not (0.98 < total < 1.02):
        return LEAGUE_RATE_VECTOR.copy()
    return vec / total
