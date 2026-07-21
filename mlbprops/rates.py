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
