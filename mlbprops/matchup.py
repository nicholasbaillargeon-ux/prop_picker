"""Combining batter, pitcher, and context into a single PA distribution.

The core operation is the **odds-ratio method** (the multinomial generalization
of Bill James's log5). For any outcome with batter rate ``b``, pitcher rate
``p``, and league rate ``l``:

    odds(matchup) = odds(b) * odds(p) / odds(l)

Dividing by the league odds is what prevents double-counting the baseline: if
both the batter and the pitcher are exactly league average, the result is
exactly league average. Every contextual effect (platoon, park, weather,
times-through-the-order) is then applied as a further multiplier on the same
odds scale, and the eight-outcome vector is renormalized once at the end.

Working in odds rather than raw probability matters at the tails. A 3% HR rate
and a park that boosts home runs 20% should not become 3.6% by naive
multiplication when the batter is already a 9% HR hitter -- the odds scale
keeps adjustments proportional to how much room the probability has left.
"""

from __future__ import annotations

import numpy as np

from .constants import (
    LEAGUE_RATE_VECTOR,
    N_OUTCOMES,
    O_1B,
    O_2B,
    O_3B,
    O_HR,
    O_K,
    OUTCOMES,
    PLATOON_SAME_HAND_OR,
    SWITCH,
    TTO_PENALTY_OR,
    WEATHER,
    WIND_XBH_DAMPING,
)

_EPS = 1e-9


def to_odds(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, _EPS, 1.0 - _EPS)
    return p / (1.0 - p)


def from_odds(o: np.ndarray) -> np.ndarray:
    o = np.clip(o, _EPS, None)
    return o / (1.0 + o)


def normalize(p: np.ndarray) -> np.ndarray:
    total = p.sum()
    if total <= 0:
        return LEAGUE_RATE_VECTOR.copy()
    return p / total


def odds_ratio_matchup(
    batter: np.ndarray,
    pitcher: np.ndarray,
    league: np.ndarray | None = None,
) -> np.ndarray:
    """Combine batter and pitcher outcome vectors via the odds-ratio method."""
    league = LEAGUE_RATE_VECTOR if league is None else league
    combined = to_odds(batter) * to_odds(pitcher) / to_odds(league)
    return normalize(from_odds(combined))


def apply_or_multipliers(probs: np.ndarray, mult: np.ndarray) -> np.ndarray:
    """Apply per-outcome odds multipliers and renormalize."""
    return normalize(from_odds(to_odds(probs) * np.clip(mult, _EPS, None)))


# ---------------------------------------------------------------------------
# Platoon
# ---------------------------------------------------------------------------

def platoon_multipliers(bats: str, throws: str) -> np.ndarray:
    """Odds multipliers for the batter given the handedness matchup.

    Switch hitters bat opposite-handed by definition, so they always get the
    platoon *advantage* side. Per-player platoon splits are not used: they need
    ~1000 PA per side to carry signal, and using the league-average effect is
    strictly better than fitting noise for all but a handful of hitters.
    """
    mult = np.ones(N_OUTCOMES, dtype=np.float64)
    if not bats or not throws:
        return mult

    bats = bats.upper()[:1]
    throws = throws.upper()[:1]

    if bats == SWITCH:
        same_hand = False
    else:
        same_hand = (bats == throws)

    for i, name in enumerate(OUTCOMES):
        same_or = PLATOON_SAME_HAND_OR.get(name, 1.0)
        # Opposite-handed gets the reciprocal effect, so the league-wide
        # average across matchups stays neutral.
        mult[i] = same_or if same_hand else (1.0 / same_or)
    return mult


# ---------------------------------------------------------------------------
# Park
# ---------------------------------------------------------------------------

def park_multipliers(park: dict | None) -> np.ndarray:
    """Odds multipliers from a park-factor record.

    Park factors are published as run/event multipliers relative to a neutral
    park; we apply them directly on the odds scale for the outcomes they
    describe and let renormalization absorb the balance into in-play outs.
    """
    mult = np.ones(N_OUTCOMES, dtype=np.float64)
    if not park:
        return mult
    mult[O_HR] = float(park.get("pf_hr", 1.0))
    mult[O_3B] = float(park.get("pf_3b", 1.0))
    mult[O_2B] = float(park.get("pf_2b", 1.0))
    mult[O_1B] = float(park.get("pf_1b", 1.0))
    mult[O_K] = float(park.get("pf_so", 1.0))
    return mult


# ---------------------------------------------------------------------------
# Weather
# ---------------------------------------------------------------------------

def wind_out_component(wind_speed_mph: float, wind_dir_deg: float,
                       cf_azimuth: float) -> float:
    """Component of wind blowing from home plate out toward center field.

    ``wind_dir_deg`` follows the meteorological convention: the direction the
    wind is coming *from*. A wind from 180 (out of the south) blows toward the
    north (0). We project that blowing-toward vector onto the home-plate ->
    center-field axis, so a crosswind contributes ~0 and a wind straight in
    from center returns a negative number.
    """
    blowing_toward = (wind_dir_deg + 180.0) % 360.0
    delta = np.radians(blowing_toward - cf_azimuth)
    return float(wind_speed_mph * np.cos(delta))


def weather_multipliers(park: dict | None, wx: dict | None) -> np.ndarray:
    """Odds multipliers from temperature, wind, humidity, and elevation.

    Elevation is applied even for domed parks (air density is air density), but
    wind and temperature are neutralized under a closed roof.
    """
    mult = np.ones(N_OUTCOMES, dtype=np.float64)
    if not park:
        return mult

    coef = WEATHER
    hr_or = 1.0

    # Elevation is intentionally absent: it lives in the park factors already.
    # See the note in constants.WEATHER.
    roof = str(park.get("roof", "open")).lower()
    roof_closed = roof == "dome" or (roof == "retractable"
                                     and bool((wx or {}).get("roof_closed")))

    wind_out = 0.0
    if wx and not roof_closed:
        temp_f = float(wx.get("temp_f", coef["ref_temp_f"]))
        hr_or *= 1.0 + coef["hr_or_per_deg_f"] * (temp_f - coef["ref_temp_f"])

        humidity = float(wx.get("humidity_pct", coef["ref_humidity_pct"]))
        hr_or *= 1.0 + coef["hr_or_per_pct_humidity"] * (
            humidity - coef["ref_humidity_pct"])

        wind_out = wind_out_component(
            float(wx.get("wind_mph", 0.0)),
            float(wx.get("wind_dir_deg", 0.0)),
            float(park.get("cf_azimuth", 0.0)),
        )
        hr_or *= 1.0 + coef["hr_or_per_mph_out"] * wind_out

    mult[O_HR] = max(hr_or, 0.2)

    # Wind that carries home runs also carries gap balls, but far less.
    if wind_out:
        xbh = 1.0 + coef["hr_or_per_mph_out"] * wind_out * WIND_XBH_DAMPING
        mult[O_2B] = max(xbh, 0.5)
        mult[O_3B] = max(xbh, 0.5)

    return mult


# ---------------------------------------------------------------------------
# Times through the order
# ---------------------------------------------------------------------------

def tto_multipliers(times_through: int) -> np.ndarray:
    """Batter-favoring odds multipliers for the Nth trip through the order.

    Applied to every offensive outcome so the *whole* distribution shifts
    toward the hitter, rather than only inflating home runs.
    """
    factor = TTO_PENALTY_OR.get(min(max(times_through, 1), 4), 1.13)
    mult = np.ones(N_OUTCOMES, dtype=np.float64)
    if factor == 1.0:
        return mult
    for i, name in enumerate(OUTCOMES):
        if name == "K":
            mult[i] = 1.0 / factor      # fewer strikeouts late
        elif name != "OUT":
            mult[i] = factor
    return mult


# ---------------------------------------------------------------------------
# Full assembly
# ---------------------------------------------------------------------------

def matchup_distribution(
    batter_probs: np.ndarray,
    pitcher_probs: np.ndarray,
    *,
    league: np.ndarray | None = None,
    bats: str = "R",
    throws: str = "R",
    park: dict | None = None,
    weather: dict | None = None,
    times_through: int = 1,
) -> np.ndarray:
    """The per-PA outcome distribution for one batter-pitcher-context matchup."""
    probs = odds_ratio_matchup(batter_probs, pitcher_probs, league)
    mult = (platoon_multipliers(bats, throws)
            * park_multipliers(park)
            * weather_multipliers(park, weather)
            * tto_multipliers(times_through))
    return apply_or_multipliers(probs, mult)
