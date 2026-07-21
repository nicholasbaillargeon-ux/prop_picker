"""League-level constants, outcome taxonomy, and shrinkage priors."""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Plate-appearance outcome taxonomy.
#
# Every PA resolves to exactly one of these. Order is fixed and is relied upon
# by the simulation engine (sim.py indexes into probability matrices by these
# integer codes), so do not reorder without updating OUT_* constants below.
# ---------------------------------------------------------------------------
OUTCOMES = ("K", "BB", "HBP", "HR", "3B", "2B", "1B", "OUT")
N_OUTCOMES = len(OUTCOMES)

O_K, O_BB, O_HBP, O_HR, O_3B, O_2B, O_1B, O_OUT = range(N_OUTCOMES)

# Outcomes that put the batter on base without a ball in play.
FREE_PASS = (O_BB, O_HBP)
HIT_OUTCOMES = (O_HR, O_3B, O_2B, O_1B)

# Total bases awarded to the batter per outcome.
TOTAL_BASES = np.array([0, 0, 0, 4, 3, 2, 1, 0], dtype=np.float64)

# ---------------------------------------------------------------------------
# League baseline rates (per plate appearance).
#
# These are the "league average batter vs league average pitcher" anchor used
# by the odds-ratio matchup method. They are refreshed from StatsAPI at runtime
# by rates.league_baseline(); the values here are a 2024-25 fallback used when
# the network is unavailable.
# ---------------------------------------------------------------------------
LEAGUE_PA_RATES = {
    "K": 0.2230,
    "BB": 0.0830,
    "HBP": 0.0115,
    "HR": 0.0310,
    "3B": 0.0042,
    "2B": 0.0455,
    "1B": 0.1420,
    "OUT": 0.4598,
}

# Sanity: the taxonomy must be exhaustive.
assert abs(sum(LEAGUE_PA_RATES.values()) - 1.0) < 1e-9, "league rates must sum to 1"

LEAGUE_RATE_VECTOR = np.array([LEAGUE_PA_RATES[o] for o in OUTCOMES], dtype=np.float64)

# ---------------------------------------------------------------------------
# Empirical-Bayes stabilization points, in plate appearances.
#
# These are the sample sizes at which a rate's observed signal reaches ~50%
# reliability (Carleton / Fangraphs "stabilization" work). We use them directly
# as the prior strength in a Beta-Binomial shrinkage: with n == stabilization,
# the estimate sits halfway between the player's observed rate and the league
# mean. Skills that stabilize fast (K%, BB%) keep more of their observed
# signal; noisy outcomes (triples, BABIP-driven singles) get pulled hard toward
# the mean.
# ---------------------------------------------------------------------------
STABILIZATION_PA = {
    "K": 60.0,
    "BB": 120.0,
    "HBP": 240.0,
    "HR": 170.0,
    "3B": 1500.0,   # essentially a speed/park artifact; trust the league mean
    "2B": 1610.0,
    "1B": 290.0,
    "OUT": 80.0,
}

# Pitcher rates stabilize at different (generally larger) sample sizes because
# the pitcher controls less of the batted-ball outcome than the batter does.
STABILIZATION_BF = {
    "K": 70.0,
    "BB": 170.0,
    "HBP": 640.0,
    "HR": 1320.0,   # pitcher HR/PA is famously unstable -> heavy shrinkage
    "3B": 1500.0,
    "2B": 1610.0,
    "1B": 670.0,
    "OUT": 100.0,
}

# ---------------------------------------------------------------------------
# Platoon split coefficients (log-odds multipliers on the odds scale).
#
# Applied when batter and pitcher share handedness (same-side = tougher for the
# batter) or oppose it. Values are league-average platoon effects; per-player
# platoon splits are extremely noisy and are shrunk toward these anchors in
# matchup.platoon_adjust().
#
# Keyed by outcome -> odds multiplier for the batter in a SAME-handed matchup.
# The opposite-handed adjustment is the reciprocal, scaled so the league-wide
# PA-weighted average is unchanged.
# ---------------------------------------------------------------------------
PLATOON_SAME_HAND_OR = {
    "K": 1.145,    # more strikeouts vs same-side pitching
    "BB": 0.865,   # fewer walks
    "HBP": 0.900,
    "HR": 0.880,
    "3B": 0.950,
    "2B": 0.930,
    "1B": 0.960,
    "OUT": 1.000,  # residual; renormalization absorbs the balance
}

# Switch hitters always bat opposite-handed to the pitcher.
SWITCH = "S"

# ---------------------------------------------------------------------------
# Base-running advancement probabilities.
#
# Simplified but empirically grounded transition model. Keys describe the
# event; values are P(the more aggressive advancement).
# ---------------------------------------------------------------------------
ADVANCE = {
    # Runner on 2nd scoring on a single (vs stopping at 3rd).
    "single_2b_scores": 0.60,
    # Runner on 1st reaching 3rd on a single (vs stopping at 2nd).
    "single_1b_to_3b": 0.28,
    # Runner on 1st scoring on a double (vs stopping at 3rd).
    "double_1b_scores": 0.42,
    # Sacrifice fly: runner on 3rd scores on a fly-ball out with < 2 outs.
    "sacfly_scores": 0.52,
    # Ground-ball double play with a runner on 1st and < 2 outs.
    "gidp": 0.115,
    # Runner on 1st advancing to 2nd on a ground out (fielder's choice aside).
    "groundout_advance": 0.35,
}

# Fraction of in-play outs that are ground balls (league average). Used to
# split O_OUT into GIDP-eligible and sac-fly-eligible branches.
LEAGUE_GB_RATE = 0.435

# Baserunning events, evaluated once per PA that begins with a runner on first
# and fewer than two outs.
#
# Without these the simulator produces ~38.9 PA per team per game against a
# real-world ~38.1: strikeouts and batted-ball outs are not the only ways an
# inning ends. Caught stealings, pickoffs, and runners retired advancing supply
# roughly half an out per team per game, and that missing out inflates every
# counting prop by ~2%. Successful steals matter in the other direction -- they
# manufacture scoring position and lift RBI props.
STEAL_ATTEMPT = {
    "sb_success": 0.114,   # runner reaches second safely
    "caught": 0.045,       # runner retired; absorbs pickoffs and other TOOTBLANs
}

# Probability of a wild pitch or passed ball, per PA that begins with any
# runner on base. All runners advance one base.
#
# This is the mechanism that produces runs *without* an RBI. Modeling it is
# what moves the simulator's RBI-to-runs ratio off an impossible 1.000 and onto
# the real ~0.955, and it supplies the ~0.2 runs per game that advancement on
# batted balls alone leaves on the table. Also absorbs balks and errors, which
# are not modeled separately.
WILD_PITCH = 0.032

# ---------------------------------------------------------------------------
# Pitcher workload model.
# ---------------------------------------------------------------------------
PITCHES_PER_PA = 3.90

# Times-through-the-order penalty, as an odds multiplier on batter success.
# The 2nd and 3rd trips through a lineup are measurably worse for the starter.
TTO_PENALTY_OR = {1: 1.00, 2: 1.045, 3: 1.095, 4: 1.13}

# Starter hook: mean and sd of the pitch count at which a starter is pulled.
# Sampled per simulated game so that "how deep does he go" uncertainty (the
# dominant variance source for strikeout and outs props) is modeled explicitly
# rather than assumed away.
STARTER_HOOK_PITCHES_MEAN = 88.0
STARTER_HOOK_PITCHES_SD = 16.0

# A starter is also pulled if he allows this many earned runs, with some noise.
STARTER_HOOK_ER = 5

# ---------------------------------------------------------------------------
# Weather / physics coefficients for home-run carry.
#
# Batted-ball carry distance responds to air density. These coefficients
# convert weather deltas into an odds multiplier on HR.
# ---------------------------------------------------------------------------
# NOTE ON ELEVATION: there is deliberately no elevation term here.
#
# Empirical park factors already contain it. Coors Field's measured HR index is
# only 1.08 -- its thin air inflates *everything*, so the resulting huge outfield
# turns would-be home runs into doubles and triples instead (2B index 1.23, 3B
# index 2.03). Adding a physics-derived altitude boost on top of pf_hr would
# multiply Coors home runs by roughly 1.44 twice over and badly misprice the
# single most distinctive park in the league. Altitude enters the model through
# the park factors, and only through them.
WEATHER = {
    # Odds multiplier on HR per degree F above the reference.
    "hr_or_per_deg_f": 0.0075,
    # Odds multiplier on HR per mph of wind blowing straight out to center.
    "hr_or_per_mph_out": 0.0300,
    # Relative humidity: humid air is *less* dense, so carry increases slightly.
    "hr_or_per_pct_humidity": 0.0009,
    # Reference conditions at which all multipliers are 1.0. These are the
    # league-average game conditions, so what the weather term actually applies
    # is each game's deviation from a typical night -- the park's own baseline
    # climate is already inside its park factors.
    "ref_temp_f": 70.0,
    "ref_humidity_pct": 50.0,
}

# Wind also nudges doubles/triples (balls in the gap) in the same direction as
# home runs, but with roughly a third of the sensitivity.
WIND_XBH_DAMPING = 0.33

__all__ = [
    "OUTCOMES", "N_OUTCOMES", "TOTAL_BASES", "HIT_OUTCOMES", "FREE_PASS",
    "O_K", "O_BB", "O_HBP", "O_HR", "O_3B", "O_2B", "O_1B", "O_OUT",
    "LEAGUE_PA_RATES", "LEAGUE_RATE_VECTOR", "STABILIZATION_PA",
    "STABILIZATION_BF", "PLATOON_SAME_HAND_OR", "SWITCH", "ADVANCE",
    "LEAGUE_GB_RATE", "PITCHES_PER_PA", "TTO_PENALTY_OR",
    "STARTER_HOOK_PITCHES_MEAN", "STARTER_HOOK_PITCHES_SD",
    "STARTER_HOOK_ER", "WEATHER", "WIND_XBH_DAMPING",
]
