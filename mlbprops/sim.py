"""Vectorized Monte Carlo game simulator.

Rather than modeling each prop in isolation, this simulates the whole game --
both lineups, both pitching staffs, base-out state, extra innings -- and reads
every prop off the same simulated games. That choice buys three things a
per-player closed-form model cannot get right:

* **Plate-appearance uncertainty.** Whether a leadoff hitter gets 4 PA or 5 is
  the single largest driver of his hits/TB distribution, and it depends on how
  the other eight hitters do. Simulating the lineup produces the correct
  joint distribution instead of assuming a fixed PA count.
* **Runs and RBI.** These are pure lineup-context stats. They are only
  well-defined if you know who bats around the player and what the base state
  looks like when he comes up.
* **Pitcher workload.** Strikeout and outs props hinge on how deep the starter
  goes, which depends on pitch count, which depends on how the opposing lineup
  performs against him. The hook is modeled explicitly and sampled per game.

Everything is vectorized across simulations: one Python-level step advances all
N games by one plate appearance simultaneously.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constants import (
    ADVANCE,
    N_OUTCOMES,
    O_1B,
    O_2B,
    O_3B,
    O_BB,
    O_HBP,
    O_HR,
    O_K,
    O_OUT,
    STARTER_HOOK_ER,
    STARTER_HOOK_PITCHES_MEAN,
    STARTER_HOOK_PITCHES_SD,
    STEAL_ATTEMPT,
    WILD_PITCH,
    TOTAL_BASES,
)

# Phase indices into the distribution table.
PHASE_BULLPEN = 4
N_PHASES = 5

# Expected pitches thrown per PA by outcome. Strikeouts and walks are long
# at-bats; balls in play are short. The PA-weighted mean lands at ~3.85.
PITCHES_BY_OUTCOME = np.array([4.8, 5.6, 3.5, 3.5, 3.4, 3.4, 3.4, 3.3])

# Runner "charged to" codes, used for earned-run attribution.
CHARGE_STARTER, CHARGE_BULLPEN, CHARGE_UNEARNED = 0, 1, 2

# Probability a runner on third scores on a ground-ball out with < 2 outs.
GROUNDOUT_3B_SCORES = 0.22

MAX_INNINGS = 12


@dataclass
class SimResult:
    """Per-simulation outcomes, indexed [sim, team, lineup_slot] or [sim, team].

    ``team`` is 0 for away, 1 for home.
    """

    pa: np.ndarray            # (N, 2, 9)
    hits: np.ndarray          # (N, 2, 9)
    total_bases: np.ndarray   # (N, 2, 9)
    home_runs: np.ndarray     # (N, 2, 9)
    runs: np.ndarray          # (N, 2, 9)
    rbi: np.ndarray           # (N, 2, 9)
    walks: np.ndarray         # (N, 2, 9)
    strikeouts: np.ndarray    # (N, 2, 9)  batter strikeouts
    singles: np.ndarray       # (N, 2, 9)
    doubles: np.ndarray       # (N, 2, 9)
    triples: np.ndarray       # (N, 2, 9)

    sp_outs: np.ndarray       # (N, 2) starting pitcher outs recorded
    sp_k: np.ndarray          # (N, 2)
    sp_er: np.ndarray         # (N, 2)
    sp_bf: np.ndarray         # (N, 2)
    sp_pitches: np.ndarray    # (N, 2)
    sp_walks: np.ndarray      # (N, 2)
    sp_hits: np.ndarray       # (N, 2)

    team_runs: np.ndarray     # (N, 2)
    innings: np.ndarray       # (N,) innings played

    @property
    def n_sims(self) -> int:
        return self.pa.shape[0]


def _sample_outcomes(probs: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Sample one categorical outcome per row of ``probs`` (shape (M, 8))."""
    cdf = np.cumsum(probs, axis=1)
    # Guard against float drift leaving the last cdf entry just under 1.
    cdf[:, -1] = 1.0
    u = rng.random((probs.shape[0], 1))
    return (u > cdf).sum(axis=1).astype(np.int8)


def simulate(
    dist: np.ndarray,
    gb_rate: np.ndarray,
    *,
    n_sims: int = 20000,
    hook_mean: np.ndarray | None = None,
    hook_sd: np.ndarray | None = None,
    seed: int | None = None,
) -> SimResult:
    """Simulate ``n_sims`` full games.

    Parameters
    ----------
    dist
        Shape ``(2, 9, 5, 8)``: for each batting team, lineup slot, and pitcher
        phase (starter times-through-order 1-4, then bullpen), the per-PA
        outcome distribution. Built by :mod:`mlbprops.pipeline`.
    gb_rate
        Shape ``(2, 9)``: ground-ball share of each batter's in-play outs,
        which controls his double-play and sacrifice-fly exposure.
    hook_mean, hook_sd
        Shape ``(2,)``: per-team starter pitch-count hook distribution.
    """
    if dist.shape != (2, 9, N_PHASES, N_OUTCOMES):
        raise ValueError(f"dist must be (2, 9, {N_PHASES}, {N_OUTCOMES}), "
                         f"got {dist.shape}")
    if gb_rate.shape != (2, 9):
        raise ValueError(f"gb_rate must be (2, 9), got {gb_rate.shape}")

    rng = np.random.default_rng(seed)
    N = int(n_sims)

    hook_mean = (np.full(2, STARTER_HOOK_PITCHES_MEAN) if hook_mean is None
                 else np.asarray(hook_mean, dtype=np.float64))
    hook_sd = (np.full(2, STARTER_HOOK_PITCHES_SD) if hook_sd is None
               else np.asarray(hook_sd, dtype=np.float64))

    # -- state ------------------------------------------------------------
    inning = np.ones(N, dtype=np.int16)
    half = np.zeros(N, dtype=np.int8)          # 0 = top (away bats), 1 = bottom
    outs = np.zeros(N, dtype=np.int8)
    alive = np.ones(N, dtype=bool)

    # Bases hold the lineup slot of the occupying runner, or -1 if empty.
    b1 = np.full(N, -1, dtype=np.int8)
    b2 = np.full(N, -1, dtype=np.int8)
    b3 = np.full(N, -1, dtype=np.int8)
    # Which pitcher is on the hook for each runner, for earned-run attribution.
    c1 = np.zeros(N, dtype=np.int8)
    c2 = np.zeros(N, dtype=np.int8)
    c3 = np.zeros(N, dtype=np.int8)

    score = np.zeros((N, 2), dtype=np.int16)
    lineup_idx = np.zeros((N, 2), dtype=np.int8)

    # Starter state, indexed by the pitching team.
    sp_in = np.ones((N, 2), dtype=bool)
    sp_bf = np.zeros((N, 2), dtype=np.int16)
    sp_outs = np.zeros((N, 2), dtype=np.int16)
    sp_k = np.zeros((N, 2), dtype=np.int16)
    sp_er = np.zeros((N, 2), dtype=np.int16)
    sp_bb = np.zeros((N, 2), dtype=np.int16)
    sp_h = np.zeros((N, 2), dtype=np.int16)
    sp_pitches = np.zeros((N, 2), dtype=np.float32)
    hook_at = np.maximum(
        rng.normal(hook_mean[None, :], hook_sd[None, :], size=(N, 2)), 35.0
    ).astype(np.float32)

    # Batter accumulators, indexed [sim, team, slot].
    shape3 = (N, 2, 9)
    acc_pa = np.zeros(shape3, dtype=np.int16)
    acc_h = np.zeros(shape3, dtype=np.int16)
    acc_tb = np.zeros(shape3, dtype=np.int16)
    acc_hr = np.zeros(shape3, dtype=np.int16)
    acc_r = np.zeros(shape3, dtype=np.int16)
    acc_rbi = np.zeros(shape3, dtype=np.int16)
    acc_bb = np.zeros(shape3, dtype=np.int16)
    acc_k = np.zeros(shape3, dtype=np.int16)
    acc_1b = np.zeros(shape3, dtype=np.int16)
    acc_2b = np.zeros(shape3, dtype=np.int16)
    acc_3b = np.zeros(shape3, dtype=np.int16)

    flat_r = acc_r.reshape(-1)
    flat_rbi = acc_rbi.reshape(-1)

    def score_runner(mask: np.ndarray, slot: np.ndarray, charge: np.ndarray,
                     bat_team: np.ndarray, sims: np.ndarray) -> None:
        """Credit a run to the runner and charge it to the right pitcher."""
        if not mask.any():
            return
        sel = np.where(mask)[0]
        s = sims[sel]
        team = bat_team[sel]
        slots = slot[sel]
        ok = slots >= 0
        if not ok.any():
            return
        s, team, slots = s[ok], team[ok], slots[ok]
        np.add.at(flat_r, (s * 2 + team) * 9 + slots, 1)
        np.add.at(score, (s, team), 1)
        # Charge the earned run to the opposing starter when he is responsible.
        ch = charge[sel][ok]
        starter_charged = ch == CHARGE_STARTER
        if starter_charged.any():
            np.add.at(sp_er, (s[starter_charged], 1 - team[starter_charged]), 1)

    # -- main loop --------------------------------------------------------
    # Each iteration advances every still-live game by exactly one PA.
    max_steps = 400
    for _ in range(max_steps):
        sims = np.where(alive)[0]
        if sims.size == 0:
            break

        bat_team = half[sims].astype(np.int64)
        pit_team = 1 - bat_team
        slot = lineup_idx[sims, bat_team].astype(np.int64)

        # Pitcher phase: starter times-through-order, or the bullpen.
        starter_here = sp_in[sims, pit_team]
        tto = np.clip(sp_bf[sims, pit_team] // 9, 0, 3).astype(np.int64)
        phase = np.where(starter_here, tto, PHASE_BULLPEN)

        probs = dist[bat_team, slot, phase]           # (M, 8)
        outcome = _sample_outcomes(probs, rng)

        m = sims.size
        r_gb = rng.random(m)
        r_adv1 = rng.random(m)
        r_adv2 = rng.random(m)
        r_adv3 = rng.random(m)

        # Current runner state for the active sims.
        s_b1, s_b2, s_b3 = b1[sims], b2[sims], b3[sims]
        s_c1, s_c2, s_c3 = c1[sims], c2[sims], c3[sims]
        s_outs = outs[sims]

        # ---- pre-PA baserunning ----------------------------------------
        # A runner on first with fewer than two outs may steal second or be
        # retired. Because this only fires with 0 or 1 outs it can never
        # produce the third out, so the plate appearance below always happens
        # and no control flow has to branch on it.
        steal_elig = (s_b1 >= 0) & (s_outs < 2)
        if steal_elig.any():
            roll = rng.random(sims.size)
            caught = steal_elig & (roll < STEAL_ATTEMPT["caught"])
            stole = (steal_elig & ~caught & (s_b2 < 0)
                     & (roll < STEAL_ATTEMPT["caught"] + STEAL_ATTEMPT["sb_success"]))
            # Caught: runner erased, out recorded against the current pitcher.
            s_outs = s_outs + caught.astype(np.int8)
            s_b1 = np.where(caught, np.int8(-1), s_b1)
            # Stolen: runner moves up, second base was verified empty.
            s_b2 = np.where(stole, s_b1, s_b2)
            s_c2 = np.where(stole, s_c1, s_c2)
            s_b1 = np.where(stole, np.int8(-1), s_b1)
            # Credit the out to the pitcher on the mound.
            if caught.any():
                cs_sims = sims[caught]
                cs_team = pit_team[caught]
                on_starter = sp_in[cs_sims, cs_team]
                if on_starter.any():
                    np.add.at(sp_outs, (cs_sims[on_starter],
                                        cs_team[on_starter]), 1)

        # ---- wild pitch / passed ball ----------------------------------
        # Every runner advances one base and the runner from third scores
        # without an RBI, which is the main way real games produce runs that
        # no batter is credited for.
        runners_on = (s_b1 >= 0) | (s_b2 >= 0) | (s_b3 >= 0)
        if runners_on.any():
            wp = runners_on & (rng.random(sims.size) < WILD_PITCH)
            if wp.any():
                score_runner(wp & (s_b3 >= 0), s_b3, s_c3, bat_team, sims)
                adv3 = wp & (s_b2 >= 0)
                adv2 = wp & (s_b1 >= 0)
                new3 = np.where(adv3, s_b2, np.where(wp, np.int8(-1), s_b3))
                new3_c = np.where(adv3, s_c2, s_c3)
                new2 = np.where(adv2, s_b1, np.where(wp, np.int8(-1), s_b2))
                new2_c = np.where(adv2, s_c1, s_c2)
                s_b3, s_c3 = new3, new3_c
                s_b2, s_c2 = new2, new2_c
                s_b1 = np.where(wp, np.int8(-1), s_b1)

        # Batter counting stats.
        flat_idx = (sims * 2 + bat_team) * 9 + slot
        np.add.at(acc_pa.reshape(-1), flat_idx, 1)
        np.add.at(acc_k.reshape(-1), flat_idx, (outcome == O_K).astype(np.int16))
        np.add.at(acc_bb.reshape(-1), flat_idx, (outcome == O_BB).astype(np.int16))
        is_hit = np.isin(outcome, (O_HR, O_3B, O_2B, O_1B))
        np.add.at(acc_h.reshape(-1), flat_idx, is_hit.astype(np.int16))
        np.add.at(acc_tb.reshape(-1), flat_idx,
                  TOTAL_BASES[outcome].astype(np.int16))
        np.add.at(acc_hr.reshape(-1), flat_idx, (outcome == O_HR).astype(np.int16))
        np.add.at(acc_1b.reshape(-1), flat_idx, (outcome == O_1B).astype(np.int16))
        np.add.at(acc_2b.reshape(-1), flat_idx, (outcome == O_2B).astype(np.int16))
        np.add.at(acc_3b.reshape(-1), flat_idx, (outcome == O_3B).astype(np.int16))

        # Pitcher counting stats (starter only; the bullpen is not a prop here).
        st = starter_here
        if st.any():
            ss, sp_t = sims[st], pit_team[st]
            np.add.at(sp_bf, (ss, sp_t), 1)
            np.add.at(sp_k, (ss, sp_t), (outcome[st] == O_K).astype(np.int16))
            np.add.at(sp_bb, (ss, sp_t), (outcome[st] == O_BB).astype(np.int16))
            np.add.at(sp_h, (ss, sp_t), is_hit[st].astype(np.int16))
            np.add.at(sp_pitches, (ss, sp_t),
                      PITCHES_BY_OUTCOME[outcome[st]].astype(np.float32))

        # Charge code for a batter who reaches base.
        charge_now = np.where(starter_here, CHARGE_STARTER,
                              CHARGE_BULLPEN).astype(np.int8)

        # New base state, built fresh each PA.
        n_b1 = np.full(m, -1, dtype=np.int8)
        n_b2 = np.full(m, -1, dtype=np.int8)
        n_b3 = np.full(m, -1, dtype=np.int8)
        n_c1 = np.zeros(m, dtype=np.int8)
        n_c2 = np.zeros(m, dtype=np.int8)
        n_c3 = np.zeros(m, dtype=np.int8)
        outs_added = np.zeros(m, dtype=np.int8)
        rbi_add = np.zeros(m, dtype=np.int16)

        occ1, occ2, occ3 = s_b1 >= 0, s_b2 >= 0, s_b3 >= 0
        batter_slot = slot.astype(np.int8)

        # ---- home run: everyone scores -------------------------------
        mask = outcome == O_HR
        if mask.any():
            score_runner(mask & occ3, s_b3, s_c3, bat_team, sims)
            score_runner(mask & occ2, s_b2, s_c2, bat_team, sims)
            score_runner(mask & occ1, s_b1, s_c1, bat_team, sims)
            score_runner(mask, batter_slot, charge_now, bat_team, sims)
            rbi_add[mask] = (1 + occ1[mask].astype(np.int16)
                             + occ2[mask].astype(np.int16)
                             + occ3[mask].astype(np.int16))

        # ---- triple: all runners score, batter to third ---------------
        mask = outcome == O_3B
        if mask.any():
            score_runner(mask & occ3, s_b3, s_c3, bat_team, sims)
            score_runner(mask & occ2, s_b2, s_c2, bat_team, sims)
            score_runner(mask & occ1, s_b1, s_c1, bat_team, sims)
            rbi_add[mask] = (occ1[mask].astype(np.int16)
                             + occ2[mask].astype(np.int16)
                             + occ3[mask].astype(np.int16))
            n_b3[mask] = batter_slot[mask]
            n_c3[mask] = charge_now[mask]

        # ---- double ---------------------------------------------------
        mask = outcome == O_2B
        if mask.any():
            score_runner(mask & occ3, s_b3, s_c3, bat_team, sims)
            score_runner(mask & occ2, s_b2, s_c2, bat_team, sims)
            # Runner from first either scores or stops at third.
            from1_scores = mask & occ1 & (r_adv1 < ADVANCE["double_1b_scores"])
            from1_third = mask & occ1 & ~from1_scores
            score_runner(from1_scores, s_b1, s_c1, bat_team, sims)
            n_b3[from1_third] = s_b1[from1_third]
            n_c3[from1_third] = s_c1[from1_third]
            rbi_add[mask] += (occ2[mask].astype(np.int16)
                              + occ3[mask].astype(np.int16))
            rbi_add[from1_scores] += 1
            n_b2[mask] = batter_slot[mask]
            n_c2[mask] = charge_now[mask]

        # ---- single ---------------------------------------------------
        mask = outcome == O_1B
        if mask.any():
            score_runner(mask & occ3, s_b3, s_c3, bat_team, sims)
            rbi_add[mask & occ3] += 1

            from2_scores = mask & occ2 & (r_adv1 < ADVANCE["single_2b_scores"])
            from2_third = mask & occ2 & ~from2_scores
            score_runner(from2_scores, s_b2, s_c2, bat_team, sims)
            rbi_add[from2_scores] += 1
            n_b3[from2_third] = s_b2[from2_third]
            n_c3[from2_third] = s_c2[from2_third]

            # The runner from first can only take third if it is still open.
            to_third = (mask & occ1 & (r_adv2 < ADVANCE["single_1b_to_3b"])
                        & ~from2_third)
            to_second = mask & occ1 & ~to_third
            n_b3[to_third] = s_b1[to_third]
            n_c3[to_third] = s_c1[to_third]
            n_b2[to_second] = s_b1[to_second]
            n_c2[to_second] = s_c1[to_second]

            n_b1[mask] = batter_slot[mask]
            n_c1[mask] = charge_now[mask]

        # ---- walk / hit by pitch: forced advancement only --------------
        mask = (outcome == O_BB) | (outcome == O_HBP)
        if mask.any():
            loaded = mask & occ1 & occ2 & occ3
            score_runner(loaded, s_b3, s_c3, bat_team, sims)
            rbi_add[loaded] += 1

            # Third stays occupied unless it was forced home.
            keep3 = mask & occ3 & ~loaded
            n_b3[keep3] = s_b3[keep3]
            n_c3[keep3] = s_c3[keep3]
            # Runner on second is forced to third only if first is occupied.
            push2 = mask & occ2 & occ1
            n_b3[push2] = s_b2[push2]
            n_c3[push2] = s_c2[push2]
            keep2 = mask & occ2 & ~occ1
            n_b2[keep2] = s_b2[keep2]
            n_c2[keep2] = s_c2[keep2]
            # Runner on first is always forced to second.
            push1 = mask & occ1
            n_b2[push1] = s_b1[push1]
            n_c2[push1] = s_c1[push1]

            n_b1[mask] = batter_slot[mask]
            n_c1[mask] = charge_now[mask]

        # ---- strikeout -------------------------------------------------
        mask = outcome == O_K
        if mask.any():
            outs_added[mask] = 1
            n_b1[mask], n_c1[mask] = s_b1[mask], s_c1[mask]
            n_b2[mask], n_c2[mask] = s_b2[mask], s_c2[mask]
            n_b3[mask], n_c3[mask] = s_b3[mask], s_c3[mask]

        # ---- ball in play, out ----------------------------------------
        mask = outcome == O_OUT
        if mask.any():
            gb_p = gb_rate[bat_team, slot]
            is_gb = mask & (r_gb < gb_p)
            is_fb = mask & ~is_gb
            can_adv = s_outs < 2

            # Carry the base state forward, then apply advancement.
            n_b1[mask], n_c1[mask] = s_b1[mask], s_c1[mask]
            n_b2[mask], n_c2[mask] = s_b2[mask], s_c2[mask]
            n_b3[mask], n_c3[mask] = s_b3[mask], s_c3[mask]
            outs_added[mask] = 1

            # Sacrifice fly.
            sf = (is_fb & can_adv & occ3
                  & (r_adv1 < ADVANCE["sacfly_scores"]))
            score_runner(sf, s_b3, s_c3, bat_team, sims)
            rbi_add[sf] += 1
            n_b3[sf] = -1

            # Ground-ball double play.
            dp = is_gb & can_adv & occ1 & (r_adv2 < ADVANCE["gidp"])
            outs_added[dp] = 2
            n_b1[dp] = -1
            # With nobody out, the runner from third scores on the DP. By rule
            # the batter is not credited with an RBI on a ground-ball double
            # play, so the run is scored without a corresponding RBI.
            dp_run = dp & (s_outs == 0) & occ3
            score_runner(dp_run, s_b3, s_c3, bat_team, sims)
            n_b3[dp_run] = -1

            # Ordinary ground out: runner from first may take second.
            gb_adv = (is_gb & ~dp & can_adv & occ1 & ~occ2
                      & (r_adv3 < ADVANCE["groundout_advance"]))
            n_b2[gb_adv] = s_b1[gb_adv]
            n_c2[gb_adv] = s_c1[gb_adv]
            n_b1[gb_adv] = -1

            # Runner from third scoring on an ordinary ground out.
            gb_score = (is_gb & ~dp & can_adv & occ3
                        & (r_adv3 > 1.0 - GROUNDOUT_3B_SCORES))
            score_runner(gb_score, s_b3, s_c3, bat_team, sims)
            rbi_add[gb_score] += 1
            n_b3[gb_score] = -1

        # ---- commit ----------------------------------------------------
        np.add.at(flat_rbi, flat_idx, rbi_add)

        if st.any():
            np.add.at(sp_outs, (sims[st], pit_team[st]),
                      outs_added[st].astype(np.int16))

        b1[sims], b2[sims], b3[sims] = n_b1, n_b2, n_b3
        c1[sims], c2[sims], c3[sims] = n_c1, n_c2, n_c3
        outs[sims] = s_outs + outs_added
        lineup_idx[sims, bat_team] = (slot + 1) % 9

        # Pull the starter once he passes his sampled hook, or blows up.
        pulled = (sp_pitches[sims, pit_team] >= hook_at[sims, pit_team]) | (
            sp_er[sims, pit_team] >= STARTER_HOOK_ER)
        if pulled.any():
            sp_in[sims[pulled], pit_team[pulled]] = False

        # ---- walk-off: home team takes the lead in the 9th or later ----
        walkoff = ((half[sims] == 1) & (inning[sims] >= 9)
                   & (score[sims, 1] > score[sims, 0]))
        if walkoff.any():
            alive[sims[walkoff]] = False

        # ---- half-inning rollover --------------------------------------
        ended = (outs[sims] >= 3) & alive[sims]
        if ended.any():
            e = sims[ended]
            outs[e] = 0
            b1[e] = b2[e] = b3[e] = -1
            was_top = half[e] == 0
            half[e] = 1 - half[e]
            # Advance the inning counter after the bottom half.
            bottom_done = e[~was_top]
            inning[bottom_done] += 1

            # Home team does not bat in the 9th when already ahead.
            top9_done = e[was_top]
            if top9_done.size:
                skip = (inning[top9_done] >= 9) & (
                    score[top9_done, 1] > score[top9_done, 0])
                alive[top9_done[skip]] = False

            # Regulation over with a decision, or the extra-inning cap hit.
            if bottom_done.size:
                decided = ((inning[bottom_done] > 9)
                           & (score[bottom_done, 0] != score[bottom_done, 1]))
                alive[bottom_done[decided]] = False
                capped = inning[bottom_done] > MAX_INNINGS
                alive[bottom_done[capped]] = False

            # Extra innings begin with an automatic runner on second, who is
            # unearned and does not belong to any pitcher.
            still = e[alive[e]]
            if still.size:
                extras = inning[still] >= 10
                ex = still[extras]
                if ex.size:
                    bt = half[ex].astype(np.int64)
                    b2[ex] = (lineup_idx[ex, bt].astype(np.int8) - 1) % 9
                    c2[ex] = CHARGE_UNEARNED

    return SimResult(
        pa=acc_pa, hits=acc_h, total_bases=acc_tb, home_runs=acc_hr,
        runs=acc_r, rbi=acc_rbi, walks=acc_bb, strikeouts=acc_k,
        singles=acc_1b, doubles=acc_2b, triples=acc_3b,
        sp_outs=sp_outs, sp_k=sp_k, sp_er=sp_er, sp_bf=sp_bf,
        sp_pitches=sp_pitches, sp_walks=sp_bb, sp_hits=sp_h,
        team_runs=score, innings=inning.astype(np.int16),
    )
