"""Tests for the modeling core.

These are deliberately weighted toward *invariants* rather than golden values:
the model's numbers should be free to improve, but the accounting identities
(runs equal team runs, probabilities sum to one, a league-average matchup
returns league-average rates) must never break.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlbprops import constants as C
from mlbprops.markets import Distribution
from mlbprops.matchup import (
    apply_or_multipliers,
    matchup_distribution,
    odds_ratio_matchup,
    platoon_multipliers,
    weather_multipliers,
    wind_out_component,
)
from mlbprops.odds import (
    american_to_decimal,
    american_to_prob,
    devig,
    expected_value,
    kelly_fraction,
    prob_to_american,
)
from mlbprops.rates import (
    counts_from_hitting,
    counts_from_pitching,
    estimate_batter,
    hit_rates,
    neutralize,
    recency_weights,
    recent_appearances,
    shrink,
)
from mlbprops.report import normalize_name
from mlbprops.sim import simulate

LEAGUE = C.LEAGUE_RATE_VECTOR


# ---------------------------------------------------------------------------
# Rates
# ---------------------------------------------------------------------------

def test_league_rates_are_a_distribution():
    assert C.LEAGUE_RATE_VECTOR.shape == (C.N_OUTCOMES,)
    assert C.LEAGUE_RATE_VECTOR.sum() == pytest.approx(1.0)


def test_hitting_counts_are_exhaustive():
    """Every PA lands in exactly one bucket, so counts must sum to PA."""
    line = {"plateAppearances": 5, "atBats": 4, "hits": 2, "doubles": 1,
            "triples": 0, "homeRuns": 1, "baseOnBalls": 1, "hitByPitch": 0,
            "strikeOuts": 1}
    counts, pa = counts_from_hitting(line)
    assert pa == 5
    assert counts.sum() == pytest.approx(pa)
    assert counts[C.O_HR] == 1
    assert counts[C.O_2B] == 1
    assert counts[C.O_1B] == 0  # 2 hits = 1 double + 1 homer, no singles


def test_pitching_counts_are_exhaustive():
    line = {"battersFaced": 25, "hits": 6, "doubles": 2, "triples": 0,
            "homeRuns": 1, "baseOnBalls": 2, "hitBatsmen": 1, "strikeOuts": 8}
    counts, bf = counts_from_pitching(line)
    assert bf == 25
    assert counts.sum() == pytest.approx(bf)
    assert counts[C.O_1B] == 3  # 6 hits - 2 doubles - 1 homer


def test_empty_stat_line_is_safe():
    counts, pa = counts_from_hitting({})
    assert pa == 0
    assert counts.sum() == 0


# ---------------------------------------------------------------------------
# Hit rates (displayed history, not model input)
# ---------------------------------------------------------------------------

def _hit_log(date, hits, pa=4, **extra):
    line = {"_date": date, "plateAppearances": pa, "atBats": pa, "hits": hits,
            "doubles": 0, "triples": 0, "homeRuns": 0, "totalBases": hits}
    line.update(extra)
    return line


def test_hit_rate_counts_only_games_over_the_line():
    # 1,2,0,3,1 hits -> three games over 0.5, two over 1.5, one over 2.5.
    logs = [_hit_log(f"2026-07-0{i + 1}", h) for i, h in enumerate([1, 2, 0, 3, 1])]
    out = hit_rates(logs, {"hits": [0.5, 1.5, 2.5]}, games=10)
    cells = out["markets"]["hits"]
    assert (cells["0.5"]["hit"], cells["0.5"]["of"]) == (4, 5)
    assert (cells["1.5"]["hit"], cells["1.5"]["of"]) == (2, 5)
    assert (cells["2.5"]["hit"], cells["2.5"]["of"]) == (1, 5)
    assert cells["1.5"]["rate"] == pytest.approx(0.4)


def test_hit_rate_window_takes_the_most_recent_games():
    """The window is the last N, not the first N -- an old hot streak must not
    outrank a current cold one."""
    logs = [_hit_log(f"2026-06-{i + 10}", 3) for i in range(10)]      # old: all overs
    logs += [_hit_log(f"2026-07-{i + 10}", 0) for i in range(10)]     # recent: none
    out = hit_rates(logs, {"hits": [0.5]}, games=10)
    assert out["games"] == 10
    assert out["markets"]["hits"]["0.5"]["hit"] == 0
    assert out["from"] == "2026-07-10"


def test_hit_rate_skips_games_the_hitter_did_not_bat_in():
    logs = [_hit_log("2026-07-01", 1), _hit_log("2026-07-02", 0, pa=0),
            _hit_log("2026-07-03", 2)]
    out = hit_rates(logs, {"hits": [0.5]}, games=10)
    assert out["games"] == 2                        # the 0-PA game is not a game
    assert out["markets"]["hits"]["0.5"]["hit"] == 2


def test_pitcher_hit_rate_counts_starts_not_appearances():
    """A starter's relief cameo must not consume one of the five slots -- it
    would drag every counting stat down for a reason unrelated to his form."""
    starts = [{"_date": f"2026-07-0{i + 1}", "gamesStarted": 1, "strikeOuts": 6,
               "outs": 18} for i in range(5)]
    relief = {"_date": "2026-07-06", "gamesStarted": 0, "strikeOuts": 0, "outs": 3}
    out = hit_rates(starts + [relief], {"pitcher_strikeouts": [5.5]},
                    games=5, starts_only=True)
    assert out["games"] == 5
    assert out["basis"] == "starts"
    assert out["markets"]["pitcher_strikeouts"]["5.5"]["hit"] == 5


def test_hit_rate_derives_singles_like_the_simulator():
    """`singles` is not a StatsAPI field; it must be hits minus extra-base hits
    so the historical column and the projected column mean the same thing."""
    logs = [_hit_log("2026-07-01", 3, doubles=1, triples=0, homeRuns=1)]  # 1 single
    out = hit_rates(logs, {"singles": [0.5, 1.5]}, games=10)
    assert out["markets"]["singles"]["0.5"]["hit"] == 1
    assert out["markets"]["singles"]["1.5"]["hit"] == 0


def test_hit_rate_is_empty_without_history():
    """Empty, not zeroed -- 'never played' and 'never cleared it' are different
    claims and the page renders them differently."""
    assert hit_rates([], {"hits": [0.5]}, games=10) == {}
    assert hit_rates([_hit_log("2026-07-01", 0, pa=0)], {"hits": [0.5]},
                     games=10) == {}


def test_hit_rate_sample_size_is_uniform_across_markets():
    """Every market is counted over the same games, which is what lets the
    artifact drop the per-cell `of` and fall back to the parent `games`."""
    logs = [_hit_log(f"2026-07-0{i + 1}", i) for i in range(5)]
    out = hit_rates(logs, {"hits": [0.5, 1.5], "runs": [0.5]}, games=10)
    for lines in out["markets"].values():
        for cell in lines.values():
            assert cell["of"] == out["games"]


def test_recent_appearances_respects_a_short_history():
    logs = [_hit_log("2026-07-01", 1), _hit_log("2026-07-02", 2)]
    assert len(recent_appearances(logs, 10)) == 2


def test_every_market_has_log_extraction():
    """A market the projection prices but the log parser cannot read would show
    a blank hit-rate column with no error anywhere."""
    from mlbprops.markets import ALL_MARKETS
    from mlbprops.rates import MARKET_LOG_VALUE
    assert set(ALL_MARKETS) == set(MARKET_LOG_VALUE)


def test_line_tables_agree_between_model_and_dashboard():
    """The dashboard's line picker and `common_lines` must offer the same lines.
    When they drifted, the picker's outer pitcher-outs lines had no precomputed
    probability and no hit rate -- a blank cell rather than a visible failure."""
    import re
    from pathlib import Path

    from mlbprops.markets import ALL_MARKETS, common_lines

    page = (Path(__file__).resolve().parent.parent
            / "mlbprops" / "web" / "dashboard.html").read_text(encoding="utf-8")
    block = re.search(r"const COMMON_LINES = \{(.*?)\n\};", page, re.S).group(1)
    js = {m.group(1): [float(x) for x in m.group(2).split(",") if x.strip()]
          for m in re.finditer(r"(\w+):\s*\[([^\]]*)\]", block)}
    for market in ALL_MARKETS:
        assert js.get(market) == common_lines(market), market


def test_shrinkage_pulls_small_samples_to_league():
    """A 10-PA sample of all strikeouts must not become a 100% K projection."""
    counts = np.zeros(C.N_OUTCOMES)
    counts[C.O_K] = 10
    out = shrink(counts, 10, LEAGUE)
    assert out.sum() == pytest.approx(1.0)
    assert out[C.O_K] < 0.45, "10 PA should stay close to the league mean"
    assert out[C.O_K] > LEAGUE[C.O_K], "but it should move in the right direction"


def test_shrinkage_respects_large_samples():
    counts = LEAGUE * 3000
    counts[C.O_K] = 3000 * 0.35
    out = shrink(counts, 3000, LEAGUE)
    assert out[C.O_K] > 0.30, "3000 PA of elite K% should mostly survive"


def test_shrinkage_ordering_follows_stabilization():
    """K% stabilizes faster than HR, so it should retain more of its signal."""
    n = 100
    k_counts = LEAGUE * n
    k_counts[C.O_K] = n * 0.40
    k_out = shrink(k_counts, n, LEAGUE)
    k_retained = (k_out[C.O_K] - LEAGUE[C.O_K]) / (0.40 - LEAGUE[C.O_K])

    hr_counts = LEAGUE * n
    hr_counts[C.O_HR] = n * 0.10
    hr_out = shrink(hr_counts, n, LEAGUE)
    hr_retained = (hr_out[C.O_HR] - LEAGUE[C.O_HR]) / (0.10 - LEAGUE[C.O_HR])

    assert k_retained > hr_retained


def test_recency_weights_decay():
    w = recency_weights(61, half_life=30.0)
    assert w[-1] == pytest.approx(1.0)          # newest game
    assert w[-31] == pytest.approx(0.5, abs=0.01)  # one half-life back
    assert w[0] < 0.3


def test_neutralize_preserves_plate_appearances():
    """Park adjustment must move events between buckets, never invent PAs."""
    counts, pa = counts_from_hitting(
        {"plateAppearances": 600, "hits": 150, "doubles": 30, "triples": 5,
         "homeRuns": 25, "baseOnBalls": 60, "strikeOuts": 140})
    coors = {"pf_hr": 1.08, "pf_3b": 2.03, "pf_2b": 1.23, "pf_1b": 1.09,
             "pf_so": 0.90}
    adjusted = neutralize(counts, coors, is_home=True)
    assert adjusted.sum() == pytest.approx(counts.sum())
    assert adjusted[C.O_3B] < counts[C.O_3B], "Coors inflates triples"
    assert adjusted[C.O_K] > counts[C.O_K], "Coors suppresses strikeouts"


def test_neutralize_skips_road_games():
    counts, _ = counts_from_hitting({"plateAppearances": 4, "hits": 1,
                                     "homeRuns": 1})
    park = {"pf_hr": 1.5}
    assert np.array_equal(neutralize(counts, park, is_home=False), counts)


def test_estimate_batter_returns_valid_distribution():
    logs = [{"plateAppearances": 4, "atBats": 4, "hits": 1, "homeRuns": 0,
             "strikeOuts": 1}] * 20
    est = estimate_batter(logs, None, league=LEAGUE)
    assert est.probs.sum() == pytest.approx(1.0)
    assert (est.probs >= 0).all()
    assert est.effective_pa > 0


# ---------------------------------------------------------------------------
# Matchup
# ---------------------------------------------------------------------------

def test_average_vs_average_returns_league():
    """The odds-ratio method must be an identity at the league baseline."""
    out = odds_ratio_matchup(LEAGUE, LEAGUE, LEAGUE)
    np.testing.assert_allclose(out, LEAGUE, atol=1e-9)


def test_matchup_moves_toward_the_stronger_side():
    batter = LEAGUE.copy()
    batter[C.O_HR] *= 3
    batter = batter / batter.sum()
    out = odds_ratio_matchup(batter, LEAGUE, LEAGUE)
    assert out[C.O_HR] > LEAGUE[C.O_HR]
    assert out.sum() == pytest.approx(1.0)


def test_two_strong_sides_compound():
    strong_bat = LEAGUE.copy(); strong_bat[C.O_K] *= 0.5
    strong_bat /= strong_bat.sum()
    weak_pitch = LEAGUE.copy(); weak_pitch[C.O_K] *= 0.5
    weak_pitch /= weak_pitch.sum()
    both = odds_ratio_matchup(strong_bat, weak_pitch, LEAGUE)
    one = odds_ratio_matchup(strong_bat, LEAGUE, LEAGUE)
    assert both[C.O_K] < one[C.O_K]


def test_platoon_is_symmetric():
    """Same- and opposite-handed effects must be reciprocal, so the league
    average across matchups stays neutral."""
    same = platoon_multipliers("R", "R")
    opp = platoon_multipliers("L", "R")
    np.testing.assert_allclose(same * opp, np.ones(C.N_OUTCOMES), atol=1e-9)


def test_switch_hitters_always_get_the_platoon_advantage():
    vs_rhp = platoon_multipliers(C.SWITCH, "R")
    vs_lhp = platoon_multipliers(C.SWITCH, "L")
    np.testing.assert_allclose(vs_rhp, vs_lhp)
    assert vs_rhp[C.O_K] < 1.0


def test_unknown_handedness_is_neutral():
    np.testing.assert_allclose(platoon_multipliers("", ""),
                               np.ones(C.N_OUTCOMES))


def test_or_multipliers_preserve_normalization():
    mult = np.full(C.N_OUTCOMES, 1.4)
    out = apply_or_multipliers(LEAGUE, mult)
    assert out.sum() == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Weather
# ---------------------------------------------------------------------------

def test_wind_straight_out_is_positive():
    # CF at due north (0). Wind FROM the south (180) blows toward the north.
    assert wind_out_component(10.0, 180.0, 0.0) == pytest.approx(10.0)


def test_wind_straight_in_is_negative():
    assert wind_out_component(10.0, 0.0, 0.0) == pytest.approx(-10.0)


def test_crosswind_is_neutral():
    assert wind_out_component(10.0, 90.0, 0.0) == pytest.approx(0.0, abs=1e-9)


def test_wind_out_boosts_home_runs():
    park = {"roof": "open", "cf_azimuth": 0, "elevation_ft": 0}
    out = weather_multipliers(park, {"wind_mph": 15, "wind_dir_deg": 180,
                                     "temp_f": 70, "humidity_pct": 50})
    assert out[C.O_HR] > 1.2


def test_dome_neutralizes_wind_and_temperature():
    park = {"roof": "dome", "cf_azimuth": 0, "elevation_ft": 0}
    out = weather_multipliers(park, {"wind_mph": 25, "wind_dir_deg": 180,
                                     "temp_f": 100, "humidity_pct": 20})
    np.testing.assert_allclose(out, np.ones(C.N_OUTCOMES))


def test_elevation_is_not_double_counted():
    """Altitude belongs to the park factors, not the weather term.

    Coors' measured HR index is only 1.08 because its thin air converts
    would-be homers into doubles and triples. A physics-derived elevation boost
    layered on top would misprice every Coors home-run prop.
    """
    sea_level = {"roof": "open", "cf_azimuth": 0, "elevation_ft": 0}
    denver = {"roof": "open", "cf_azimuth": 0, "elevation_ft": 5180}
    calm = {"wind_mph": 0, "wind_dir_deg": 0, "temp_f": 70, "humidity_pct": 50}
    np.testing.assert_allclose(weather_multipliers(sea_level, calm),
                               weather_multipliers(denver, calm))


def test_hot_weather_helps_home_runs():
    park = {"roof": "open", "cf_azimuth": 0}
    hot = weather_multipliers(park, {"temp_f": 95, "wind_mph": 0,
                                     "wind_dir_deg": 0, "humidity_pct": 50})
    cold = weather_multipliers(park, {"temp_f": 45, "wind_mph": 0,
                                      "wind_dir_deg": 0, "humidity_pct": 50})
    assert hot[C.O_HR] > cold[C.O_HR]


def test_full_matchup_stays_a_distribution():
    park = {"pf_hr": 1.26, "pf_1b": 1.0, "pf_2b": 0.93, "pf_3b": 0.64,
            "pf_so": 1.01, "roof": "open", "cf_azimuth": 30}
    wx = {"temp_f": 88, "wind_mph": 12, "wind_dir_deg": 210, "humidity_pct": 60}
    for tto in (1, 2, 3, 4):
        out = matchup_distribution(LEAGUE, LEAGUE, league=LEAGUE, bats="L",
                                   throws="R", park=park, weather=wx,
                                   times_through=tto)
        assert out.sum() == pytest.approx(1.0)
        assert (out > 0).all()


def test_times_through_the_order_favors_the_batter():
    first = matchup_distribution(LEAGUE, LEAGUE, league=LEAGUE, times_through=1)
    third = matchup_distribution(LEAGUE, LEAGUE, league=LEAGUE, times_through=3)
    assert third[C.O_K] < first[C.O_K]
    assert third[C.O_HR] > first[C.O_HR]


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def league_sim():
    dist = np.tile(LEAGUE, (2, 9, 5, 1))
    gb = np.full((2, 9), C.LEAGUE_GB_RATE)
    return simulate(dist, gb, n_sims=6000, seed=42)


def test_sim_run_accounting(league_sim):
    """Individual runs scored must reconcile with the team totals exactly."""
    assert league_sim.runs.sum() == league_sim.team_runs.sum()


def test_sim_rbi_never_exceeds_runs(league_sim):
    """Wild pitches and double plays produce runs with no RBI, never the
    reverse."""
    assert league_sim.rbi.sum() <= league_sim.runs.sum()


def test_sim_total_bases_consistent(league_sim):
    tb = (league_sim.singles + 2 * league_sim.doubles
          + 3 * league_sim.triples + 4 * league_sim.home_runs)
    np.testing.assert_array_equal(tb, league_sim.total_bases)


def test_sim_hits_equal_hit_types(league_sim):
    hits = (league_sim.singles + league_sim.doubles + league_sim.triples
            + league_sim.home_runs)
    np.testing.assert_array_equal(hits, league_sim.hits)


def test_sim_scoring_environment_is_realistic(league_sim):
    """Against league-average inputs the sim must reproduce league baseball."""
    assert 4.0 < league_sim.team_runs.mean() < 4.8
    assert 37.0 < league_sim.pa.sum(axis=2).mean() < 39.5


def test_sim_starter_workload_is_realistic(league_sim):
    assert 4.7 < league_sim.sp_outs.mean() / 3 < 5.7
    assert 4.4 < league_sim.sp_k.mean() < 5.6
    assert 2.0 < league_sim.sp_er.mean() < 3.1
    assert 78 < league_sim.sp_pitches.mean() < 96


def test_batting_order_gets_descending_plate_appearances(league_sim):
    """The leadoff spot must out-bat the ninth -- this is the effect a
    per-player model without lineup context cannot capture."""
    pa = league_sim.pa[:, 0, :].mean(axis=0)
    assert np.all(np.diff(pa) < 0)
    assert pa[0] - pa[8] > 0.6


def test_home_team_bats_less_often(league_sim):
    """The home team skips the ninth when it is already ahead."""
    assert league_sim.pa[:, 1, :].sum() < league_sim.pa[:, 0, :].sum()


def test_sim_rejects_bad_shapes():
    gb = np.full((2, 9), 0.4)
    with pytest.raises(ValueError):
        simulate(np.tile(LEAGUE, (2, 8, 5, 1)), gb, n_sims=10)
    with pytest.raises(ValueError):
        simulate(np.tile(LEAGUE, (2, 9, 5, 1)), np.full((2, 8), 0.4), n_sims=10)


def test_better_hitters_produce_more_offense():
    gb = np.full((2, 9), C.LEAGUE_GB_RATE)
    good = LEAGUE.copy()
    good[C.O_HR] *= 2.5
    good[C.O_1B] *= 1.3
    good = good / good.sum()

    base = simulate(np.tile(LEAGUE, (2, 9, 5, 1)), gb, n_sims=4000, seed=1)
    boosted_dist = np.tile(LEAGUE, (2, 9, 5, 1))
    boosted_dist[0] = np.tile(good, (9, 5, 1))
    boosted = simulate(boosted_dist, gb, n_sims=4000, seed=1)
    assert boosted.team_runs[:, 0].mean() > base.team_runs[:, 0].mean() + 1.0


# ---------------------------------------------------------------------------
# Markets
# ---------------------------------------------------------------------------

def test_distribution_probabilities_are_coherent():
    samples = np.array([0, 0, 1, 1, 1, 2, 2, 3])
    d = Distribution(market="hits", samples=samples)
    assert d.over(1.5) + d.under(1.5) == pytest.approx(1.0)
    assert d.mean == pytest.approx(1.25)


def test_push_is_removed_from_the_base():
    """On a whole-number line the stake is returned on a tie, so the bet is a
    wager on the non-push outcomes only."""
    samples = np.array([0, 1, 1, 2, 2, 2, 3, 3])  # 3 of 8 land exactly on 2
    d = Distribution(market="hits", samples=samples)
    assert d.push(2) == pytest.approx(3 / 8)
    over, under = d.resolve(2, "Over"), d.resolve(2, "Under")
    assert over + under == pytest.approx(1.0)
    assert over == pytest.approx((2 / 8) / (5 / 8))


def test_half_point_lines_have_no_push():
    d = Distribution(market="hits", samples=np.array([0, 1, 2, 3]))
    assert d.push(1.5) == 0.0
    assert d.resolve(1.5, "Over") == pytest.approx(d.over(1.5))


# ---------------------------------------------------------------------------
# Odds
# ---------------------------------------------------------------------------

def test_american_decimal_roundtrip():
    for a in (-250, -110, 100, 150, 400):
        assert prob_to_american(american_to_prob(a)) == pytest.approx(a, abs=1)


def test_favorite_and_underdog_conversions():
    assert american_to_decimal(100) == pytest.approx(2.0)
    assert american_to_decimal(-110) == pytest.approx(1.9091, abs=1e-4)
    assert american_to_prob(-110) == pytest.approx(0.5238, abs=1e-4)


@pytest.mark.parametrize("method", ["multiplicative", "power", "shin"])
def test_devig_produces_a_distribution(method):
    raw = [american_to_prob(-130), american_to_prob(105)]
    assert sum(raw) > 1.0, "the raw pair should carry vig"
    fair = devig(raw, method)
    assert sum(fair) == pytest.approx(1.0)
    assert all(0 < p < 1 for p in fair)


def test_devig_preserves_ordering():
    raw = [american_to_prob(-200), american_to_prob(160)]
    for method in ("multiplicative", "power", "shin"):
        fair = devig(raw, method)
        assert fair[0] > fair[1]


def test_shin_and_power_shift_longshots_more_than_proportional():
    """On a lopsided market the favourite-longshot correction should assign the
    longshot a lower fair probability than proportional devigging does."""
    raw = [american_to_prob(-450), american_to_prob(340)]
    prop = devig(raw, "multiplicative")
    shin = devig(raw, "shin")
    power = devig(raw, "power")
    assert shin[1] <= prop[1] + 1e-9
    assert power[1] <= prop[1] + 1e-9


def test_devig_rejects_unknown_method():
    with pytest.raises(ValueError):
        devig([0.5, 0.55], "nonsense")


def test_expected_value_signs():
    assert expected_value(0.60, 100) == pytest.approx(0.20)
    assert expected_value(0.40, -110) < 0


def test_kelly_is_zero_without_an_edge():
    assert kelly_fraction(0.45, -110) == 0.0
    assert kelly_fraction(0.50, -200) == 0.0


def test_kelly_is_capped():
    assert kelly_fraction(0.95, 200, multiplier=1.0, cap=0.02) == 0.02


def test_kelly_scales_with_the_multiplier():
    full = kelly_fraction(0.60, 100, multiplier=1.0, cap=1.0)
    quarter = kelly_fraction(0.60, 100, multiplier=0.25, cap=1.0)
    assert quarter == pytest.approx(full * 0.25)


# ---------------------------------------------------------------------------
# Name matching
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("a,b", [
    ("José Ramírez", "Jose Ramirez"),
    ("Ronald Acuña Jr.", "Ronald Acuna"),
    ("Michael Harris II", "Michael Harris"),
    ("J.T. Realmuto", "JT Realmuto"),
])
def test_name_normalization_matches_books_and_statsapi(a, b):
    assert normalize_name(a) == normalize_name(b)


def test_distinct_players_do_not_collide():
    assert normalize_name("Will Smith") != normalize_name("Willy Smith")


# ---------------------------------------------------------------------------
# Best-bet selection
# ---------------------------------------------------------------------------

from mlbprops.report import Recommendation, best_bets, score_bet  # noqa: E402


def _rec(**kw):
    base = dict(
        game="A @ B", player="Test Player", team="A", market="hits", line=1.5,
        side="Over", model_prob=0.40, market_prob=0.35, edge=0.05,
        best_book="dk", best_american=120, model_fair_american=150,
        ev=0.08, kelly=0.01, n_books=5, mean_outcome=1.1,
        lineup_slot=3, is_pitcher=False, confirmed_lineup=True,
        context={"effective_pa": 400},
    )
    base.update(kw)
    return Recommendation(**base)


def test_confirmed_lineup_scores_higher_than_projected():
    a = score_bet(_rec(confirmed_lineup=True))
    b = score_bet(_rec(confirmed_lineup=False))
    assert a.score > b.score
    assert "lineup confirmed" in a.reasons
    assert "lineup projected" in b.reasons


def test_thin_sample_is_penalized():
    deep = score_bet(_rec(context={"effective_pa": 500}))
    thin = score_bet(_rec(context={"effective_pa": 40}))
    assert deep.score > thin.score
    assert "thin sample" in thin.reasons


def test_more_books_scores_higher():
    assert score_bet(_rec(n_books=6)).score > score_bet(_rec(n_books=2)).score


def test_implausible_edges_are_damped():
    """A model claiming 4x the market's probability is far more likely wrong
    than right, so raw EV must not carry it to the top."""
    sane = score_bet(_rec(model_prob=0.40, market_prob=0.34, ev=0.15))
    wild = score_bet(_rec(model_prob=0.16, market_prob=0.04, ev=1.80))
    assert wild.plausibility < 0.5
    assert sane.plausibility == 1.0
    # Despite 12x the EV, the implausible play must not dominate outright.
    assert wild.score < sane.score * 6
    assert any("verify the line" in r for r in wild.reasons)


def test_tail_probabilities_are_discounted():
    mid = score_bet(_rec(model_prob=0.50, market_prob=0.46))
    tail = score_bet(_rec(model_prob=0.06, market_prob=0.05))
    assert tail.confidence < mid.confidence
    assert "tail probability" in tail.reasons


def test_best_bets_returns_one_per_game():
    recs = [
        _rec(game="A @ B", player="Low", ev=0.04),
        _rec(game="A @ B", player="High", ev=0.12),
        _rec(game="C @ D", player="Only", ev=0.06),
    ]
    overall, per_game = best_bets(recs)
    assert len(per_game) == 2
    assert {b.recommendation.game for b in per_game} == {"A @ B", "C @ D"}
    assert overall is per_game[0]
    assert overall.scope == "slate"
    # Within a game, the stronger play wins.
    a_game = next(b for b in per_game if b.recommendation.game == "A @ B")
    assert a_game.recommendation.player == "High"


def test_best_bets_sorted_by_score():
    overall, per_game = best_bets([
        _rec(game="A @ B", ev=0.05),
        _rec(game="C @ D", ev=0.20),
        _rec(game="E @ F", ev=0.10),
    ])
    scores = [b.score for b in per_game]
    assert scores == sorted(scores, reverse=True)
    assert overall.recommendation.game == "C @ D"


def test_best_bets_respects_thresholds():
    recs = [_rec(ev=0.01, n_books=5)]
    assert best_bets(recs, min_ev=0.05)[0] is None
    assert best_bets([_rec(ev=0.20, n_books=1)], min_books=3)[0] is None


def test_best_bets_handles_empty_input():
    overall, per_game = best_bets([])
    assert overall is None
    assert per_game == []


def test_best_bet_prefers_solid_play_over_longshot():
    """The headline selection must not be a thin-sample tail price."""
    solid = _rec(game="A @ B", player="Solid", model_prob=0.55,
                 market_prob=0.48, edge=0.07, ev=0.14, n_books=6,
                 confirmed_lineup=True, context={"effective_pa": 480})
    longshot = _rec(game="C @ D", player="Longshot", market="home_runs",
                    line=0.5, model_prob=0.13, market_prob=0.035, edge=0.095,
                    ev=1.90, best_american=2500, n_books=2,
                    confirmed_lineup=False, context={"effective_pa": 55})
    overall, _ = best_bets([solid, longshot])
    assert overall.recommendation.player == "Solid"


# ---------------------------------------------------------------------------
# Bullpen splits
# ---------------------------------------------------------------------------

from mlbprops.pipeline import Projector  # noqa: E402


class _FakeAPI:
    """Minimal StatsAPI stand-in for exercising bullpen resolution offline."""

    def __init__(self, rp=None, total=None, raise_split=False):
        self._rp, self._total, self._raise = rp, total, raise_split
        self.split_calls = 0

    def team_split_stats(self, team_id, season, sit_codes="rp", group="pitching"):
        self.split_calls += 1
        if self._raise:
            raise RuntimeError("upstream down")
        return {"rp": self._rp} if self._rp else {}

    def team_season_stats(self, team_id, season, group="pitching"):
        return self._total or {}

    def team_venues(self, season):
        return {}

    def league_hitting(self, season):
        return {}


def _relief_line(bf=1400, k=360):
    return {"battersFaced": bf, "strikeOuts": k, "baseOnBalls": 130,
            "hits": 290, "doubles": 55, "triples": 4, "homeRuns": 38,
            "hitBatsmen": 18, "gamesStarted": 0}


def test_bullpen_uses_the_relief_split_when_available():
    api = _FakeAPI(rp=_relief_line())
    est = Projector(api=api).bullpen_rates(114, 2026)
    assert est.meta["source"] == "relief split"
    assert api.split_calls == 1
    assert est.probs.sum() == pytest.approx(1.0)


def test_bullpen_reflects_team_specific_strikeout_rates():
    """The old flat multiplier priced every bullpen identically; a real split
    must separate a high-strikeout pen from a low-strikeout one."""
    high = Projector(api=_FakeAPI(rp=_relief_line(k=400))).bullpen_rates(1, 2026)
    low = Projector(api=_FakeAPI(rp=_relief_line(k=250))).bullpen_rates(2, 2026)
    assert high.probs[C.O_K] > low.probs[C.O_K] + 0.04


def test_bullpen_falls_back_when_split_is_missing():
    api = _FakeAPI(rp=None, total={"battersFaced": 6000, "strikeOuts": 1400,
                                   "baseOnBalls": 500, "hits": 1300,
                                   "homeRuns": 160, "hitBatsmen": 50})
    est = Projector(api=api).bullpen_rates(114, 2026)
    assert est.meta["source"] == "team total (fallback)"
    assert est.probs.sum() == pytest.approx(1.0)


def test_bullpen_falls_back_when_split_errors():
    api = _FakeAPI(raise_split=True,
                   total={"battersFaced": 6000, "strikeOuts": 1400,
                          "baseOnBalls": 500, "hits": 1300, "homeRuns": 160})
    est = Projector(api=api).bullpen_rates(114, 2026)
    assert est.meta["source"] == "team total (fallback)"


def test_bullpen_rejects_a_too_thin_split():
    """An early-season split off 40 batters is worse than the team total."""
    api = _FakeAPI(rp=_relief_line(bf=40, k=12),
                   total={"battersFaced": 6000, "strikeOuts": 1400,
                          "baseOnBalls": 500, "hits": 1300, "homeRuns": 160})
    est = Projector(api=api).bullpen_rates(114, 2026)
    assert est.meta["source"] == "team total (fallback)"


# ---------------------------------------------------------------------------
# Lineup watcher
# ---------------------------------------------------------------------------

import json as _json  # noqa: E402

import requests  # noqa: E402

from mlbprops.watch import SlateWatcher  # noqa: E402


class _StubWatcher(SlateWatcher):
    """Watcher with the network and the projector stubbed out."""

    def __init__(self, tmp_path, signatures):
        self._signatures = list(signatures)
        self._poll_calls = 0
        self.refreshed: list[set] = []
        super().__init__(date="2026-07-21", out_dir=tmp_path,
                         projector=object(), api=object(),
                         cache=_NullCache(), interval=20)

    def poll_signature(self):
        i = min(self._poll_calls, len(self._signatures) - 1)
        self._poll_calls += 1
        return self._signatures[i]

    def refresh(self, only=None):
        self.refreshed.append(set(only) if only else None)
        return len(only) if only else 0

    def write_payload(self):
        self.slate_path.write_text("{}", encoding="utf-8")


class _NullCache:
    def clear(self, namespace=None):
        return 0


def test_watcher_ignores_a_slate_with_no_changes(tmp_path):
    sig = {1: (False, "Pre-Game"), 2: (True, "Pre-Game")}
    w = _StubWatcher(tmp_path, [sig, sig])
    w.poll_signature()          # seed the baseline
    w._signature = sig
    assert w.tick() is False
    assert w.refreshed == []
    assert w.status.generation == 0


def test_watcher_rebuilds_only_the_game_that_changed(tmp_path):
    before = {1: (False, "Pre-Game"), 2: (False, "Pre-Game")}
    after = {1: (True, "Pre-Game"), 2: (False, "Pre-Game")}
    w = _StubWatcher(tmp_path, [after])
    w._signature = before
    assert w.tick() is True
    assert w.refreshed == [{1}], "only the confirmed game should re-project"
    assert w.status.generation == 1
    assert w.status.games_confirmed == 1


def test_watcher_bumps_generation_so_the_browser_refetches(tmp_path):
    w = _StubWatcher(tmp_path, [{1: (True, "Pre-Game")}])
    w._signature = {1: (False, "Pre-Game")}
    w.tick()
    first = w.status.generation
    w._signatures = [{1: (True, "Pre-Game"), 2: (True, "Pre-Game")}]
    w._poll_calls = 0
    w._signature = {1: (True, "Pre-Game"), 2: (False, "Pre-Game")}
    w.tick()
    assert w.status.generation == first + 1


def test_watcher_writes_status_the_server_can_serve(tmp_path):
    w = _StubWatcher(tmp_path, [{1: (True, "Pre-Game")}])
    w._signature = {1: (False, "Pre-Game")}
    w.tick()
    status = _json.loads(w.status_path.read_text())
    assert status["generation"] == 1
    assert status["games_confirmed"] == 1
    assert "changed_games" in status


def test_watcher_treats_postponement_as_a_change(tmp_path):
    w = _StubWatcher(tmp_path, [{1: (True, "Postponed")}])
    w._signature = {1: (True, "Pre-Game")}
    assert w.tick() is True
    assert w.refreshed == [{1}]


def test_watcher_counts_locked_games_as_not_pending(tmp_path):
    w = _StubWatcher(tmp_path, [{
        1: (True, "In Progress"), 2: (False, "Final"),
        3: (False, "Pre-Game"), 4: (False, "Postponed"),
    }])
    w._signature = {1: (True, "In Progress"), 2: (False, "Final"),
                    3: (False, "Pre-Game"), 4: (False, "Postponed")}
    w.tick()
    # Only the Pre-Game one can still change.
    assert w.status.games_pending == 1


def test_watcher_survives_a_failed_poll(tmp_path):
    class _Failing(_StubWatcher):
        def poll_signature(self):
            raise requests.RequestException("network down")

    import requests  # noqa: F811
    w = _Failing(tmp_path, [{}])
    assert w.tick() is False
    assert "poll failed" in w.status.message
