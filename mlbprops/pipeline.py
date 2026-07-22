"""End-to-end slate projection.

Pulls schedule, lineups, player histories, park, and weather; builds the
per-matchup PA distributions; simulates each game; and returns projections for
every batter and starting pitcher.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import date as date_cls
from datetime import timedelta

import numpy as np

from .constants import LEAGUE_GB_RATE, LEAGUE_RATE_VECTOR, N_OUTCOMES, O_K
from .markets import (
    BATTER_MARKETS,
    PITCHER_MARKETS,
    PlayerProjection,
    batter_distributions,
    common_lines,
    game_markets,
    pitcher_distributions,
)
from .matchup import matchup_distribution, normalize
from .parks import get_park
from .rates import (
    RateEstimate,
    estimate_batter,
    estimate_pitcher,
    half_season_park,
    hit_rates,
    league_baseline,
    recent_form,
)
from .sim import N_PHASES, simulate
from .statsapi import Game, Lineup, Player, StatsAPI
from .weather import WeatherClient

log = logging.getLogger(__name__)

# Lookback windows for the displayed hit rates. Ten games is the bettor's
# convention for hitters and lands on a roughly two-week span. Pitchers get five
# *starts* instead: a starter appears every fifth day, so ten would reach back
# nearly two months and stop describing anything current.
BATTER_HIT_RATE_GAMES = 10
PITCHER_HIT_RATE_STARTS = 5

# Which lines to count against, per market -- the same standard lines the
# projection already precomputes, so the historical and modelled columns on a
# board row always refer to the identical number.
BATTER_HIT_RATE_LINES = {m: common_lines(m) for m in BATTER_MARKETS}
PITCHER_HIT_RATE_LINES = {m: common_lines(m) for m in PITCHER_MARKETS}


def _bvp_summary(stat: dict | None) -> dict:
    """Compact career batter-vs-pitcher line, or empty when never faced."""
    if not stat:
        return {}
    pa = int(stat.get("plateAppearances", 0) or 0)
    if pa <= 0:
        return {}
    hits = int(stat.get("hits", 0) or 0)
    ab = int(stat.get("atBats", 0) or 0)
    return {
        "pa": pa,
        "ab": ab,
        "hits": hits,
        "hr": int(stat.get("homeRuns", 0) or 0),
        "bb": int(stat.get("baseOnBalls", 0) or 0),
        "k": int(stat.get("strikeOuts", 0) or 0),
        "avg": round(hits / ab, 3) if ab else None,
    }

# Relievers miss more bats than starters do, largely because they face hitters
# once and can throw max-effort. Applied as an odds multiplier to the team's
# overall pitching line when using it as a bullpen proxy.
BULLPEN_K_BOOST = 1.09

MAX_WORKERS = 8


@dataclass
class GameProjection:
    game: Game
    park: dict
    weather: dict
    projections: list[PlayerProjection] = field(default_factory=list)
    lineups_confirmed: bool = False
    notes: list[str] = field(default_factory=list)
    team_runs: dict[str, float] = field(default_factory=dict)
    bullpen: dict = field(default_factory=dict)
    game_markets: dict = field(default_factory=dict)
    lineup_order: dict = field(default_factory=dict)


class Projector:
    def __init__(
        self,
        api: StatsAPI | None = None,
        weather_client: WeatherClient | None = None,
        *,
        n_sims: int = 20000,
        seed: int | None = 20260721,
        include_bvp: bool = True,
    ):
        self.api = api or StatsAPI()
        self.weather = weather_client or WeatherClient()
        self.n_sims = n_sims
        self.seed = seed
        self.include_bvp = include_bvp
        self._league: np.ndarray | None = None
        self._player_cache: dict[int, Player] = {}
        self._team_venues: dict[int, int] | None = None

    def home_park(self, team_id: int, season: int) -> dict | None:
        """The park a team plays its home games in, used to de-bias its
        players' raw stat lines before the game's own park is applied."""
        if self._team_venues is None:
            try:
                self._team_venues = self.api.team_venues(season)
            except Exception:  # noqa: BLE001
                self._team_venues = {}
        venue = self._team_venues.get(int(team_id))
        return get_park(venue) if venue else None

    # -- baselines ---------------------------------------------------------

    def league_vector(self, season: int) -> np.ndarray:
        """League-average PA outcome vector, fetched once per run."""
        if self._league is None:
            try:
                totals = self.api.league_hitting(season)
                self._league = league_baseline(totals)
                log.info("league baseline: K=%.3f BB=%.3f HR=%.3f",
                         self._league[0], self._league[1], self._league[3])
            except Exception as exc:  # noqa: BLE001 - degrade, do not crash
                log.warning("league baseline fetch failed (%s); using defaults", exc)
                self._league = LEAGUE_RATE_VECTOR.copy()
        return self._league

    # -- per-player estimation --------------------------------------------

    def batter_rates(self, player_id: int, season: int,
                     home_park: dict | None = None) -> RateEstimate:
        logs = self.api.game_log(player_id, season, "hitting")
        prior = {}
        try:
            prior = self.api.season_stats(player_id, season - 1, "hitting")
        except Exception:  # noqa: BLE001
            pass
        return estimate_batter(logs, prior, league=self.league_vector(season),
                               home_park=home_park)

    def pitcher_rates(self, player_id: int, season: int,
                      home_park: dict | None = None) -> RateEstimate:
        logs = self.api.game_log(player_id, season, "pitching")
        prior = {}
        try:
            prior = self.api.season_stats(player_id, season - 1, "pitching")
        except Exception:  # noqa: BLE001
            pass
        return estimate_pitcher(logs, prior, league=self.league_vector(season),
                                home_park=home_park)

    def _pitching_logs(self, player_id: int, season: int) -> list[dict]:
        """Pitching game logs for display, never fatal.

        ``pitcher_rates`` has already pulled these for this player earlier in the
        same projection, so this is a disk-cache hit rather than a second network
        call -- the same reason ``_hook_params`` refetches rather than threading
        the list through. A hit-rate column is decoration; if the fetch fails,
        the projection itself is still sound, so this swallows the error and
        returns nothing to display.
        """
        try:
            return self.api.game_log(player_id, season, "pitching")
        except Exception:  # noqa: BLE001
            log.warning("no pitching logs for %s; hit rates omitted", player_id)
            return []

    def bullpen_rates(self, team_id: int, season: int) -> RateEstimate:
        """The team's actual relief corps, from StatsAPI's ``rp`` situation split.

        This is a real measured aggregate of every relief appearance the team
        has thrown, not a proxy. Three things follow from that:

        * **Swingmen are handled correctly.** A pitcher's relief outings count
          here and his starts do not, which a team-total line cannot separate.
        * **It is usage-weighted by construction.** Relievers who actually
          pitch contribute proportionally more batters faced, so the aggregate
          already reflects how the pen is really deployed.
        * **The strikeout rate is measured per team**, replacing the flat
          league-wide boost this used to apply. Real relief K/BF runs about 12%
          above the same club's rotation, but the gap varies by team, and a
          constant multiplier priced every bullpen identically.

        Falls back to the team total with the old flat boost if the split is
        unavailable -- early in a season, or on an API hiccup.
        """
        league = self.league_vector(season)
        park = self.home_park(team_id, season)

        stat: dict = {}
        used_split = False
        try:
            splits = self.api.team_split_stats(team_id, season, "rp", "pitching")
            candidate = splits.get("rp") or {}
            # Guard against a split that exists but is too thin to lean on.
            if float(candidate.get("battersFaced", 0) or 0) >= 100:
                stat, used_split = candidate, True
        except Exception as exc:  # noqa: BLE001
            log.info("relief split unavailable for team %s (%s)", team_id, exc)

        if not used_split:
            try:
                stat = self.api.team_season_stats(team_id, season, "pitching")
            except Exception:  # noqa: BLE001
                stat = {}

        # A season aggregate is roughly half home games, so it carries about
        # half the home park's effect. Flagging the line as a home game lets
        # the half-strength park record remove the right fraction.
        if stat:
            stat = dict(stat, _home=True)

        est = estimate_pitcher([stat] if stat else [], None, league=league,
                               home_park=half_season_park(park))

        if not used_split:
            # Without a real split, restore the old approximation: the team
            # total blends rotation and bullpen, so nudge strikeouts up.
            probs = est.probs.copy()
            probs[O_K] *= BULLPEN_K_BOOST
            est.probs = normalize(probs)

        est.meta["source"] = "relief split" if used_split else "team total (fallback)"
        return est

    # -- lineups -----------------------------------------------------------

    def resolve_lineups(self, game: Game,
                        season: int) -> tuple[Lineup, Lineup, bool, list[str]]:
        """Confirmed lineups when posted, otherwise the team's last-used order."""
        notes: list[str] = []
        away, home = self.api.lineups(game.game_pk)
        confirmed = away.confirmed and home.confirmed
        if confirmed:
            return away, home, True, notes

        for side, lineup, team_id in (("away", away, game.away_team_id),
                                      ("home", home, game.home_team_id)):
            if not lineup.confirmed:
                projected = self._last_lineup(team_id, game.official_date, season)
                if projected:
                    lineup.team_id = team_id
                    lineup.batters = projected
                    notes.append(f"{side} lineup projected from last start")
                else:
                    notes.append(f"{side} lineup unavailable")
        return away, home, False, notes

    def _last_lineup(self, team_id: int, before: str,
                     season: int) -> list[Player]:
        """The batting order this team used in its most recent completed game."""
        try:
            end = date_cls.fromisoformat(before) - timedelta(days=1)
        except ValueError:
            return []
        for back in range(0, 10):
            day = (end - timedelta(days=back)).isoformat()
            try:
                games = self.api.schedule(day)
            except Exception:  # noqa: BLE001
                continue
            for g in games:
                if team_id not in (g.home_team_id, g.away_team_id):
                    continue
                if not g.is_final:
                    continue
                try:
                    a, h = self.api.lineups(g.game_pk)
                except Exception:  # noqa: BLE001
                    continue
                lu = h if g.home_team_id == team_id else a
                if len(lu) == 9:
                    return lu.batters
        return []

    # -- assembly ----------------------------------------------------------

    def project_game(self, game: Game, season: int) -> GameProjection | None:
        # A postponed or suspended game will not be played as scheduled, so any
        # props on it are void. Projecting it anyway would put dead rows on the
        # board that look identical to live ones.
        status = (game.status or "").lower()
        if any(bad in status for bad in ("postponed", "suspended", "cancelled",
                                         "canceled")):
            log.info("skipping %s @ %s: %s", game.away_team, game.home_team,
                     game.status)
            return None

        park = get_park(game.venue_id) or {}
        if not park:
            log.warning("no park record for venue %s (%s)",
                        game.venue_id, game.venue_name)

        wx = self.weather.forecast(park, game.game_date) if park else {}
        away_lu, home_lu, confirmed, notes = self.resolve_lineups(game, season)

        if len(away_lu) != 9 or len(home_lu) != 9:
            log.warning("skipping %s @ %s: incomplete lineups",
                        game.away_team, game.home_team)
            return None
        if not game.away_probable_id or not game.home_probable_id:
            log.warning("skipping %s @ %s: probable pitcher not announced",
                        game.away_team, game.home_team)
            return None

        # Resolve handedness for everyone in one batch.
        ids = ([b.id for b in away_lu.batters] + [b.id for b in home_lu.batters]
               + [game.away_probable_id, game.home_probable_id])
        people = self.api.players(ids)
        for lu in (away_lu, home_lu):
            for b in lu.batters:
                info = people.get(b.id)
                if info:
                    b.bats, b.throws = info.bats, info.throws
                    b.name = info.name or b.name

        # Fetch every rate estimate concurrently; these are the slow calls.
        away_home_park = self.home_park(game.away_team_id, season)
        home_home_park = self.home_park(game.home_team_id, season)

        logs_by_id: dict[int, list[dict]] = {}

        def batter_job(item: tuple[int, dict | None]) -> tuple[int, RateEstimate]:
            pid, hp = item
            logs = self.api.game_log(pid, season, "hitting")
            logs_by_id[pid] = logs
            return pid, self.batter_rates(pid, season, hp)

        batter_ids = ([(b.id, away_home_park) for b in away_lu.batters]
                      + [(b.id, home_home_park) for b in home_lu.batters])
        rates: dict[int, RateEstimate] = {}
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            for pid, est in pool.map(batter_job, batter_ids):
                rates[pid] = est

        # Batter-vs-pitcher history, fetched for display only.
        #
        # This is deliberately kept out of the projection. A hitter faces a
        # given starter three or four times a season, so career BvP samples run
        # 5-30 PA -- at 15 PA the 95% interval around a .300 average spans
        # roughly .080 to .560. Folding that into the model would add variance
        # with no predictive content and let a 1-for-12 history override a full
        # season of evidence. It is shown because people reasonably want to see
        # it, with the sample size next to it so it reads as trivia rather than
        # signal.
        bvp: dict[int, dict] = {}
        if self.include_bvp:
            def bvp_job(item):
                bid, pid = item
                try:
                    return bid, self.api.batter_vs_pitcher(bid, pid)
                except Exception:  # noqa: BLE001
                    return bid, {}
            pairs = ([(b.id, game.home_probable_id) for b in away_lu.batters]
                     + [(b.id, game.away_probable_id) for b in home_lu.batters])
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
                for bid, stat in pool.map(bvp_job, pairs):
                    if stat:
                        bvp[bid] = stat

        away_sp = self.pitcher_rates(game.away_probable_id, season, away_home_park)
        home_sp = self.pitcher_rates(game.home_probable_id, season, home_home_park)
        away_pen = self.bullpen_rates(game.away_team_id, season)
        home_pen = self.bullpen_rates(game.home_team_id, season)
        bullpen_info = {
            game.away_team: {
                "source": away_pen.meta.get("source", ""),
                "bf": away_pen.raw_pa,
                "k_rate": round(float(away_pen.probs[O_K]), 4),
            },
            game.home_team: {
                "source": home_pen.meta.get("source", ""),
                "bf": home_pen.raw_pa,
                "k_rate": round(float(home_pen.probs[O_K]), 4),
            },
        }

        sp_people = {
            0: people.get(game.away_probable_id),
            1: people.get(game.home_probable_id),
        }

        league = self.league_vector(season)

        # dist[batting_team, slot, phase] -- the away lineup (team 0) faces the
        # home starter, and vice versa.
        dist = np.zeros((2, 9, N_PHASES, N_OUTCOMES), dtype=np.float64)
        gb = np.full((2, 9), LEAGUE_GB_RATE)

        lineups = {0: away_lu, 1: home_lu}
        opposing_sp = {0: home_sp, 1: away_sp}
        opposing_sp_hand = {
            0: (sp_people[1].throws if sp_people[1] else "R"),
            1: (sp_people[0].throws if sp_people[0] else "R"),
        }
        opposing_pen = {0: home_pen, 1: away_pen}

        for team in (0, 1):
            for slot, batter in enumerate(lineups[team].batters):
                est = rates.get(batter.id)
                if est is None:
                    est = RateEstimate(probs=league.copy(), effective_pa=0.0,
                                       raw_pa=0)
                gb[team, slot] = est.gb_rate
                for tto in range(1, 5):
                    dist[team, slot, tto - 1] = matchup_distribution(
                        est.probs, opposing_sp[team].probs,
                        league=league,
                        bats=batter.bats,
                        throws=opposing_sp_hand[team],
                        park=park, weather=wx, times_through=tto,
                    )
                # Bullpen: handedness is unknown in advance, so the platoon
                # term is left neutral rather than guessed.
                dist[team, slot, 4] = matchup_distribution(
                    est.probs, opposing_pen[team].probs,
                    league=league, bats=batter.bats, throws="",
                    park=park, weather=wx, times_through=1,
                )

        hook_mean, hook_sd = self._hook_params(
            [game.away_probable_id, game.home_probable_id], season)

        result = simulate(dist, gb, n_sims=self.n_sims,
                          hook_mean=hook_mean, hook_sd=hook_sd, seed=self.seed)

        projections: list[PlayerProjection] = []
        for team in (0, 1):
            team_name = game.away_team if team == 0 else game.home_team
            for slot, batter in enumerate(lineups[team].batters):
                est = rates.get(batter.id)
                projections.append(PlayerProjection(
                    player_id=batter.id,
                    name=batter.name,
                    team=team_name,
                    is_home=bool(team),
                    lineup_slot=slot + 1,
                    is_pitcher=False,
                    distributions=batter_distributions(result, team, slot),
                    context={
                        "bats": batter.bats,
                        "opp_sp_throws": opposing_sp_hand[team],
                        "effective_pa": round(est.effective_pa, 1) if est else 0,
                        "raw_pa": est.raw_pa if est else 0,
                        "mean_pa": round(float(result.pa[:, team, slot].mean()), 2),
                        "last10": recent_form(logs_by_id.get(batter.id, []),
                                              BATTER_HIT_RATE_GAMES),
                        "hit_rates": hit_rates(
                            logs_by_id.get(batter.id, []),
                            BATTER_HIT_RATE_LINES,
                            games=BATTER_HIT_RATE_GAMES),
                        "vs_pitcher": _bvp_summary(bvp.get(batter.id)),
                    },
                ))

        for team, pid in ((0, game.away_probable_id), (1, game.home_probable_id)):
            info = sp_people[team]
            est = away_sp if team == 0 else home_sp
            projections.append(PlayerProjection(
                player_id=pid,
                name=info.name if info else str(pid),
                team=game.away_team if team == 0 else game.home_team,
                is_home=bool(team),
                lineup_slot=None,
                is_pitcher=True,
                distributions=pitcher_distributions(result, team),
                context={
                    "throws": info.throws if info else "R",
                    "effective_bf": round(est.effective_pa, 1),
                    "raw_bf": est.raw_pa,
                    "mean_pitches": round(float(result.sp_pitches[:, team].mean()), 1),
                    "mean_ip": round(float(result.sp_outs[:, team].mean() / 3), 2),
                    "hit_rates": hit_rates(
                        self._pitching_logs(pid, season),
                        PITCHER_HIT_RATE_LINES,
                        games=PITCHER_HIT_RATE_STARTS,
                        starts_only=True),
                },
            ))

        gm = game_markets(result, game.away_team, game.home_team)
        lineup_order = {
            game.away_team: [
                {"slot": i + 1, "name": b.name, "id": b.id, "bats": b.bats,
                 "pos": b.position}
                for i, b in enumerate(away_lu.batters)],
            game.home_team: [
                {"slot": i + 1, "name": b.name, "id": b.id, "bats": b.bats,
                 "pos": b.position}
                for i, b in enumerate(home_lu.batters)],
        }

        return GameProjection(
            game=game, park=park, weather=wx, projections=projections,
            game_markets=gm, lineup_order=lineup_order,
            lineups_confirmed=confirmed, notes=notes, bullpen=bullpen_info,
            team_runs={
                game.away_team: round(float(result.team_runs[:, 0].mean()), 2),
                game.home_team: round(float(result.team_runs[:, 1].mean()), 2),
            },
        )

    def _hook_params(self, pitcher_ids: list[int],
                     season: int) -> tuple[np.ndarray, np.ndarray]:
        """Per-starter pitch-count hook, learned from his own workload history.

        A pitcher on an innings limit and an established 100-pitch horse have
        very different strikeout distributions even off identical rate stats,
        so the hook is fit per pitcher rather than assumed league-average.
        """
        means, sds = [], []
        for pid in pitcher_ids:
            try:
                logs = self.api.game_log(pid, season, "pitching")
            except Exception:  # noqa: BLE001
                logs = []
            counts = [float(g.get("numberOfPitches", 0) or 0)
                      for g in logs
                      if float(g.get("gamesStarted", 0) or 0) > 0
                      and float(g.get("numberOfPitches", 0) or 0) > 20]
            if len(counts) >= 3:
                arr = np.array(counts[-12:])
                # Shrink toward the league hook when the sample is short.
                w = min(len(arr) / 12.0, 1.0)
                means.append(w * float(arr.mean()) + (1 - w) * 88.0)
                sds.append(max(float(arr.std()), 8.0))
            else:
                means.append(88.0)
                sds.append(16.0)
        return np.array(means), np.array(sds)

    # -- slate -------------------------------------------------------------

    def project_slate(self, date: str) -> list[GameProjection]:
        season = int(date[:4])
        games = self.api.schedule(date)
        log.info("%s: %d games on the schedule", date, len(games))
        out: list[GameProjection] = []
        for g in games:
            try:
                gp = self.project_game(g, season)
            except Exception as exc:  # noqa: BLE001
                log.warning("failed to project %s @ %s: %s",
                            g.away_team, g.home_team, exc)
                continue
            if gp:
                out.append(gp)
                log.info("projected %s @ %s (%s)", g.away_team, g.home_team,
                         "confirmed" if gp.lineups_confirmed else "projected")
        return out
