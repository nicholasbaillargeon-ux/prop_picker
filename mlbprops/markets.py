"""Turning simulated games into prop probabilities.

Every market reads off the same set of simulated games, so the numbers are
mutually consistent: a player's hits, total-bases, and RBI probabilities all
come from the same joint distribution rather than three separate models that
might disagree about how often he bats.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .sim import SimResult

# Internal market name -> (SimResult attribute, is_pitcher_market)
BATTER_MARKETS = {
    "hits": "hits",
    "total_bases": "total_bases",
    "home_runs": "home_runs",
    "rbi": "rbi",
    "runs": "runs",
    "singles": "singles",
    "doubles": "doubles",
    "triples": "triples",
    "walks": "walks",
    "batter_strikeouts": "strikeouts",
    # Derived below rather than read from an attribute; see
    # batter_distributions().
    "hits_runs_rbis": None,
}

PITCHER_MARKETS = {
    "pitcher_strikeouts": "sp_k",
    "pitcher_outs": "sp_outs",
    "pitcher_earned_runs": "sp_er",
    "pitcher_hits_allowed": "sp_hits",
    "pitcher_walks": "sp_walks",
}

ALL_MARKETS = {**BATTER_MARKETS, **PITCHER_MARKETS}


@dataclass
class Distribution:
    """The simulated distribution of one player's outcome in one market."""

    market: str
    samples: np.ndarray            # integer outcomes, one per simulated game
    mean: float = 0.0
    median: float = 0.0
    pmf: dict[int, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        s = np.asarray(self.samples)
        self.mean = float(s.mean())
        self.median = float(np.median(s))
        values, counts = np.unique(s, return_counts=True)
        total = counts.sum()
        # Trim the tail so the payload stays small; anything below 0.05% is
        # noise at typical simulation counts anyway.
        self.pmf = {
            int(v): float(c / total)
            for v, c in zip(values, counts)
            if c / total >= 0.0005
        }

    def over(self, line: float) -> float:
        """P(outcome > line). Half-point lines make this unambiguous."""
        return float((self.samples > line).mean())

    def under(self, line: float) -> float:
        return float((self.samples < line).mean())

    def push(self, line: float) -> float:
        """P(exact tie) -- nonzero only for whole-number lines."""
        return float((self.samples == line).mean())

    def resolve(self, line: float, side: str) -> float:
        """Win probability for a side, with pushes removed from the base.

        On a whole-number line a push returns the stake, so the bet is really a
        wager on the non-push outcomes. Renormalizing here is what keeps EV
        comparable between a 1.5 line and a 2.0 line.
        """
        push = self.push(line)
        live = 1.0 - push
        if live <= 1e-12:
            return 0.0
        raw = self.over(line) if side.lower().startswith("o") else self.under(line)
        return float(raw / live)

    def quantile(self, q: float) -> float:
        return float(np.quantile(self.samples, q))


@dataclass
class PlayerProjection:
    """All simulated markets for one player in one game."""

    player_id: int
    name: str
    team: str
    is_home: bool
    lineup_slot: int | None          # None for pitchers
    is_pitcher: bool
    distributions: dict[str, Distribution] = field(default_factory=dict)
    context: dict = field(default_factory=dict)

    def get(self, market: str) -> Distribution | None:
        return self.distributions.get(market)


def batter_distributions(result: SimResult, team: int,
                         slot: int) -> dict[str, Distribution]:
    """Extract every batter market for one lineup slot."""
    out: dict[str, Distribution] = {}
    for market, attr in BATTER_MARKETS.items():
        if attr is None:
            continue
        samples = getattr(result, attr)[:, team, slot]
        out[market] = Distribution(market=market, samples=samples)

    # Hits + runs + RBI. Summing the three per simulated game (rather than
    # combining three separate distributions) is what makes this correct: the
    # components are strongly correlated within a game. A home run is
    # simultaneously a hit, a run, and an RBI -- worth 3 to this total from one
    # swing -- so treating the pieces as independent would badly understate the
    # upper tail.
    out["hits_runs_rbis"] = Distribution(
        market="hits_runs_rbis",
        samples=(result.hits[:, team, slot]
                 + result.runs[:, team, slot]
                 + result.rbi[:, team, slot]),
    )
    return out


def pitcher_distributions(result: SimResult, team: int) -> dict[str, Distribution]:
    """Extract every starting-pitcher market for one team.

    ``team`` is the pitcher's own team index, matching how ``sim`` stores
    starter accumulators.
    """
    out: dict[str, Distribution] = {}
    for market, attr in PITCHER_MARKETS.items():
        samples = getattr(result, attr)[:, team]
        out[market] = Distribution(market=market, samples=samples)
    return out


# ---------------------------------------------------------------------------
# Game-level markets
#
# These come free from the same simulated games as the player props: the
# simulator already tracks both teams' runs inning by inning, so moneyline,
# totals, and run line are just different readings of the final score
# distribution. Because they share the simulation, a team's moneyline price is
# consistent with its hitters' RBI props by construction.
# ---------------------------------------------------------------------------

COMMON_TOTALS = [6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.5]


def game_markets(result: SimResult, away_team: str, home_team: str) -> dict:
    """Moneyline, total runs, and run line from the simulated final scores."""
    away = result.team_runs[:, 0].astype(np.int32)
    home = result.team_runs[:, 1].astype(np.int32)
    total = away + home
    margin = home - away

    # The simulator caps extra innings, so a small share of games finish level.
    # A real game cannot, so ties are removed from the base rather than being
    # silently scored as a loss for one side.
    home_win = float((margin > 0).mean())
    away_win = float((margin < 0).mean())
    decided = home_win + away_win
    p_home = home_win / decided if decided > 0 else 0.5

    total_dist = Distribution(market="game_total", samples=total)
    margin_dist = Distribution(market="run_line", samples=margin)

    totals = {}
    for line in COMMON_TOTALS:
        totals[str(line)] = {
            "over": round(total_dist.resolve(line, "Over"), 4),
            "under": round(total_dist.resolve(line, "Under"), 4),
            "push": round(total_dist.push(line), 4),
        }

    return {
        "moneyline": {
            "home": round(p_home, 4),
            "away": round(1.0 - p_home, 4),
            "home_team": home_team,
            "away_team": away_team,
        },
        "total": {
            "mean": round(float(total.mean()), 2),
            "median": float(np.median(total)),
            "p10": float(np.quantile(total, 0.10)),
            "p90": float(np.quantile(total, 0.90)),
            "pmf": {str(k): round(v, 5) for k, v in total_dist.pmf.items()},
            "lines": totals,
        },
        "run_line": {
            # -1.5 on the home team means home must win by two or more.
            "home_-1.5": round(float((margin >= 2).mean()), 4),
            "home_+1.5": round(float((margin >= -1).mean()), 4),
            "away_-1.5": round(float((margin <= -2).mean()), 4),
            "away_+1.5": round(float((margin <= 1).mean()), 4),
        },
        "team_runs": {
            away_team: {
                "mean": round(float(away.mean()), 2),
                "pmf": {str(k): round(v, 5) for k, v in
                        Distribution(market="runs", samples=away).pmf.items()},
            },
            home_team: {
                "mean": round(float(home.mean()), 2),
                "pmf": {str(k): round(v, 5) for k, v in
                        Distribution(market="runs", samples=home).pmf.items()},
            },
        },
    }


def summarize(dist: Distribution, lines: list[float] | None = None) -> dict:
    """A compact, JSON-serializable view of a distribution."""
    payload = {
        "market": dist.market,
        "mean": round(dist.mean, 3),
        "median": dist.median,
        "p10": dist.quantile(0.10),
        "p90": dist.quantile(0.90),
        "pmf": {str(k): round(v, 5) for k, v in dist.pmf.items()},
    }
    if lines:
        payload["lines"] = {
            str(ln): {
                "over": round(dist.over(ln), 5),
                "under": round(dist.under(ln), 5),
                "push": round(dist.push(ln), 5),
            }
            for ln in lines
        }
    return payload


def common_lines(market: str) -> list[float]:
    """Typical sportsbook lines for a market, used to pre-compute probabilities."""
    table = {
        "hits": [0.5, 1.5, 2.5],
        "total_bases": [0.5, 1.5, 2.5, 3.5],
        "home_runs": [0.5, 1.5],
        "rbi": [0.5, 1.5, 2.5],
        "runs": [0.5, 1.5],
        "singles": [0.5, 1.5],
        "doubles": [0.5],
        "triples": [0.5],
        "walks": [0.5, 1.5],
        "batter_strikeouts": [0.5, 1.5, 2.5],
        "hits_runs_rbis": [0.5, 1.5, 2.5, 3.5],
        "pitcher_strikeouts": [3.5, 4.5, 5.5, 6.5, 7.5, 8.5],
        # Matches the dashboard's own picker range. It offered 11.5 through 21.5
        # while this stopped at 14.5-19.5, so the five outer lines had no
        # precomputed probabilities -- and, once hit rates keyed off this table,
        # no hit rate either.
        "pitcher_outs": [11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5, 19.5,
                         20.5, 21.5],
        "pitcher_earned_runs": [1.5, 2.5, 3.5],
        "pitcher_hits_allowed": [3.5, 4.5, 5.5, 6.5],
        "pitcher_walks": [1.5, 2.5],
    }
    return table.get(market, [0.5, 1.5, 2.5])
