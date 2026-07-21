"""Sportsbook odds: fetching, vig removal, and bet sizing.

A model probability is only actionable next to a price. This module pulls
player-prop lines from The Odds API, strips the bookmaker's margin to recover
the market's true implied probability, and turns the gap between that and the
model into an expected value and a stake.

**Why the devig method matters.** A two-way prop priced -130/+105 carries ~4%
margin. How you distribute that margin across the two sides changes the implied
probability by a percentage point or more, which is often larger than the edge
being claimed. Proportional (multiplicative) devigging assumes the book applies
margin evenly, which is demonstrably wrong for longshots -- a 12% home-run prop
carries far more margin than the 88% other side. Shin and power devigging both
correct for this, and the default here is Shin because it is derived from an
explicit model of informed betting rather than a curve fit.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Iterable

import requests
from scipy import optimize

from .cache import TTL, DiskCache

log = logging.getLogger(__name__)

BASE = "https://api.the-odds-api.com/v4"
SPORT = "baseball_mlb"

# The Odds API market keys we support, mapped to this project's internal
# market names.
MARKET_MAP = {
    "batter_hits": "hits",
    "batter_total_bases": "total_bases",
    "batter_home_runs": "home_runs",
    "batter_rbis": "rbi",
    "batter_runs_scored": "runs",
    "batter_singles": "singles",
    "batter_doubles": "doubles",
    "batter_triples": "triples",
    "batter_walks": "walks",
    "batter_strikeouts": "batter_strikeouts",
    "batter_hits_runs_rbis": "hits_runs_rbis",
    "pitcher_strikeouts": "pitcher_strikeouts",
    "pitcher_outs": "pitcher_outs",
    "pitcher_earned_runs": "pitcher_earned_runs",
    "pitcher_hits_allowed": "pitcher_hits_allowed",
    "pitcher_walks": "pitcher_walks",
}

DEFAULT_MARKETS = tuple(MARKET_MAP.keys())


class OddsAPIError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Odds conversions
# ---------------------------------------------------------------------------

def american_to_decimal(american: float) -> float:
    if american is None:
        raise ValueError("missing odds")
    a = float(american)
    if a >= 0:
        return a / 100.0 + 1.0
    return 100.0 / abs(a) + 1.0


def decimal_to_american(decimal: float) -> float:
    d = float(decimal)
    if d <= 1.0:
        raise ValueError(f"decimal odds must exceed 1.0, got {d}")
    if d >= 2.0:
        return round((d - 1.0) * 100.0)
    return round(-100.0 / (d - 1.0))


def american_to_prob(american: float) -> float:
    """Raw implied probability, still containing the bookmaker's margin."""
    return 1.0 / american_to_decimal(american)


def prob_to_american(p: float) -> float:
    if not 0.0 < p < 1.0:
        raise ValueError(f"probability must be in (0, 1), got {p}")
    return decimal_to_american(1.0 / p)


# ---------------------------------------------------------------------------
# Vig removal
# ---------------------------------------------------------------------------

def devig_multiplicative(raw: Iterable[float]) -> list[float]:
    """Scale probabilities down proportionally so they sum to 1."""
    raw = list(raw)
    total = sum(raw)
    if total <= 0:
        raise ValueError("implied probabilities must be positive")
    return [p / total for p in raw]


def devig_power(raw: Iterable[float]) -> list[float]:
    """Find k such that sum(p_i ** k) == 1.

    Removes proportionally more margin from longshots than the multiplicative
    method does, which matters for home-run and low-probability props.
    """
    raw = list(raw)
    if any(p <= 0 for p in raw):
        raise ValueError("implied probabilities must be positive")
    if abs(sum(raw) - 1.0) < 1e-9:
        return raw

    def objective(k: float) -> float:
        return sum(p ** k for p in raw) - 1.0

    try:
        k = optimize.brentq(objective, 0.2, 5.0, maxiter=200)
    except (ValueError, RuntimeError):
        log.debug("power devig failed to converge; using multiplicative")
        return devig_multiplicative(raw)
    return [p ** k for p in raw]


def devig_shin(raw: Iterable[float]) -> list[float]:
    """Shin (1992): back out the implied share of insider money.

    Models the book as setting prices against a fraction ``z`` of informed
    bettors. Solving for ``z`` and inverting gives fair probabilities that
    correct the favourite-longshot bias more principledly than a fitted
    exponent.
    """
    raw = list(raw)
    n = len(raw)
    if n < 2 or any(p <= 0 for p in raw):
        raise ValueError("need at least two positive probabilities")
    total = sum(raw)
    if abs(total - 1.0) < 1e-9:
        return raw

    def implied(z: float) -> list[float]:
        return [
            (((z * z + 4.0 * (1.0 - z) * p * p / total) ** 0.5) - z)
            / (2.0 * (1.0 - z))
            for p in raw
        ]

    def objective(z: float) -> float:
        return sum(implied(z)) - 1.0

    try:
        z = optimize.brentq(objective, 1e-9, 0.35, maxiter=200)
    except (ValueError, RuntimeError):
        log.debug("shin devig failed to converge; using power")
        return devig_power(raw)

    out = implied(z)
    s = sum(out)
    return [p / s for p in out]


DEVIG_METHODS = {
    "multiplicative": devig_multiplicative,
    "power": devig_power,
    "shin": devig_shin,
}


def devig(raw: Iterable[float], method: str = "shin") -> list[float]:
    fn = DEVIG_METHODS.get(method)
    if fn is None:
        raise ValueError(f"unknown devig method {method!r}; "
                         f"choose from {sorted(DEVIG_METHODS)}")
    return fn(raw)


# ---------------------------------------------------------------------------
# Edge and staking
# ---------------------------------------------------------------------------

def expected_value(model_prob: float, american: float) -> float:
    """EV per unit staked. 0.05 means +5% expected return on the wager."""
    dec = american_to_decimal(american)
    return model_prob * (dec - 1.0) - (1.0 - model_prob)


def kelly_fraction(model_prob: float, american: float,
                   multiplier: float = 0.25, cap: float = 0.02) -> float:
    """Fractional Kelly stake as a share of bankroll.

    Full Kelly is far too aggressive against model probabilities that carry
    estimation error: it assumes the edge is known exactly. Quarter Kelly with
    a hard 2% cap is the default, and negative-edge bets return 0.
    """
    dec = american_to_decimal(american)
    b = dec - 1.0
    if b <= 0:
        return 0.0
    f = (model_prob * b - (1.0 - model_prob)) / b
    if f <= 0:
        return 0.0
    return float(min(f * multiplier, cap))


# ---------------------------------------------------------------------------
# API client
# ---------------------------------------------------------------------------

@dataclass
class PropLine:
    """One side of one player prop at one book."""

    event_id: str
    book: str
    market: str            # internal market name
    player: str
    point: float           # the line, e.g. 1.5
    side: str              # "Over" or "Under"
    american: float
    raw_prob: float = 0.0
    fair_prob: float = 0.0  # populated after devigging against the other side

    def __post_init__(self) -> None:
        if not self.raw_prob:
            self.raw_prob = american_to_prob(self.american)


@dataclass
class EventOdds:
    event_id: str
    home_team: str
    away_team: str
    commence_time: str
    lines: list[PropLine] = field(default_factory=list)


class OddsClient:
    def __init__(self, api_key: str | None = None,
                 cache: DiskCache | None = None, timeout: float = 20.0):
        self.api_key = api_key or os.environ.get("ODDS_API_KEY", "")
        self.cache = cache or DiskCache()
        self.timeout = timeout
        self._session = requests.Session()
        self.last_quota: dict[str, str] = {}

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    def _get(self, path: str, params: dict, namespace: str, ttl: float):
        if not self.enabled:
            raise OddsAPIError(
                "No odds API key. Set ODDS_API_KEY, or pass --no-odds to run "
                "the model without prices."
            )
        params = dict(params, apiKey=self.api_key)
        url = f"{BASE}{path}"
        # Keep the key out of the cache filename.
        key = url + "?" + "&".join(
            f"{k}={v}" for k, v in sorted(params.items()) if k != "apiKey"
        )

        def fetch():
            resp = self._session.get(url, params=params, timeout=self.timeout)
            if resp.status_code == 401:
                raise OddsAPIError("odds API rejected the key (401)")
            if resp.status_code == 429:
                raise OddsAPIError("odds API quota exhausted (429)")
            resp.raise_for_status()
            self.last_quota = {
                "remaining": resp.headers.get("x-requests-remaining", "?"),
                "used": resp.headers.get("x-requests-used", "?"),
            }
            return resp.json()

        try:
            return self.cache.get_or_fetch(namespace, key, fetch, ttl)
        except requests.RequestException as exc:
            raise OddsAPIError(f"odds request failed: {exc}") from exc

    def events(self, date_iso: str | None = None) -> list[dict]:
        """Upcoming MLB events with their odds-API event ids."""
        return self._get(f"/sports/{SPORT}/events", {}, "odds", TTL["odds"])

    def event_props(
        self,
        event_id: str,
        markets: Iterable[str] = DEFAULT_MARKETS,
        regions: str = "us",
        books: str | None = None,
    ) -> EventOdds:
        """Player props for a single event, across books."""
        params = {
            "regions": regions,
            "markets": ",".join(markets),
            "oddsFormat": "american",
        }
        if books:
            params["bookmakers"] = books
        data = self._get(f"/sports/{SPORT}/events/{event_id}/odds", params,
                         "odds", TTL["odds"])

        ev = EventOdds(
            event_id=data.get("id", event_id),
            home_team=data.get("home_team", ""),
            away_team=data.get("away_team", ""),
            commence_time=data.get("commence_time", ""),
        )
        for bk in data.get("bookmakers", []):
            book = bk.get("key", "")
            for mk in bk.get("markets", []):
                internal = MARKET_MAP.get(mk.get("key", ""))
                if internal is None:
                    continue
                for oc in mk.get("outcomes", []):
                    price = oc.get("price")
                    if price is None:
                        continue
                    ev.lines.append(PropLine(
                        event_id=ev.event_id,
                        book=book,
                        market=internal,
                        player=oc.get("description") or oc.get("name", ""),
                        point=float(oc.get("point", 0.0)),
                        side=oc.get("name", ""),
                        american=float(price),
                    ))
        return ev


# ---------------------------------------------------------------------------
# Pairing and fair-price construction
# ---------------------------------------------------------------------------

def attach_fair_probs(lines: list[PropLine], method: str = "shin") -> None:
    """Devig each Over/Under pair in place.

    Lines are grouped by (book, market, player, point). Only complete two-sided
    pairs can be devigged; an orphaned side keeps its raw, margin-inclusive
    probability and is flagged by leaving ``fair_prob`` equal to ``raw_prob``.
    """
    groups: dict[tuple, list[PropLine]] = {}
    for ln in lines:
        groups.setdefault((ln.book, ln.market, ln.player, ln.point), []).append(ln)

    for key, group in groups.items():
        if len(group) == 2:
            try:
                fair = devig([g.raw_prob for g in group], method)
            except (ValueError, ZeroDivisionError):
                fair = [g.raw_prob for g in group]
            for ln, p in zip(group, fair):
                ln.fair_prob = p
        else:
            for ln in group:
                ln.fair_prob = ln.raw_prob


def consensus_fair_prob(lines: list[PropLine]) -> dict[tuple, float]:
    """Average fair probability across books for each (market, player, point, side).

    A single book's line is noisy and can simply be stale. Averaging devigged
    probabilities across books approximates the market consensus, which is the
    honest benchmark to measure a model edge against.
    """
    bucket: dict[tuple, list[float]] = {}
    for ln in lines:
        if ln.fair_prob > 0:
            bucket.setdefault((ln.market, ln.player, ln.point, ln.side),
                              []).append(ln.fair_prob)
    return {k: sum(v) / len(v) for k, v in bucket.items() if v}


def best_price(lines: list[PropLine]) -> dict[tuple, PropLine]:
    """The most favourable available American price for each prop side.

    Line shopping is not a detail. The difference between the best and median
    price on a prop routinely exceeds the model's entire claimed edge, so every
    recommendation is quoted at the best book found.
    """
    best: dict[tuple, PropLine] = {}
    for ln in lines:
        key = (ln.market, ln.player, ln.point, ln.side)
        cur = best.get(key)
        if cur is None or american_to_decimal(ln.american) > american_to_decimal(cur.american):
            best[key] = ln
    return best
