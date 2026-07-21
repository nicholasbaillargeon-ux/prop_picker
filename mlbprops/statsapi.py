"""Client for the free, unauthenticated MLB StatsAPI.

Endpoint shapes verified against statsapi.mlb.com. No API key required.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import requests

from .cache import TTL, DiskCache

log = logging.getLogger(__name__)

BASE = "https://statsapi.mlb.com/api"
USER_AGENT = "mlbprops/0.1 (player prop research)"


# ---------------------------------------------------------------------------
# Domain objects
# ---------------------------------------------------------------------------

@dataclass
class Player:
    id: int
    name: str
    bats: str = "R"          # "R", "L", or "S" (switch)
    throws: str = "R"
    position: str = ""

    @property
    def is_pitcher(self) -> bool:
        return self.position in ("P", "SP", "RP", "TWP")


@dataclass
class Game:
    game_pk: int
    game_date: str           # ISO-8601 UTC start time
    official_date: str
    status: str
    venue_id: int
    venue_name: str
    home_team_id: int
    home_team: str
    away_team_id: int
    away_team: str
    home_probable_id: int | None = None
    home_probable_name: str = ""
    away_probable_id: int | None = None
    away_probable_name: str = ""
    day_night: str = "day"

    @property
    def is_final(self) -> bool:
        return self.status.lower().startswith("final")


@dataclass
class Lineup:
    """A team's batting order for a game.

    ``confirmed`` distinguishes an official posted lineup from a projection.
    This matters: an unconfirmed lineup carries batting-order uncertainty that
    the caller should widen its distributions for (or simply skip the slate).
    """
    team_id: int
    batters: list[Player] = field(default_factory=list)
    confirmed: bool = False

    def __len__(self) -> int:
        return len(self.batters)


class StatsAPIError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class StatsAPI:
    def __init__(self, cache: DiskCache | None = None, timeout: float = 20.0):
        self.cache = cache or DiskCache()
        self.timeout = timeout
        self._session = requests.Session()
        self._session.headers["User-Agent"] = USER_AGENT

    def _get(self, path: str, params: dict | None = None,
             *, namespace: str, ttl: float, retries: int = 3) -> Any:
        url = f"{BASE}{path}"
        key = url + "?" + "&".join(f"{k}={v}" for k, v in sorted((params or {}).items()))

        def fetch() -> Any:
            last: Exception | None = None
            for attempt in range(retries):
                try:
                    resp = self._session.get(url, params=params, timeout=self.timeout)
                    resp.raise_for_status()
                    return resp.json()
                except (requests.RequestException, ValueError) as exc:
                    last = exc
                    if attempt < retries - 1:
                        time.sleep(0.6 * (2 ** attempt))
            raise StatsAPIError(f"GET {url} failed after {retries} attempts: {last}")

        return self.cache.get_or_fetch(namespace, key, fetch, ttl)

    # -- schedule ----------------------------------------------------------

    def schedule(self, date: str) -> list[Game]:
        """Games for a date (YYYY-MM-DD), with probable pitchers and venue."""
        data = self._get(
            "/v1/schedule",
            {
                "sportId": 1,
                "date": date,
                "hydrate": "probablePitcher,team,venue,linescore",
            },
            namespace="schedule",
            ttl=TTL["schedule"],
        )
        games: list[Game] = []
        for day in data.get("dates", []):
            for g in day.get("games", []):
                teams = g.get("teams", {})
                home, away = teams.get("home", {}), teams.get("away", {})
                hp = home.get("probablePitcher") or {}
                ap = away.get("probablePitcher") or {}
                games.append(Game(
                    game_pk=g["gamePk"],
                    game_date=g.get("gameDate", ""),
                    official_date=g.get("officialDate", date),
                    status=(g.get("status") or {}).get("detailedState", ""),
                    venue_id=(g.get("venue") or {}).get("id", 0),
                    venue_name=(g.get("venue") or {}).get("name", ""),
                    home_team_id=(home.get("team") or {}).get("id", 0),
                    home_team=(home.get("team") or {}).get("name", ""),
                    away_team_id=(away.get("team") or {}).get("id", 0),
                    away_team=(away.get("team") or {}).get("name", ""),
                    home_probable_id=hp.get("id"),
                    home_probable_name=hp.get("fullName", ""),
                    away_probable_id=ap.get("id"),
                    away_probable_name=ap.get("fullName", ""),
                    day_night=g.get("dayNight", "day"),
                ))
        return games

    # -- people ------------------------------------------------------------

    def player(self, player_id: int) -> Player:
        data = self._get(f"/v1/people/{player_id}", namespace="player",
                         ttl=TTL["player"])
        people = data.get("people") or []
        if not people:
            raise StatsAPIError(f"no player {player_id}")
        p = people[0]
        return Player(
            id=p["id"],
            name=p.get("fullName", str(player_id)),
            bats=(p.get("batSide") or {}).get("code", "R"),
            throws=(p.get("pitchHand") or {}).get("code", "R"),
            position=(p.get("primaryPosition") or {}).get("abbreviation", ""),
        )

    def players(self, player_ids: list[int]) -> dict[int, Player]:
        """Batch player lookup. StatsAPI accepts a comma-separated id list."""
        if not player_ids:
            return {}
        out: dict[int, Player] = {}
        ids = sorted(set(player_ids))
        for i in range(0, len(ids), 60):
            chunk = ids[i:i + 60]
            data = self._get(
                "/v1/people",
                {"personIds": ",".join(str(x) for x in chunk)},
                namespace="player",
                ttl=TTL["player"],
            )
            for p in data.get("people", []):
                out[p["id"]] = Player(
                    id=p["id"],
                    name=p.get("fullName", ""),
                    bats=(p.get("batSide") or {}).get("code", "R"),
                    throws=(p.get("pitchHand") or {}).get("code", "R"),
                    position=(p.get("primaryPosition") or {}).get("abbreviation", ""),
                )
        return out

    # -- stats -------------------------------------------------------------

    def game_log(self, player_id: int, season: int,
                 group: str = "hitting") -> list[dict]:
        """Per-game stat lines, oldest first. Drives the recency weighting."""
        data = self._get(
            f"/v1/people/{player_id}",
            {"hydrate": f"stats(group={group},type=gameLog,season={season})"},
            namespace="gamelog",
            ttl=TTL["gamelog"],
        )
        people = data.get("people") or []
        if not people:
            return []
        splits: list[dict] = []
        for block in people[0].get("stats", []):
            for sp in block.get("splits", []):
                stat = dict(sp.get("stat", {}))
                stat["_date"] = sp.get("date", "")
                stat["_opponent_id"] = (sp.get("opponent") or {}).get("id")
                stat["_home"] = sp.get("isHome", None)
                splits.append(stat)
        splits.sort(key=lambda s: s.get("_date", ""))
        return splits

    def season_stats(self, player_id: int, season: int,
                     group: str = "hitting") -> dict:
        data = self._get(
            f"/v1/people/{player_id}",
            {"hydrate": f"stats(group={group},type=season,season={season})"},
            namespace="season_stats",
            ttl=TTL["season_stats"],
        )
        people = data.get("people") or []
        if not people:
            return {}
        for block in people[0].get("stats", []):
            for sp in block.get("splits", []):
                return dict(sp.get("stat", {}))
        return {}

    def team_season_stats(self, team_id: int, season: int,
                          group: str = "hitting") -> dict:
        """Team aggregate stats. Used for bullpen and lineup-quality baselines."""
        data = self._get(
            f"/v1/teams/{team_id}/stats",
            {"season": season, "stats": "season", "group": group},
            namespace="season_stats",
            ttl=TTL["season_stats"],
        )
        for block in data.get("stats", []):
            for sp in block.get("splits", []):
                return dict(sp.get("stat", {}))
        return {}

    def league_hitting(self, season: int) -> dict:
        """League-wide hitting totals, for the odds-ratio baseline."""
        data = self._get(
            "/v1/teams/stats",
            {"season": season, "sportIds": 1, "stats": "season", "group": "hitting"},
            namespace="season_stats",
            ttl=TTL["season_stats"],
        )
        totals: dict[str, float] = {}
        for block in data.get("stats", []):
            for sp in block.get("splits", []):
                for k, v in (sp.get("stat") or {}).items():
                    if isinstance(v, (int, float)):
                        totals[k] = totals.get(k, 0.0) + float(v)
        return totals

    # -- live feed / lineups ----------------------------------------------

    def live_feed(self, game_pk: int, ttl: float | None = None) -> dict:
        return self._get(f"/v1.1/game/{game_pk}/feed/live", namespace="lineup",
                         ttl=TTL["lineup"] if ttl is None else ttl)

    def lineups(self, game_pk: int) -> tuple[Lineup, Lineup]:
        """Return (away_lineup, home_lineup) from the live feed.

        Before lineups are posted the feed's ``battingOrder`` arrays are empty;
        we surface that via ``confirmed=False`` and an empty batter list rather
        than silently inventing an order. The pipeline falls back to a
        projection in that case.
        """
        feed = self.live_feed(game_pk)
        boxscore = (feed.get("liveData") or {}).get("boxscore") or {}
        out: list[Lineup] = []
        for side in ("away", "home"):
            team_block = (boxscore.get("teams") or {}).get(side) or {}
            team_id = ((team_block.get("team") or {}).get("id")) or 0
            order = team_block.get("battingOrder") or []
            players_block = team_block.get("players") or {}
            batters: list[Player] = []
            for pid in order[:9]:
                info = players_block.get(f"ID{pid}") or {}
                person = info.get("person") or {}
                pos = (info.get("position") or {}).get("abbreviation", "")
                batters.append(Player(
                    id=pid,
                    name=person.get("fullName", str(pid)),
                    position=pos,
                ))
            out.append(Lineup(team_id=team_id, batters=batters,
                              confirmed=len(batters) == 9))
        return out[0], out[1]

    def roster(self, team_id: int, season: int) -> list[Player]:
        data = self._get(
            f"/v1/teams/{team_id}/roster",
            {"season": season, "rosterType": "active"},
            namespace="roster",
            ttl=TTL["roster"],
        )
        players: list[Player] = []
        for entry in data.get("roster", []):
            person = entry.get("person") or {}
            players.append(Player(
                id=person.get("id", 0),
                name=person.get("fullName", ""),
                position=(entry.get("position") or {}).get("abbreviation", ""),
            ))
        return players

    def batter_vs_pitcher(self, batter_id: int, pitcher_id: int) -> dict:
        """Career batter-vs-pitcher totals.

        Displayed as context only -- see the note in ``pipeline`` for why this
        never touches the projection.
        """
        data = self._get(
            f"/v1/people/{batter_id}",
            {"hydrate": f"stats(group=hitting,type=vsPlayerTotal,"
                        f"opposingPlayerId={pitcher_id})"},
            namespace="gamelog",
            ttl=TTL["gamelog"],
        )
        people = data.get("people") or []
        if not people:
            return {}
        for block in people[0].get("stats", []):
            if block.get("type", {}).get("displayName") != "vsPlayerTotal":
                continue
            for sp in block.get("splits", []):
                return dict(sp.get("stat", {}))
        return {}

    def team_split_stats(self, team_id: int, season: int,
                         sit_codes: str = "rp",
                         group: str = "pitching") -> dict[str, dict]:
        """Team stats broken out by situation code, keyed by code.

        The codes that matter here are ``sp`` (starters) and ``rp``
        (relievers), which is the only place StatsAPI exposes a genuine
        starter/reliever split. It handles swingmen correctly: a pitcher's
        relief outings land in ``rp`` and his starts in ``sp``, rather than
        both being smeared into one team total.
        """
        data = self._get(
            f"/v1/teams/{team_id}/stats",
            {"season": season, "stats": "statSplits", "group": group,
             "sitCodes": sit_codes},
            namespace="season_stats",
            ttl=TTL["season_stats"],
        )
        out: dict[str, dict] = {}
        for block in data.get("stats", []):
            for split in block.get("splits", []):
                code = (split.get("split") or {}).get("code")
                if code:
                    out[str(code)] = dict(split.get("stat", {}))
        return out

    def team_venues(self, season: int) -> dict[int, int]:
        """Map team id -> home venue id, for park-neutralizing player stats."""
        data = self._get("/v1/teams", {"sportId": 1, "season": season},
                         namespace="roster", ttl=TTL["venue"])
        out: dict[int, int] = {}
        for t in data.get("teams", []):
            venue = (t.get("venue") or {}).get("id")
            if t.get("id") and venue:
                out[int(t["id"])] = int(venue)
        return out

    def venues(self) -> dict[int, dict]:
        data = self._get("/v1/venues", {"sportId": 1}, namespace="venue",
                         ttl=TTL["venue"])
        return {v["id"]: v for v in data.get("venues", [])}
