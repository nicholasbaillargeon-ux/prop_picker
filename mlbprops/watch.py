"""Auto-refresh: re-project the slate as lineups get confirmed.

Lineups land piecemeal in the two or three hours before first pitch, and they
are the single input that most changes a projection — batting order drives
plate-appearance counts, which drive every counting prop. A slate built at noon
is largely guesswork; the same slate rebuilt after lineups post is the real one.

The watcher polls one cheap endpoint for the whole slate, notices when a game's
lineup flips from projected to confirmed, and re-projects only the games that
actually changed.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import requests

from .cache import DiskCache
from .pipeline import GameProjection, Projector
from .report import best_bets, build_recommendations, rank, slate_payload
from .statsapi import Game, StatsAPI

log = logging.getLogger(__name__)

SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule"

# Statuses that mean a game's lineup can no longer change.
LOCKED_STATUSES = ("in progress", "final", "game over", "completed")
DEAD_STATUSES = ("postponed", "suspended", "cancelled", "canceled")


@dataclass
class WatchStatus:
    """Published at /api/status so the browser knows when to re-fetch."""

    generation: int = 0
    updated_utc: str = ""
    checked_utc: str = ""
    games_total: int = 0
    games_confirmed: int = 0
    games_pending: int = 0
    last_change: str = ""
    interval_seconds: int = 90
    running: bool = True
    message: str = ""
    changed_games: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class SlateWatcher:
    """Polls for lineup confirmations and rebuilds the slate when they land."""

    def __init__(
        self,
        date: str,
        out_dir: Path,
        *,
        projector: Projector | None = None,
        api: StatsAPI | None = None,
        cache: DiskCache | None = None,
        odds_builder: Callable[[GameProjection], list] | None = None,
        interval: int = 90,
        min_ev: float = 0.03,
        min_edge: float = 0.02,
        min_books: int = 2,
        meta: dict | None = None,
    ):
        self.date = date
        self.season = int(date[:4])
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.cache = cache or DiskCache()
        self.api = api or StatsAPI(cache=self.cache)
        self.projector = projector or Projector(api=self.api)
        self.odds_builder = odds_builder
        self.interval = max(int(interval), 20)
        self.min_ev = min_ev
        self.min_edge = min_edge
        self.min_books = min_books
        self.base_meta = meta or {}

        self.status = WatchStatus(interval_seconds=self.interval)
        self._projections: dict[int, GameProjection] = {}
        self._signature: dict[int, tuple[bool, str]] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._session = requests.Session()

    # -- paths -------------------------------------------------------------

    @property
    def slate_path(self) -> Path:
        return self.out_dir / "slate-latest.json"

    @property
    def status_path(self) -> Path:
        return self.out_dir / "status.json"

    # -- polling -----------------------------------------------------------

    def poll_signature(self) -> dict[int, tuple[bool, str]]:
        """One request for the whole slate's lineup state.

        Hydrating ``lineups`` onto the schedule returns every game's posted
        batting order in a single ~70 KB response. Polling each game's live
        feed instead would be one request per game for the same information.
        """
        params = {
            "sportId": 1,
            "date": self.date,
            "hydrate": "lineups,probablePitcher,team",
        }
        resp = self._session.get(SCHEDULE_URL, params=params, timeout=20)
        resp.raise_for_status()
        data = resp.json()

        out: dict[int, tuple[bool, str]] = {}
        for day in data.get("dates", []):
            for g in day.get("games", []):
                lineups = g.get("lineups") or {}
                away = lineups.get("awayPlayers") or []
                home = lineups.get("homePlayers") or []
                confirmed = len(away) >= 9 and len(home) >= 9
                status = (g.get("status") or {}).get("detailedState", "")
                out[int(g["gamePk"])] = (confirmed, status)
        return out

    def _label(self, game_pk: int) -> str:
        gp = self._projections.get(game_pk)
        if gp:
            return f"{gp.game.away_team} @ {gp.game.home_team}"
        return f"game {game_pk}"

    # -- projection --------------------------------------------------------

    def _fresh_games(self) -> dict[int, Game]:
        # Lineup and schedule responses must not come from cache here, or the
        # watcher would keep rebuilding from the same stale payload it just
        # detected a change against. Player histories stay cached — those are
        # the expensive calls and they do not change intraday.
        self.cache.clear("schedule")
        self.cache.clear("lineup")
        return {g.game_pk: g for g in self.api.schedule(self.date)}

    def refresh(self, only: set[int] | None = None) -> int:
        """Re-project games. Returns how many were rebuilt."""
        games = self._fresh_games()
        targets = [pk for pk in games if only is None or pk in only]
        rebuilt = 0
        for pk in targets:
            try:
                gp = self.projector.project_game(games[pk], self.season)
            except Exception as exc:  # noqa: BLE001
                log.warning("re-projection failed for %s: %s", pk, exc)
                continue
            if gp is None:
                # Postponed, or lineups/probables not yet available.
                self._projections.pop(pk, None)
                continue
            self._projections[pk] = gp
            rebuilt += 1
        return rebuilt

    def write_payload(self) -> None:
        """Rebuild the slate JSON from the current projections."""
        projections = [self._projections[pk] for pk in sorted(self._projections)]

        recommendations = []
        if self.odds_builder:
            for gp in projections:
                try:
                    recommendations.extend(self.odds_builder(gp))
                except Exception as exc:  # noqa: BLE001
                    log.warning("odds for %s failed: %s", gp.game.game_pk, exc)

        overall, per_game = best_bets(recommendations, min_ev=self.min_ev,
                                      min_books=self.min_books)
        rank(recommendations, min_ev=self.min_ev, min_edge=self.min_edge)

        meta = dict(self.base_meta)
        meta.update({
            "generated_utc": _now(),
            "auto_refresh": True,
            "generation": self.status.generation,
            "games_projected": len(projections),
        })
        payload = slate_payload(projections, recommendations, date=self.date,
                                meta=meta, best=overall, per_game_best=per_game)
        payload["generated_utc"] = meta["generated_utc"]

        tmp = self.slate_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        # Atomic replace so the server never serves a half-written file.
        tmp.replace(self.slate_path)
        (self.out_dir / f"slate-{self.date}.json").write_text(
            json.dumps(payload), encoding="utf-8")

    def write_status(self) -> None:
        tmp = self.status_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.status.to_dict()), encoding="utf-8")
        tmp.replace(self.status_path)

    # -- loop --------------------------------------------------------------

    def tick(self) -> bool:
        """One poll. Returns True if the slate was rebuilt."""
        try:
            signature = self.poll_signature()
        except (requests.RequestException, ValueError) as exc:
            self.status.message = f"poll failed: {exc}"
            self.status.checked_utc = _now()
            self.write_status()
            log.warning("lineup poll failed: %s", exc)
            return False

        changed: set[int] = set()
        for pk, (confirmed, status) in signature.items():
            was = self._signature.get(pk)
            if was is None:
                continue  # first pass is handled by the initial full build
            if was != (confirmed, status):
                # Only rebuild for changes that matter: a lineup being posted,
                # or a game becoming (or ceasing to be) playable.
                was_confirmed, was_status = was
                if confirmed != was_confirmed:
                    changed.add(pk)
                elif any(d in status.lower() for d in DEAD_STATUSES):
                    changed.add(pk)

        self._signature = signature
        self.status.checked_utc = _now()
        self.status.games_total = len(signature)
        self.status.games_confirmed = sum(1 for c, _ in signature.values() if c)
        self.status.games_pending = sum(
            1 for c, s in signature.values()
            if not c and not any(d in s.lower() for d in LOCKED_STATUSES + DEAD_STATUSES))

        if not changed:
            self.status.message = "no lineup changes"
            self.write_status()
            return False

        labels = [self._label(pk) for pk in changed]
        log.info("lineups changed: %s", ", ".join(labels))
        rebuilt = self.refresh(only=changed)
        if rebuilt:
            self.status.generation += 1
            self.status.updated_utc = _now()
            self.status.last_change = ", ".join(labels)
            self.status.changed_games = labels
            self.status.message = f"rebuilt {rebuilt} game(s)"
            self.write_payload()
        self.write_status()
        return bool(rebuilt)

    def initial_build(self) -> None:
        log.info("building initial slate for %s", self.date)
        self._signature = {}
        try:
            self._signature = self.poll_signature()
        except (requests.RequestException, ValueError) as exc:
            log.warning("initial poll failed: %s", exc)
        self.refresh(only=None)
        self.status.generation = 1
        self.status.updated_utc = _now()
        self.status.checked_utc = _now()
        self.status.games_total = len(self._signature) or len(self._projections)
        self.status.games_confirmed = sum(
            1 for c, _ in self._signature.values() if c)
        self.status.games_pending = sum(
            1 for c, s in self._signature.values()
            if not c and not any(d in s.lower()
                                 for d in LOCKED_STATUSES + DEAD_STATUSES))
        self.status.message = "initial build complete"
        self.write_payload()
        self.write_status()

    def loop(self) -> None:
        """Poll until stopped, or until no game can change again."""
        while not self._stop.is_set():
            if self._stop.wait(self.interval):
                break
            with self._lock:
                self.tick()
                if self.status.games_pending == 0 and self.status.games_total:
                    self.status.message = (
                        "all lineups locked — no further refreshes needed")
                    self.status.running = False
                    self.write_status()
                    log.info("every game locked; watcher going idle")
                    break
        self.status.running = False
        self.write_status()

    def start(self) -> threading.Thread:
        thread = threading.Thread(target=self.loop, name="slate-watcher",
                                  daemon=True)
        thread.start()
        return thread

    def stop(self) -> None:
        self._stop.set()
