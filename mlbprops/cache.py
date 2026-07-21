"""Disk cache with TTL for upstream API responses.

Every network call in this project goes through here. Slate-building hits the
same endpoints repeatedly (a player's game log is needed once per prop, not
once per player), and the upstream APIs are free but rate-limited by courtesy.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any, Callable

log = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache"


class DiskCache:
    """Namespaced JSON/text cache keyed by a hash of the request identity."""

    def __init__(self, root: Path | str = DEFAULT_CACHE_DIR):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, namespace: str, key: str, suffix: str) -> Path:
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]
        ns_dir = self.root / namespace
        ns_dir.mkdir(parents=True, exist_ok=True)
        return ns_dir / f"{digest}{suffix}"

    def get_or_fetch(
        self,
        namespace: str,
        key: str,
        fetch: Callable[[], Any],
        ttl_seconds: float,
        *,
        as_text: bool = False,
    ) -> Any:
        """Return a cached value, or call ``fetch`` and cache the result.

        ``ttl_seconds`` of 0 or less forces a refetch. Negative-cached failures
        are not stored: if ``fetch`` raises, the exception propagates and
        nothing is written, so a transient outage does not poison the cache.
        """
        suffix = ".txt" if as_text else ".json"
        path = self._path(namespace, key, suffix)

        if ttl_seconds > 0 and path.exists():
            age = time.time() - path.stat().st_mtime
            if age < ttl_seconds:
                try:
                    raw = path.read_text(encoding="utf-8")
                    return raw if as_text else json.loads(raw)
                except (OSError, json.JSONDecodeError):
                    log.warning("corrupt cache entry %s; refetching", path)

        value = fetch()
        try:
            payload = value if as_text else json.dumps(value)
            path.write_text(payload, encoding="utf-8")
        except (OSError, TypeError) as exc:
            log.warning("could not cache %s/%s: %s", namespace, key, exc)
        return value

    def clear(self, namespace: str | None = None) -> int:
        """Delete cached entries. Returns the number of files removed."""
        target = self.root / namespace if namespace else self.root
        if not target.exists():
            return 0
        removed = 0
        for path in target.rglob("*"):
            if path.is_file():
                path.unlink()
                removed += 1
        return removed


# TTLs, in seconds, tuned to how fast each resource actually changes.
TTL = {
    "schedule": 15 * 60,        # probable pitchers shift during the day
    "lineup": 3 * 60,           # confirmed lineups are the time-critical input
    "gamelog": 6 * 60 * 60,     # only changes after a game completes
    "season_stats": 6 * 60 * 60,
    "savant": 24 * 60 * 60,     # leaderboards refresh nightly
    "roster": 24 * 60 * 60,
    "player": 7 * 24 * 60 * 60,  # bio/handedness is essentially static
    "weather": 30 * 60,
    "odds": 60,                 # lines move; keep this short
    "venue": 30 * 24 * 60 * 60,
}
