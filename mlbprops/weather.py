"""Game-time weather from Open-Meteo (free, no API key).

We need conditions at first pitch, not a daily average: a 7:10pm start in
Denver is a very different hitting environment than that afternoon. The client
pulls the hourly forecast for the park's coordinates and selects the hour
containing the scheduled start.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import requests

from .cache import TTL, DiskCache

log = logging.getLogger(__name__)

ENDPOINT = "https://api.open-meteo.com/v1/forecast"

# Conditions assumed when a park is domed or the forecast is unavailable.
NEUTRAL = {
    "temp_f": 72.0,
    "humidity_pct": 50.0,
    "wind_mph": 0.0,
    "wind_dir_deg": 0.0,
    "roof_closed": True,
    "source": "neutral",
}


def _to_f(celsius: float) -> float:
    return celsius * 9.0 / 5.0 + 32.0


def _kmh_to_mph(kmh: float) -> float:
    return kmh * 0.621371


class WeatherClient:
    def __init__(self, cache: DiskCache | None = None, timeout: float = 15.0):
        self.cache = cache or DiskCache()
        self.timeout = timeout
        self._session = requests.Session()

    def forecast(self, park: dict, game_time_utc: str) -> dict:
        """Weather at first pitch for a park.

        ``game_time_utc`` is the ISO-8601 start time from StatsAPI. Domed parks
        short-circuit to neutral conditions; retractable-roof parks are treated
        as open, since whether the roof is shut is not published in advance and
        assuming "open" is the higher-variance, more common case.
        """
        roof = str(park.get("roof", "open")).lower()
        if roof == "dome":
            return dict(NEUTRAL, source="dome")

        lat, lon = park.get("lat"), park.get("lon")
        if lat is None or lon is None:
            return dict(NEUTRAL, source="no-coords")

        try:
            start = datetime.fromisoformat(game_time_utc.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            return dict(NEUTRAL, source="bad-time")

        # Open-Meteo only forecasts ~16 days out and cannot serve the past.
        days_ahead = (start - datetime.now(timezone.utc)).days
        if days_ahead > 15:
            return dict(NEUTRAL, source="too-far-out")

        params = {
            "latitude": round(float(lat), 4),
            "longitude": round(float(lon), 4),
            "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,"
                      "wind_direction_10m,precipitation_probability",
            "timezone": "UTC",
            "forecast_days": max(2, min(days_ahead + 2, 16)),
        }
        key = f"{params['latitude']},{params['longitude']},{start:%Y-%m-%dT%H}"

        def fetch() -> dict:
            resp = self._session.get(ENDPOINT, params=params, timeout=self.timeout)
            resp.raise_for_status()
            return resp.json()

        try:
            data = self.cache.get_or_fetch("weather", key, fetch, TTL["weather"])
        except (requests.RequestException, ValueError) as exc:
            log.warning("weather fetch failed for %s: %s", park.get("name"), exc)
            return dict(NEUTRAL, source="fetch-failed")

        hourly = data.get("hourly") or {}
        times = hourly.get("time") or []
        if not times:
            return dict(NEUTRAL, source="empty-forecast")

        target = start.replace(minute=0, second=0, microsecond=0)
        stamp = target.strftime("%Y-%m-%dT%H:00")
        try:
            i = times.index(stamp)
        except ValueError:
            # Fall back to the nearest available hour.
            i = min(
                range(len(times)),
                key=lambda j: abs(
                    datetime.fromisoformat(times[j]).replace(tzinfo=timezone.utc)
                    - target
                ),
            )

        def at(field: str, default: float) -> float:
            series = hourly.get(field) or []
            if i < len(series) and series[i] is not None:
                return float(series[i])
            return default

        return {
            "temp_f": _to_f(at("temperature_2m", 22.2)),
            "humidity_pct": at("relative_humidity_2m", 50.0),
            "wind_mph": _kmh_to_mph(at("wind_speed_10m", 0.0)),
            "wind_dir_deg": at("wind_direction_10m", 0.0),
            "precip_pct": at("precipitation_probability", 0.0),
            "roof_closed": False,
            "source": "open-meteo",
            "valid_time": times[i] if i < len(times) else stamp,
        }
