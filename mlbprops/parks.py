"""Ballpark reference data and lookup.

Park factors, coordinates, elevation, roof type, and field orientation for all
30 current MLB venues. Venue ids, names, coordinates, and roof types are taken
directly from StatsAPI; park factors are Baseball Savant's 2024-2026 three-year
rolling Statcast indices divided by 100.

Two caveats worth knowing before trusting a number here:

* ``cf_azimuth`` (the home-plate-to-center-field bearing, which resolves wind
  into a blowing-out or blowing-in component) is accurate only to about +/-15
  degrees. No public dataset publishes numeric MLB field orientations, so these
  come from published orientation diagrams. loanDepot park and Sutter Health
  Park have no source at all and are estimates. Because the wind term scales
  with the cosine of the angle, a 15-degree error is worth only a few percent
  for a wind near the CF axis, but it matters more for a crosswind.
* Sutter Health Park has no three-year Savant entry (the Athletics moved there
  in 2025), so its factors are a PA-weighted blend of 2025 and partial-2026
  regressed 25% toward neutral.
"""

from __future__ import annotations

from .parks_data import PARKS

# Neutral fallback so an unknown venue degrades to league-average context
# rather than raising.
NEUTRAL_PARK = {
    "venue_id": 0,
    "name": "Unknown Park",
    "team": "",
    "lat": None,
    "lon": None,
    "elevation_ft": 0,
    "roof": "open",
    "cf_azimuth": 0,
    "pf_hr": 1.0,
    "pf_1b": 1.0,
    "pf_2b": 1.0,
    "pf_3b": 1.0,
    "pf_runs": 1.0,
    "pf_so": 1.0,
}

# Venues whose orientation is a low-confidence estimate. Surfaced in the
# dashboard so a wind-driven edge at these parks can be discounted.
LOW_CONFIDENCE_AZIMUTH = {3313, 3289, 2392, 5325, 4169, 2529}


def get_park(venue_id: int) -> dict:
    """Park record for a StatsAPI venue id, or a neutral park if unknown."""
    park = PARKS.get(int(venue_id))
    if park is None:
        return dict(NEUTRAL_PARK)
    out = dict(park)
    out["azimuth_confidence"] = (
        "low" if int(venue_id) in LOW_CONFIDENCE_AZIMUTH else "approximate"
    )
    return out


def by_team(abbr: str) -> dict | None:
    """Park record for a team abbreviation (e.g. ``"COL"``)."""
    target = abbr.upper()
    for park in PARKS.values():
        if str(park.get("team", "")).upper() == target:
            return dict(park)
    return None


__all__ = ["PARKS", "NEUTRAL_PARK", "get_park", "by_team"]
