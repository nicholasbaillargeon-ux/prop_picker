"""MLB player prop model.

A full-game Monte Carlo simulator that prices batter and starting-pitcher props
from matchup, recent form, park, weather, lineup context, and market odds.
"""

__version__ = "0.1.0"

from .markets import ALL_MARKETS, Distribution, PlayerProjection
from .pipeline import GameProjection, Projector
from .rates import RateEstimate, estimate_batter, estimate_pitcher
from .sim import SimResult, simulate

__all__ = [
    "ALL_MARKETS",
    "Distribution",
    "GameProjection",
    "PlayerProjection",
    "Projector",
    "RateEstimate",
    "SimResult",
    "estimate_batter",
    "estimate_pitcher",
    "simulate",
]
