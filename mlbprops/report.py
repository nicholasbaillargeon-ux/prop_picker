"""Joining model projections to market prices, and ranking the result.

This is where a probability becomes a bet. For every prop offered by a book we
compare the model's win probability against the devigged market consensus, quote
the best available price, and size the stake.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import asdict, dataclass, field

from .markets import PlayerProjection, common_lines, summarize
from .odds import (
    EventOdds,
    PropLine,
    american_to_prob,
    attach_fair_probs,
    best_price,
    consensus_fair_prob,
    expected_value,
    kelly_fraction,
    prob_to_american,
)
from .pipeline import GameProjection

log = logging.getLogger(__name__)

_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


def normalize_name(name: str) -> str:
    """Fold a player name to a comparable key.

    Books and StatsAPI disagree on accents, punctuation, and suffixes
    ("Jose Ramirez" vs "José Ramírez", "Ronald Acuna Jr."), and a missed match
    silently drops a prop from the report rather than failing loudly -- so this
    is deliberately aggressive.
    """
    if not name:
        return ""
    decomposed = unicodedata.normalize("NFKD", name)
    ascii_name = "".join(c for c in decomposed if not unicodedata.combining(c))
    # Periods are dropped rather than turned into spaces so that initials
    # collapse the same way both feeds write them: "J.T." and "JT" must both
    # become "jt", or every player known by initials silently falls off the
    # board. Hyphens do become spaces, since those separate real name parts.
    ascii_name = ascii_name.lower().replace(".", "").replace("-", " ")
    ascii_name = re.sub(r"[^a-z ]", " ", ascii_name)
    parts = [p for p in ascii_name.split() if p and p not in _SUFFIXES]
    return " ".join(parts)


@dataclass
class Recommendation:
    """One prop side, priced by both the model and the market."""

    game: str
    player: str
    team: str
    market: str
    line: float
    side: str

    model_prob: float
    market_prob: float          # devigged consensus across books
    edge: float                 # model_prob - market_prob

    best_book: str
    best_american: float
    model_fair_american: float
    ev: float                   # per unit staked
    kelly: float                # fraction of bankroll

    n_books: int
    mean_outcome: float
    lineup_slot: int | None = None
    is_pitcher: bool = False
    confirmed_lineup: bool = False
    context: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def build_recommendations(
    game_proj: GameProjection,
    event: EventOdds | None,
    *,
    devig_method: str = "shin",
    kelly_multiplier: float = 0.25,
    kelly_cap: float = 0.02,
    min_books: int = 2,
) -> list[Recommendation]:
    """Price every offered prop for one game against the model."""
    if event is None or not event.lines:
        return []

    attach_fair_probs(event.lines, devig_method)
    consensus = consensus_fair_prob(event.lines)
    best = best_price(event.lines)

    by_name: dict[str, PlayerProjection] = {}
    for p in game_proj.projections:
        by_name[normalize_name(p.name)] = p

    book_counts: dict[tuple, set[str]] = {}
    for ln in event.lines:
        book_counts.setdefault(
            (ln.market, ln.player, ln.point, ln.side), set()).add(ln.book)

    label = f"{game_proj.game.away_team} @ {game_proj.game.home_team}"
    out: list[Recommendation] = []
    unmatched: set[str] = set()

    for key, price_line in best.items():
        market, player, point, side = key
        proj = by_name.get(normalize_name(player))
        if proj is None:
            unmatched.add(player)
            continue
        dist = proj.get(market)
        if dist is None:
            continue

        books = book_counts.get(key, set())
        if len(books) < min_books:
            # A price quoted by a single book is as likely to be stale as sharp.
            continue

        model_p = dist.resolve(point, side)
        if not 0.0 < model_p < 1.0:
            continue

        market_p = consensus.get(key, price_line.fair_prob)
        ev = expected_value(model_p, price_line.american)

        out.append(Recommendation(
            game=label,
            player=proj.name,
            team=proj.team,
            market=market,
            line=point,
            side=side,
            model_prob=round(model_p, 4),
            market_prob=round(market_p, 4),
            edge=round(model_p - market_p, 4),
            best_book=price_line.book,
            best_american=price_line.american,
            model_fair_american=prob_to_american(min(max(model_p, 1e-4), 1 - 1e-4)),
            ev=round(ev, 4),
            kelly=round(kelly_fraction(model_p, price_line.american,
                                       kelly_multiplier, kelly_cap), 4),
            n_books=len(books),
            mean_outcome=round(dist.mean, 2),
            lineup_slot=proj.lineup_slot,
            is_pitcher=proj.is_pitcher,
            confirmed_lineup=game_proj.lineups_confirmed,
            context=dict(proj.context, park=game_proj.park.get("name", ""),
                         weather=game_proj.weather.get("source", "")),
        ))

    if unmatched:
        log.info("%s: %d players in odds feed had no projection (%s)",
                 label, len(unmatched), ", ".join(sorted(unmatched)[:5]))
    return out


def rank(recommendations: list[Recommendation],
         *, min_ev: float = 0.0, min_edge: float = 0.0) -> list[Recommendation]:
    """Filter to plays that clear both thresholds, best EV first.

    Requiring *both* a positive EV and a probability edge is intentional. EV
    alone can be manufactured by one book's stale price; an edge against the
    devigged consensus alone can exist at a price too poor to bet. A real play
    needs the model to disagree with the market *and* a price that pays for it.
    """
    keep = [r for r in recommendations if r.ev >= min_ev and r.edge >= min_edge]
    return sorted(keep, key=lambda r: r.ev, reverse=True)


# ---------------------------------------------------------------------------
# Best bets
# ---------------------------------------------------------------------------

# Sample sizes at which a player's history stops meaningfully limiting
# confidence in the projection.
FULL_SAMPLE_PA = 250.0
FULL_SAMPLE_BF = 350.0

# Books quoting a prop before market agreement is treated as established.
FULL_BOOK_COUNT = 4.0


@dataclass
class BestBet:
    """A single highlighted play, with the reasoning that selected it."""

    recommendation: Recommendation
    score: float
    confidence: float
    plausibility: float
    reasons: list[str] = field(default_factory=list)
    scope: str = "game"          # "game" or "slate"

    def to_dict(self) -> dict:
        return {
            **self.recommendation.to_dict(),
            "score": round(self.score, 4),
            "confidence": round(self.confidence, 3),
            "plausibility": round(self.plausibility, 3),
            "reasons": self.reasons,
            "scope": self.scope,
        }


def _confidence(rec: Recommendation) -> tuple[float, list[str]]:
    """How much the inputs behind a recommendation can be trusted.

    Separate from the edge itself. A 6% edge off a confirmed lineup, a full
    season of plate appearances, and six books quoting is a different
    proposition than the same 6% off a projected lineup and one week of data.
    """
    reasons: list[str] = []

    books = min(rec.n_books / FULL_BOOK_COUNT, 1.0)
    if rec.n_books >= FULL_BOOK_COUNT:
        reasons.append(f"{rec.n_books} books quoting")
    elif rec.n_books <= 2:
        reasons.append(f"only {rec.n_books} books")

    if rec.confirmed_lineup:
        lineup = 1.0
        reasons.append("lineup confirmed")
    else:
        lineup = 0.70
        reasons.append("lineup projected")

    ctx = rec.context or {}
    if rec.is_pitcher:
        observed = float(ctx.get("effective_bf") or 0.0)
        sample = min(observed / FULL_SAMPLE_BF, 1.0)
    else:
        observed = float(ctx.get("effective_pa") or 0.0)
        sample = min(observed / FULL_SAMPLE_PA, 1.0)
    sample = max(sample, 0.35)
    if observed < 80:
        reasons.append("thin sample")

    # Probabilities near 0 or 1 are where model error is largest relative to the
    # quantity being estimated, so a coin-flip-ish line is the more trustworthy
    # place to claim an edge.
    p = rec.model_prob
    band = 1.0 - min(abs(p - 0.5) / 0.5, 1.0) ** 2 * 0.55
    if p < 0.15 or p > 0.85:
        reasons.append("tail probability")

    return books * lineup * sample * band, reasons


def _plausibility(rec: Recommendation) -> tuple[float, list[str]]:
    """Discount edges too large to be believable.

    A model that thinks a prop is four times more likely than a liquid market
    does is far more often wrong than right -- that gap usually means a stale
    line, a limit-shaded number, an injury the model has not seen, or a bug.
    Ranking by raw EV would put exactly those cases at the top of the board, so
    implausible disagreement is damped rather than rewarded.
    """
    reasons: list[str] = []
    if rec.market_prob <= 0:
        return 0.5, ["no market baseline"]
    ratio = rec.model_prob / rec.market_prob
    if ratio <= 1.5:
        return 1.0, reasons
    # Decays smoothly: ~0.65 at 2x, ~0.35 at 3x, ~0.2 at 4x.
    plaus = max(1.0 / (1.0 + (ratio - 1.5) ** 1.6), 0.15)
    if ratio >= 2.0:
        reasons.append(f"model {ratio:.1f}x the market — verify the line")
    return plaus, reasons


def score_bet(rec: Recommendation) -> BestBet:
    """Risk-adjusted ranking score for a single recommendation."""
    confidence, reasons = _confidence(rec)
    plausibility, plaus_reasons = _plausibility(rec)
    return BestBet(
        recommendation=rec,
        score=rec.ev * confidence * plausibility,
        confidence=confidence,
        plausibility=plausibility,
        reasons=reasons + plaus_reasons,
    )


def best_bets(
    recommendations: list[Recommendation],
    *,
    min_ev: float = 0.0,
    min_books: int = 2,
) -> tuple[BestBet | None, list[BestBet]]:
    """Pick the model's favourite play per game, and the best overall.

    Selection is by risk-adjusted score rather than raw EV. Ranking on EV alone
    reliably surfaces long-shot home-run props off thin samples against
    one-book prices -- the plays most likely to reflect model error rather than
    genuine edge.

    Returns ``(overall, per_game)`` where ``per_game`` is sorted best first and
    ``overall`` is the same object as the strongest entry in it.
    """
    eligible = [r for r in recommendations
                if r.ev >= min_ev and r.n_books >= min_books]
    if not eligible:
        return None, []

    by_game: dict[str, BestBet] = {}
    for rec in eligible:
        scored = score_bet(rec)
        current = by_game.get(rec.game)
        if current is None or scored.score > current.score:
            by_game[rec.game] = scored

    ranked = sorted(by_game.values(), key=lambda b: b.score, reverse=True)
    if not ranked:
        return None, []

    overall = ranked[0]
    overall.scope = "slate"
    return overall, ranked


def slate_payload(
    game_projections: list[GameProjection],
    recommendations: list[Recommendation],
    *,
    date: str,
    meta: dict | None = None,
    best: BestBet | None = None,
    per_game_best: list[BestBet] | None = None,
) -> dict:
    """Assemble the JSON payload consumed by the dashboard."""
    games = []
    for gp in game_projections:
        players = []
        for p in gp.projections:
            markets = {}
            for name, dist in p.distributions.items():
                markets[name] = summarize(dist, common_lines(name))
            players.append({
                "player_id": p.player_id,
                "name": p.name,
                "team": p.team,
                "is_home": p.is_home,
                "lineup_slot": p.lineup_slot,
                "is_pitcher": p.is_pitcher,
                "context": p.context,
                "markets": markets,
            })
        games.append({
            "game_pk": gp.game.game_pk,
            "label": f"{gp.game.away_team} @ {gp.game.home_team}",
            "away_team": gp.game.away_team,
            "home_team": gp.game.home_team,
            "start_utc": gp.game.game_date,
            "status": gp.game.status,
            "venue": gp.game.venue_name,
            "park": gp.park,
            "weather": gp.weather,
            "lineups_confirmed": gp.lineups_confirmed,
            "notes": gp.notes,
            "team_runs": gp.team_runs,
            "bullpen": gp.bullpen,
            "game_markets": gp.game_markets,
            "lineup_order": gp.lineup_order,
            "away_sp": gp.game.away_probable_name,
            "home_sp": gp.game.home_probable_name,
            "players": players,
        })

    return {
        "date": date,
        "generated_utc": (meta or {}).get("generated_utc", ""),
        "meta": meta or {},
        "games": games,
        "recommendations": [r.to_dict() for r in recommendations],
        "best_bet": best.to_dict() if best else None,
        "best_by_game": [b.to_dict() for b in (per_game_best or [])],
    }
