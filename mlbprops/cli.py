"""Command line interface."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from .cache import DiskCache
from .odds import DEFAULT_MARKETS, OddsAPIError, OddsClient
from .pipeline import Projector
from .report import (
    best_bets,
    build_recommendations,
    normalize_name,
    rank,
    slate_payload,
)
from .serve import export_standalone, serve
from .statsapi import StatsAPI
from .watch import SlateWatcher

log = logging.getLogger("mlbprops")

DEFAULT_OUT = Path("out")


def _today() -> str:
    return datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d")


def _match_events(games, events) -> dict[int, str]:
    """Map StatsAPI game_pk -> Odds API event id by team names and date.

    The two feeds use different team-name conventions ("Athletics" vs "Oakland
    Athletics"), so matching is on the last word of each club name, which is
    the nickname in every case, plus the calendar date.
    """
    def key(name: str) -> str:
        return normalize_name(name).split()[-1] if name else ""

    index: dict[tuple, str] = {}
    for ev in events or []:
        index[(key(ev.get("home_team", "")), key(ev.get("away_team", "")))] = ev.get("id", "")

    out: dict[int, str] = {}
    for g in games:
        ev_id = index.get((key(g.home_team), key(g.away_team)))
        if ev_id:
            out[g.game_pk] = ev_id
        else:
            log.info("no odds event matched for %s @ %s", g.away_team, g.home_team)
    return out


def cmd_run(args: argparse.Namespace) -> int:
    date = args.date or _today()
    cache = DiskCache()
    if args.no_cache:
        cleared = cache.clear()
        log.info("cleared %d cached files", cleared)

    api = StatsAPI(cache=cache)
    projector = Projector(api=api, n_sims=args.sims, seed=args.seed)

    print(f"Projecting {date} ({args.sims:,} sims/game)...")
    game_projections = projector.project_slate(date)
    if not game_projections:
        print("No games could be projected. Lineups or probable pitchers may "
              "not be posted yet.", file=sys.stderr)
        return 1

    recommendations = []
    odds_note = "odds disabled"
    if not args.no_odds:
        client = OddsClient(api_key=args.odds_key, cache=cache)
        if not client.enabled:
            odds_note = "no ODDS_API_KEY set"
            print("\n! No odds API key found. Running model-only.\n"
                  "  Set ODDS_API_KEY=... (get one free at the-odds-api.com) "
                  "to price edges.\n", file=sys.stderr)
        else:
            try:
                events = client.events()
                mapping = _match_events([gp.game for gp in game_projections], events)
                markets = args.markets.split(",") if args.markets else DEFAULT_MARKETS
                for gp in game_projections:
                    ev_id = mapping.get(gp.game.game_pk)
                    if not ev_id:
                        continue
                    try:
                        event = client.event_props(ev_id, markets, args.regions,
                                                   args.books)
                    except OddsAPIError as exc:
                        log.warning("odds for %s: %s", gp.game.game_pk, exc)
                        continue
                    recommendations.extend(build_recommendations(
                        gp, event,
                        devig_method=args.devig,
                        kelly_multiplier=args.kelly,
                        kelly_cap=args.kelly_cap,
                        min_books=args.min_books,
                    ))
                odds_note = (f"{len(recommendations)} priced props; "
                             f"quota remaining {client.last_quota.get('remaining', '?')}")
            except OddsAPIError as exc:
                odds_note = f"odds unavailable: {exc}"
                print(f"! {odds_note}", file=sys.stderr)

    ranked = rank(recommendations, min_ev=args.min_ev, min_edge=args.min_edge)
    overall_best, game_bests = best_bets(recommendations, min_ev=args.min_ev,
                                         min_books=args.min_books)

    payload = slate_payload(
        game_projections, recommendations, date=date,
        best=overall_best, per_game_best=game_bests,
        meta={
            "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "n_sims": args.sims,
            "devig": args.devig,
            "kelly_multiplier": args.kelly,
            "kelly_cap": args.kelly_cap,
            "odds": odds_note,
            "games_projected": len(game_projections),
        },
    )
    payload["generated_utc"] = payload["meta"]["generated_utc"]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    slate_path = out_dir / f"slate-{date}.json"
    slate_path.write_text(json.dumps(payload), encoding="utf-8")
    (out_dir / "slate-latest.json").write_text(json.dumps(payload), encoding="utf-8")

    html_path = None
    if args.html:
        html_path = export_standalone(payload, out_dir / f"props-{date}.html")

    # -- console summary --------------------------------------------------
    print(f"\n{len(game_projections)} games projected · {odds_note}")
    print(f"slate: {slate_path}")
    if html_path:
        print(f"html:  {html_path}")

    if overall_best:
        r = overall_best.recommendation
        print("\n" + "=" * 74)
        print("BEST BET OF THE SLATE")
        print("=" * 74)
        print(f"  {r.player} — {r.market.replace('_', ' ')} {r.side} {r.line}")
        print(f"  {r.game}")
        print(f"  {r.best_american:+.0f} at {r.best_book}   "
              f"model {r.model_prob:.1%} vs market {r.market_prob:.1%}   "
              f"EV {r.ev:+.1%}   stake {r.kelly:.2%}")
        print(f"  confidence {overall_best.confidence:.0%}"
              + (f" — {'; '.join(overall_best.reasons)}" if overall_best.reasons else ""))

    if game_bests:
        print("\nBest bet per game:\n")
        hdr = (f"{'game':<42}{'player':<20}{'market':<16}{'line':>5}"
               f"{'side':>6}{'price':>8}{'EV':>8}{'conf':>6}")
        print(hdr)
        print("-" * len(hdr))
        for b in game_bests:
            r = b.recommendation
            print(f"{r.game[:41]:<42}{r.player[:19]:<20}"
                  f"{r.market.replace('_',' ')[:15]:<16}{r.line:>5}{r.side:>6}"
                  f"{r.best_american:>+8.0f}{r.ev:>+8.1%}{b.confidence:>6.0%}")

    if ranked:
        print(f"\nTop plays (EV >= {args.min_ev:.0%}, edge >= {args.min_edge:.0%}):\n")
        hdr = (f"{'player':<22}{'market':<18}{'line':>6}{'side':>7}"
               f"{'model':>8}{'mkt':>8}{'price':>8}{'EV':>8}{'stake':>7}  book")
        print(hdr)
        print("-" * len(hdr))
        for r in ranked[:args.top]:
            price = f"{r.best_american:+.0f}"
            print(f"{r.player[:21]:<22}{r.market[:17]:<18}{r.line:>6}{r.side:>7}"
                  f"{r.model_prob:>8.1%}{r.market_prob:>8.1%}{price:>8}"
                  f"{r.ev:>+8.1%}{r.kelly:>7.2%}  {r.best_book}")
        unconfirmed = sum(1 for r in ranked[:args.top] if not r.confirmed_lineup)
        if unconfirmed:
            print(f"\n  ! {unconfirmed} of these have projected (not confirmed) "
                  "lineups -- batting order and PA counts are estimates.")
    elif recommendations:
        print(f"\nNo plays cleared the thresholds "
              f"(EV >= {args.min_ev:.0%}, edge >= {args.min_edge:.0%}). "
              f"{len(recommendations)} props priced.")

    if args.serve:
        print()
        serve(out_dir / "slate-latest.json", port=args.port,
              open_browser=not args.no_browser)
    return 0


def _odds_builder(client, mapping_getter, args):
    """Build a per-game odds callback for the watcher.

    Odds are re-pulled on each rebuild because lines move while lineups are
    being posted -- a projection refreshed against stale prices would show
    edges that no longer exist.
    """
    if client is None or not client.enabled:
        return None

    markets = args.markets.split(",") if args.markets else DEFAULT_MARKETS

    def build(gp):
        ev_id = mapping_getter().get(gp.game.game_pk)
        if not ev_id:
            return []
        try:
            event = client.event_props(ev_id, markets, args.regions, args.books)
        except OddsAPIError as exc:
            log.warning("odds refresh failed for %s: %s", gp.game.game_pk, exc)
            return []
        return build_recommendations(
            gp, event, devig_method=args.devig, kelly_multiplier=args.kelly,
            kelly_cap=args.kelly_cap, min_books=args.min_books)

    return build


def cmd_watch(args: argparse.Namespace) -> int:
    date = args.date or _today()
    cache = DiskCache()
    api = StatsAPI(cache=cache)
    projector = Projector(api=api, n_sims=args.sims, seed=args.seed)

    client = None
    mapping: dict[int, str] = {}
    if not args.no_odds:
        client = OddsClient(api_key=args.odds_key, cache=cache)
        if not client.enabled:
            print("! No ODDS_API_KEY — running model-only.", file=sys.stderr)
            client = None
        else:
            try:
                events = client.events()
                games = api.schedule(date)
                mapping = _match_events(games, events)
            except OddsAPIError as exc:
                print(f"! odds unavailable: {exc}", file=sys.stderr)
                client = None

    watcher = SlateWatcher(
        date=date,
        out_dir=Path(args.out),
        projector=projector,
        api=api,
        cache=cache,
        odds_builder=_odds_builder(client, lambda: mapping, args),
        interval=args.interval,
        min_ev=args.min_ev,
        min_edge=args.min_edge,
        min_books=args.min_books,
        meta={"n_sims": args.sims, "devig": args.devig,
              "kelly_multiplier": args.kelly, "kelly_cap": args.kelly_cap},
    )

    print(f"Building initial slate for {date} ({args.sims:,} sims/game)...")
    watcher.initial_build()
    st = watcher.status
    print(f"  {st.games_confirmed}/{st.games_total} lineups confirmed, "
          f"{st.games_pending} still pending")
    if st.games_pending == 0:
        print("  all lineups already locked — nothing left to watch")
    else:
        print(f"  re-checking every {args.interval}s; the page updates itself")

    watcher.start()
    try:
        serve(watcher.slate_path, host=args.host, port=args.port,
              open_browser=not args.no_browser,
              status_path=watcher.status_path)
    finally:
        watcher.stop()
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    slate = Path(args.slate) if args.slate else Path(args.out) / "slate-latest.json"
    if not slate.exists():
        print(f"No slate at {slate}. Run `mlbprops run` first.", file=sys.stderr)
        return 1
    serve(slate, host=args.host, port=args.port,
          open_browser=not args.no_browser,
          status_path=Path(args.out) / "status.json")
    return 0


def cmd_clear_cache(args: argparse.Namespace) -> int:
    n = DiskCache().clear(args.namespace)
    print(f"cleared {n} cached files"
          + (f" from {args.namespace}" if args.namespace else ""))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="mlbprops",
        description="MLB player prop model: simulate a slate, price props, "
                    "and serve an interactive board.")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="project a slate and price props")
    run.add_argument("--date", help="YYYY-MM-DD (default: today)")
    run.add_argument("--sims", type=int, default=20000,
                     help="simulated games per matchup (default 20000)")
    run.add_argument("--seed", type=int, default=20260721,
                     help="RNG seed; fixed by default so runs are reproducible")
    run.add_argument("--out", default=str(DEFAULT_OUT), help="output directory")
    run.add_argument("--html", action="store_true",
                     help="also write a standalone HTML board")
    run.add_argument("--serve", action="store_true",
                     help="serve the dashboard when the run finishes")
    run.add_argument("--port", type=int, default=8765)
    run.add_argument("--no-browser", action="store_true")
    run.add_argument("--no-odds", action="store_true",
                     help="skip the odds API entirely")
    run.add_argument("--odds-key", default=os.environ.get("ODDS_API_KEY"))
    run.add_argument("--regions", default="us")
    run.add_argument("--books", help="comma-separated bookmaker keys")
    run.add_argument("--markets", help="comma-separated odds-API market keys")
    run.add_argument("--devig", default="shin",
                     choices=("shin", "power", "multiplicative"))
    run.add_argument("--kelly", type=float, default=0.25,
                     help="Kelly multiplier (default quarter Kelly)")
    run.add_argument("--kelly-cap", type=float, default=0.02,
                     help="max stake as a fraction of bankroll")
    run.add_argument("--min-ev", type=float, default=0.03)
    run.add_argument("--min-edge", type=float, default=0.02)
    run.add_argument("--min-books", type=int, default=2,
                     help="ignore props quoted by fewer books than this")
    run.add_argument("--top", type=int, default=25)
    run.add_argument("--no-cache", action="store_true",
                     help="clear the disk cache before running")
    run.set_defaults(func=cmd_run)

    srv = sub.add_parser("serve", help="serve an existing slate")
    srv.add_argument("--slate", help="path to a slate JSON")
    srv.add_argument("--out", default=str(DEFAULT_OUT))
    srv.add_argument("--port", type=int, default=8765)
    srv.add_argument("--host", default="127.0.0.1")
    srv.add_argument("--no-browser", action="store_true")
    srv.set_defaults(func=cmd_serve)

    w = sub.add_parser(
        "watch",
        help="project, serve, and auto-rebuild as lineups get confirmed")
    w.add_argument("--date", help="YYYY-MM-DD (default: today)")
    w.add_argument("--interval", type=int, default=90,
                   help="seconds between lineup checks (default 90)")
    w.add_argument("--sims", type=int, default=20000)
    w.add_argument("--seed", type=int, default=20260721)
    w.add_argument("--out", default=str(DEFAULT_OUT))
    w.add_argument("--port", type=int, default=8765)
    w.add_argument("--host", default="127.0.0.1",
                   help="use 0.0.0.0 to reach it from another device")
    w.add_argument("--no-browser", action="store_true")
    w.add_argument("--no-odds", action="store_true")
    w.add_argument("--odds-key", default=os.environ.get("ODDS_API_KEY"))
    w.add_argument("--regions", default="us")
    w.add_argument("--books")
    w.add_argument("--markets")
    w.add_argument("--devig", default="shin",
                   choices=("shin", "power", "multiplicative"))
    w.add_argument("--kelly", type=float, default=0.25)
    w.add_argument("--kelly-cap", type=float, default=0.02)
    w.add_argument("--min-ev", type=float, default=0.03)
    w.add_argument("--min-edge", type=float, default=0.02)
    w.add_argument("--min-books", type=int, default=2)
    w.set_defaults(func=cmd_watch)

    cc = sub.add_parser("clear-cache", help="drop cached API responses")
    cc.add_argument("--namespace", help="e.g. lineup, odds, gamelog")
    cc.set_defaults(func=cmd_clear_cache)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
