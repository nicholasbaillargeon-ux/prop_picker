# MLB Player Prop Model

A full-game Monte Carlo simulator that prices MLB player props from matchup,
recent form, park, weather, lineup context, and market odds — with an
interactive board for reading the output.

```bash
python -m mlbprops watch                  # project, serve, auto-refresh on lineups
```

---

## Why it simulates whole games

Most prop models price each player in isolation: estimate a rate, assume a
plate-appearance count, evaluate a binomial. That approach gets three things
structurally wrong, and this model exists to avoid them.

**Plate appearances are not a constant.** Whether a leadoff hitter bats 4 or 5
times is the single largest driver of his hits and total-bases distribution, and
it depends on how the other eight hitters do. Simulating the lineup produces the
real joint distribution instead of an assumed PA count.

**Runs and RBI are pure lineup context.** They are only well defined if you know
who bats around a player and what the base state looks like when he comes up.
There is no sensible standalone RBI model.

**Pitcher props hinge on the hook.** Strikeout and outs markets are dominated by
how deep the starter goes, which depends on pitch count, which depends on how the
opposing lineup performs against him. The hook is sampled per game from the
pitcher's own workload history, so "will he even face the order a third time"
becomes part of the distribution rather than an assumption.

So the simulator plays the whole game: both lineups, both staffs, base-out state,
double plays, wild pitches, extra-inning ghost runners, and the home team
skipping the ninth when it leads. Every market is read off the same simulated
games, which means a player's hits, total-bases, and RBI numbers are mutually
consistent by construction.

## How a probability gets built

```
game logs ──► recency weighting ──► park neutralization ──► EB shrinkage
                                                                  │
league baseline ──────────────────────────────────────────────────┤
                                                                  ▼
                                              odds-ratio matchup (log5)
                                                                  │
              platoon · park factors · weather · times-through-order
                                                                  ▼
                                        per-PA outcome distribution
                                                                  ▼
                                    20,000 simulated games ──► markets
                                                                  ▼
                                    devigged odds ──► edge, EV, Kelly
```

Every plate appearance resolves to exactly one of eight outcomes —
`K, BB, HBP, HR, 3B, 2B, 1B, OUT` — so the categories are exhaustive and the
accounting always closes.

**Recency weighting.** Game logs decay exponentially with a 30-game half-life.
Long enough that most short streaks are treated as the noise they are, short
enough that a real change in talent shows up.

**Park neutralization.** Raw stat lines are *not* park neutral. A Rockies hitter
has half his games at Coors, so his observed rates already contain that park —
and applying the game's park factor on top would count it twice. Home-game counts
are divided back out before anything else touches them.

**Empirical-Bayes shrinkage.** Each outcome is shrunk toward the league mean
using that outcome's own stabilization point as prior strength. Strikeout rate
(~60 PA) keeps most of its signal; pitcher home-run rate (~1300 BF) collapses
almost entirely to league average. This is what stops the model from paying up
for a 4-homer week.

**Odds-ratio matchup.** Batter and pitcher are combined on the odds scale and
divided by the league baseline, so two average players return exactly the league
rate. Context effects multiply on that same scale, which keeps a park boost
proportional to how much room a probability actually has left.

**Real bullpen splits.** Relief innings come from StatsAPI's `rp` situation
split — an actual aggregate of every relief appearance a team has thrown, not
the team pitching line with a fudge factor. It separates swingmen correctly
(relief outings count, starts do not) and is usage-weighted by construction,
since the relievers who pitch most contribute the most batters faced. This
matters because bullpens genuinely differ: measured relief strikeout rates on
the current slate span .205 to .252, a spread a flat league-wide multiplier
erased entirely. Falls back to the team total if the split is too thin to trust.

**Devigging.** Two-sided props are stripped of margin with Shin's method by
default (power and multiplicative also available). This matters more than it
sounds: a -130/+105 prop carries ~4% margin, and how you distribute it across the
two sides moves the implied probability by more than most claimed edges.

## Data sources

All free; only odds require a key.

| Source | Used for | Key |
|---|---|---|
| MLB StatsAPI | schedule, lineups, game logs, season stats, handedness, venues | none |
| Open-Meteo | first-pitch temperature, wind speed/direction, humidity | none |
| Baseball Savant | park factors (2024–26 rolling Statcast indices) | bundled |
| StatsAPI `sitCodes=rp` | real relief-corps aggregates per team | none |
| The Odds API | player prop lines across US books | `ODDS_API_KEY` |

```bash
export ODDS_API_KEY=...        # free tier at the-odds-api.com
```

Without a key the model still runs and the **All projections** tab works — you
just get probabilities with no prices to bet into.

## Usage

```bash
# the main one: build, serve, and rebuild automatically as lineups post
python -m mlbprops watch
python -m mlbprops watch --interval 60 --host 0.0.0.0   # reachable from your phone

# one-shot run, no watching
python -m mlbprops run --serve

# a specific date, more simulations, standalone HTML file
python -m mlbprops run --date 2026-07-21 --sims 50000 --html

# model only, no odds calls
python -m mlbprops run --no-odds

# tighter thresholds, specific books
python -m mlbprops run --min-ev 0.05 --min-edge 0.03 --books draftkings,fanduel

# serve a slate you already generated
python -m mlbprops serve
```

Useful flags: `--devig {shin,power,multiplicative}`, `--kelly` (multiplier,
default quarter), `--kelly-cap` (default 2% of bankroll), `--min-books` (ignore
props quoted by fewer books, default 2), `--no-cache`, `--verbose`.

Output lands in `out/`: `slate-<date>.json`, `slate-latest.json`, and optionally
`props-<date>.html`.

## The dashboard

Four views, all filterable by game, market, side, role, player, and minimum EV:

- **Best bets** — the model's single favourite play per game, plus one headline
  pick for the slate. Ranked by *risk-adjusted* score rather than raw EV: the
  score discounts for thin samples, unconfirmed lineups, few books quoting, and
  tail probabilities, then damps edges too large to believe. A model claiming
  four times the market's probability is usually looking at a stale line, a
  limit-shaded number, or an injury it has not seen — ranking on raw EV puts
  exactly those at the top, so it does not. Each pick shows the reasoning that
  selected it.
- **Betting board** — every priced prop with model probability, devigged market
  probability, edge, best available price and book, EV, and stake. Click any row
  for the simulated distribution and the full context behind it.
- **All projections** — raw model output with no odds involved. Works with no
  API key.
- **Games** — park, roof, first-pitch weather, projected team runs, starters,
  and whether lineups are confirmed.

Sorting is by column click, there is a CSV export, and the theme follows your
system with a manual toggle. Served from `localhost` unless you pass `--host`.

### Hit rates

Every prop row carries how often the player has actually cleared *that line*
recently — **last 10 games** for a hitter, **last 5 starts** for a pitcher.
Pitchers are counted in starts rather than games because a starter appears every
fifth day, so ten games would reach back nearly two months and stop describing
anything current; relief cameos are excluded via `gamesStarted`, since an
opener appearance is not a start and would drag the counts down for a reason
that has nothing to do with form.

The column shows the count (`7/10`) next to the rate, because "70%" and "7 of
10" invite very different confidence and only one of them is honest about the
sample. It is styled more quietly than the model's probability on purpose.

**This is displayed, not modelled.** It deliberately feeds nothing: the
projection already weights recent games via exponential decay across the whole
season, which is a strictly better estimator than a hard cutoff at 10. A hit
rate ignores the price, the opponent, the park, and the batting order the player
happened to occupy, and at n=10 — let alone n=5 — its standard error is large
enough that a 7/10 and a 5/10 are rarely distinguishable. It is on the board
because it is the most-quoted number in prop betting and worth seeing *beside*
the model rather than instead of it; where the two disagree is the interesting
part, and usually the model is right about why.

### Auto-refresh

Lineups post piecemeal in the couple of hours before first pitch, and they are
the input that most changes a projection — batting order drives plate-appearance
counts, which drive every counting prop. A slate built at noon is largely
guesswork; the same slate after lineups land is the real one.

`mlbprops watch` polls for that and rebuilds when it happens:

- **One request per poll for the whole slate.** Hydrating `lineups` onto the
  schedule returns every game's posted order in ~70 KB. Polling each game's live
  feed would be one request per game for the same information.
- **Only changed games re-project.** A confirmation in one game does not rebuild
  the other twelve.
- **Odds are re-pulled on rebuild**, since lines move while lineups are being
  posted and a fresh projection against stale prices invents edges.
- **The page updates itself.** The browser polls a small `/api/status` endpoint
  and re-downloads the slate only when the generation moves, then re-renders in
  place — your tab, filters, and sort survive the update. A live badge shows how
  many lineups are in, and a toast names the games that changed.
- **It stops when there is nothing left to watch.** Once every game is locked
  (started, final, or postponed) the watcher goes idle and the browser stops
  polling.

The dashboard degrades cleanly without it: the standalone `--html` export has no
endpoints to poll and simply never shows the badge.

## Calibration

Run against league-average inputs, the simulator reproduces real baseball:

| Metric | Model | Actual |
|---|---|---|
| PA per team per game | 38.5 | 38.1 |
| Runs per team per game | 4.34 | 4.40 |
| RBI-to-runs ratio | 0.98 | 0.955 |
| Starter innings | 5.17 | 5.20 |
| Starter strikeouts | 4.97 | 5.10 |
| Starter earned runs | 2.52 | 2.60 |
| Starter pitches | 86 | 87 |
| Games to extra innings | 6.7% | 8.5% |

Park factors reproduce independently: forcing league-average players through
Coors yields a 1.26 implied run factor against a published 1.28, and T-Mobile
0.83 against 0.83.

```bash
python -m pytest tests/ -q        # 79 tests
```

The suite is weighted toward invariants rather than golden numbers — runs must
reconcile with team runs, probabilities must sum to one, a league-average matchup
must return league-average rates — so the model is free to improve without
breaking its own accounting.

## Known limitations

Read these before betting real money.

- **Bullpens do not model individual relievers.** The relief aggregate is real
  and usage-weighted, but it treats the pen as one average arm: it cannot know
  that a closer takes the ninth in a save spot while a mop-up arm takes it in a
  blowout, and it has no fatigue or availability feed, so a reliever who threw
  40 pitches yesterday looks identical to a rested one.
- **Field orientations are ±15°.** No public dataset publishes numeric MLB
  home-plate-to-center-field bearings, so `cf_azimuth` comes from published
  diagrams. Since the wind term scales with the cosine of the angle, a crosswind
  is where this hurts most. loanDepot park and Sutter Health Park are unsourced
  estimates; `parks.LOW_CONFIDENCE_AZIMUTH` lists the shakiest entries.
- **Retractable roofs are assumed open**, since closure is not published in
  advance.
- **No batter-vs-pitcher history.** Deliberately: BvP samples are far too small
  to carry signal, and using them would be fitting noise.
- **No injury, weather-delay, or bullpen-usage feed.** A reliever who threw 40
  pitches yesterday looks identical to a rested one.
- **Weather is applied against a global reference**, not each park's own seasonal
  norm, so a small part of the temperature effect is already inside the park
  factor.
- **Unconfirmed lineups fall back** to a team's last-used batting order. Those
  rows are flagged in the UI and via `confirmed_lineup`; batting order drives PA
  counts, so treat them as provisional.
- **Best bets still depend on odds.** With no API key the tab is empty, because
  a favourite play is meaningless without a price to compare against.
- **Kelly stakes are computed per prop, independently.** Props within a game are
  strongly correlated, so summed stakes overstate safe exposure. Size per game.

A model edge is not a guarantee, and the market is sharp. The most common way
this loses money is betting into a stale line the model thinks it beats.

## Layout

```
mlbprops/
  constants.py   outcome taxonomy, league rates, stabilization points, coefficients
  cache.py       TTL disk cache — every network call goes through it
  statsapi.py    MLB StatsAPI client
  savant.py      park factors (bundled data in parks_data.py)
  parks.py       venue lookup, factors, coordinates, orientation
  weather.py     Open-Meteo first-pitch conditions
  rates.py       recency weighting, park neutralization, EB shrinkage, hit rates
  matchup.py     odds-ratio combination, platoon, park, weather, TTO
  sim.py         vectorized full-game Monte Carlo
  markets.py     simulated distributions -> prop probabilities
  odds.py        odds API, devigging, EV, Kelly
  report.py      model/market join, ranking, JSON payload
  pipeline.py    slate orchestration
  serve.py       local server, status endpoint, standalone HTML export
  watch.py       lineup polling and incremental re-projection
  cli.py         command line interface
  web/           dashboard
tests/           79 tests
```
