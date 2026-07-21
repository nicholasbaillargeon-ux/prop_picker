#!/usr/bin/env bash
#
# Rebuild today's slate and produce a publishable artifact.
#
# This is what the scheduled cloud routine runs. It is deliberately a plain
# shell script rather than logic embedded in the routine's prompt, so the
# refresh can be run and debugged by hand exactly as the routine runs it.
#
# Publishing is NOT done here: writing the HTML is a local operation, but
# pushing it to the hosted artifact URL requires Claude's Artifact tool, so the
# routine's prompt handles that step after this script exits.
#
# Usage:  scripts/daily_refresh.sh [YYYY-MM-DD]

set -euo pipefail

cd "$(dirname "$0")/.."

DATE="${1:-$(date -u +%F)}"
SIMS="${SIMS:-20000}"

echo "=== MLB prop refresh: ${DATE} (${SIMS} sims/game) ==="

# Dependencies. Installation is best-effort: on a PEP 668 "externally managed"
# Python (Debian/Ubuntu system interpreters) a plain pip install is refused, and
# on a prepared image the packages are already present. So try the reasonable
# variants, then verify by import and fail only if something is genuinely
# missing -- a blocked installer is not itself an error.
if [ -f requirements.txt ]; then
  python3 -m pip install --quiet --disable-pip-version-check -r requirements.txt 2>/dev/null \
    || python3 -m pip install --quiet --disable-pip-version-check --user -r requirements.txt 2>/dev/null \
    || python3 -m pip install --quiet --disable-pip-version-check --break-system-packages -r requirements.txt 2>/dev/null \
    || echo "note: pip install unavailable; using preinstalled packages"
fi

missing=$(python3 - <<'PY'
import importlib
missing = [m for m in ("numpy", "scipy", "requests")
           if not importlib.util.find_spec(m)]
print(" ".join(missing))
PY
)
if [ -n "${missing}" ]; then
  echo "error: required packages missing and could not be installed: ${missing}" >&2
  echo "       install them with: python3 -m pip install ${missing}" >&2
  exit 1
fi

# Odds are optional. With ODDS_API_KEY set the board gains real prices, edge,
# EV, and stake sizing; without it the model still produces every projection.
ODDS_FLAG=""
if [ -z "${ODDS_API_KEY:-}" ]; then
  echo "note: ODDS_API_KEY not set — running model-only (no prices, no EV)"
  ODDS_FLAG="--no-odds"
fi

python3 -m mlbprops run --date "${DATE}" --sims "${SIMS}" ${ODDS_FLAG}

python3 scripts/build_artifact.py out/slate-latest.json out/artifact.html

# A short summary the routine can quote back without re-reading the payload.
python3 - <<'PY'
import json
from pathlib import Path

slate = json.loads(Path("out/slate-latest.json").read_text())
games = slate.get("games", [])
confirmed = sum(1 for g in games if g.get("lineups_confirmed"))
size = Path("out/artifact.html").stat().st_size / 1024

print()
print(f"games projected : {len(games)}")
print(f"lineups confirmed: {confirmed}/{len(games)}")
print(f"artifact         : out/artifact.html ({size:.0f} KB)")

best = slate.get("best_bet")
if best:
    print(f"best bet         : {best['player']} {best['market']} "
          f"{best['side']} {best['line']} ({best['ev']:+.1%} EV)")
else:
    print("best bet         : none (no odds loaded)")

Path("out/summary.txt").write_text(
    f"{len(games)} games, {confirmed} confirmed lineups, {size:.0f} KB artifact\n")
PY

echo "=== done ==="
