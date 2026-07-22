"""Build a hosted-artifact version of the dashboard.

The local dashboard talks to a server for live data. A published artifact has
no server and a strict CSP, so this produces a self-contained snapshot:

* the slate is embedded rather than fetched;
* precomputed ``lines`` blocks are dropped and derived from the PMF in the
  browser instead, which is where most of the payload weight lives;
* the document scaffolding is stripped, since the artifact host supplies it;
* server-only affordances (Reload, the live badge) are replaced with an honest
  snapshot stamp.

Usage:  python scripts/build_artifact.py [slate.json] [out.html]
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = ROOT / "mlbprops" / "web" / "dashboard.html"


def trim(payload: dict) -> dict:
    """Shrink the payload without losing anything the page displays.

    ``lines`` is pure redundancy: every over/under/push probability in it can
    be recomputed from the PMF, which the page already ships for its charts.
    Dropping it roughly halves the file.

    Hit-rate cells are redundant the same way. ``of`` is the player's game count,
    identical for every line and already on the parent as ``games``, and ``rate``
    is just ``hit / of`` -- so each cell collapses from a three-key object to a
    bare integer, which is ~350 KB on a full slate. The slate JSON keeps the
    self-describing form; only the shipped page pays for brevity, and the reader
    of the page never sees the difference.
    """
    out = json.loads(json.dumps(payload))  # deep copy
    for game in out.get("games", []):
        for player in game.get("players", []):
            rates = (player.get("context") or {}).get("hit_rates") or {}
            for lines in (rates.get("markets") or {}).values():
                for ln, cell in list(lines.items()):
                    lines[ln] = cell["hit"]
            for market in player.get("markets", {}).values():
                market.pop("lines", None)
                pmf = market.get("pmf") or {}
                market["pmf"] = {
                    k: round(float(v), 4)
                    for k, v in pmf.items()
                    if float(v) >= 0.0008
                }
    return out


def build(slate_path: Path, out_path: Path) -> Path:
    payload = json.loads(slate_path.read_text(encoding="utf-8"))
    trimmed = trim(payload)
    html = TEMPLATE.read_text(encoding="utf-8")

    # 1. Strip the document scaffolding the artifact host provides itself.
    style = re.search(r"<style>(.*?)</style>", html, re.S)
    body = re.search(r"<body>(.*?)</body>", html, re.S)
    if not style or not body:
        raise SystemExit("template structure changed; cannot extract")
    css, markup = style.group(1), body.group(1)

    # 2. Server-only affordances have nothing to talk to on a hosted page.
    markup = markup.replace(
        '<button class="ghost" id="reload" title="Re-fetch from the local server">'
        '↻ Reload</button>', "")
    markup = markup.replace(
        '<span class="live" id="live" hidden><i class="dot" id="dot"></i>'
        '<span id="livetext">—</span></span>',
        '<span class="live" id="live"><i class="dot off"></i>'
        '<span id="livetext">Snapshot</span></span>')

    script = re.search(r"<script>(.*?)</script>", markup, re.S)
    js = script.group(1)
    markup = markup[:script.start()] + "__SCRIPT__" + markup[script.end():]

    # 3. Neutralize the reload/polling wiring; there are no endpoints.
    js = js.replace('$("#reload").addEventListener("click", async () => {',
                    'if (false) ($("#reload") || {addEventListener(){}}).addEventListener("click", async () => {')
    js = js.replace('if (status.running) pollTimer = setInterval(poll, POLL_MS);',
                    'if (false) pollTimer = setInterval(poll, POLL_MS);')
    js = js.replace('function setLive(status) {',
                    'function setLive(_ignored) { return; }\nfunction _unusedSetLive(status) {')

    # The dashboard's own overAt()/marketLines() fall back to the PMF whenever
    # a `lines` block is absent, so dropping it at build time needs no shim.
    js = (f"window.SLATE_DATA = {json.dumps(trimmed, separators=(',', ':'))};\n"
          + js)
    markup = markup.replace("__SCRIPT__", f"<script>{js}</script>")

    meta = trimmed.get("meta", {})
    stamp = (trimmed.get("generated_utc") or "").replace("T", " ")[:16]
    banner = f"""
<p class="note" style="margin:0 0 14px">
  <b>Snapshot</b> of the {trimmed.get('date', '')} slate, built
  {stamp} UTC from {meta.get('n_sims', 20000):,} simulated games per matchup.
  Hosted pages cannot reach the MLB feed, so this does not refresh itself &mdash;
  run <code>python -m mlbprops watch</code> locally for the auto-updating board.
</p>"""
    markup = markup.replace('<div id="view"></div>',
                            f'{banner}\n  <div id="view"></div>')

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(f"<style>{css}</style>\n{markup}", encoding="utf-8")
    return out_path


if __name__ == "__main__":
    slate = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "out" / "slate-latest.json"
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "out" / "artifact.html"
    path = build(slate, out)
    print(f"{path}  ({path.stat().st_size / 1024:.0f} KB)")
