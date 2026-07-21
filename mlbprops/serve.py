"""Local web server for the dashboard.

Serves the dashboard shell and the slate JSON from the same origin so the
page's Reload button picks up a fresh model run without a rebuild. Binds to
localhost only -- this exposes betting positions and should not be reachable
from the network.
"""

from __future__ import annotations

import json
import logging
import threading
import webbrowser
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

log = logging.getLogger(__name__)

WEB_DIR = Path(__file__).resolve().parent / "web"
DASHBOARD = WEB_DIR / "dashboard.html"


class SlateHandler(BaseHTTPRequestHandler):
    """Serves the dashboard and the current slate payload."""

    def __init__(self, *args, slate_path: Path, status_path: Path | None = None,
                 **kwargs):
        self.slate_path = slate_path
        self.status_path = status_path or slate_path.parent / "status.json"
        super().__init__(*args, **kwargs)

    def log_message(self, fmt: str, *args) -> None:  # quieter default logging
        log.debug("%s - %s", self.address_string(), fmt % args)

    def _send(self, body: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 - stdlib naming
        path = self.path.split("?", 1)[0]

        if path in ("/", "/index.html", "/dashboard.html"):
            if not DASHBOARD.exists():
                self._send(b"dashboard.html missing", "text/plain", 500)
                return
            self._send(DASHBOARD.read_bytes(), "text/html; charset=utf-8")
            return

        if path == "/api/status":
            # Cheap poll target: the browser hits this on a timer and only
            # re-downloads the (much larger) slate when the generation moves.
            if self.status_path.exists():
                self._send(self.status_path.read_bytes(), "application/json")
            else:
                self._send(json.dumps({"generation": 0, "running": False,
                                       "message": "no watcher running"}).encode(),
                           "application/json")
            return

        if path in ("/api/slate", "/slate.json"):
            if not self.slate_path.exists():
                self._send(
                    json.dumps({
                        "date": "", "games": [], "recommendations": [],
                        "error": f"no slate at {self.slate_path}; run the model first",
                    }).encode(),
                    "application/json", 404)
                return
            self._send(self.slate_path.read_bytes(), "application/json")
            return

        self._send(b"not found", "text/plain", 404)


def serve(slate_path: Path, host: str = "127.0.0.1", port: int = 8765,
          open_browser: bool = True, status_path: Path | None = None,
          block: bool = True) -> ThreadingHTTPServer:
    handler = partial(SlateHandler, slate_path=Path(slate_path),
                      status_path=Path(status_path) if status_path else None)
    httpd = ThreadingHTTPServer((host, port), handler)
    url = f"http://{host}:{port}/"
    print(f"Dashboard: {url}")
    print(f"Slate:     {slate_path}")
    print("Ctrl-C to stop.")
    if open_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    if not block:
        threading.Thread(target=httpd.serve_forever, daemon=True,
                         name="slate-server").start()
        return httpd
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        httpd.server_close()
    return httpd


def export_standalone(payload: dict, out_path: Path) -> Path:
    """Write a single self-contained HTML file with the slate embedded.

    Useful for sharing a slate or keeping a dated snapshot: the result needs no
    server and no network.
    """
    if not DASHBOARD.exists():
        raise FileNotFoundError(f"missing template: {DASHBOARD}")
    html = DASHBOARD.read_text(encoding="utf-8")
    # </script> inside the JSON would close the tag early.
    blob = json.dumps(payload).replace("</", "<\\/")
    injected = f"<script>window.SLATE_DATA = {blob};</script>\n</head>"
    html = html.replace("</head>", injected, 1)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    return out_path
