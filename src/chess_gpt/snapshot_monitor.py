"""Serve a live logarithmic loss chart for a local snapshot training run."""

# ruff: noqa: E501

from __future__ import annotations

import argparse
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

MONITOR_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Chess GPT training loss</title>
  <style>
    :root { color-scheme: light dark; --bg: #fafafa; --fg: #171717; --muted: #666; --grid: #d4d4d4; --raw: #9ca3af; --line: #2563eb; --button: #171717; --button-fg: #fff; }
    @media (prefers-color-scheme: dark) { :root { --bg: #111; --fg: #f5f5f5; --muted: #aaa; --grid: #404040; --raw: #737373; --line: #60a5fa; --button: #f5f5f5; --button-fg: #111; } }
    * { box-sizing: border-box; }
    body { margin: 0; padding: 24px; background: var(--bg); color: var(--fg); font: 16px system-ui, sans-serif; }
    main { max-width: 1000px; margin: auto; }
    header { display: flex; gap: 16px; align-items: center; justify-content: space-between; flex-wrap: wrap; }
    h1 { margin: 0; font-size: 1.25rem; font-weight: 500; }
    #status { color: var(--muted); }
    button { appearance: none; border: 0; border-radius: 6px; padding: 10px 14px; background: var(--button); color: var(--button-fg); font: inherit; cursor: pointer; }
    button:disabled { opacity: .55; cursor: default; }
    svg { display: block; width: 100%; height: auto; margin-top: 18px; }
    .grid { stroke: var(--grid); stroke-width: 1; }
    .axis-label, .tick { fill: var(--muted); font-size: 13px; }
    .raw { fill: none; stroke: var(--raw); stroke-width: 1; opacity: .45; }
    .smooth { fill: none; stroke: var(--line); stroke-width: 2.5; }
    .endpoint { fill: var(--line); }
    @media (max-width: 520px) { body { padding: 14px; } }
  </style>
</head>
<body>
  <main>
    <header>
      <div><h1>Logarithmic training loss</h1><div id="status" aria-live="polite">Waiting for the first update…</div></div>
      <button id="stop" type="button">End training</button>
    </header>
    <svg id="chart" viewBox="0 0 900 420" role="img" aria-label="Batch and smoothed cross-entropy loss over training time on a logarithmic scale"></svg>
  </main>
  <script>
    const svg = document.getElementById("chart");
    const status = document.getElementById("status");
    const stopButton = document.getElementById("stop");
    const ns = "http://www.w3.org/2000/svg";
    const add = (name, attrs, text = "") => {
      const element = document.createElementNS(ns, name);
      for (const [key, value] of Object.entries(attrs)) element.setAttribute(key, value);
      if (text) element.textContent = text;
      svg.appendChild(element);
      return element;
    };
    const pathFor = (events, field, x, y) => events.map((event, index) => `${index ? "L" : "M"}${x(event.elapsed_seconds).toFixed(1)},${y(event[field]).toFixed(1)}`).join(" ");
    function render(events) {
      svg.replaceChildren();
      const valid = events.filter(event => event.loss > 0 && event.smoothed_loss > 0);
      if (!valid.length) return;
      const left = 70, right = 875, top = 18, bottom = 370;
      const maxTime = Math.max(1, ...valid.map(event => event.elapsed_seconds));
      const values = valid.flatMap(event => [event.loss, event.smoothed_loss]);
      const minLog = Math.floor(Math.log10(Math.min(...values)) * 2) / 2;
      const maxLog = Math.ceil(Math.log10(Math.max(...values)) * 2) / 2;
      const span = Math.max(.5, maxLog - minLog);
      const x = seconds => left + (seconds / maxTime) * (right - left);
      const y = loss => bottom - ((Math.log10(loss) - minLog) / span) * (bottom - top);
      for (let step = 0; step <= 4; step += 1) {
        const seconds = maxTime * step / 4;
        const px = x(seconds);
        add("line", { x1: px, x2: px, y1: top, y2: bottom, class: "grid" });
        add("text", { x: px, y: 394, "text-anchor": "middle", class: "tick" }, `${(seconds / 60).toFixed(0)}m`);
      }
      for (let value = minLog; value <= maxLog + .001; value += .5) {
        const loss = 10 ** value;
        const py = y(loss);
        add("line", { x1: left, x2: right, y1: py, y2: py, class: "grid" });
        add("text", { x: left - 10, y: py + 4, "text-anchor": "end", class: "tick" }, loss >= 1 ? loss.toFixed(1) : loss.toPrecision(2));
      }
      add("path", { d: pathFor(valid, "loss", x, y), class: "raw" });
      add("path", { d: pathFor(valid, "smoothed_loss", x, y), class: "smooth" });
      const latest = valid.at(-1);
      add("circle", { cx: x(latest.elapsed_seconds), cy: y(latest.smoothed_loss), r: 4, class: "endpoint" });
      add("text", { x: 472, y: 416, "text-anchor": "middle", class: "axis-label" }, "Elapsed training time");
      add("text", { x: 18, y: 194, transform: "rotate(-90 18 194)", "text-anchor": "middle", class: "axis-label" }, "Cross-entropy loss (log scale)");
      status.textContent = `Update ${latest.update.toLocaleString()} · ${latest.positions.toLocaleString()} positions · smoothed loss ${latest.smoothed_loss.toFixed(4)}`;
    }
    async function refresh() {
      try { render(await (await fetch("/events", { cache: "no-store" })).json()); }
      catch { status.textContent = "Waiting for the trainer…"; }
    }
    stopButton.addEventListener("click", async () => {
      stopButton.disabled = true;
      await fetch("/stop", { method: "POST" });
      stopButton.textContent = "Stop requested";
      status.textContent = "Finishing the current update and saving a checkpoint…";
    });
    refresh();
    setInterval(refresh, 2000);
  </script>
</body>
</html>
"""


def load_loss_events(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "losses.jsonl"
    if not path.is_file():
        return []
    events: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            events.append(event)
    return events


def request_stop(run_dir: Path) -> None:
    """Ask the trainer to stop after its current update and save a checkpoint."""
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "STOP").write_text("graceful stop requested\n")


def create_monitor_server(
    run_dir: Path, *, host: str = "127.0.0.1", port: int = 8765
) -> ThreadingHTTPServer:
    """Create, but do not start, a local monitor for one run directory."""

    class MonitorHandler(BaseHTTPRequestHandler):
        def _send(self, payload: bytes, content_type: str) -> None:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self) -> None:
            if self.path == "/events":
                payload = json.dumps(load_loss_events(run_dir), separators=(",", ":")).encode()
                self._send(payload, "application/json")
                return
            if self.path == "/":
                self._send(MONITOR_HTML.encode(), "text/html; charset=utf-8")
                return
            self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:
            if self.path != "/stop":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            request_stop(run_dir)
            self._send(b'{"stop_requested":true}', "application/json")

        def log_message(self, format: str, *args: object) -> None:
            return

    return ThreadingHTTPServer((host, port), MonitorHandler)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    server = create_monitor_server(args.run, host=args.host, port=args.port)
    print(f"Training monitor: http://{args.host}:{server.server_port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
