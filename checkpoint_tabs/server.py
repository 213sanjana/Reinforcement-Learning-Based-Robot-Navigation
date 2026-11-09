"""Local page that starts one checkpoint navigation per click.

The page binds to localhost. Gazebo motion still requires the live
authorization flag and environment variable. This server does not train.
"""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from checkpoint_tabs.catalog import checkpoint_tabs
from checkpoint_tabs.page import render_page
from checkpoint_tabs.selector import CheckpointSelector
from jackal_rl_navigation.warehouse_config import load_warehouse_config
from jackal_rl_navigation.warehouse_estop import live_authorized

ROOT = Path(__file__).resolve().parents[1]
RECORD = Path(__file__).resolve().parent / "four_route_record.html"
DEFAULT_CHECKPOINT = (
    ROOT / "artifacts/warehouse_ppo/runs/run_20261103T054820Z_bf036311"
)
DEFAULT_OUTPUT = ROOT / "artifacts/warehouse_ppo/checkpoint_tabs"


class CheckpointTabHandler(BaseHTTPRequestHandler):
    """HTTP routes for the checkpoint page. The selector is set on the class."""

    selector: CheckpointSelector

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/":
            self._send(200, "text/html; charset=utf-8", render_page().encode("utf-8"))
        elif path == "/record":
            self._send(200, "text/html; charset=utf-8", RECORD.read_bytes())
        elif path == "/api/checkpoints":
            body = json.dumps(checkpoint_tabs()).encode("utf-8")
            self._send(200, "application/json", body)
        elif path == "/api/status":
            body = json.dumps(self.selector.status()).encode("utf-8")
            self._send(200, "application/json", body)
        else:
            self._send(404, "application/json", b'{"error":"not found"}')

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path != "/api/go":
            self._send(404, "application/json", b'{"error":"not found"}')
            return
        length = int(self.headers.get("Content-Length", "0"))
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            status = self.selector.start(str(payload["id"]))
        except KeyError:
            self._send(404, "application/json", b'{"state":"error","error":"Unknown checkpoint"}')
            return
        except SystemExit as exc:
            body = json.dumps({"state": "error", "error": str(exc)}).encode("utf-8")
            self._send(403, "application/json", body)
            return
        except RuntimeError as exc:
            body = json.dumps({"state": "error", "error": str(exc)}).encode("utf-8")
            self._send(409, "application/json", body)
            return
        except (json.JSONDecodeError, TypeError, ValueError):
            self._send(400, "application/json", b'{"state":"error","error":"Checkpoint id is required"}')
            return
        self._send(202, "application/json", json.dumps(status).encode("utf-8"))

    def log_message(self, fmt: str, *args) -> None:
        return

    def _send(self, status: int, content_type: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def serve(selector: CheckpointSelector, host: str = "127.0.0.1", port: int = 8766) -> ThreadingHTTPServer:
    CheckpointTabHandler.selector = selector
    server = ThreadingHTTPServer((host, port), CheckpointTabHandler)
    return server


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Open clickable warehouse checkpoint tabs. One click drives the Jackal to that checkpoint.",
    )
    parser.add_argument("--config", default=str(ROOT / "configs/warehouse_ppo.yaml"))
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--authorize-live", action="store_true")
    args = parser.parse_args(argv)
    config = load_warehouse_config(args.config)
    selector = CheckpointSelector(
        config,
        Path(args.checkpoint),
        authorize_live=live_authorized(args.authorize_live),
        seed=args.seed,
        output_dir=Path(args.output),
    )
    server = serve(selector, args.host, args.port)
    print(f"Checkpoint tabs: http://{args.host}:{args.port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
