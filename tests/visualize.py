#!/usr/bin/env python3

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from cache_client import CacheClient


HTML_PATH = Path(__file__).with_name("visualize.html")


def make_handler(rust_host: str, rust_port: int) -> type[BaseHTTPRequestHandler]:
    client = CacheClient(rust_host, rust_port)

    class VisualizeHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            path = urlparse(self.path).path
            if path == "/":
                self.send_file(HTML_PATH, "text/html; charset=utf-8")
                return

            if path == "/api/tvcache":
                try:
                    payload = client.dump_all()
                    task_ids = sorted(payload.keys()) if isinstance(payload, dict) else []
                    print("Received TVCache from Rust:")
                    print(f"  task_ids: {task_ids}")
                    print(json.dumps(payload))
                    self.send_json(payload, status=200)
                except Exception as error:
                    self.send_json({"error": str(error)}, status=502)
                return

            self.send_json({"error": "not found"}, status=404)

        def do_POST(self) -> None:
            path = urlparse(self.path).path
            if path != "/api/find":
                self.send_json({"error": "not found"}, status=404)
                return

            try:
                body = json.loads(self.rfile.read(self.content_length()))
                task_id = body.get("task_id")
                find_path = body.get("path")
                if not isinstance(task_id, str) or not isinstance(find_path, list):
                    raise ValueError("expected JSON body with task_id string and path list")
                if not all(isinstance(entry, str) for entry in find_path):
                    raise ValueError("path entries must all be strings")

                payload = client.find(task_id, find_path)
                print("Find result from Rust:")
                print(f"  task_id: {task_id}")
                print(f"  path: {find_path}")
                print(json.dumps(payload))
                self.send_json(payload, status=200)
            except Exception as error:
                self.send_json({"error": str(error)}, status=502)

        def log_message(self, format: str, *args: object) -> None:
            print(f"{self.address_string()} - {format % args}")

        def content_length(self) -> int:
            raw_length = self.headers.get("Content-Length", "0")
            return int(raw_length)

        def send_file(self, path: Path, content_type: str) -> None:
            encoded = path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(encoded)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(encoded)

        def send_json(self, body: object, status: int) -> None:
            encoded = json.dumps(body).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(encoded)

    return VisualizeHandler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8080, type=int)
    parser.add_argument("--rust-host", default="127.0.0.1")
    parser.add_argument("--rust-port", default=7878, type=int)
    args = parser.parse_args()

    handler = make_handler(args.rust_host, args.rust_port)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"Serving TCG visualizer at http://{args.host}:{args.port}")
    print(f"Serving HTML from {HTML_PATH}")
    print(f"Reading TVCache from Rust TCP server at {args.rust_host}:{args.rust_port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
