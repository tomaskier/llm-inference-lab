"""Expose only vLLM's /metrics on another port, for the cloudflared tunnel.

    python scripts/metrics_proxy.py            # serves :9101/metrics from :8000/metrics

Tunnelling vLLM itself would put the whole OpenAI API on a public URL. This
proxy answers /metrics and nothing else, so Prometheus can scrape from the
laptop while the API stays on localhost.
"""

import argparse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def handler_for(upstream: str) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path != "/metrics":
                self.send_error(404)
                return
            try:
                with urllib.request.urlopen(upstream, timeout=10) as resp:
                    body = resp.read()
                    ctype = resp.headers.get("Content-Type", "text/plain")
            except OSError:
                self.send_error(502, "vLLM not reachable")
                return
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args: object) -> None:  # keep the notebook quiet
            pass

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=9101)
    parser.add_argument("--upstream", default="http://localhost:8000/metrics")
    args = parser.parse_args()
    ThreadingHTTPServer(("0.0.0.0", args.port), handler_for(args.upstream)).serve_forever()


if __name__ == "__main__":
    main()
