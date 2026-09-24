"""Check that every metric the dashboard and alerts use exists on a live server.

    python scripts/check_metrics.py --url http://localhost:8000

vLLM renames metrics between versions, and a renamed metric doesn't break
anything loudly: the panel just goes empty and the alert never fires.
"""

import argparse
import re
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCES = [ROOT / "observability/alerts.yml", ROOT / "observability/grafana/dashboards/vllm.json"]
NAME = re.compile(r"\b(vllm:[a-z_:]+|http_requests_total)\b")


def used_metrics() -> set[str]:
    names = set()
    for path in SOURCES:
        names |= set(NAME.findall(path.read_text()))
    # histogram_quantile(..._bucket) only needs the base histogram to exist
    return {n.removesuffix("_bucket") for n in names}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8000")
    args = parser.parse_args()

    with urllib.request.urlopen(f"{args.url.rstrip('/')}/metrics", timeout=30) as resp:
        text = resp.read().decode()
    exposed = set(re.findall(r"^# TYPE (\S+)", text, re.MULTILINE))
    # Counters show up as "x_total" in samples but "x" in the TYPE line on some clients.
    exposed |= {f"{n}_total" for n in exposed}

    missing = sorted(n for n in used_metrics() if n not in exposed)
    for n in sorted(used_metrics()):
        print(f"{'MISSING' if n in missing else 'ok     '} {n}")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
