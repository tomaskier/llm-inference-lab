"""Read raw result files into flat pandas tables.

Benchmarks are GuideLLM 0.7 JSON reports: one file per run, one entry in
"benchmarks" per concurrency level. Only the summary stats are read, so files
saved with --metrics sample_size=0 work fine.
"""

import json
from pathlib import Path
from typing import Any

import pandas as pd


def _find(obj: Any, key: str) -> Any:
    """First value for `key` anywhere inside nested dicts/lists."""
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        obj = list(obj.values())
    if isinstance(obj, list):
        for item in obj:
            found = _find(item, key)
            if found is not None:
                return found
    return None


def _ok(metric: dict) -> dict:
    return metric["successful"]


def benchmark_points(path: Path) -> list[dict]:
    report = json.loads(path.read_text())
    points = []
    for bench in report["benchmarks"]:
        m = bench["metrics"]
        streams = _find(bench.get("config", {}), "streams")
        if isinstance(streams, list):  # the profile lists all levels; the strategy has one
            streams = _find(bench["config"].get("strategy", {}), "streams")
        if not isinstance(streams, int):
            streams = round(_ok(m["request_concurrency"])["mean"])

        ttft = _ok(m["time_to_first_token_ms"])["percentiles"]
        itl = _ok(m["inter_token_latency_ms"])["percentiles"]
        e2e = _ok(m["request_latency"])["percentiles"]  # seconds
        totals = m["request_totals"]
        total = totals.get("total") or sum(totals.get(k, 0) for k in ("successful", "errored", "incomplete"))

        points.append(
            {
                "concurrency": streams,
                "requests_per_s": _ok(m["requests_per_second"])["mean"],
                "output_tok_per_s": _ok(m["output_tokens_per_second"])["mean"],
                **{f"ttft_{p}_ms": ttft[p] for p in ("p50", "p95", "p99")},
                **{f"itl_{p}_ms": itl[p] for p in ("p50", "p95", "p99")},
                **{f"e2e_{p}_ms": e2e[p] * 1000 for p in ("p50", "p95", "p99")},
                "requests": total,
                "error_rate": totals.get("errored", 0) / total if total else 0.0,
            }
        )
    return points


def load_benchmarks(results: Path) -> pd.DataFrame:
    """results/benchmarks/<variant>/<cell>/run_<n>.json -> one row per load point."""
    rows = []
    for path in sorted((results / "benchmarks").glob("*/*/run_*.json")):
        variant, cell = path.parts[-3], path.parts[-2]
        run = int(path.stem.removeprefix("run_"))
        for point in benchmark_points(path):
            rows.append({"variant": variant, "cell": cell, "run": run, **point})
    return pd.DataFrame(rows)


def load_quantization(results: Path) -> pd.DataFrame:
    rows = [json.loads(p.read_text()) for p in sorted((results / "quantization").glob("*.json"))]
    return pd.DataFrame(rows).set_index("variant") if rows else pd.DataFrame()


def load_eval(results: Path) -> pd.DataFrame:
    """results/eval/<variant>.json -> one row per (variant, task)."""
    rows = []
    for path in sorted((results / "eval").glob("*.json")):
        data = json.loads(path.read_text())
        for task, r in data["tasks"].items():
            rows.append(
                {"variant": data["variant"], "task": task, "value": r["value"], "stderr": r["stderr"]}
            )
    return pd.DataFrame(rows)
