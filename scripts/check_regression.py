"""Compare a result set against baselines/baseline.json and fail on regressions.

    python scripts/check_regression.py --results results
    python scripts/check_regression.py --results results --write   # new baseline

A check fails if p95 TTFT gets worse or throughput drops by more than the
tolerance (10% by default) at any baseline load point, or if average accuracy
drops by more than the allowed points.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from analysis.load import load_benchmarks, load_eval  # noqa: E402
from analysis.report import aggregate  # noqa: E402

BASELINE = ROOT / "baselines" / "baseline.json"
# One mid-load point per cell keeps the check quick and less noisy than the extremes.
CHECK_CONCURRENCY = 8


def current_numbers(results: Path) -> dict:
    agg = aggregate(load_benchmarks(results))
    points = agg[agg["concurrency"] == CHECK_CONCURRENCY]
    cells = [
        {
            "variant": r.variant,
            "cell": r.cell,
            "concurrency": int(r.concurrency),
            "ttft_p95_ms": round(float(r.ttft_p95_ms), 2),
            "output_tok_per_s": round(float(r.output_tok_per_s), 2),
        }
        for r in points.itertuples()
    ]
    ev = load_eval(results)
    accuracy = (ev.groupby("variant")["value"].mean() * 100).round(2).to_dict() if not ev.empty else {}
    return {"cells": cells, "accuracy": accuracy}


def compare(baseline: dict, current: dict, partial: bool = False) -> list[str]:
    tol = baseline["tolerance"]
    now = {(c["variant"], c["cell"], c["concurrency"]): c for c in current["cells"]}
    failures = []
    for ref in baseline["cells"]:
        key = (ref["variant"], ref["cell"], ref["concurrency"])
        name = "/".join(map(str, key))
        cur = now.get(key)
        if cur is None:
            if not partial:
                failures.append(f"{name}: missing from the new results")
            continue
        if cur["ttft_p95_ms"] > ref["ttft_p95_ms"] * (1 + tol["latency"]):
            failures.append(f"{name}: p95 TTFT {ref['ttft_p95_ms']} -> {cur['ttft_p95_ms']} ms")
        if cur["output_tok_per_s"] < ref["output_tok_per_s"] * (1 - tol["throughput"]):
            failures.append(
                f"{name}: throughput {ref['output_tok_per_s']} -> {cur['output_tok_per_s']} tok/s"
            )
    for variant, ref_acc in baseline.get("accuracy", {}).items():
        cur_acc = current["accuracy"].get(variant)
        if cur_acc is not None and cur_acc < ref_acc - tol["accuracy_points"]:
            failures.append(f"{variant}: average accuracy {ref_acc} -> {cur_acc}")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--baseline", type=Path, default=BASELINE)
    parser.add_argument("--write", action="store_true", help="save these results as the new baseline")
    parser.add_argument("--partial", action="store_true", help="only check cells present in --results")
    args = parser.parse_args()

    current = current_numbers(args.results)
    if args.write:
        data = {
            "source": args.results.resolve().relative_to(ROOT).as_posix(),
            "tolerance": {"latency": 0.10, "throughput": 0.10, "accuracy_points": 2.0},
            **current,
        }
        args.baseline.write_text(json.dumps(data, indent=2) + "\n")
        print(f"Wrote {args.baseline} with {len(current['cells'])} load points")
        return 0

    failures = compare(json.loads(args.baseline.read_text()), current, args.partial)
    for f in failures:
        print(f"REGRESSION {f}")
    print("OK: no regressions" if not failures else f"{len(failures)} regression(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
