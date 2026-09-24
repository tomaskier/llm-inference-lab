"""Generate a small, fake result set in the same format as the real one.

    python -m analysis.make_sample

CI uses results/sample/ to check that the report still builds without a GPU.
The numbers come from a toy latency model, not from any measurement.
"""

import json
import shutil
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "sample"

# Toy model knobs: decode speed-up, prefill speed-up, and how early the queue builds.
VARIANTS = {
    "base": {"decode": 1.00, "prefill": 1.00, "room": 1.00},
    "w8": {"decode": 1.20, "prefill": 1.35, "room": 1.10},
    "w4": {"decode": 1.45, "prefill": 0.90, "room": 1.20},
}


def point(v: dict, prompt: int, output: int, c: int, rng: np.random.Generator) -> dict:
    jitter = rng.normal(1, 0.03)
    prefill_ms = 0.09 * prompt / v["prefill"] + 15
    knee = 24 * v["room"] * (512 / (prompt + output)) ** 0.5
    ttft = prefill_ms * (1 + 3 * (c / knee) ** 2) * jitter
    itl = 11 / v["decode"] * (1 + c / 48) * jitter
    e2e = ttft + output * itl
    tput = c * output / (e2e / 1000)

    def dist(x: float, tail: float) -> dict:
        return {"mean": x, "median": x, "percentiles": {"p50": x, "p95": x * tail, "p99": x * tail * 1.25}}

    n = int(tput / output * 40)
    return {
        "config": {"strategy": {"type_": "concurrent", "streams": c}},
        "metrics": {
            "request_totals": {"successful": n, "errored": 0, "incomplete": 0, "total": n},
            "requests_per_second": {"successful": dist(tput / output, 1)},
            "output_tokens_per_second": {"successful": dist(tput, 1)},
            "request_concurrency": {"successful": dist(c, 1)},
            "time_to_first_token_ms": {"successful": dist(ttft, 1.7)},
            "inter_token_latency_ms": {"successful": dist(itl, 1.3)},
            "request_latency": {"successful": dist(e2e / 1000, 1.4)},
        },
    }


def main() -> None:
    shutil.rmtree(OUT, ignore_errors=True)
    rng = np.random.default_rng(0)

    for path in sorted((ROOT / "workloads").glob("*.yaml")):
        wl = yaml.safe_load(path.read_text())
        options = wl["prefix_caching"]
        for cache in options:
            cell = f"{path.stem}_cache-{'on' if cache else 'off'}" if len(options) > 1 else path.stem
            prefix = wl["data"].get("prefix_tokens", 0)
            prompt = wl["data"]["prompt_tokens"] + (0 if cache else prefix)
            for name, v in VARIANTS.items():
                folder = OUT / "benchmarks" / name / cell
                folder.mkdir(parents=True)
                for run in (1, 2, 3):
                    benches = [
                        point(v, prompt, wl["data"]["output_tokens"], c, rng) for c in wl["concurrency"]
                    ]
                    report = {"metadata": {"note": "synthetic sample"}, "benchmarks": benches}
                    (folder / f"run_{run}.json").write_text(json.dumps(report), newline="\n")

    quant = {
        "base": {"disk_gib": 2.88, "weights_gib": 2.89, "kv_cache_tokens": 309056, "perplexity": 9.61},
        "w8": {"disk_gib": 1.73, "weights_gib": 1.75, "kv_cache_tokens": 350720, "perplexity": 9.70},
        "w4": {"disk_gib": 1.12, "weights_gib": 1.16, "kv_cache_tokens": 372224, "perplexity": 10.35},
    }
    scores = {  # (value, stderr)
        "base": {"mmlu": (0.600, 0.014), "arc_challenge": (0.470, 0.022), "hellaswag": (0.660, 0.021), "gsm8k": (0.620, 0.031)},
        "w8": {"mmlu": (0.597, 0.014), "arc_challenge": (0.468, 0.022), "hellaswag": (0.657, 0.021), "gsm8k": (0.608, 0.031)},
        "w4": {"mmlu": (0.574, 0.014), "arc_challenge": (0.451, 0.022), "hellaswag": (0.641, 0.021), "gsm8k": (0.548, 0.031)},
    }  # fmt: skip
    for name in VARIANTS:
        for kind, data in (
            ("quantization", {"variant": name, **quant[name]}),
            (
                "eval",
                {
                    "variant": name,
                    "tasks": {t: {"value": s, "stderr": e} for t, (s, e) in scores[name].items()},
                },
            ),
        ):
            f = OUT / kind / f"{name}.json"
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(json.dumps(data, indent=2) + "\n", newline="\n")

    (OUT / "README.md").write_text(
        "Fake results made by `python -m analysis.make_sample`, used by CI to test the report.\n"
        "Nothing in this folder was measured.\n",
        newline="\n",
    )
    print(f"Wrote sample results to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
