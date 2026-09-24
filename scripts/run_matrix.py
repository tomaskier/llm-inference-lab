"""Run the benchmark matrix: variant x workload (x prefix caching) x repeats.

    python scripts/run_matrix.py                      # everything
    python scripts/run_matrix.py --variants base --workloads chat_short --runs 1

Resumable (FR-7.2): a run whose JSON already exists is skipped, and results are
written to a temp file first, so a Kaggle session that dies mid-run leaves no
half-written file behind.

Output: results/benchmarks/<variant>/<cell>/run_<n>.json plus server_config.json.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    RESULTS,
    ROOT,
    VARIANTS,
    cell_name,
    resolve_model,
    serving_config,
    workload_config,
    workload_names,
)

GUIDELLM = os.environ.get("GUIDELLM", str(ROOT / ".venvs/bench/bin/guidellm"))
SERVE = [sys.executable, str(ROOT / "scripts/serve.py")]
TARGET = "http://localhost:8000"


def guidellm(workload: dict, streams: list[int], seconds: int, seed: int, out: Path) -> None:
    profile = {"kind": "concurrent", "streams": streams, "warmup": 0.1, "cooldown": 0.1}
    data = {"kind": "synthetic_text", **workload["data"]}
    cmd = [
        GUIDELLM, "run",
        # Plain completions, so prompt sizes are exactly what the workload says
        # (the chat endpoint would add template tokens).
        "--backend", f"kind=openai_http,target={TARGET},request_format=/v1/completions",
        "--profile", json.dumps(profile),
        "--constraint", f"kind=max_duration,seconds={seconds}",
        "--data", json.dumps(data),
        "--seed", f"kind=static,value={seed}",
        "--metrics", "kind=generative,sample_size=0",  # stats only, keeps files small
        "--output", f"kind=json,path={out}",
        "--disable-console-interactive",
    ]  # fmt: skip
    subprocess.run(cmd, check=True)


def warmup(workload: dict) -> None:
    """Short throwaway run in the workload's shape (FR-3.4). vLLM compiles some
    Triton kernels on the first request of each shape, which shows up as a
    latency spike if it happens inside a measured run."""
    out = ROOT / "build" / "warmup.json"
    guidellm(workload, [4], seconds=20, seed=0, out=out)
    out.unlink(missing_ok=True)


def run_dir(variant: str, cell: str) -> Path:
    return RESULTS / "benchmarks" / variant / cell


def pending(variant: str, cells: list[tuple[str, str]], runs: int) -> bool:
    return any(
        not (run_dir(variant, cell) / f"run_{n}.json").exists()
        for _, cell in cells
        for n in range(1, runs + 1)
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--variants", nargs="+", default=VARIANTS)
    parser.add_argument("--workloads", nargs="+", default=workload_names())
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()

    for variant in args.variants:
        model = resolve_model(serving_config(variant)["model"])
        if model.startswith("/") and not Path(model).exists():
            print(f"skip {variant}: {model} not found (set MODELS_DIR?)")
            continue

        # One server start per prefix-caching setting, not per workload.
        for cache in (False, True):
            cells = []
            for name in args.workloads:
                options = workload_config(name)["prefix_caching"]
                if cache in options:
                    cells.append((name, cell_name(name, cache, options)))
            if not cells or not pending(variant, cells, args.runs):
                continue

            flag = "on" if cache else "off"
            subprocess.run(SERVE + ["start", variant, "--prefix-cache", flag], check=True)
            try:
                for name, cell in cells:
                    wl = workload_config(name)
                    folder = run_dir(variant, cell)
                    folder.mkdir(parents=True, exist_ok=True)
                    shutil.copy(ROOT / "build/server_config.json", folder / "server_config.json")
                    warmup(wl)
                    for n in range(1, args.runs + 1):
                        final = folder / f"run_{n}.json"
                        if final.exists():
                            continue
                        print(f"== {variant} / {cell} / run {n}")
                        tmp = folder / f".run_{n}.partial.json"
                        guidellm(wl, wl["concurrency"], wl["seconds_per_point"], seed=n, out=tmp)
                        tmp.rename(final)
            finally:
                subprocess.run(SERVE + ["stop"], check=False)


if __name__ == "__main__":
    main()
