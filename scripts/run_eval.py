"""Run lm-eval against the running vLLM server and save a compact summary.

    python scripts/serve.py start w8 --prefix-cache off
    python scripts/run_eval.py --variant w8

Every variant gets the same tasks, few-shot counts, limits and seed from
configs/eval.yaml (FR-4.2). Raw lm-eval output stays in build/eval/<variant>/;
the summary goes to results/eval/<variant>.json (FR-4.4).
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CONFIGS, RESULTS, ROOT, load_yaml, resolve_model, serving_config  # noqa: E402

LM_EVAL = os.environ.get("LM_EVAL", str(ROOT / ".venvs/eval/bin/lm_eval"))


def run_task(task: dict, cfg: dict, model: str, out_dir: Path) -> dict:
    model_args = ",".join(
        [
            f"model={model}",
            "base_url=http://localhost:8000/v1/completions",
            f"num_concurrent={cfg['num_concurrent']}",
            "max_retries=3",
            "tokenized_requests=False",
            f"tokenizer={cfg['tokenizer']}",
        ]
    )
    task_dir = out_dir / task["name"]
    cmd = [
        LM_EVAL,
        "--model", "local-completions",
        "--model_args", model_args,
        "--tasks", task["name"],
        "--num_fewshot", str(task["num_fewshot"]),
        "--limit", str(task["limit"]),
        "--seed", str(cfg["seed"]),
        "--output_path", str(task_dir),
    ]  # fmt: skip
    subprocess.run(cmd, check=True)

    latest = max(task_dir.rglob("results_*.json"), key=lambda p: p.stat().st_mtime)
    raw = json.loads(latest.read_text())
    metric = task["metric"]
    name, filt = metric.split(",")
    scores = raw["results"][task["name"]]
    return {
        "metric": metric,
        "value": scores[metric],
        "stderr": scores.get(f"{name}_stderr,{filt}"),
        "num_fewshot": task["num_fewshot"],
        "limit": task["limit"],
        "lm_eval_version": raw.get("lm_eval_version") or raw.get("versions", {}).get(task["name"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--tasks", nargs="+", help="subset of tasks in configs/eval.yaml")
    args = parser.parse_args()

    cfg = load_yaml(CONFIGS / "eval.yaml")
    model = resolve_model(serving_config(args.variant)["model"])
    out = RESULTS / "eval" / f"{args.variant}.json"
    summary = json.loads(out.read_text()) if out.exists() else {"variant": args.variant, "tasks": {}}

    for task in cfg["tasks"]:
        if args.tasks and task["name"] not in args.tasks:
            continue
        print(f"== {args.variant} / {task['name']}")
        summary["tasks"][task["name"]] = run_task(task, cfg, model, ROOT / "build/eval" / args.variant)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2))  # save after each task

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
