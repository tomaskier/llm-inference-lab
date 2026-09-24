"""Start and stop a vLLM server for one variant, from configs/serving/<variant>.yaml.

    python scripts/serve.py start base --prefix-cache off
    python scripts/serve.py stop

The server runs in the background (Kaggle notebooks can't background a shell
command with '&'). Its PID goes to build/server.pid, its log to build/vllm.log,
and the exact launch config to build/server_config.json (FR-2.5).
"""

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, resolve_model, serving_config  # noqa: E402

BUILD = ROOT / "build"
PID_FILE = BUILD / "server.pid"
LOG_FILE = BUILD / "vllm.log"
CONFIG_FILE = BUILD / "server_config.json"


def build_command(variant: str, prefix_cache: bool, port: int, api_key: str | None) -> list[str]:
    cfg = serving_config(variant)
    cmd = [
        "vllm", "serve", resolve_model(cfg["model"]),
        "--dtype", cfg["dtype"],
        "--max-model-len", str(cfg["max_model_len"]),
        "--gpu-memory-utilization", str(cfg["gpu_memory_utilization"]),
        "--max-num-seqs", str(cfg["max_num_seqs"]),
        "--seed", str(cfg["seed"]),
        "--port", str(port),
        # Always explicit: vLLM 0.30 turns prefix caching on by default (FR-2.3).
        "--enable-prefix-caching" if prefix_cache else "--no-enable-prefix-caching",
    ]  # fmt: skip
    if cfg.get("revision"):
        cmd += ["--revision", cfg["revision"]]
    if api_key:
        cmd += ["--api-key", api_key]
    return cmd


def log_stats(log_path: Path) -> dict[str, Any]:
    """Pull memory and kernel facts out of the vLLM startup log (FR-1.2)."""
    text = log_path.read_text(errors="replace") if log_path.exists() else ""
    stats: dict[str, Any] = {}
    if m := re.search(r"Model loading took ([\d.]+) GiB", text):
        stats["weights_gib"] = float(m.group(1))
    if m := re.search(r"Available KV cache memory: ([\d.]+) GiB", text):
        stats["kv_cache_gib"] = float(m.group(1))
    if m := re.search(r"GPU KV cache size: ([\d,]+) tokens", text):
        stats["kv_cache_tokens"] = int(m.group(1).replace(",", ""))
    if m := re.search(r"Maximum concurrency for ([\d,]+) tokens per request: ([\d.]+)x", text):
        stats["max_concurrency_at_max_len"] = float(m.group(2))
    kernels = re.findall(r"(?:Selected|Using) (\w+Kernel) for (\w+)", text)
    if kernels:
        stats["linear_kernels"] = sorted({f"{k} ({q})" for k, q in kernels})
    if m := re.search(r"V1 LLM engine \(v([\w.+-]+)\)", text):
        stats["vllm_version"] = m.group(1)
    return stats


def wait_healthy(port: int, proc: subprocess.Popen[bytes] | None, timeout: int) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(f"vLLM exited with code {proc.returncode}, see {LOG_FILE}")
        try:
            urllib.request.urlopen(f"http://localhost:{port}/health", timeout=5)
            return
        except OSError:
            time.sleep(5)
    raise TimeoutError(f"vLLM not healthy after {timeout}s, see {LOG_FILE}")


def start(args: argparse.Namespace) -> None:
    if PID_FILE.exists():
        sys.exit(f"A server seems to be running (PID file {PID_FILE}). Stop it first.")
    BUILD.mkdir(exist_ok=True)
    prefix_cache = args.prefix_cache == "on"
    cmd = build_command(args.variant, prefix_cache, args.port, args.api_key)

    env = {**os.environ}
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")  # one GPU, even on Kaggle's 2xT4

    with open(LOG_FILE, "w") as log:
        proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    PID_FILE.write_text(str(proc.pid))
    print(f"Started vLLM for {args.variant} (prefix cache {args.prefix_cache}), PID {proc.pid}")

    try:
        wait_healthy(args.port, proc, args.timeout)
    except Exception:
        stop(args)
        raise

    shown = [("***" if prev == "--api-key" else c) for prev, c in zip([""] + cmd, cmd, strict=False)]
    config = {
        "variant": args.variant,
        "prefix_caching": prefix_cache,
        "command": shown,
        "serving_config": serving_config(args.variant),
        "cuda_visible_devices": env["CUDA_VISIBLE_DEVICES"],
        **log_stats(LOG_FILE),
    }
    CONFIG_FILE.write_text(json.dumps(config, indent=2))
    print(f"Ready. Launch config saved to {CONFIG_FILE.relative_to(ROOT)}")


def stop(args: argparse.Namespace | None = None) -> None:
    if not PID_FILE.exists():
        print("No server running.")
        return
    pid = int(PID_FILE.read_text())
    try:
        os.killpg(pid, signal.SIGTERM)
        for _ in range(60):
            os.kill(pid, 0)  # raises once the process is gone
            time.sleep(1)
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    PID_FILE.unlink()
    print(f"Stopped server {pid}.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="action", required=True)
    s = sub.add_parser("start")
    s.add_argument("variant")
    s.add_argument("--prefix-cache", choices=["on", "off"], required=True)
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--api-key", default=os.environ.get("VLLM_API_KEY"))
    s.add_argument("--timeout", type=int, default=600)
    sub.add_parser("stop")
    args = parser.parse_args()
    start(args) if args.action == "start" else stop(args)


if __name__ == "__main__":
    main()
