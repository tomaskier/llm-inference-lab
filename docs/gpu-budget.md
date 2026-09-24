# GPU budget log

Kaggle free tier: ~30 GPU-hours/week, T4 16 GB. Estimated total for the project: ~16 h (SPEC §7).

| Date | Session | What ran | Hours |
|---|---|---|---|
| 2026-09-24 | 1 | Milestone 1: vLLM 0.30.0 install, offline test (0.5B), `vllm serve` base 1.5B FP16, chat request | ~1 |

**Used so far:** ~1 h

## Findings

- vLLM 0.30.0 / torch 2.13.0+cu130 / Python 3.12.13 runs on the T4 (driver 580, CUDA 13.0).
- `base` FP16 at `gpu_memory_utilization=0.85`: 8.25 GiB KV cache = 309,056 tokens, max concurrency 37.73x at 8,192 tokens/request.
- Kaggle notebooks reject background shell jobs (`!cmd &`); start the server with `subprocess.Popen`.
- pip packages are lost when the session stops; reinstall from `requirements-gpu.txt` each session.
