# GPU budget log

Kaggle free tier: ~30 GPU-hours/week, T4 16 GB. Estimated total for the project: ~16 h (SPEC §7).

| Date | Session | What ran | Hours |
|---|---|---|---|
| 2026-09-24 | 1 | Milestone 1: vLLM 0.30.0 install, offline test (0.5B), `vllm serve` base 1.5B FP16, chat request | ~1 |
| 2026-09-24 | 2 | FR-1.0: quantize 0.5B to W8A8 and W4A16 (64 samples), serve both, check kernels | ~1 |

**Used so far:** ~2 h (estimates; check the quota page on Kaggle)

## Findings

- vLLM 0.30.0 / torch 2.13.0+cu130 / Python 3.12.13 runs on the T4 (driver 580, CUDA 13.0).
- `base` FP16 at `gpu_memory_utilization=0.85`: 8.25 GiB KV cache = 309,056 tokens, max concurrency 37.73x at 8,192 tokens/request.
- Kaggle notebooks reject background shell jobs (`!cmd &`); start the server with `subprocess.Popen`.
- pip packages are lost when the session stops; reinstall from `requirements-gpu.txt` each session.
- FR-1.0 (0.5B test models): W8A8 → `CutlassInt8ScaledMMLinearKernel`, W4A16 → `MarlinLinearKernel`. Both load and serve on the T4.
- Test sizes on disk: 0.5B W8A8 613 MB, W4A16 447 MB (FP16 is ~990 MB). Embeddings and `lm_head` stay in FP16.
- llmcompressor 0.14.0 can't be installed next to vllm 0.30.0; it lives in its own venv.
- vLLM 0.30 enables prefix caching by default.
