# LLM Inference Tradeoff Lab

Which model configuration should we ship, and what does it cost?

This project takes one open-source model (Qwen2.5-1.5B-Instruct), builds three versions of it (FP16, INT8, 4-bit), serves each one with vLLM and compares them on speed, accuracy and cost under three traffic patterns. The goal is a recommendation per use case, not just a table of numbers.

**Status:** the pipeline is built and tested on sample data, and the T4 is confirmed to run all three formats. The full benchmark runs are next. The conclusion and the main chart will go here once they're done.

## What gets compared

| Variant | Format | Why it's here |
|---|---|---|
| `base` | FP16 | Reference for speed and accuracy |
| `w8` | INT8 weights and activations (SmoothQuant + GPTQ) | Middle ground |
| `w4` | 4-bit weights (GPTQ) | Smallest, leaves the most room for the KV cache |

| Workload | Shape | What it shows |
|---|---|---|
| `chat_short` | 128 tokens in, 128 out, up to 64 users | Continuous batching |
| `rag_shared_prefix` | 2,048-token shared context + short question, cache on and off | Prefix caching |
| `long_context` | 4,096 tokens in, 256 out | KV cache pressure |

Every workload is swept from 1 user to saturation, three times. Accuracy comes from lm-eval (MMLU, ARC-Challenge, HellaSwag, GSM8K) and perplexity on WikiText-2. Cost is measured at a latency target (p95 time to first token), not at peak throughput.

## Hardware

Everything that needs a GPU runs on a free Kaggle T4. Analysis, the report and monitoring run on a laptop without an NVIDIA GPU. The T4 is from 2018, so the absolute numbers are modest. What matters is how the variants compare. [SPEC §12](SPEC.md#12-platform-decisions) explains what that choice changes.

## Reproduce

**Build the report from saved results (no GPU):**

```bash
pip install -r requirements-dev.txt
python -m analysis.report                     # writes REPORT.md and analysis/figures/
python -m analysis.report --results results/sample --out build/sample   # fake data, for testing
```

**Run the GPU stages on Kaggle:** open `notebooks/kaggle_runner.ipynb` in a Kaggle notebook with a T4 and internet on, then run the stages in order: setup, quantize, measure, evaluate, benchmark, pack results. Each stage skips work that's already done, so a session that times out just picks up where it stopped. Download `results.zip`, unzip it into the repo and commit `results/`.

**Watch it live:** start a server and the tunnel from the notebook (stage E), copy `observability/vllm-target.example.json` to `observability/targets/vllm.json` with the tunnel host, then:

```bash
docker compose -f observability/compose.yml up -d    # Grafana on http://localhost:3000
```

## Layout

```
configs/        model, quantization, serving, eval and cost settings (validated against schemas/)
workloads/      the three traffic patterns
scripts/        quantize, serve, measure, benchmark, evaluate, regression check
analysis/       turns results/ into REPORT.md and charts
results/        raw result files (results/sample/ is fake data for CI)
observability/  Prometheus, alert rules, Grafana dashboard
docs/           runbook, lessons learned, GPU hours log
```

## CI

Every push runs lint, type checks, config validation, unit tests, a report build on the sample data, a regression check against `baselines/baseline.json`, and `promtool` on the alert rules. There's no GPU runner, so `.github/workflows/gpu-benchmark.yml` is a template; the same regression check runs by hand on Kaggle output.

## Versions

vLLM 0.30.0, LLM Compressor 0.14.0, GuideLLM 0.7.4, lm-eval 0.4.13, model revision `989aa79`. Kaggle image: Python 3.12, CUDA 13.0, driver 580.

More detail: [SPEC.md](SPEC.md), [runbook](docs/runbook.md), [lessons learned](docs/lessons-learned.md).
