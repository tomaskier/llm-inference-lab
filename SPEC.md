# LLM Inference Tradeoff Lab: Project Spec

**Status:** Draft v2 (2026-09-22): adapted to a zero-budget platform (free Kaggle T4 + local laptop). v1 is kept in `docs/SPEC_v1.md`.
**Owner:** _your name_
**One-line pitch:** A reproducible optimize → deploy → benchmark pipeline that answers: *"Which model configuration should we ship, and what does it cost?"*

---

## 1. Goals and Non-Goals

### Goals
- Run the full production workflow (quantize with LLM Compressor, serve with vLLM, benchmark with GuideLLM, evaluate with lm-eval) on a real open-source model.
- Compare multiple model variants under **realistic traffic shapes**, not just one synthetic test.
- Produce a **decision**, not just numbers: a per-use-case recommendation backed by an accuracy-vs-speed-vs-cost analysis.
- Make everything **reproducible**: one command per stage, configs in version control, results saved as files so reviewers don't need a GPU.
- Do all of it for **$0**, and document honestly what that constraint changes.
- Include production-style extras: live observability (Prometheus + Grafana with alerts), CI regression checks, and a written lessons-learned section.

### Non-Goals
- Training or fine-tuning models.
- Multi-GPU serving (Kaggle offers 2×T4, but we use exactly one GPU so results are comparable to a single-GPU deployment).
- Building a chat UI or product on top of the model.
- Beating published benchmarks or reporting "fast" absolute numbers. The T4 is an old GPU; the value is in the **relative** comparison and the methodology.

### Success Criteria
- [ ] A reviewer can read the README top section and understand the conclusion in under 2 minutes.
- [ ] Every chart in the report can be regenerated from raw result files with one command, on a laptop without a GPU.
- [ ] At least 3 model variants and 3 workloads are benchmarked, with accuracy measured for every variant.
- [ ] The Grafana dashboard shows live vLLM metrics during a load test, with at least 2 alert rules that fired.
- [ ] CI runs on every PR and fails on config errors or benchmark regressions (tested on checked-in sample results).
- [ ] The GPU matrix can be resumed after a Kaggle session ends without redoing finished cells.

---

## 2. Target Audience and Skills Demonstrated

| Skill | Where it shows up |
|---|---|
| Model compression (quantization) and hardware-support analysis | Sections 4.1, 12 |
| LLM serving internals (continuous batching, PagedAttention, prefix caching) | Sections 4.2, 4.3 |
| Performance benchmarking under load | Section 4.3 |
| Model quality evaluation | Section 4.4 |
| Cost and capacity analysis | Section 4.5 |
| Observability and alerting across networks | Section 5.1 |
| CI/CD and reproducibility under resource limits | Sections 5.2, 4.7 |
| Technical communication | Sections 4.6, 5.3, 12 |

Relevant roles: LLM/ML engineer, MLOps/platform engineer, inference/AI infrastructure engineer, solutions engineer.

---

## 3. Platform

| Environment | Hardware | Used for |
|---|---|---|
| **Kaggle Notebook** (free) | 1× NVIDIA T4 16 GB (Turing, compute capability 7.5), ~30 GPU-h/week, sessions ≤ 12 h | Quantization, vLLM serving, GuideLLM, lm-eval, perplexity |
| **Laptop** (Windows 11 + WSL2 Ubuntu + Docker) | Intel Arc 140V, no CUDA | Analysis, report, Prometheus + Grafana, local CI checks |
| **GitHub Actions** (free) | CPU runners | Lint, schema validation, analysis on sample results, promtool |

**T4 constraints that shape the design:**
- No BF16 → `base` runs in **FP16**.
- No FP8 tensor cores → `w8` uses **INT8 (W8A8)**.
- 4-bit and INT8 kernel support on sm75 had to be **verified before building the variants** (FR-1.0). Both work; see §12.
- No Docker inside Kaggle → vLLM runs as a plain process (`vllm serve`); the Docker Compose file is kept as a documented template for a GPU VM.

### 3.1 Architecture

```
                 ┌──────────────────────┐
                 │  Base model (HF Hub) │  pinned revision
                 └──────────┬───────────┘
 ══════════ KAGGLE (T4) ════╪════════════════════════════════════════
                            │  LLM Compressor
           ┌────────────────┼────────────────┐
           ▼                ▼                ▼
       FP16 base         INT8 W8A8       4-bit weights
           └───────┬────────┴────────┬───────┘
                   ▼  (saved as a private Kaggle Dataset, reused across sessions)
           ┌───────────────┐
           │  vllm serve   │── /metrics ──┐
           │ (OpenAI API)  │              │ cloudflared tunnel (during load tests)
           └───────┬───────┘              │
         ┌─────────┴──────────┐           │
         ▼                    ▼           │
    GuideLLM sweeps       lm-eval         │
         └─────────┬──────────┘           │
                   ▼                      │
      /kaggle/working/results → download  │
 ══════════════════╪══════════════════════╪═════ LAPTOP ═════════════
                   ▼                      ▼
        results/ (raw JSON, in git)   Prometheus → Grafana (+ alerts)
                   ▼                  (docker compose, local)
        analysis/ → charts + tables + REPORT.md
                   ▼
        GitHub Actions (CI on sample results)
```

---

## 4. Core Requirements

### 4.1 Model Optimization

**Base model:** `Qwen/Qwen2.5-1.5B-Instruct` with the HF revision (commit hash) pinned in `configs/model.yaml`. Fall back to `Qwen2.5-0.5B-Instruct` only if 1.5B does not fit the time budget; document it if so.

**Variants:**

| ID | Scheme | Method | Purpose |
|---|---|---|---|
| `base` | FP16 | none | Reference for speed and accuracy |
| `w8` | INT8 W8A8 | SmoothQuant + GPTQ (LLM Compressor) | Balanced option; tests INT8 compute on Turing |
| `w4` | 4-bit weights, 16-bit activations (W4A16) | GPTQ (LLM Compressor) | Maximum memory savings |

**Requirements:**
- FR-1.0: **Hardware support check first.** Before the full quantization run, produce a tiny quantized checkpoint of each scheme and confirm vLLM loads and serves it on the T4. Record which vLLM kernel was selected (from the server log). If W4A16 is unsupported, pick the closest supported 4-bit path and document why.
- FR-1.1: `scripts/quantize.py --variant <id>` outputs a model directory. Calibration dataset, sample count, max sequence length, and seed live in `configs/quantization/<id>.yaml`.
- FR-1.2: For each variant, record on-disk size, GPU memory used by weights after loading, **number of KV cache blocks vLLM allocates** (quantized weights free memory for KV cache), and **perplexity** on a fixed evaluation set (fixed dataset split, fixed number of tokens).
- FR-1.3: Write results to `results/quantization/<variant>.json`.
- FR-1.4: Publish quantized models as private Kaggle Datasets (or private HF repos) and record their identifiers, so later sessions never re-quantize.

### 4.2 Serving with vLLM

- FR-2.1: Each variant has a vLLM config file `configs/serving/<variant>.yaml` (model path, `dtype`, `max-model-len`, `gpu-memory-utilization`, `max-num-seqs`, `seed`). `scripts/serve.py start <variant> --prefix-cache on|off` launches `vllm serve` from it in the background.
- FR-2.2: All settings not under test are **identical across variants**, in particular `gpu-memory-utilization` and `max-model-len`.
- FR-2.3: **Prefix caching is always set explicitly** (`--enable-prefix-caching` or `--no-enable-prefix-caching`), never left to the version default.
- FR-2.4: Serve through the OpenAI-compatible API. `scripts/smoke_test.py` sends one request, checks for a non-empty response, and exits non-zero otherwise.
- FR-2.5: Each benchmark run saves the exact launch command, resolved config, and `vllm --version` next to its results (`server_config.json`).
- FR-2.6: Pin the vLLM version in `requirements-gpu.txt` after confirming it works on the T4 (FR-1.0). Record the Kaggle image/CUDA version too.
- FR-2.7: `docker/compose.yml` defines the same server for a GPU VM, parameterized by `.env` (`MODEL_PATH`, `MAX_MODEL_LEN`, `GPU_MEMORY_UTILIZATION`, `PREFIX_CACHING`). It is validated in CI (`docker compose config`) but not run on Kaggle.

### 4.3 Benchmarking with GuideLLM

**Workloads** (each defined by a config file in `workloads/`):

| ID | Shape (starting point, tune to T4) | What it demonstrates |
|---|---|---|
| `chat_short` | ~128 prompt tokens, ~128 output tokens, high concurrency | Continuous batching, throughput scaling |
| `rag_shared_prefix` | ~2,000-token shared prefix + ~64-token unique question, ~128 output | Prefix caching benefit (run with cache on and off) |
| `long_context` | ~4,000 prompt tokens, ~256 output tokens | KV cache pressure, PagedAttention, memory limits |

**Requirements:**
- FR-3.1: For each (variant × workload), sweep request rate or concurrency from low load to saturation (at least 6 load points).
- FR-3.2: Capture TTFT, inter-token latency, end-to-end latency (p50/p95/p99), output tokens/s, requests/s, and error rate.
- FR-3.3: Run each configuration 3 times and report the median and spread (min–max or IQR).
- FR-3.4: Warm up before measuring (e.g. 20 requests discarded after every server start) and document the procedure.
- FR-3.5: Save raw output to `results/benchmarks/<variant>/<workload>[_cache-on|_cache-off]/run_<n>.json`.
- FR-3.6: Pin the GuideLLM version and check its docs for current CLI flags. GuideLLM 0.7's synthetic data supports `prefix_tokens` / `prefix_count`, so the shared-prefix workload needs no custom dataset.
- FR-3.7: The GuideLLM client runs in the same Kaggle session as the server (localhost), so tunnel latency never appears in benchmark numbers.

### 4.4 Quality Evaluation with lm-eval

- FR-4.1: Evaluate every variant on the same tasks: an MMLU subset (knowledge), ARC-Challenge (reasoning), HellaSwag (commonsense), and GSM8K (math).
- FR-4.2: Use identical few-shot settings, seeds, and `--limit` values across variants. Choose limits so one variant's evaluation fits in about 1 GPU-hour.
- FR-4.3: Report accuracy **delta vs. `base`** for each variant, with lm-eval's stderr.
- FR-4.4: Save to `results/eval/<variant>.json`.
- FR-4.5: Flag in the report any difference smaller than about 2× the combined stderr as "within noise".

### 4.5 Cost and Capacity Analysis

Kaggle is free, so costs use a **reference cloud price** for the same hardware.

- FR-5.1: Cost model:

  `cost_per_1M_output_tokens = (gpu_price_per_hour / (tokens_per_second × 3600)) × 1,000,000`

- FR-5.2: `configs/cost.yaml` stores the reference price, provider, region, and the date it was checked (e.g. an on-demand T4 instance on a major cloud).
- FR-5.3: Compute cost at a **latency target** (e.g. p95 TTFT ≤ 1 s for chat), not at maximum throughput. Interpolate between sweep points and state the method.
- FR-5.4: Report **max concurrency (or request rate) that meets the latency target** per variant and workload.

### 4.6 Analysis and Decision Report

`make report` runs on the laptop (no GPU) and generates from `results/`:
- A Pareto chart of accuracy delta vs. throughput (or cost), one point per variant.
- Latency-vs-load curves per workload.
- A prefix caching on/off comparison for the RAG workload.
- A summary table: size, weight memory, KV blocks, perplexity, accuracy delta, throughput at SLO, TTFT p95, cost per 1M tokens.
- `REPORT.md`, with a **recommendation per use case** (e.g. "For latency-sensitive chat, ship X; for high-volume batch jobs, ship Y; here's why").

### 4.7 Running on Kaggle (session management)

- FR-7.1: `notebooks/kaggle_runner.ipynb` is a thin wrapper that clones the repo, runs `scripts/kaggle_setup.sh` (vLLM in the system Python; LLM Compressor, GuideLLM and lm-eval in separate venvs because their pins clash with vLLM's), and calls `make` targets. All logic lives in `scripts/`, never in the notebook.
- FR-7.2: `scripts/run_matrix.py` is **resumable**: it skips any cell whose `run_<n>.json` already exists, so a session that dies mid-matrix loses at most one run.
- FR-7.3: Results are written to `/kaggle/working/results/` and zipped at the end of every stage for download. Committing them to git is a manual step on the laptop.
- FR-7.4: Keep a GPU-hour log in `docs/gpu-budget.md` (date, session, what ran, hours used).

---

## 5. Extras

### 5.1 Observability: Prometheus + Grafana

**Setup**
- Prometheus and Grafana run **on the laptop** via `observability/compose.yml`.
- During an observability session, the Kaggle notebook exposes vLLM through a `cloudflared` quick tunnel. The tunnel points at `scripts/metrics_proxy.py`, which serves `/metrics` and nothing else, so the OpenAI API never leaves localhost. Prometheus reads the tunnel host from `observability/targets/vllm.json`, which is never committed.
- Close the tunnel when the session ends.
- Grafana is **provisioned as code** (datasource + dashboard JSON in `observability/grafana/`), so `docker compose up` gives a working dashboard with no manual clicks.
- Observability runs are **separate** from the measured benchmark runs (FR-3.7), so scraping never affects the numbers.

**Dashboard panels** (verify metric names against the pinned vLLM version's `/metrics` output first):
- Requests running vs. waiting (queue depth)
- KV cache usage %
- Prefix cache hit rate
- TTFT and inter-token latency (p50/p95/p99)
- Tokens/s (prompt and generation)
- Request success/error rate

**Alert rules** (at least two must fire during a documented load test):

| Alert | Condition (example) | Why it matters |
|---|---|---|
| High TTFT | p95 TTFT above SLO for 2 min | Users feel slow starts |
| KV cache near full | KV cache usage > 90% for 2 min | Requests will queue or be preempted |
| Queue building | Waiting requests > N for 1 min | Server is saturated |
| Scrape down | `up == 0` for 1 min | Tunnel or server died (common with Kaggle) |

- `observability/alerts.yml` holds the rules. The README documents how each alert was triggered, with a screenshot.
- `docs/runbook.md`: for each alert, what it means and what to check first.

### 5.2 CI: GitHub Actions

**On every PR (`ci.yml`, CPU only):**
- Lint (ruff) and type-check (mypy) `scripts/` and `analysis/`.
- Validate all YAML configs against JSON schemas in `schemas/`.
- Run `make report` on `results/sample/` (a small checked-in subset) to confirm charts and the report still generate.
- Run `scripts/check_regression.py` against `baselines/baseline.json` with a sample that should pass and one that should fail, which tests the regression logic without a GPU.
- `promtool check config` / `promtool check rules` and `docker compose config` for both compose files.

**GPU regression (`gpu-benchmark.yml`, template):**
- A `workflow_dispatch` job that would run a small benchmark on a self-hosted GPU runner and fail if p95 latency worsens or throughput drops by more than 10%, or accuracy drops beyond noise.
- **No GPU runner exists for this project.** The workflow is kept as a template, and the same check is run manually on Kaggle output. The README says so plainly.

### 5.3 Lessons Learned

`docs/lessons-learned.md` (summarized in the README) answers:
- Where did prefix caching help a lot, and where did it barely matter?
- Which quantization scheme gave the best tradeoff on a T4, and where did accuracy visibly break?
- How did old-GPU kernel support change the plan?
- What surprised you (unexpected bottleneck, misleading metric, noisy result)?
- What would you test next with a newer GPU (FP8, Marlin kernels, speculative decoding)?

---

## 6. Repository Structure

```
llm-inference-lab/
├── README.md                 # Conclusion + key chart at the top
├── SPEC.md                   # This file
├── REPORT.md                 # Generated decision report
├── Makefile                  # make quantize | serve | smoke | bench | eval | report | lint
├── requirements-gpu.txt      # Kaggle system Python: vllm (pinned)
├── requirements-quant.txt    # Kaggle venv: llmcompressor
├── requirements-bench.txt    # Kaggle venv: guidellm
├── requirements-eval.txt     # Kaggle venv: lm-eval
├── requirements-dev.txt      # Laptop/CI: pandas, matplotlib, ruff, mypy, jsonschema (pinned)
├── configs/
│   ├── model.yaml            # base model id + pinned revision
│   ├── quantization/         # one config per variant
│   ├── serving/              # vLLM settings per variant
│   ├── eval.yaml
│   └── cost.yaml
├── schemas/                  # JSON schemas for every config type
├── workloads/                # GuideLLM workload definitions
├── scripts/
│   ├── quantize.py
│   ├── serve.py              # start/stop vLLM from a serving config
│   ├── measure_variant.py    # size, memory, KV cache, perplexity
│   ├── run_eval.py
│   ├── metrics_proxy.py
│   ├── check_metrics.py
│   ├── kaggle_setup.sh
│   ├── smoke_test.py
│   ├── run_matrix.py         # variant × workload × repeats, resumable
│   └── check_regression.py
├── notebooks/
│   └── kaggle_runner.ipynb   # thin wrapper, no logic
├── docker/
│   └── compose.yml           # vLLM on a GPU VM (template, validated in CI)
├── observability/
│   ├── compose.yml           # Prometheus + Grafana on the laptop
│   ├── prometheus.yml
│   ├── alerts.yml
│   └── grafana/              # provisioned datasource + dashboards
├── analysis/                 # scripts → charts + tables + REPORT.md
├── results/                  # raw JSON (quantization, benchmarks, eval) + sample/
├── baselines/baseline.json
├── docs/
│   ├── SPEC_v1.md
│   ├── gpu-budget.md
│   ├── runbook.md
│   └── lessons-learned.md
└── .github/workflows/        # ci.yml, gpu-benchmark.yml
```

---

## 7. Experiment Matrix

| | `chat_short` | `rag_shared_prefix` (cache off) | `rag_shared_prefix` (cache on) | `long_context` |
|---|---|---|---|---|
| `base` | ✔ | ✔ | ✔ | ✔ |
| `w8` | ✔ | ✔ | ✔ | ✔ |
| `w4` | ✔ | ✔ | ✔ | ✔ |

Every cell gets 3 runs and a load sweep. Each variant also gets one lm-eval run and one perplexity run.

**GPU-hour budget (estimate, T4):**

| Stage | Estimate |
|---|---|
| Setup + FR-1.0 kernel checks | 2 h |
| Quantization (w8, w4) + perplexity (×3) | 1.5 h |
| Benchmark matrix: 36 sweeps × ~10 min | 6 h |
| lm-eval (×3) | 3 h |
| Observability session(s) | 1 h |
| Debugging margin | 3 h |
| **Total** | **~16 h (≈ 1–2 weeks of free quota)** |

If the budget runs short, cut sweep points or `--limit` first, never the number of runs (FR-3.3).

---

## 8. Metric Definitions

| Metric | Definition |
|---|---|
| TTFT | Time from request sent to first token received |
| Inter-token latency | Average time between consecutive output tokens |
| E2E latency | Time from request sent to last token received |
| Throughput | Output tokens per second across all concurrent requests |
| KV cache usage | Fraction of allocated KV cache blocks in use |
| KV blocks | Number of KV cache blocks allocated at startup (capacity) |
| Prefix cache hit rate | Fraction of prompt tokens served from cache |
| Perplexity | Language-model quality proxy on a fixed text set (lower is better) |
| Accuracy delta | Task accuracy minus `base` accuracy |

---

## 9. Milestones

| # | Milestone | Where | Done when |
|---|---|---|---|
| 0 | Repo skeleton | Laptop | Git repo, folder structure, `.gitignore`, dev venv, first commit |
| 1 | Baseline serving | Kaggle | `base` served with pinned vLLM; smoke test passes; versions recorded |
| 2 | Quantization | Kaggle | FR-1.0 checks done; 3 variants built and stored; size, memory, KV blocks, perplexity recorded |
| 3 | Benchmark harness | Kaggle | One workload × one variant runs end to end; JSON saved and downloaded |
| 4 | Early analysis | Laptop | First latency-vs-load chart generated from milestone 3 output |
| 5 | Evaluation | Kaggle | lm-eval results for all variants |
| 6 | Full matrix | Kaggle | All workloads × variants × 3 runs complete |
| 7 | Analysis + report | Laptop | Charts and `REPORT.md` regenerate from raw data with `make report` |
| 8 | Observability | Both | Grafana dashboard live over tunnel; 2+ alerts fired and screenshotted |
| 9 | CI | GitHub | PR checks green; GPU workflow template documented |
| 10 | Polish | Laptop | README top section, lessons learned, runbook |

---

## 10. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Recent vLLM versions drop or break T4 (sm75) support | FR-1.0 first; pin the newest version that works and record it |
| 4-bit scheme unsupported on T4 | Test with a tiny checkpoint first; switch format and document |
| Kaggle session ends mid-run (12 h limit, idle timeout) | Resumable matrix (FR-7.2), zip results after each stage |
| Weekly GPU quota runs out | Budget table (§7), GPU-hour log, small model, reuse quantized checkpoints |
| Shared/virtualized cloud GPU adds noise | Warmup, 3 runs, medians + spread, same session per variant where possible |
| Metric names differ across vLLM versions | Pin the version; verify against `/metrics` before building dashboards |
| Tunnel exposes the server publicly | Tunnel only a `/metrics` proxy, only during observability sessions; URL never committed |
| Unfair comparisons | Same prompts, seeds, and server settings apart from the variable under test |
| Scope creep | Ship milestones 0–7 first; extras are a second pass |

---

## 11. README Outline

1. **Conclusion** (3–4 sentences) and the Pareto chart
2. Key results table
3. What this project is and why
4. Quickstart: reproduce the report without a GPU (`make report`), then optionally rerun on Kaggle
5. Platform and constraints (T4, $0 budget, what changed vs. a production GPU)
6. Methodology (variants, workloads, metrics, cost model)
7. Dashboard screenshot + alert demo
8. Lessons learned (short)
9. Limitations and future work (newer GPUs/FP8, multi-GPU, speculative decoding, other model families)
10. Reproducing results (pinned versions, hardware, dates)

---

## 12. Platform Decisions

**Hardware.** The project runs on a free Kaggle T4 because there is no budget for a paid GPU and my laptop has an Intel GPU, which vLLM and LLM Compressor don't support. The T4 is old (2018, Turing), so absolute numbers are modest. The comparison between variants is what matters.

**What the T4 allows** (checked on 2026-09-24 with vLLM 0.30.0, FR-1.0):

| Variant | Scheme | Kernel vLLM picked | Notes |
|---|---|---|---|
| `base` | FP16 | n/a | No BF16 on Turing |
| `w8` | INT8 W8A8 | `CutlassInt8ScaledMMLinearKernel` | FP8 would need Ada or Hopper |
| `w4` | W4A16 | `MarlinLinearKernel` | Works on the T4, which I didn't expect |

**What changes compared with a paid GPU VM:**

- No Docker on Kaggle. vLLM runs as a plain process started by `scripts/serve.py`. `docker/compose.yml` is the equivalent for a VM, validated in CI but not used.
- Four Python environments instead of one. llmcompressor 0.14.0 and vllm 0.30.0 pin incompatible versions of the same dependency, so quantization, benchmarking and evaluation each have their own venv.
- Prometheus and Grafana run on the laptop and reach Kaggle through a cloudflared tunnel that only exposes `/metrics`.
- Sessions stop after 12 h or when idle, and there are about 30 GPU hours a week. Everything that runs on the GPU is resumable and saves results after each step.
- Cost uses a reference cloud price for a T4 (`configs/cost.yaml`), since Kaggle itself is free.

---

## 13. Definition of Done

- All success criteria in Section 1 are checked.
- README conclusion matches the numbers in `REPORT.md`.
- A fresh clone can regenerate the report from saved results on a CPU-only machine, and reproduce at least one benchmark cell on Kaggle following the documented steps.
- Versions (vLLM, GuideLLM, LLM Compressor, lm-eval, model revision, Kaggle image/CUDA) are pinned and recorded.
