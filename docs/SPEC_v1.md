# LLM Inference Tradeoff Lab: Project Spec

**Status:** Draft v1
**Owner:** _your name_
**One-line pitch:** A reproducible optimize → deploy → benchmark pipeline that answers: *"Which model configuration should we ship, and what does it cost?"*

---

## 1. Goals and Non-Goals

### Goals
- Run the full workflow from the course (quantize with LLM Compressor, serve with vLLM, benchmark with GuideLLM, evaluate with lm-eval) on a real open-source model.
- Compare multiple model variants under **realistic traffic shapes**, not just one synthetic test.
- Produce a **decision**, not just numbers: a per-use-case recommendation backed by an accuracy-vs-speed-vs-cost analysis.
- Make everything **reproducible**: one command per stage, configs in version control, results saved as files so reviewers don't need a GPU.
- Include production-style extras: live observability (Prometheus + Grafana with alerts), CI regression checks, and a written lessons-learned section.

### Non-Goals
- Training or fine-tuning models.
- Multi-node / multi-GPU tensor-parallel serving (may be listed as future work).
- Building a chat UI or product on top of the model.
- Beating published benchmarks. The goal is a sound methodology, not a leaderboard position.

### Success Criteria
- [ ] A reviewer can read the README top section and understand the conclusion in under 2 minutes.
- [ ] Every chart in the report can be regenerated from raw result files with one command.
- [ ] At least 3 model variants and 3 workloads are benchmarked, with accuracy measured for every variant.
- [ ] The Grafana dashboard shows live vLLM metrics during a load test, with at least 2 working alert rules.
- [ ] CI runs on every PR and fails on config errors or benchmark regressions.

---

## 2. Target Audience and Skills Demonstrated

| Skill | Where it shows up |
|---|---|
| Model compression (quantization) | Section 4.1 |
| LLM serving internals (continuous batching, PagedAttention, prefix caching) | Sections 4.2, 4.3 |
| Performance benchmarking under load | Section 4.3 |
| Model quality evaluation | Section 4.4 |
| Cost and capacity analysis | Section 4.5 |
| Observability and alerting | Section 5.1 |
| CI/CD and reproducibility | Section 5.2 |
| Technical communication | Sections 4.6, 5.3 |

Relevant roles: LLM/ML engineer, MLOps/platform engineer, inference/AI infrastructure engineer, solutions engineer.

---

## 3. Architecture

```
                ┌──────────────────────┐
                │  Base model (HF Hub) │
                └──────────┬───────────┘
                           │  LLM Compressor
          ┌────────────────┼────────────────┐
          ▼                ▼                ▼
      FP16 base        INT8/FP8          4-bit weights
          │                │                │
          └───────┬────────┴────────┬───────┘
                  ▼   (one at a time, via Docker Compose)
          ┌───────────────┐   /metrics    ┌────────────┐   ┌─────────┐
          │  vLLM server  │──────────────▶│ Prometheus │──▶│ Grafana │
          │ (OpenAI API)  │               └────────────┘   └─────────┘
          └───────┬───────┘
        ┌─────────┴──────────┐
        ▼                    ▼
   GuideLLM load test     lm-eval quality run
   (3 workloads)          (accuracy tasks)
        │                    │
        └─────────┬──────────┘
                  ▼
        results/ (raw JSON, versioned)
                  ▼
        analysis/ → charts + tables + REPORT.md
```

---

## 4. Core Requirements

### 4.1 Model Optimization

**Inputs:** one open-source base model (e.g. a Qwen model in the 0.5B–7B range, sized to your GPU).

**Variants to produce:**

| ID | Description | Purpose |
|---|---|---|
| `base` | Full-precision baseline (FP16/BF16) | Reference for speed and accuracy |
| `w8` | 8-bit quantization (INT8 or FP8, depending on GPU support) | Balanced option |
| `w4` | 4-bit weight quantization | Maximum memory savings |

**Requirements:**
- FR-1.1: A script (`scripts/quantize.py`) takes a variant ID and outputs a model directory. Calibration dataset, sample count, and seed are set in a config file.
- FR-1.2: For each variant, record on-disk size, GPU memory footprint after loading, and **perplexity** on a fixed evaluation set.
- FR-1.3: Write results to `results/quantization/<variant>.json`.
- FR-1.4: Document which quantization schemes your GPU supports and why you chose them.

### 4.2 Serving with vLLM

- FR-2.1: One Docker Compose service definition per variant, parameterized by a `.env` file (`MODEL_PATH`, `MAX_MODEL_LEN`, `GPU_MEMORY_UTILIZATION`, etc.).
- FR-2.2: Serve through the OpenAI-compatible API. Include a `scripts/smoke_test.py` that sends one request and verifies a response.
- FR-2.3: **Prefix caching must be an explicit, recorded setting** so on/off comparisons are unambiguous (check your vLLM version's default and set it explicitly).
- FR-2.4: Each benchmark run saves the exact server launch config alongside its results.
- FR-2.5: Pin the vLLM version and container image tag in the repo.

### 4.3 Benchmarking with GuideLLM

**Workloads** (each defined by a config file in `workloads/`):

| ID | Shape | What it demonstrates |
|---|---|---|
| `chat_short` | Short prompts, short outputs, many concurrent users | Continuous batching, throughput scaling |
| `rag_shared_prefix` | Long shared system prompt/context + short unique question | Prefix caching benefit (run with cache on and off) |
| `long_context` | Long prompts, moderate outputs | KV cache pressure, PagedAttention, memory limits |

**Requirements:**
- FR-3.1: For each (variant × workload), sweep request rate or concurrency from low load to saturation.
- FR-3.2: Capture: TTFT, inter-token latency, end-to-end latency (p50/p95/p99), output tokens/s, requests/s, and error rate.
- FR-3.3: Run each configuration at least 3 times and report the median and spread.
- FR-3.4: Warm up the server before measuring and document the warmup procedure.
- FR-3.5: Save raw output to `results/benchmarks/<variant>/<workload>/run_<n>.json`.
- FR-3.6: Check the GuideLLM docs for current CLI flags and pin its version.

### 4.4 Quality Evaluation with lm-eval

- FR-4.1: Evaluate every variant on the same task set (choose 3–4 tasks covering knowledge, reasoning, and math, e.g. an MMLU subset, ARC, HellaSwag, GSM8K).
- FR-4.2: Use identical few-shot settings, seeds, and sample limits across variants.
- FR-4.3: Report accuracy **delta vs. `base`** for each variant, with the stderr lm-eval provides.
- FR-4.4: Save to `results/eval/<variant>.json`.
- FR-4.5: Note in the report when a difference is within noise.

### 4.5 Cost and Capacity Analysis

- FR-5.1: Define the cost model in the report:

  `cost_per_1M_output_tokens = (gpu_price_per_hour / (tokens_per_second × 3600)) × 1,000,000`

- FR-5.2: Compute cost at a fixed latency target (e.g. p95 TTFT under a threshold you choose), not at maximum throughput, since max throughput often violates any real SLO.
- FR-5.3: Report **max concurrent users at the latency target** per variant and workload.
- FR-5.4: State your GPU type and hourly price and the date you priced it.

### 4.6 Analysis and Decision Report

`analysis/` generates the following from raw results:
- Pareto chart: accuracy delta vs. throughput (or cost), one point per variant.
- Latency-vs-load curves per workload.
- Prefix caching on/off comparison for the RAG workload.
- Summary table: size, memory, perplexity, accuracy delta, throughput, TTFT p95, cost per 1M tokens.
- `REPORT.md` with a **recommendation per use case**, for example: "For latency-sensitive chat, ship X; for high-volume batch summarization, ship Y; here's why."

---

## 5. Extras

### 5.1 Observability: Prometheus + Grafana

**Setup**
- Prometheus scrapes the vLLM `/metrics` endpoint. Config lives in `observability/prometheus.yml`.
- Grafana is **provisioned as code** (datasource and dashboard JSON committed to the repo) so `docker compose up` gives a working dashboard with no manual clicks.

**Dashboard panels** (verify exact metric names against your vLLM version's `/metrics` output, since they change between versions):
- Requests running vs. waiting (queue depth)
- KV cache usage %
- Prefix cache hit rate
- TTFT and inter-token latency (p50/p95/p99)
- Tokens/s (prompt and generation)
- Request success/error rate
- GPU memory utilization (optional: add a GPU exporter such as DCGM or nvidia-smi based)

**Alert rules** (at least two are required):

| Alert | Condition (example) | Why it matters |
|---|---|---|
| High TTFT | p95 TTFT above your SLO for 5 min | Users feel slow starts |
| KV cache near full | KV cache usage above ~90% for 5 min | Requests will queue or be preempted |
| Queue building | Waiting requests above N for 2 min | Server is saturated |
| Error spike | Error rate above 1% for 5 min | Reliability |

- Include the alert rule file in `observability/alerts.yml`, and document how you triggered each one during a load test (screenshot in the README).
- Add a short runbook (`docs/runbook.md`): for each alert, what it means and what to check first.

### 5.2 CI: GitHub Actions

**On every PR (no GPU needed):**
- Lint (ruff) and type-check the scripts.
- Validate all config files against a schema.
- Run analysis on a small **checked-in sample of results** to confirm charts and the report still generate.
- Validate Prometheus config and alert rules (`promtool check`).

**Manual / scheduled (GPU needed, workflow_dispatch):**
- Run a small benchmark against a self-hosted or rented GPU runner.
- Compare against `baselines/baseline.json` and **fail if** p95 latency worsens or throughput drops beyond a set threshold (e.g. 10%), or accuracy drops beyond noise.
- Upload result artifacts.

If you don't have GPU CI, document that clearly and keep the manual job as a template. Being honest here reads better than faking it.

### 5.3 Lessons Learned

A `docs/lessons-learned.md` (also summarized in the README) answering:
- Where did prefix caching help a lot, and where did it barely matter?
- Which quantization scheme gave the best tradeoff, and where did accuracy visibly break?
- What surprised you (unexpected bottleneck, misleading metric, noisy result)?
- What would you do differently or test next?

---

## 6. Repository Structure

```
llm-inference-lab/
├── README.md                 # Conclusion + key chart at the top
├── SPEC.md                   # This file
├── REPORT.md                 # Generated decision report
├── Makefile                  # make quantize | serve | bench | eval | report
├── configs/
│   ├── quantization/         # one config per variant
│   ├── serving/              # vLLM settings per variant
│   └── eval.yaml
├── workloads/                # GuideLLM workload definitions
├── scripts/
│   ├── quantize.py
│   ├── smoke_test.py
│   └── run_matrix.sh         # runs variant × workload × repeats
├── docker/
│   └── compose.yml           # vLLM + Prometheus + Grafana
├── observability/
│   ├── prometheus.yml
│   ├── alerts.yml
│   └── grafana/              # provisioned datasource + dashboards
├── analysis/                 # notebooks/scripts → charts + tables
├── results/                  # raw JSON (quantization, benchmarks, eval)
├── baselines/baseline.json
├── docs/
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

Every cell: 3 runs, load sweep, plus lm-eval per variant (once, not per workload).

---

## 8. Metric Definitions

| Metric | Definition |
|---|---|
| TTFT | Time from request sent to first token received |
| Inter-token latency | Average time between consecutive output tokens |
| E2E latency | Time from request sent to last token received |
| Throughput | Output tokens per second across all concurrent requests |
| KV cache usage | Fraction of allocated KV cache blocks in use |
| Prefix cache hit rate | Fraction of prompt tokens served from cache |
| Perplexity | Language-model quality proxy on a fixed text set (lower is better) |
| Accuracy delta | Task accuracy minus `base` accuracy |

---

## 9. Milestones

| # | Milestone | Done when |
|---|---|---|
| 1 | Repo skeleton + baseline serving | `base` model served, smoke test passes |
| 2 | Quantization | 3 variants built, size and perplexity recorded |
| 3 | Benchmark harness | One workload runs end to end with saved JSON |
| 4 | Full matrix | All workloads × variants × 3 runs complete |
| 5 | Evaluation | lm-eval results for all variants |
| 6 | Analysis + report | Charts and `REPORT.md` regenerate from raw data |
| 7 | Observability | Grafana dashboard + 2 alerts triggered and screenshotted |
| 8 | CI | PR checks green; GPU workflow documented |
| 9 | Polish | README top section, lessons learned, runbook |

Suggested order: 1 → 2 → 3 → 5 → 4 → 6 (get a thin end-to-end result early), then 7–9.

---

## 10. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| GPU cost or availability | Use a small model; rent GPU by the hour (Colab, RunPod, Modal); save results as files so work isn't repeated |
| Quantization scheme unsupported on your GPU | Check hardware support first; swap the variant and document why |
| Noisy benchmark results | Warmup, 3+ runs, report medians and spread, fix seeds |
| Metric names differ across vLLM versions | Pin the version; verify against `/metrics` before building dashboards |
| Scope creep | Ship milestones 1–6 first; treat extras as a second pass |
| Unfair comparisons | Same prompts, same seeds, same server settings apart from the variable under test |

---

## 11. README Outline

1. **Conclusion** (3–4 sentences) and the Pareto chart
2. Key results table
3. What this project is and why
4. Quickstart (`make` commands, hardware needed)
5. Methodology (variants, workloads, metrics, cost model)
6. Dashboard screenshot + alert demo
7. Lessons learned (short)
8. Limitations and future work (multi-GPU, speculative decoding, other model families)
9. Reproducing results (versions pinned, hardware, dates)

---

## 12. Definition of Done

- All success criteria in Section 1 are checked.
- README conclusion matches the numbers in `REPORT.md`.
- A fresh clone plus documented setup can reproduce at least one benchmark cell and regenerate the report from saved results.
- Versions (vLLM, GuideLLM, LLM Compressor, lm-eval, model revision) are pinned and recorded.
