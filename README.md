# LLM Inference Tradeoff Lab

Which model configuration should we ship, and what does it cost?

This project takes one open-source model (Qwen2.5-1.5B-Instruct), builds three versions of it (FP16, INT8, 4-bit), serves each one with vLLM and compares them on speed, accuracy and cost under different traffic patterns. The goal is a recommendation per use case, not just a table of numbers.

**Status:** work in progress. There are no results yet.

## Stack

- Quantization: LLM Compressor
- Serving: vLLM
- Load testing: GuideLLM
- Accuracy: lm-eval
- Monitoring: Prometheus + Grafana

## Hardware

GPU work runs on a free Kaggle T4. Analysis, reports and monitoring run on a laptop with no NVIDIA GPU. The T4 is old, so absolute numbers will be modest. What matters here is how the variants compare to each other.

## Plan

See [SPEC.md](SPEC.md) for requirements, milestones and design decisions.
