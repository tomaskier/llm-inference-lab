# Runbook

What each alert means and what to look at first. Rules live in `observability/alerts.yml`, the dashboard is "vLLM on Kaggle T4" in Grafana.

## HighTTFT

p95 time to first token has been over 500 ms for 2 minutes.

Users wait too long for the first word. Almost always a load problem, not a model problem.

1. Look at **Requests running vs waiting**. If *waiting* is above zero, requests queue before they start. That's saturation; see QueueBuilding.
2. If nothing is waiting, look at **Tokens per second → prompt**. A burst of long prompts makes prefill expensive, and prefill is what TTFT measures.
3. On a RAG-style workload, check **Prefix cache hit rate**. If it dropped, the shared prefix is no longer being reused (cache turned off, or the prefix changed between requests).

Fixes, cheapest first: lower `max_num_seqs` so fewer requests share the GPU, switch to a variant with more KV room (`w4`), or add capacity.

## KVCacheNearFull

KV cache usage above 90% for 2 minutes.

Each running request keeps its attention keys and values in GPU memory. When the space runs out, vLLM preempts requests and recomputes them later, which hurts latency for everyone.

1. Check the prompt sizes. Long contexts (the `long_context` workload) fill the cache with only a few requests.
2. Compare with the KV cache size in `results/quantization/<variant>.json`. Quantized weights leave more room: `w4` holds more tokens than `base` at the same `gpu_memory_utilization`.

Fixes: cap concurrency, cap `max_model_len`, or serve a variant with smaller weights.

## QueueBuilding

More than 10 requests waiting for 1 minute.

The server takes requests faster than it can finish them. Throughput has hit its ceiling and latency will keep rising until the load drops.

1. Check **Tokens per second → generation**. If it is flat while waiting grows, the GPU is at its limit.
2. Check **KV cache usage**. If it's near full, the queue is caused by memory, not compute (see KVCacheNearFull).

Fixes: shed or rate-limit load, or add capacity. The report's "users at SLO" column tells you roughly where this starts for each variant.

## ErrorRateHigh

More than 1% of HTTP requests returned 5xx for 2 minutes.

1. Read the end of `build/vllm.log` on Kaggle. Out-of-memory errors and engine crashes show up there.
2. If every request fails, the engine is probably dead. Restart with `make stop && make serve`.

## ScrapeDown

Prometheus can't reach the metrics endpoint for 1 minute.

On this setup it usually means the Kaggle session stopped (idle timeout or the 12 h limit) or the cloudflared tunnel dropped.

1. Check the Kaggle session is still running.
2. Quick tunnels get a new URL every time they restart. Put the new host in `observability/targets/vllm.json`; Prometheus picks it up without a restart.

## Triggering the alerts on purpose

The observability section of `notebooks/kaggle_runner.ipynb` has a load test: 64 concurrent requests with 4,096-token prompts for 5 minutes. On a T4 that queues requests and fills the KV cache, which trips QueueBuilding, KVCacheNearFull and HighTTFT. ScrapeDown is easy: stop the tunnel.
