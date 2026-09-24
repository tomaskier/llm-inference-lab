# Lessons learned

Notes as the project goes. The first part is what already came up while setting things up; the questions at the end get answered once the full results are in.

## Setup and tooling

- **Check hardware support, don't assume it.** I expected the fast 4-bit kernels (Marlin) to need a newer GPU than the T4. vLLM 0.30 picked `MarlinLinearKernel` for W4A16 and `CutlassInt8ScaledMMLinearKernel` for INT8 W8A8 on the T4 without complaints. A 10-minute test with a tiny model settled it.
- **Defaults change under you.** vLLM 0.30 turns prefix caching on by default. A "cache off" run that doesn't pass `--no-enable-prefix-caching` would have been a "cache on" run with the wrong label.
- **The first request of each shape is slow.** vLLM compiles some Triton kernels during the first requests (`kernel_unified_attention`). Without a warmup in the same shape as the benchmark, that shows up as a fake latency spike in the lowest-load point.
- **The tools don't share an environment.** llmcompressor 0.14 and vllm 0.30 pin incompatible versions of the same package. Quantization, benchmarking and evaluation each get their own venv, which is closer to how it would run in production anyway.
- **Quantized models shrink less than the bit count suggests.** On Qwen2.5-0.5B, W4A16 only cut the size by 55%, not 75%. The embedding table (~150k tokens) and `lm_head` stay in FP16, and in a small model they are a big part of the total.
- **Kaggle quirks.** `!command &` is refused, so the server is started from Python. Installed packages disappear when the session stops.

## Questions to answer with the results

- Where did prefix caching help a lot, and where did it barely matter?
- Which quantization scheme gave the best tradeoff on a T4, and where did accuracy visibly break?
- What surprised me: an unexpected bottleneck, a misleading metric, a noisy result?
- What would I test next with a newer GPU (FP8, speculative decoding, a bigger model)?
