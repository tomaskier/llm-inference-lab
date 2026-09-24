"""Record size, memory and perplexity for the variant that is being served (FR-1.2).

    python scripts/serve.py start w4 --prefix-cache off
    python scripts/measure_variant.py --variant w4

Perplexity is measured through the running server (prompt logprobs via
/v1/completions with echo), so it uses the same kernels as the benchmarks.
Writes results/quantization/<variant>.json.
"""

import argparse
import json
import math
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RESULTS, ROOT, resolve_model, serving_config  # noqa: E402

PPL_DATASET = ("wikitext", "wikitext-2-raw-v1", "test")
PPL_CHUNKS = 40
PPL_CHUNK_CHARS = 6000  # about 1.4k tokens each


def text_chunks() -> list[str]:
    from datasets import load_dataset

    name, config, split = PPL_DATASET
    text = "".join(load_dataset(name, config, split=split)["text"])
    return [text[i * PPL_CHUNK_CHARS : (i + 1) * PPL_CHUNK_CHARS] for i in range(PPL_CHUNKS)]


def prompt_logprobs(base_url: str, model: str, text: str) -> list[float]:
    body = {"model": model, "prompt": text, "max_tokens": 1, "echo": True, "logprobs": 0}
    req = urllib.request.Request(
        f"{base_url}/v1/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        reply = json.load(resp)
    n_prompt = reply["usage"]["prompt_tokens"]
    values = reply["choices"][0]["logprobs"]["token_logprobs"][:n_prompt]
    return [v for v in values[1:] if v is not None]  # the first token has no context


def perplexity(base_url: str, model: str) -> dict:
    logprobs: list[float] = []
    for chunk in text_chunks():
        logprobs += prompt_logprobs(base_url, model, chunk)
    return {
        "perplexity": round(math.exp(-sum(logprobs) / len(logprobs)), 4),
        "tokens": len(logprobs),
        "dataset": "/".join(PPL_DATASET),
        "chunks": PPL_CHUNKS,
        "chunk_chars": PPL_CHUNK_CHARS,
    }


def disk_size_gib(model: str, revision: str | None) -> float:
    path = Path(model)
    if path.exists():
        size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    else:  # HF id: add up the weight files of the pinned revision
        from huggingface_hub import HfApi

        info = HfApi().model_info(model, revision=revision, files_metadata=True)
        size = sum(s.size or 0 for s in info.siblings or [] if s.rfilename.endswith(".safetensors"))
    return round(size / 2**30, 3)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--url", default="http://localhost:8000")
    args = parser.parse_args()

    server = json.loads((ROOT / "build" / "server_config.json").read_text())
    if server["variant"] != args.variant:
        sys.exit(f"The running server is {server['variant']}, not {args.variant}.")

    cfg = serving_config(args.variant)
    model = resolve_model(cfg["model"])
    result = {
        "variant": args.variant,
        "disk_gib": disk_size_gib(model, cfg.get("revision")),
        **{k: server[k] for k in server if k not in ("command", "serving_config", "variant")},
        **perplexity(args.url.rstrip("/"), model),
    }

    out = RESULTS / "quantization" / f"{args.variant}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
