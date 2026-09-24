"""Quantize the base model into one variant with LLM Compressor (FR-1.1).

    .venvs/quant/bin/python scripts/quantize.py --variant w8

Reads configs/model.yaml and configs/quantization/<variant>.yaml and writes the
compressed model to $MODELS_DIR/<variant> (default ./models/<variant>).
Runs in its own venv because llmcompressor and vllm pin incompatible packages.
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CONFIGS, load_yaml, models_dir  # noqa: E402


def calibration_data(cfg: dict, tokenizer):
    from datasets import load_dataset

    # Sample from the first 5k rows so we don't download the whole dataset.
    ds = load_dataset(cfg["calibration_dataset"], split="train_sft[:5000]")
    ds = ds.shuffle(seed=cfg["seed"]).select(range(cfg["num_samples"]))

    def tokenize(row):
        text = tokenizer.apply_chat_template(row["messages"], tokenize=False)
        return tokenizer(text, max_length=cfg["max_seq_len"], truncation=True, add_special_tokens=False)

    return ds.map(tokenize, remove_columns=ds.column_names)


def recipe(scheme: str) -> list:
    from llmcompressor.modifiers.quantization import GPTQModifier

    steps: list = [GPTQModifier(targets="Linear", scheme=scheme, ignore=["lm_head"])]
    if scheme == "W8A8":
        # INT8 activations have outliers; SmoothQuant moves them into the weights first.
        try:
            from llmcompressor.modifiers.smoothquant import SmoothQuantModifier
        except ImportError:
            from llmcompressor.modifiers.transform.smoothquant import SmoothQuantModifier
        steps.insert(0, SmoothQuantModifier(smoothing_strength=0.8))
    return steps


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True, choices=["w8", "w4"])
    args = parser.parse_args()

    import torch
    from llmcompressor import oneshot
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_cfg = load_yaml(CONFIGS / "model.yaml")
    cfg = load_yaml(CONFIGS / "quantization" / f"{args.variant}.yaml")
    out = models_dir() / args.variant
    torch.manual_seed(cfg["seed"])

    started = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["model_id"], revision=model_cfg["revision"], torch_dtype=torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["model_id"], revision=model_cfg["revision"])

    oneshot(
        model=model,
        dataset=calibration_data(cfg, tokenizer),
        recipe=recipe(cfg["scheme"]),
        max_seq_length=cfg["max_seq_len"],
        num_calibration_samples=cfg["num_samples"],
    )
    model.save_pretrained(out, save_compressed=True)
    tokenizer.save_pretrained(out)

    import llmcompressor

    meta = {
        "variant": args.variant,
        "base_model": model_cfg,
        "quantization": cfg,
        "llmcompressor_version": llmcompressor.__version__,
        "minutes": round((time.time() - started) / 60, 1),
    }
    (out / "quantization_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Saved {args.variant} to {out} in {meta['minutes']} min")


if __name__ == "__main__":
    main()
