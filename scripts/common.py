"""Small helpers shared by the scripts: paths and config loading."""

import os
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIGS = ROOT / "configs"
WORKLOADS = ROOT / "workloads"
RESULTS = ROOT / "results"

VARIANTS = ["base", "w8", "w4"]


def load_yaml(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} is not a YAML mapping")
    return data


def models_dir() -> Path:
    # Kaggle keeps quantized models in an input dataset, so the location varies.
    return Path(os.environ.get("MODELS_DIR", ROOT / "models"))


def resolve_model(model: str) -> str:
    """Map 'models/<name>' to $MODELS_DIR/<name>; leave HF ids untouched."""
    if model.startswith("models/"):
        return str(models_dir() / model.removeprefix("models/"))
    return model


def serving_config(variant: str) -> dict[str, Any]:
    return load_yaml(CONFIGS / "serving" / f"{variant}.yaml")


def workload_config(name: str) -> dict[str, Any]:
    return load_yaml(WORKLOADS / f"{name}.yaml")


def workload_names() -> list[str]:
    return sorted(p.stem for p in WORKLOADS.glob("*.yaml"))


def cell_name(workload: str, prefix_caching: bool, options: list[bool]) -> str:
    """Folder name for one matrix cell. Only add the cache suffix when the
    workload is run both ways, so single-setting workloads keep a short name."""
    if len(options) > 1:
        return f"{workload}_cache-{'on' if prefix_caching else 'off'}"
    return workload
