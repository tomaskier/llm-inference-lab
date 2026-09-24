"""Validate every YAML config against its JSON schema.

schemas/<name>.schema.json validates configs/<name>.yaml, or every
configs/<name>/*.yaml if <name> is a folder (e.g. quantization/).
The workload schema is the exception: it validates workloads/*.yaml.
"""

import json
import sys
from pathlib import Path

import yaml
from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parent.parent
SCHEMAS = ROOT / "schemas"
CONFIGS = ROOT / "configs"
ELSEWHERE = {"workload": ROOT / "workloads"}


def configs_for(name: str) -> list[Path]:
    if name in ELSEWHERE:
        return sorted(ELSEWHERE[name].glob("*.yaml"))
    single = CONFIGS / f"{name}.yaml"
    if single.exists():
        return [single]
    return sorted((CONFIGS / name).glob("*.yaml"))


def main() -> int:
    errors = 0
    for schema_path in sorted(SCHEMAS.glob("*.schema.json")):
        name = schema_path.name.removesuffix(".schema.json")
        validator = Draft202012Validator(json.loads(schema_path.read_text()))

        paths = configs_for(name)
        if not paths:
            print(f"FAIL {name}: no config found for this schema")
            errors += 1

        for path in paths:
            config = yaml.safe_load(path.read_text())
            problems = list(validator.iter_errors(config))
            rel = path.relative_to(ROOT).as_posix()
            if problems:
                errors += 1
                for p in problems:
                    field = ".".join(str(k) for k in p.path) or "(root)"
                    print(f"FAIL {rel} [{field}]: {p.message}")
            else:
                print(f"OK   {rel}")

    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
