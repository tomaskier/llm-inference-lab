#!/usr/bin/env bash
# One-time setup for each Kaggle session (pip installs don't survive a restart).
# vLLM goes in the system Python; the other tools get their own venvs because
# their pinned dependencies clash with vLLM's.
set -euo pipefail
cd "$(dirname "$0")/.."

pip install -q uv
uv pip install -q --system -r requirements-gpu.txt

for env in quant bench eval; do
  if [ ! -x ".venvs/$env/bin/python" ]; then
    uv venv -q ".venvs/$env" --python 3.12
  fi
  # wrapt: Kaggle's sitecustomize imports it in every interpreter.
  uv pip install -q --python ".venvs/$env/bin/python" -r "requirements-$env.txt" wrapt
done

python -c "import vllm; print('vllm', vllm.__version__)"
.venvs/quant/bin/python -c "import llmcompressor; print('llmcompressor', llmcompressor.__version__)"
.venvs/bench/bin/guidellm --version || true
.venvs/eval/bin/lm_eval --help > /dev/null && echo "lm-eval ok"
