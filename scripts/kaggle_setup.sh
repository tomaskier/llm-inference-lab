#!/usr/bin/env bash
# One-time setup for each Kaggle session (pip installs don't survive a restart).
# vLLM goes in the system Python; the other tools get their own venvs because
# their pinned dependencies clash with vLLM's.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== uv"
pip install -q uv

for env in quant bench eval; do
  echo "== venv $env"
  if [ ! -x ".venvs/$env/bin/python" ]; then
    uv venv -q ".venvs/$env" --python 3.12
  fi
  # wrapt: Kaggle's sitecustomize imports it in every interpreter.
  uv pip install -q --python ".venvs/$env/bin/python" -r "requirements-$env.txt" wrapt
done

echo "== vllm (system python)"
# Plain pip here: it's what worked in the first session, and pip tolerates the
# conflicts with Kaggle's preinstalled packages that we don't use.
pip install -q -r requirements-gpu.txt

echo "== versions"
python -c "import vllm; print('vllm', vllm.__version__)"
.venvs/quant/bin/python -c "import llmcompressor; print('llmcompressor', llmcompressor.__version__)"
.venvs/bench/bin/python -c "import guidellm; print('guidellm', guidellm.__version__)"
.venvs/eval/bin/python -c "import lm_eval; print('lm-eval', lm_eval.__version__)"
