# GPU targets run on Kaggle (Linux). Laptop targets run anywhere with the dev venv.
# On Windows without make, run the python commands shown next to each target.

PY ?= python
VARIANT ?= base
CACHE ?= off
QUANT_PY := .venvs/quant/bin/python

.PHONY: setup quantize serve stop smoke measure bench eval report sample check lint test validate

# ---- Kaggle (GPU)

setup:                      ## install vLLM + the quant/bench/eval venvs
	bash scripts/kaggle_setup.sh

quantize:                   ## make quantize VARIANT=w8
	$(QUANT_PY) scripts/quantize.py --variant $(VARIANT)

serve:                      ## make serve VARIANT=w4 CACHE=on
	$(PY) scripts/serve.py start $(VARIANT) --prefix-cache $(CACHE)

stop:
	$(PY) scripts/serve.py stop

smoke:
	$(PY) scripts/smoke_test.py

measure:                    ## size, memory, perplexity for the served variant
	$(PY) scripts/measure_variant.py --variant $(VARIANT)

bench:                      ## full matrix, resumable
	$(PY) scripts/run_matrix.py

eval:                       ## lm-eval for the served variant
	$(PY) scripts/run_eval.py --variant $(VARIANT)

# ---- Laptop / CI (no GPU)

report:                     ## python -m analysis.report
	$(PY) -m analysis.report

sample:                     ## python -m analysis.make_sample
	$(PY) -m analysis.make_sample

validate:
	$(PY) scripts/validate_configs.py

lint:
	ruff check .
	ruff format --check .
	mypy

test:
	pytest -q

check: validate lint test   ## everything CI runs, minus Docker
	$(PY) -m analysis.report --results results/sample --out build/sample
	$(PY) scripts/check_regression.py --results results/sample
