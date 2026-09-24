"""Tests for the parts that decide what we recommend: capacity, cost, regressions."""

import copy
import math
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from check_regression import compare, current_numbers  # noqa: E402

from analysis.report import accuracy_table, capacity, cost_per_million  # noqa: E402

SAMPLE = ROOT / "results" / "sample"


def curve(ttft: list[float], tput: list[float], errors: list[float] | None = None) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "concurrency": [1, 2, 4, 8][: len(ttft)],
            "ttft_p95_ms": ttft,
            "output_tok_per_s": tput,
            "error_rate": errors or [0.0] * len(ttft),
        }
    )


def test_capacity_interpolates_between_points():
    c = capacity(curve([100, 200, 400, 800], [10, 20, 40, 60]), slo_ms=600)
    # 600 ms is halfway between 400 (c=4) and 800 (c=8)
    assert c.concurrency == pytest.approx(6)
    assert c.tput == pytest.approx(50)
    assert not c.capped


def test_capacity_capped_when_slo_always_met():
    c = capacity(curve([100, 110, 120, 130], [10, 20, 40, 60]), slo_ms=500)
    assert c.concurrency == 8 and c.capped


def test_capacity_zero_when_slo_never_met():
    c = capacity(curve([900, 1000, 1100, 1200], [10, 20, 40, 60]), slo_ms=500)
    assert c.concurrency == 0 and math.isnan(c.tput)


def test_errors_count_as_failing():
    c = capacity(curve([100, 110, 120, 130], [10, 20, 40, 60], [0, 0, 0.05, 0]), slo_ms=500)
    assert c.concurrency < 4


def test_cost_formula():
    # $3.60/h at 1000 tok/s -> 3.6 / 3.6e6 * 1e6 = $1 per million tokens
    assert cost_per_million(1000, 3.60) == pytest.approx(1.0)
    assert math.isnan(cost_per_million(0, 3.60))


def test_accuracy_noise_flag():
    ev = pd.DataFrame(
        [
            {"variant": "base", "task": "t", "value": 0.60, "stderr": 0.02},
            {"variant": "w8", "task": "t", "value": 0.59, "stderr": 0.02},
            {"variant": "w4", "task": "t", "value": 0.45, "stderr": 0.02},
        ]
    )
    acc = accuracy_table(ev).set_index("variant")
    assert acc.loc["w8", "within_noise"]
    assert not acc.loc["w4", "within_noise"]
    assert acc.loc["w4", "delta"] == pytest.approx(-15)


@pytest.fixture
def sample_numbers():
    if not SAMPLE.exists():
        pytest.skip("run `python -m analysis.make_sample` first")
    return current_numbers(SAMPLE)


def baseline_from(numbers: dict) -> dict:
    return {
        "tolerance": {"latency": 0.10, "throughput": 0.10, "accuracy_points": 2.0},
        **copy.deepcopy(numbers),
    }


def test_regression_check_passes_on_same_data(sample_numbers):
    assert compare(baseline_from(sample_numbers), sample_numbers) == []


def test_regression_check_catches_slower_run(sample_numbers):
    worse = copy.deepcopy(sample_numbers)
    worse["cells"][0]["ttft_p95_ms"] *= 1.25
    worse["cells"][1]["output_tok_per_s"] *= 0.8
    failures = compare(baseline_from(sample_numbers), worse)
    assert len(failures) == 2


def test_regression_check_ignores_small_noise(sample_numbers):
    noisy = copy.deepcopy(sample_numbers)
    for c in noisy["cells"]:
        c["ttft_p95_ms"] *= 1.05
    assert compare(baseline_from(sample_numbers), noisy) == []
