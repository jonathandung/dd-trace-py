"""Layer 2 (step-level profiling) overhead benchmark.

Excluded from default test runs by the ``benchmark`` marker; run explicitly:
    pytest -m benchmark tests/contrib/pytorch/bench_layer_two.py
"""

import os
import time

import pytest
import torch

from ddtrace.contrib.internal.pytorch.patch import patch
from ddtrace.contrib.internal.pytorch.patch import unpatch


_ITERS = 200
_OVERHEAD_BUDGET_MS_PER_STEP = 1.0  # design target ~0.3 ms; 1 ms gate for CI noise


def _run_step_loop():
    params = [torch.nn.Parameter(torch.randn(64))]
    opt = torch.optim.AdamW(params, lr=1e-3)
    for _ in range(_ITERS):
        opt.zero_grad()
        loss = (params[0] ** 2).sum()
        loss.backward()
        opt.step()


def _time_run() -> float:
    start = time.perf_counter()
    _run_step_loop()
    return time.perf_counter() - start


@pytest.mark.benchmark
def test_layer_two_overhead_under_budget():
    """Assert that enabling Layer 2 adds < 1 ms / step on a single-rank,
    CPU-only optimizer-step loop. The design target is ~0.3 ms; the 1 ms
    gate absorbs CI noise.
    """
    # Baseline: integration disabled.
    unpatch()
    os.environ.pop("DD_PYTORCH_PROFILING", None)
    baseline = _time_run()

    # Traced: integration enabled with Layer 2.
    os.environ["DD_PYTORCH_PROFILING"] = "true"
    import importlib

    from ddtrace.contrib.internal.pytorch import _hooks

    importlib.reload(_hooks)
    patch()
    try:
        traced = _time_run()
    finally:
        unpatch()
        del os.environ["DD_PYTORCH_PROFILING"]
    per_step_ms = (traced - baseline) / _ITERS * 1000.0
    assert per_step_ms < _OVERHEAD_BUDGET_MS_PER_STEP, "Layer 2 overhead %.3f ms / step exceeds %.1f ms budget" % (
        per_step_ms,
        _OVERHEAD_BUDGET_MS_PER_STEP,
    )
