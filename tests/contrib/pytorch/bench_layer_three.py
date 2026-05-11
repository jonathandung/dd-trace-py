"""Layer 3 (CUDA kernel profiling) overhead benchmark.

Excluded from default test runs by the ``benchmark`` marker; run explicitly:
    pytest -m benchmark tests/contrib/pytorch/bench_layer_three.py

This is a CPU-only sanity check — real perf review needs CUDA + the
profiler active-window schedule. The Layer 3 active-window budget per the
design is 5-15 %; we assert < 30 % as a coarse CI gate.
"""

import os
import time

import pytest
import torch

from ddtrace.contrib.internal.pytorch.patch import patch
from ddtrace.contrib.internal.pytorch.patch import unpatch


_ITERS = 200
_OVERHEAD_BUDGET = 0.30  # 30%


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
def test_layer_three_overhead_under_budget():
    """Layer 3 active-window overhead should stay under the 30 % gate even
    on a single-rank, CPU-only loop where the profiler does no useful work.
    """
    unpatch()
    os.environ.pop("DD_PYTORCH_PROFILING", None)
    os.environ.pop("DD_PYTORCH_KERNEL_PROFILING", None)
    baseline = _time_run()

    os.environ["DD_PYTORCH_PROFILING"] = "true"
    os.environ["DD_PYTORCH_KERNEL_PROFILING"] = "true"
    import importlib

    from ddtrace.contrib.internal.pytorch import _hooks
    from ddtrace.contrib.internal.pytorch import _profiler

    importlib.reload(_hooks)
    importlib.reload(_profiler)
    patch()
    try:
        traced = _time_run()
    finally:
        unpatch()
        del os.environ["DD_PYTORCH_PROFILING"]
        del os.environ["DD_PYTORCH_KERNEL_PROFILING"]
    overhead = (traced - baseline) / max(baseline, 1e-9)
    assert overhead < _OVERHEAD_BUDGET, "Layer 3 overhead %.2f%% exceeds %.0f%% budget" % (
        overhead * 100,
        _OVERHEAD_BUDGET * 100,
    )
