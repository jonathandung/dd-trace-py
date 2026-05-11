"""Layer 1 overhead benchmark for the PyTorch contrib integration.

Excluded from default test runs by the ``benchmark`` marker; run explicitly:
    pytest -m benchmark tests/contrib/pytorch/bench_layer1.py
"""

import time

import pytest
import torch

from ddtrace.contrib.internal.pytorch.patch import patch
from ddtrace.contrib.internal.pytorch.patch import unpatch


_ITERS = 200
_OVERHEAD_BUDGET = 0.02  # 2 %


def _run_step():
    params = [torch.nn.Parameter(torch.randn(64))]
    opt = torch.optim.SGD(params, lr=0.01)
    for _ in range(_ITERS):
        opt.zero_grad()
        loss = (params[0] ** 2).sum()
        loss.backward()
        opt.step()


def _time_run() -> float:
    start = time.perf_counter()
    _run_step()
    return time.perf_counter() - start


@pytest.mark.benchmark
def test_layer1_overhead_under_budget():
    """Assert that enabling Layer 1 instrumentation adds < 2% overhead on a
    simple single-rank optimizer-step loop.

    Layer 1 should be near-zero on a non-distributed, non-CUDA workload: the
    collective wrappers never fire, and the optimizer-step wrap is a pure
    pass-through.
    """
    unpatch()
    baseline = _time_run()
    patch()
    try:
        traced = _time_run()
    finally:
        unpatch()
    overhead = (traced - baseline) / max(baseline, 1e-9)
    assert overhead < _OVERHEAD_BUDGET, "Layer 1 overhead %.2f%% exceeds %.0f%% budget" % (
        overhead * 100,
        _OVERHEAD_BUDGET * 100,
    )
