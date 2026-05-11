import time
from unittest import mock

import torch

from ddtrace.contrib.internal.pytorch._distributed import _broadcast_job_id_with_timeout
from ddtrace.contrib.internal.pytorch._utils import _instrumentation_bypass
from ddtrace.contrib.internal.pytorch.patch import patch
from ddtrace.contrib.internal.pytorch.patch import unpatch


def test_init_process_group_triggers_bootstrap_once(monkeypatch):
    bootstrap = mock.Mock()
    monkeypatch.setattr(
        "ddtrace.contrib.internal.pytorch._distributed._bootstrap_distributed",
        bootstrap,
    )
    fake_init = mock.Mock()
    monkeypatch.setattr(torch.distributed, "init_process_group", fake_init)
    patch()
    try:
        torch.distributed.init_process_group(backend="gloo")
        torch.distributed.init_process_group(backend="gloo")  # second call
        assert bootstrap.call_count == 1
        assert fake_init.call_count == 2
    finally:
        unpatch()


def test_broadcast_returns_local_id_on_timeout(monkeypatch):
    def hang(*args, **kwargs):
        time.sleep(60)

    monkeypatch.setattr(torch.distributed, "broadcast_object_list", hang)
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setenv("DD_PYTORCH_JOB_ID_BROADCAST_TIMEOUT", "0.1")
    out = _broadcast_job_id_with_timeout("local-uuid", rank=0)
    assert out == "local-uuid"


def test_late_patch_runs_bootstrap_when_distributed_already_initialized(monkeypatch):
    """Codex P2 regression guard: if patch() runs after init_process_group,
    we still bootstrap (resolve job_id, capture rank/world_size, start the
    CUDA event resolver).
    """
    bootstrap = mock.Mock()
    monkeypatch.setattr(
        "ddtrace.contrib.internal.pytorch._distributed._bootstrap_distributed",
        bootstrap,
    )
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "init_process_group", mock.Mock())
    patch()
    try:
        assert bootstrap.call_count == 1
    finally:
        unpatch()


def test_late_patch_skips_bootstrap_when_not_initialized(monkeypatch):
    bootstrap = mock.Mock()
    monkeypatch.setattr(
        "ddtrace.contrib.internal.pytorch._distributed._bootstrap_distributed",
        bootstrap,
    )
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    monkeypatch.setattr(torch.distributed, "init_process_group", mock.Mock())
    patch()
    try:
        # Not yet initialized -> wrapper will run bootstrap on first init call.
        assert bootstrap.call_count == 0
    finally:
        unpatch()


def test_broadcast_returns_id_when_successful(monkeypatch):
    def fast(obj_list, src=0):
        obj_list[0] = "broadcast-id"

    monkeypatch.setattr(torch.distributed, "broadcast_object_list", fast)
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    out = _broadcast_job_id_with_timeout("rank0-uuid", rank=1)
    assert out == "broadcast-id"


def test_broadcast_short_circuits_when_distributed_not_initialized(monkeypatch):
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    out = _broadcast_job_id_with_timeout("local-uuid", rank=0)
    assert out == "local-uuid"


def test_broadcast_returns_local_id_on_exception(monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("nccl stalled")

    monkeypatch.setattr(torch.distributed, "broadcast_object_list", boom)
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    out = _broadcast_job_id_with_timeout("local-uuid", rank=0)
    assert out == "local-uuid"


def test_all_reduce_emits_span(test_spans, monkeypatch):
    fake_all_reduce = mock.Mock()
    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)
    patch()
    try:
        tensor = mock.Mock(is_cuda=False, numel=lambda: 10, element_size=lambda: 4)
        torch.distributed.all_reduce(tensor)
        spans = test_spans.pop()
        names = [s.name for s in spans]
        assert "pytorch.allreduce" in names
        ar = next(s for s in spans if s.name == "pytorch.allreduce")
        assert ar.get_metric("bytes") == 40
        assert fake_all_reduce.called
    finally:
        unpatch()


def test_broadcast_emits_span(test_spans, monkeypatch):
    monkeypatch.setattr(torch.distributed, "broadcast", mock.Mock())
    patch()
    try:
        tensor = mock.Mock(is_cuda=False, numel=lambda: 5, element_size=lambda: 8)
        torch.distributed.broadcast(tensor, src=0)
        names = [s.name for s in test_spans.pop()]
        assert "pytorch.broadcast" in names
    finally:
        unpatch()


def test_barrier_emits_span_without_bytes(test_spans, monkeypatch):
    monkeypatch.setattr(torch.distributed, "barrier", mock.Mock())
    patch()
    try:
        torch.distributed.barrier()
        barriers = [s for s in test_spans.pop() if s.name == "pytorch.barrier"]
        assert len(barriers) == 1
        # No bytes metric for barrier (no tensor arg).
        assert barriers[0].get_metric("bytes") is None
    finally:
        unpatch()


def test_collective_bypass_skips_span(test_spans, monkeypatch):
    monkeypatch.setattr(torch.distributed, "all_reduce", mock.Mock())
    patch()
    try:
        with _instrumentation_bypass():
            tensor = mock.Mock(is_cuda=False, numel=lambda: 4, element_size=lambda: 4)
            torch.distributed.all_reduce(tensor)
        spans = test_spans.pop()
        assert all(s.name != "pytorch.allreduce" for s in spans)
    finally:
        unpatch()


def test_all_gather_into_tensor_emits_span(test_spans, monkeypatch):
    if not hasattr(torch.distributed, "all_gather_into_tensor"):
        import pytest

        pytest.skip("all_gather_into_tensor unavailable on this torch")
    monkeypatch.setattr(torch.distributed, "all_gather_into_tensor", mock.Mock())
    patch()
    try:
        out = mock.Mock(is_cuda=False, numel=lambda: 8, element_size=lambda: 4)
        inp = mock.Mock(is_cuda=False, numel=lambda: 2, element_size=lambda: 4)
        torch.distributed.all_gather_into_tensor(out, inp)
        spans = test_spans.pop()
        names = [s.name for s in spans]
        assert "pytorch.allgather_into_tensor" in names
        # bytes should come from the input tensor (index 1), i.e. 2 * 4 = 8.
        s = next(s for s in spans if s.name == "pytorch.allgather_into_tensor")
        assert s.get_metric("bytes") == 8
    finally:
        unpatch()


def test_reduce_scatter_sums_input_list_bytes(test_spans, monkeypatch):
    """Codex P2 regression guard: reduce_scatter must report the full
    input_list size, not just the output tensor.
    """
    monkeypatch.setattr(torch.distributed, "reduce_scatter", mock.Mock())
    patch()
    try:
        output = mock.Mock(is_cuda=False, numel=lambda: 4, element_size=lambda: 4)
        input_list = [
            mock.Mock(is_cuda=False, numel=lambda: 4, element_size=lambda: 4),
            mock.Mock(is_cuda=False, numel=lambda: 4, element_size=lambda: 4),
            mock.Mock(is_cuda=False, numel=lambda: 4, element_size=lambda: 4),
        ]
        torch.distributed.reduce_scatter(output, input_list)
        rs = next(s for s in test_spans.pop() if s.name == "pytorch.reducescatter")
        # 3 input tensors × 4 elements × 4 bytes = 48 (world-aggregated input).
        assert rs.get_metric("bytes") == 48
    finally:
        unpatch()


def test_reduce_scatter_tensor_emits_span(test_spans, monkeypatch):
    if not hasattr(torch.distributed, "reduce_scatter_tensor"):
        import pytest

        pytest.skip("reduce_scatter_tensor unavailable on this torch")
    monkeypatch.setattr(torch.distributed, "reduce_scatter_tensor", mock.Mock())
    patch()
    try:
        out = mock.Mock(is_cuda=False, numel=lambda: 2, element_size=lambda: 4)
        inp = mock.Mock(is_cuda=False, numel=lambda: 8, element_size=lambda: 4)
        torch.distributed.reduce_scatter_tensor(out, inp)
        spans = test_spans.pop()
        names = [s.name for s in spans]
        assert "pytorch.reducescatter_tensor" in names
        # bytes from input tensor (index 1): 8 * 4 = 32.
        s = next(s for s in spans if s.name == "pytorch.reducescatter_tensor")
        assert s.get_metric("bytes") == 32
    finally:
        unpatch()


def test_fsdp_collectives_skip_cleanly_when_attribute_absent(monkeypatch):
    # Simulate older torch lacking the functional collective; install/uninstall
    # must not raise.
    if hasattr(torch.distributed, "all_gather_into_tensor"):
        monkeypatch.delattr(torch.distributed, "all_gather_into_tensor", raising=False)
    if hasattr(torch.distributed, "reduce_scatter_tensor"):
        monkeypatch.delattr(torch.distributed, "reduce_scatter_tensor", raising=False)
    patch()
    unpatch()  # round-trip without error


def test_install_skips_cleanly_when_distributed_unavailable(monkeypatch):
    # Simulate a torch built with USE_DISTRIBUTED=0: is_available() returns
    # False and collective APIs may be missing. patch()/unpatch() must not raise.
    monkeypatch.setattr(torch.distributed, "is_available", lambda: False)
    patch()
    unpatch()


def test_collective_span_carries_job_id_and_rank(test_spans, monkeypatch):
    monkeypatch.setattr(torch.distributed, "all_reduce", mock.Mock())
    patch()
    try:
        # Simulate post-bootstrap state without actually running init_process_group.
        from ddtrace.contrib.internal.pytorch import _distributed as _d

        _d._state["job_id"] = "job-abc"
        _d._state["rank"] = 3
        _d._state["world_size"] = 8
        try:
            tensor = mock.Mock(is_cuda=False, numel=lambda: 2, element_size=lambda: 4)
            torch.distributed.all_reduce(tensor)
            ar = next(s for s in test_spans.pop() if s.name == "pytorch.allreduce")
            assert ar.get_tag("job_id") == "job-abc"
            assert ar.get_metric("rank") == 3
            assert ar.get_metric("world_size") == 8
        finally:
            _d._state.update({"job_id": None, "rank": 0, "world_size": 1})
    finally:
        unpatch()


def test_unpatch_resets_bootstrap_state(monkeypatch):
    bootstrap = mock.Mock()
    monkeypatch.setattr(
        "ddtrace.contrib.internal.pytorch._distributed._bootstrap_distributed",
        bootstrap,
    )
    monkeypatch.setattr(torch.distributed, "init_process_group", mock.Mock())
    patch()
    torch.distributed.init_process_group(backend="gloo")
    unpatch()
    patch()
    torch.distributed.init_process_group(backend="gloo")
    assert bootstrap.call_count == 2
    unpatch()
