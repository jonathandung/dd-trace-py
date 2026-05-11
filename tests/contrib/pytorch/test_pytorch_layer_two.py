"""Layer 2 (step-level profiling) unit tests."""

import importlib
import threading
from unittest import mock


def _reload_hooks():
    """Reload `_hooks` so each test starts from a clean module state.

    Resets ``PROFILING_ENABLED``, ``STEP_OPTIMIZER_NAME``,
    ``_designated_optimizer_ref``, ``_designated_warning_emitted``, and
    ``_step_counter`` to module-load values.
    """
    from ddtrace.contrib.internal.pytorch import _hooks

    return importlib.reload(_hooks)


def _reset_state():
    """Reset thread-local state shared by Layer 2 tests."""
    from ddtrace.contrib.internal.pytorch import _utils

    if hasattr(_utils._LAST_OPTIMIZER_STEP_END_NS, "value"):
        del _utils._LAST_OPTIMIZER_STEP_END_NS.value
    if hasattr(_utils._amp_skip_state, "in_amp"):
        del _utils._amp_skip_state.in_amp
    if hasattr(_utils._amp_skip_state, "step_executed"):
        del _utils._amp_skip_state.step_executed


# ---------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------


def test_gate_disabled_by_default(monkeypatch):
    monkeypatch.delenv("DD_PYTORCH_PROFILING", raising=False)
    hooks = _reload_hooks()
    assert hooks.PROFILING_ENABLED is False


def test_gate_enabled_via_env(monkeypatch):
    monkeypatch.setenv("DD_PYTORCH_PROFILING", "true")
    hooks = _reload_hooks()
    assert hooks.PROFILING_ENABLED is True


def test_attach_layer_two_hooks_is_noop_when_gate_disabled(monkeypatch):
    monkeypatch.delenv("DD_PYTORCH_PROFILING", raising=False)
    hooks = _reload_hooks()
    model = mock.MagicMock()
    model.register_forward_pre_hook = mock.Mock()
    hooks.attach_layer_two_hooks(model)
    model.register_forward_pre_hook.assert_not_called()


# ---------------------------------------------------------------------------
# Span emission
# ---------------------------------------------------------------------------


def test_forward_hook_emits_pytorch_forward_span(test_spans, monkeypatch):
    monkeypatch.setenv("DD_PYTORCH_PROFILING", "true")
    hooks = _reload_hooks()
    _reset_state()

    fake_model = mock.MagicMock()
    hooks._forward_pre_hook(fake_model, (mock.sentinel.t,))
    hooks._forward_hook(fake_model, (mock.sentinel.t,), mock.sentinel.out)
    # Close the open step.
    hooks._close_step(skipped=False)

    spans = test_spans.pop()
    forward = [s for s in spans if s.name == "pytorch.forward"]
    assert len(forward) == 1
    assert forward[0].duration is not None


def test_full_backward_hook_emits_pytorch_backward_span(test_spans, monkeypatch):
    monkeypatch.setenv("DD_PYTORCH_PROFILING", "true")
    hooks = _reload_hooks()
    _reset_state()

    fake_model = mock.MagicMock()
    hooks._full_backward_hook(fake_model, (mock.sentinel.gin,), (mock.sentinel.gout,))
    hooks._close_step(skipped=False)

    bwd = [s for s in test_spans.pop() if s.name == "pytorch.backward"]
    assert len(bwd) == 1


def test_data_load_span_measures_gap_after_first_iteration(test_spans, monkeypatch):
    monkeypatch.setenv("DD_PYTORCH_PROFILING", "true")
    hooks = _reload_hooks()
    _reset_state()

    hooks._mark_optimizer_step_end_now()
    fake_model = mock.MagicMock()
    hooks._forward_pre_hook(fake_model, (mock.sentinel.t,))
    hooks._forward_hook(fake_model, (mock.sentinel.t,), mock.sentinel.out)
    hooks._close_step(skipped=False)

    data_load = [s for s in test_spans.pop() if s.name == "pytorch.data_load"]
    assert len(data_load) == 1


def test_data_load_span_absent_on_first_iteration(test_spans, monkeypatch):
    monkeypatch.setenv("DD_PYTORCH_PROFILING", "true")
    hooks = _reload_hooks()
    _reset_state()

    fake_model = mock.MagicMock()
    hooks._forward_pre_hook(fake_model, (mock.sentinel.t,))
    hooks._forward_hook(fake_model, (mock.sentinel.t,), mock.sentinel.out)
    hooks._close_step(skipped=False)

    assert not [s for s in test_spans.pop() if s.name == "pytorch.data_load"]


def test_optimizer_step_emits_pytorch_optimizer_span_with_class_tag(test_spans, monkeypatch):
    monkeypatch.setenv("DD_PYTORCH_PROFILING", "true")
    hooks = _reload_hooks()
    _reset_state()

    class FakeAdamW:
        def step(self, closure=None):
            return None

    optimizer = FakeAdamW()
    hooks.optimizer_step(optimizer.step, optimizer, (), {})

    opt_spans = [s for s in test_spans.pop() if s.name == "pytorch.optimizer"]
    assert len(opt_spans) == 1
    assert opt_spans[0].get_tag("optimizer") == "FakeAdamW"


def test_pytorch_step_root_wraps_forward_backward_optimizer(test_spans, monkeypatch):
    monkeypatch.setenv("DD_PYTORCH_PROFILING", "true")
    hooks = _reload_hooks()
    _reset_state()

    class FakeAdamW:
        def step(self, closure=None):
            return None

    optimizer = FakeAdamW()
    fake_model = mock.MagicMock()
    hooks._forward_pre_hook(fake_model, (mock.sentinel.t,))
    hooks._forward_hook(fake_model, (mock.sentinel.t,), mock.sentinel.out)
    hooks._full_backward_hook(fake_model, (mock.sentinel.gin,), (mock.sentinel.gout,))
    hooks.optimizer_step(optimizer.step, optimizer, (), {})

    spans = test_spans.pop()
    by_name = {s.name: s for s in spans}
    assert "pytorch.step" in by_name
    step = by_name["pytorch.step"]
    children = [s for s in spans if s.parent_id == step.span_id]
    names = sorted(c.name for c in children)
    assert names == ["pytorch.backward", "pytorch.forward", "pytorch.optimizer"]


def test_gradient_accumulation_yields_single_pytorch_step(test_spans, monkeypatch):
    monkeypatch.setenv("DD_PYTORCH_PROFILING", "true")
    hooks = _reload_hooks()
    _reset_state()

    class FakeAdamW:
        def step(self, closure=None):
            return None

    optimizer = FakeAdamW()
    fake_model = mock.MagicMock()
    for _ in range(3):
        hooks._forward_pre_hook(fake_model, (mock.sentinel.t,))
        hooks._forward_hook(fake_model, (mock.sentinel.t,), mock.sentinel.out)
        hooks._full_backward_hook(fake_model, (mock.sentinel.gin,), (mock.sentinel.gout,))
    hooks.optimizer_step(optimizer.step, optimizer, (), {})

    spans = test_spans.pop()
    assert len([s for s in spans if s.name == "pytorch.step"]) == 1
    assert len([s for s in spans if s.name == "pytorch.forward"]) == 3
    assert len([s for s in spans if s.name == "pytorch.backward"]) == 3
    assert len([s for s in spans if s.name == "pytorch.optimizer"]) == 1
    assert hooks._step_counter == 1


# ---------------------------------------------------------------------------
# Multi-optimizer designation
# ---------------------------------------------------------------------------


def test_designation_via_env_var_picks_first_matching_class_instance(test_spans, monkeypatch):
    monkeypatch.setenv("DD_PYTORCH_PROFILING", "true")
    monkeypatch.setenv("DD_PYTORCH_STEP_OPTIMIZER", "AdamW")
    hooks = _reload_hooks()
    _reset_state()

    class SGD:
        def step(self, closure=None):
            return None

    class AdamW:
        def step(self, closure=None):
            return None

    for o in (SGD(), SGD()):
        hooks.optimizer_step(o.step, o, (), {})
    assert hooks._step_counter == 0

    adamw1 = AdamW()
    hooks.optimizer_step(adamw1.step, adamw1, (), {})
    assert hooks._step_counter == 1

    adamw2 = AdamW()
    hooks.optimizer_step(adamw2.step, adamw2, (), {})
    # Only the first matching instance is designated.
    assert hooks._step_counter == 1


def test_designation_auto_picks_first_step_caller(test_spans, monkeypatch):
    monkeypatch.setenv("DD_PYTORCH_PROFILING", "true")
    monkeypatch.delenv("DD_PYTORCH_STEP_OPTIMIZER", raising=False)
    hooks = _reload_hooks()
    _reset_state()

    class Lion:
        def step(self, closure=None):
            return None

    class Adagrad:
        def step(self, closure=None):
            return None

    lion = Lion()
    adagrad = Adagrad()
    hooks.optimizer_step(lion.step, lion, (), {})
    hooks.optimizer_step(adagrad.step, adagrad, (), {})
    hooks.optimizer_step(lion.step, lion, (), {})

    # First caller (Lion) becomes the designated optimizer for step counting.
    # ddtrace's logger is rate-limited and may suppress the user-facing
    # warning across tests in the same run, so we assert the underlying
    # one-shot flag instead.
    assert hooks._designated_warning_emitted is True
    assert hooks._designated_optimizer_ref() is lion
    assert hooks._step_counter == 2  # 2 calls from Lion; Adagrad isn't designated.


def test_designation_re_runs_after_weakref_gc(test_spans, monkeypatch):
    monkeypatch.setenv("DD_PYTORCH_PROFILING", "true")
    monkeypatch.delenv("DD_PYTORCH_STEP_OPTIMIZER", raising=False)
    hooks = _reload_hooks()
    _reset_state()

    class Adam:
        def step(self, closure=None):
            return None

    a = Adam()
    hooks.optimizer_step(a.step, a, (), {})
    assert hooks._step_counter == 1
    del a
    import gc

    gc.collect()
    b = Adam()
    hooks.optimizer_step(b.step, b, (), {})
    assert hooks._step_counter == 2
    assert hooks._designated_optimizer_ref() is b


def test_designation_thread_safe_under_concurrent_first_step(test_spans, monkeypatch):
    monkeypatch.setenv("DD_PYTORCH_PROFILING", "true")
    monkeypatch.delenv("DD_PYTORCH_STEP_OPTIMIZER", raising=False)
    hooks = _reload_hooks()
    _reset_state()

    class Adam:
        def step(self, closure=None):
            return None

    optimizers = [Adam() for _ in range(8)]
    barrier = threading.Barrier(len(optimizers))

    def race(o):
        barrier.wait()
        hooks.optimizer_step(o.step, o, (), {})

    threads = [threading.Thread(target=race, args=(o,)) for o in optimizers]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert hooks._step_counter == 1
    assert hooks._designated_optimizer_ref() in optimizers


# ---------------------------------------------------------------------------
# AMP overflow skip integration
# ---------------------------------------------------------------------------


def test_amp_skip_emits_pytorch_step_skipped_true_with_no_optimizer_span(test_spans, monkeypatch):
    monkeypatch.setenv("DD_PYTORCH_PROFILING", "true")
    hooks = _reload_hooks()
    _reset_state()

    from ddtrace.contrib.internal.pytorch._utils import _amp_skip_state

    class AdamW:
        def step(self, closure=None):
            return None

    optimizer = AdamW()
    fake_model = mock.MagicMock()

    # Simulate `scaler.step(optimizer)` where GradScaler skips due to overflow:
    # in_amp=True, the optimizer wrapper would have been called or not, but
    # `gradscaler_emit_step_outcome(..., skipped=True)` is invoked from the
    # GradScaler wrapper afterward.
    _amp_skip_state.in_amp = True
    try:
        hooks._forward_pre_hook(fake_model, (mock.sentinel.t,))
        hooks._forward_hook(fake_model, (mock.sentinel.t,), mock.sentinel.out)
        hooks._full_backward_hook(fake_model, (mock.sentinel.gin,), (mock.sentinel.gout,))
    finally:
        _amp_skip_state.in_amp = False
    initial_counter = hooks._step_counter
    hooks.gradscaler_emit_step_outcome(optimizer, skipped=True)

    spans = test_spans.pop()
    step_spans = [s for s in spans if s.name == "pytorch.step"]
    opt_spans = [s for s in spans if s.name == "pytorch.optimizer"]
    assert len(step_spans) == 1
    assert step_spans[0].get_tag("skipped") in ("true", True, "True")
    assert opt_spans == []
    assert hooks._step_counter == initial_counter


def test_amp_non_skip_emits_step_with_optimizer_span(test_spans, monkeypatch):
    """AMP path, no overflow: optimizer wrapper emits `pytorch.optimizer`
    (covering the inner step), and `gradscaler_emit_step_outcome(skipped=False)`
    closes `pytorch.step` and increments the counter.
    """
    monkeypatch.setenv("DD_PYTORCH_PROFILING", "true")
    hooks = _reload_hooks()
    _reset_state()

    from ddtrace.contrib.internal.pytorch._utils import _amp_skip_state

    class AdamW:
        def step(self, closure=None):
            return None

    optimizer = AdamW()
    fake_model = mock.MagicMock()

    _amp_skip_state.in_amp = True
    _amp_skip_state.step_executed = False
    try:
        hooks._forward_pre_hook(fake_model, (mock.sentinel.t,))
        hooks._forward_hook(fake_model, (mock.sentinel.t,), mock.sentinel.out)
        hooks._full_backward_hook(fake_model, (mock.sentinel.gin,), (mock.sentinel.gout,))
        hooks.optimizer_step(optimizer.step, optimizer, (), {})
        assert _amp_skip_state.step_executed is True
    finally:
        _amp_skip_state.in_amp = False
        _amp_skip_state.step_executed = False
    hooks.gradscaler_emit_step_outcome(optimizer, skipped=False)

    spans = test_spans.pop()
    step_spans = [s for s in spans if s.name == "pytorch.step"]
    opt_spans = [s for s in spans if s.name == "pytorch.optimizer"]
    assert len(step_spans) == 1
    assert len(opt_spans) == 1
    assert opt_spans[0].get_tag("optimizer") == "AdamW"
    assert hooks._step_counter == 1


def test_amp_skip_on_non_designated_optimizer_does_not_close_step(test_spans, monkeypatch):
    """Gemini P1 regression guard: GradScaler skip with a non-designated
    optimizer must not prematurely close the designated optimizer's
    pytorch.step span.
    """
    monkeypatch.setenv("DD_PYTORCH_PROFILING", "true")
    monkeypatch.setenv("DD_PYTORCH_STEP_OPTIMIZER", "Designated")
    hooks = _reload_hooks()
    _reset_state()

    class Designated:
        def step(self, closure=None):
            return None

    class Other:
        def step(self, closure=None):
            return None

    designated = Designated()
    other = Other()
    fake_model = mock.MagicMock()

    hooks._forward_pre_hook(fake_model, (mock.sentinel.t,))
    hooks._forward_hook(fake_model, (mock.sentinel.t,), mock.sentinel.out)
    # Non-designated optimizer's AMP skip arrives mid-step.
    hooks.gradscaler_emit_step_outcome(other, skipped=True)
    hooks._full_backward_hook(fake_model, (mock.sentinel.gin,), (mock.sentinel.gout,))
    hooks.optimizer_step(designated.step, designated, (), {})

    step_spans = [s for s in test_spans.pop() if s.name == "pytorch.step"]
    assert len(step_spans) == 1
    assert step_spans[0].get_tag("skipped") is None
    assert hooks._step_counter == 1


# ---------------------------------------------------------------------------
# LBFGS closure + idempotent hook registration
# ---------------------------------------------------------------------------


def test_lbfgs_closure_yields_one_step_with_repeated_fwd_bwd(test_spans, monkeypatch):
    monkeypatch.setenv("DD_PYTORCH_PROFILING", "true")
    hooks = _reload_hooks()
    _reset_state()

    fake_model = mock.MagicMock()

    class FakeLBFGS:
        def step(self, closure):
            for _ in range(3):
                closure()
            return None

    optimizer = FakeLBFGS()

    def closure():
        hooks._forward_pre_hook(fake_model, (mock.sentinel.t,))
        hooks._forward_hook(fake_model, (mock.sentinel.t,), mock.sentinel.out)
        hooks._full_backward_hook(fake_model, (mock.sentinel.gin,), (mock.sentinel.gout,))
        return mock.sentinel.loss

    hooks.optimizer_step(optimizer.step, optimizer, (closure,), {})

    spans = test_spans.pop()
    assert len([s for s in spans if s.name == "pytorch.step"]) == 1
    assert len([s for s in spans if s.name == "pytorch.forward"]) == 3
    assert len([s for s in spans if s.name == "pytorch.backward"]) == 3
    assert len([s for s in spans if s.name == "pytorch.optimizer"]) == 1


def test_attach_layer_two_hooks_is_idempotent(monkeypatch):
    monkeypatch.setenv("DD_PYTORCH_PROFILING", "true")
    hooks = _reload_hooks()

    class FakeModel:
        def __init__(self):
            self.registrations = []

        def register_forward_pre_hook(self, hook):
            self.registrations.append(("pre", hook))

        def register_forward_hook(self, hook):
            self.registrations.append(("fwd", hook))

        def register_full_backward_hook(self, hook):
            self.registrations.append(("bwd", hook))

    model = FakeModel()
    hooks.attach_layer_two_hooks(model)
    hooks.attach_layer_two_hooks(model)

    kinds = [k for (k, _) in model.registrations]
    assert kinds.count("pre") == 1
    assert kinds.count("fwd") == 1
    assert kinds.count("bwd") == 1
