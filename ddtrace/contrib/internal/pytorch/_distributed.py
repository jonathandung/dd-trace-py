import collections
import os
import sys
import threading
from typing import Any
from typing import Optional
import weakref

import torch
import wrapt

from ddtrace import config
from ddtrace import tracer
from ddtrace.contrib.internal.pytorch._utils import _enter_framework
from ddtrace.contrib.internal.pytorch._utils import _get_active_framework
from ddtrace.contrib.internal.pytorch._utils import (
    _instrumentation_bypass,  # noqa: F401  (re-exported for grad_comm hook)
)
from ddtrace.contrib.internal.pytorch._utils import _should_record_cuda_event
from ddtrace.contrib.internal.pytorch._utils import is_instrumentation_bypassed
from ddtrace.contrib.internal.pytorch._utils import register_framework
from ddtrace.contrib.internal.pytorch._utils import resolve_job_id_from_env
from ddtrace.contrib.internal.trace_utils import unwrap as _unwrap
from ddtrace.contrib.internal.trace_utils import wrap as _wrap
from ddtrace.internal.logger import get_logger


log = get_logger(__name__)

_DEFAULT_CAPACITY = 1024
_OVERFLOW_WARN_EVERY = 64


_DEFAULT_BROADCAST_TIMEOUT_S = 10.0


def _broadcast_timeout_s() -> float:
    """Read `DD_PYTORCH_JOB_ID_BROADCAST_TIMEOUT` lazily.

    Reading on each call (rather than caching at module load) lets tests and
    users mutate the env var after `import ddtrace` but before
    `init_process_group`.
    """
    raw = os.environ.get("DD_PYTORCH_JOB_ID_BROADCAST_TIMEOUT")
    if not raw:
        return _DEFAULT_BROADCAST_TIMEOUT_S
    try:
        value = float(raw)
    except ValueError:
        log.warning(
            "pytorch: invalid DD_PYTORCH_JOB_ID_BROADCAST_TIMEOUT=%r; using default 10s",
            raw,
        )
        return _DEFAULT_BROADCAST_TIMEOUT_S
    if value <= 0:
        log.warning(
            "pytorch: DD_PYTORCH_JOB_ID_BROADCAST_TIMEOUT=%r must be > 0; using default 10s",
            raw,
        )
        return _DEFAULT_BROADCAST_TIMEOUT_S
    return value


_state: dict[str, Any] = {
    "bootstrapped": False,
    "job_id": None,
    "rank": 0,
    "world_size": 1,
    "resolver": None,
}
# Guards the wrapper-side "bootstrapped" check so concurrent
# `init_process_group` calls (rare, but possible across user threads)
# don't double-run the bootstrap.
_bootstrap_lock = threading.Lock()


class CudaEventResolver:
    """Background polling of ``torch.cuda.Event`` pairs.

    Spans are held open until ``end_event.query()`` returns True; we then
    compute ``start_event.elapsed_time(end_event)`` (ms) and finish the span.
    Overflow drops the oldest entry and finishes that span with
    ``_dd.error_reason="cuda_event_overflow"``. Shutdown joins the thread with
    a bounded timeout and finishes any remaining spans with
    ``_dd.error_reason="cuda_event_unresolved"``.
    """

    def __init__(self, poll_interval: float = 0.005, capacity: int = _DEFAULT_CAPACITY):
        self._poll_interval = poll_interval
        self._capacity = capacity
        self._queue: collections.deque = collections.deque()
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = None
        self._overflow_count = 0

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop_event.clear()
        t = threading.Thread(target=self._run, name="dd-pytorch-cuda-resolver", daemon=True)
        self._thread = t
        t.start()

    def submit(self, span, start_event, end_event) -> None:
        # Hold the lock only to mutate the queue / counter. Span finalization
        # (set_tag + finish) runs outside the lock because it may invoke user
        # span processors, exporters, or even recursively re-enter the resolver.
        evicted = None
        overflow_total = 0
        with self._lock:
            if len(self._queue) >= self._capacity:
                evicted, _, _ = self._queue.popleft()
                self._overflow_count += 1
                overflow_total = self._overflow_count
            self._queue.append((span, start_event, end_event))
        if evicted is not None:
            if overflow_total % _OVERFLOW_WARN_EVERY == 1:
                log.warning(
                    "pytorch: cuda event queue overflow (count=%d); dropping oldest span",
                    overflow_total,
                )
            self._finish_unresolved(evicted, reason="cuda_event_overflow")

    def _run(self) -> None:
        # Use stop_event.wait so stop() interrupts the sleep promptly instead
        # of waiting up to poll_interval before exiting.
        while not self._stop_event.wait(self._poll_interval):
            self._drain_ready()
        self._drain_ready()
        self._flush_remaining()

    def _drain_ready(self) -> None:
        with self._lock:
            pending = list(self._queue)
            self._queue.clear()
        keep: collections.deque = collections.deque()
        for span, start, end in pending:
            try:
                ready = end.query()
            except Exception:
                self._finish_unresolved(span, reason="cuda_event_query_error")
                continue
            if ready:
                try:
                    duration_ms = float(start.elapsed_time(end))
                    span.set_metric("gpu.duration_ms", duration_ms)
                except Exception:
                    span.set_tag("_dd.error_reason", "cuda_event_elapsed_error")
                span.finish()
            else:
                keep.append((span, start, end))
        with self._lock:
            keep.extend(self._queue)
            self._queue = keep

    def _flush_remaining(self) -> None:
        with self._lock:
            remaining = list(self._queue)
            self._queue.clear()
        for span, _, _ in remaining:
            self._finish_unresolved(span, reason="cuda_event_unresolved")

    @staticmethod
    def _finish_unresolved(span, reason: str) -> None:
        try:
            span.set_tag("_dd.error_reason", reason)
        finally:
            span.finish()

    def stop(self, timeout: float = 2.0) -> None:
        self._stop_event.set()
        t = self._thread
        if t is not None:
            t.join(timeout=timeout)
        self._thread = None
        self._flush_remaining()


def _broadcast_job_id_with_timeout(local_job_id: str, rank: int) -> str:
    """Broadcast rank-0's ``local_job_id`` to all ranks, bounded by
    ``_broadcast_timeout_s()``.

    Falls back to the caller's local id on timeout or any error. The broadcast
    runs in a **daemon** background thread because
    ``torch.distributed.broadcast_object_list`` does not honor a per-call
    timeout and can block indefinitely if the collective stalls; a non-daemon
    leaked worker would also prevent process exit. We accept a leaked daemon
    thread as the cost of guaranteeing the user's training job does not hang.
    """
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return local_job_id

    obj_list = [local_job_id if rank == 0 else None]
    captured: list = [None, None]  # [result, exception]

    def do_broadcast():
        try:
            torch.distributed.broadcast_object_list(obj_list, src=0)
            captured[0] = obj_list[0]
        except BaseException as exc:  # noqa: BLE001
            captured[1] = exc

    thread = threading.Thread(
        target=do_broadcast,
        name="dd-pytorch-job-id-broadcast",
        daemon=True,
    )
    thread.start()
    thread.join(timeout=_broadcast_timeout_s())
    if thread.is_alive():
        log.warning(
            "pytorch: job_id broadcast exceeded %.1fs; falling back to local UUID per rank",
            _broadcast_timeout_s(),
        )
        return local_job_id
    if captured[1] is not None:
        log.exception("pytorch: job_id broadcast failed; using local UUID", exc_info=captured[1])
        return local_job_id
    return captured[0] or local_job_id


def _bootstrap_distributed() -> None:
    """One-shot bootstrap: resolve job_id, capture rank/world_size, broadcast
    job_id to all ranks, start resolver.

    Idempotency is enforced by the call site (`_wrapped_init_process_group`),
    not here — this function is the unit of work that runs exactly once per
    patched lifetime.
    """
    _state["job_id"] = resolve_job_id_from_env()
    try:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            _state["rank"] = torch.distributed.get_rank()
            _state["world_size"] = torch.distributed.get_world_size()
            _state["job_id"] = _broadcast_job_id_with_timeout(_state["job_id"], _state["rank"])
    except Exception:
        log.exception("pytorch: failed to capture rank/world_size; defaulting to single-rank")
    resolver = CudaEventResolver()
    resolver.start()
    _state["resolver"] = resolver


def _wrapped_init_process_group(wrapped, instance, args, kwargs):
    result = wrapped(*args, **kwargs)
    with _bootstrap_lock:
        already = _state["bootstrapped"]
        if not already:
            _state["bootstrapped"] = True
    if not already:
        try:
            _bootstrap_distributed()
        except Exception:
            log.exception("pytorch: distributed bootstrap failed")
    return result


# (span_name, tensor_arg_index). For collectives whose first positional is a
# list/aggregated tensor, point at the input arg so `bytes` reflects the full
# communication volume (the list-aware `_tensor_bytes` sums entries):
#   all_gather(tensor_list, tensor, ...)         -> per-rank input at 1
#   reduce_scatter(output, input_list, ...)      -> full input_list at 1
_COLLECTIVES: dict[str, tuple[str, int]] = {
    "all_reduce": ("pytorch.allreduce", 0),
    "all_gather": ("pytorch.allgather", 1),
    "broadcast": ("pytorch.broadcast", 0),
    "reduce_scatter": ("pytorch.reducescatter", 1),
    "barrier": ("pytorch.barrier", 0),
}


def _tensor_bytes(tensor) -> int:
    """Best-effort byte count for a single tensor or a list/tuple of tensors."""
    if isinstance(tensor, (list, tuple)):
        total = 0
        for t in tensor:
            total += _tensor_bytes(t)
        return total
    try:
        return int(tensor.numel()) * int(tensor.element_size())
    except Exception:
        return 0


def _make_collective_wrapper(span_name: str, tensor_arg_index: int = 0):
    def wrapper(wrapped, instance, args, kwargs):
        if is_instrumentation_bypassed():
            return wrapped(*args, **kwargs)
        tensor = args[tensor_arg_index] if len(args) > tensor_arg_index else None
        group = kwargs.get("group", None)
        # `child_of=tracer.current_span()` ties the collective span to the
        # currently-active application span without making it the new active
        # span. We deliberately do NOT pass `activate=True` because the CUDA
        # path defers `span.finish()` to a background resolver thread; if the
        # span were active, it would remain so on the caller thread until the
        # async finish ran, mis-parenting any subsequent spans on this thread.
        span = tracer.start_span(
            span_name,
            service=config.pytorch.service,
            child_of=tracer.current_span(),
        )
        span.set_tag("framework", _get_active_framework() or "none")
        span.set_metric("rank", _state["rank"])
        span.set_metric("world_size", _state["world_size"])
        if _state.get("job_id"):
            span.set_tag("job_id", _state["job_id"])
        if tensor is not None and span_name != "pytorch.barrier":
            span.set_metric("bytes", _tensor_bytes(tensor))

        resolver = _state.get("resolver")
        record_cuda = resolver is not None and _should_record_cuda_event(group, tensor)
        start_event = end_event = None
        if record_cuda:
            try:
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
            except Exception:
                start_event = end_event = None
        wrapped_raised = False
        try:
            return wrapped(*args, **kwargs)
        except BaseException:
            wrapped_raised = True
            span.set_exc_info(*sys.exc_info())
            raise
        finally:
            submitted = False
            # Only submit events when the call succeeded; recording end_event on
            # a never-launched kernel produces a duration that misrepresents the
            # collective.
            if not wrapped_raised and start_event is not None and end_event is not None and resolver is not None:
                try:
                    end_event.record()
                    resolver.submit(span, start_event, end_event)
                    submitted = True
                except Exception:
                    span.set_tag("_dd.error_reason", "cuda_event_record_error")
            if not submitted:
                span.finish()

    return wrapper


def _distributed_available() -> bool:
    """True only when `torch.distributed` was compiled in (USE_DISTRIBUTED=1).

    On torch builds without distributed support, the collective APIs are
    absent and `_wrap` would raise — short-circuit the whole install path.
    """
    try:
        return bool(torch.distributed.is_available())
    except Exception:
        return False


def _install_collectives() -> None:
    if not _distributed_available():
        return
    for fn_name, (span_name, tensor_idx) in _COLLECTIVES.items():
        if not hasattr(torch.distributed, fn_name):
            continue
        _wrap("torch.distributed", fn_name, _make_collective_wrapper(span_name, tensor_idx))


def _uninstall_collectives() -> None:
    if not _distributed_available():
        return
    for fn_name in _COLLECTIVES:
        if not hasattr(torch.distributed, fn_name):
            continue
        try:
            _unwrap(torch.distributed, fn_name)
        except Exception:
            log.debug("pytorch: failed to unwrap torch.distributed.%s", fn_name, exc_info=True)


# FSDP-style functional collective variants. Available on torch >= 2.0 but
# wrapped behind `hasattr` because PyTorch sometimes ships these with internal
# underscored names or removes them in pre-release builds.
# Both take `(output_tensor, input_tensor, ...)`; we tag bytes from the input
# (per-rank contribution) at index 1.
_FSDP_COLLECTIVES: dict[str, tuple[str, int]] = {
    "all_gather_into_tensor": ("pytorch.allgather_into_tensor", 1),
    "reduce_scatter_tensor": ("pytorch.reducescatter_tensor", 1),
}


def _install_fsdp_collectives() -> None:
    if not _distributed_available():
        return
    for fn_name, (span_name, tensor_idx) in _FSDP_COLLECTIVES.items():
        if not hasattr(torch.distributed, fn_name):
            continue
        _wrap("torch.distributed", fn_name, _make_collective_wrapper(span_name, tensor_idx))


def _uninstall_fsdp_collectives() -> None:
    if not _distributed_available():
        return
    for fn_name in _FSDP_COLLECTIVES:
        if not hasattr(torch.distributed, fn_name):
            continue
        try:
            _unwrap(torch.distributed, fn_name)
        except Exception:
            log.debug("pytorch: failed to unwrap torch.distributed.%s", fn_name, exc_info=True)


def _wrapped_ddp_init(wrapped, instance, args, kwargs):
    """Run the original DDP __init__, then tag the instance as `ddp` in the
    framework registry so collective spans inside DDP backward/comm hooks pick
    up the right `framework` tag.
    """
    result = wrapped(*args, **kwargs)
    try:
        register_framework(instance, "ddp")
    except Exception:
        # Non-actionable for the user (the model still works); log at warning
        # without a traceback to avoid polluting training output.
        log.warning("pytorch: failed to register DDP framework", exc_info=True)
    return result


def _install_ddp() -> None:
    try:
        import torch.nn.parallel.distributed  # noqa: F401
    except Exception:
        return
    if not hasattr(torch.nn.parallel.distributed, "DistributedDataParallel"):
        return
    _wrap(
        "torch.nn.parallel.distributed",
        "DistributedDataParallel.__init__",
        _wrapped_ddp_init,
    )


def _uninstall_ddp() -> None:
    try:
        import torch.nn.parallel.distributed  # noqa: F401
    except Exception:
        return
    if not hasattr(torch.nn.parallel.distributed, "DistributedDataParallel"):
        return
    try:
        _unwrap(torch.nn.parallel.distributed.DistributedDataParallel, "__init__")
    except Exception:
        log.debug("pytorch: failed to unwrap DDP.__init__", exc_info=True)


def _wrapped_fsdp_init(wrapped, instance, args, kwargs):
    result = wrapped(*args, **kwargs)
    try:
        register_framework(instance, "fsdp")
    except Exception:
        log.warning("pytorch: failed to register FSDP framework", exc_info=True)
    return result


def _wrapped_fsdp_forward(wrapped, instance, args, kwargs):
    """Open a framework context around FSDP forward so collectives emitted
    during the sharded forward pass are tagged `framework=fsdp`.
    """
    with _enter_framework(instance):
        return wrapped(*args, **kwargs)


def _install_fsdp() -> None:
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel  # noqa: F401
    except Exception:
        return
    _wrap("torch.distributed.fsdp", "FullyShardedDataParallel.__init__", _wrapped_fsdp_init)
    _wrap("torch.distributed.fsdp", "FullyShardedDataParallel.forward", _wrapped_fsdp_forward)


def _uninstall_fsdp() -> None:
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel
    except Exception:
        return
    for attr in ("__init__", "forward"):
        try:
            _unwrap(FullyShardedDataParallel, attr)
        except Exception:
            log.debug("pytorch: failed to unwrap FSDP.%s", attr, exc_info=True)


def _wrapped_deepspeed_init(wrapped, instance, args, kwargs):
    result = wrapped(*args, **kwargs)
    try:
        register_framework(instance, "deepspeed")
    except Exception:
        log.warning("pytorch: failed to register deepspeed framework", exc_info=True)
    return result


def _wrapped_deepspeed_method(name: str):
    def wrapper(wrapped, instance, args, kwargs):
        with _enter_framework(instance):
            return wrapped(*args, **kwargs)

    return wrapper


def _install_deepspeed() -> None:
    try:
        import deepspeed  # type: ignore[import-not-found]  # noqa: F401
    except Exception:
        return
    if not hasattr(deepspeed, "DeepSpeedEngine"):
        return
    _wrap("deepspeed", "DeepSpeedEngine.__init__", _wrapped_deepspeed_init)
    for m in ("forward", "backward", "step"):
        if hasattr(deepspeed.DeepSpeedEngine, m):
            _wrap("deepspeed", "DeepSpeedEngine.%s" % m, _wrapped_deepspeed_method(m))


def _uninstall_deepspeed() -> None:
    try:
        import deepspeed  # type: ignore[import-not-found]
    except Exception:
        return
    if not hasattr(deepspeed, "DeepSpeedEngine"):
        return
    for m in ("__init__", "forward", "backward", "step"):
        try:
            _unwrap(deepspeed.DeepSpeedEngine, m)
        except Exception:
            log.debug("pytorch: failed to unwrap DeepSpeedEngine.%s", m, exc_info=True)


# Stores the original `step` bound method for every wrapped optimizer. The
# WeakKeyDictionary auto-evicts entries for garbage-collected optimizers and
# also acts as our "is this instance wrapped" registry (presence ⇔ wrapped).
_step_originals: "weakref.WeakKeyDictionary[Any, Any]" = weakref.WeakKeyDictionary()


def _instance_step_wrapper(wrapped, instance, args, kwargs):
    return wrapped(*args, **kwargs)


def _wrapped_optimizer_init(wrapped, instance, args, kwargs):
    """Wrap `optimizer.step` per-instance after construction.

    Most PyTorch optimizers override `Optimizer.step`, so wrapping the base
    class method does NOT intercept subclass calls. Instance-level wrapping
    catches all optimizers without enumerating subclasses.
    """
    result = wrapped(*args, **kwargs)
    try:
        if instance in _step_originals:
            # Already wrapped (e.g. base Optimizer.__init__ chained through a
            # subclass __init__ that also calls super().__init__()). Skip.
            return result
        original = instance.step
        _step_originals[instance] = original
        instance.step = wrapt.FunctionWrapper(original, _instance_step_wrapper)
    except Exception:
        log.warning("pytorch: failed to wrap optimizer.step", exc_info=True)
    return result


def _install_optimizer() -> None:
    _wrap("torch.optim.optimizer", "Optimizer.__init__", _wrapped_optimizer_init)


def _uninstall_optimizer() -> None:
    try:
        _unwrap(torch.optim.Optimizer, "__init__")
    except Exception:
        log.debug("pytorch: failed to unwrap Optimizer.__init__", exc_info=True)
    # Restore the original `step` on every already-constructed instance.
    for opt, original_step in list(_step_originals.items()):
        try:
            opt.step = original_step
        except Exception:
            log.debug("pytorch: failed to restore optimizer.step", exc_info=True)
    _step_originals.clear()


_amp_skip_state = threading.local()
_wrapped_gradscaler_targets: list = []


def _is_amp_step_in_progress() -> bool:
    return getattr(_amp_skip_state, "in_amp", False)


def _wrapped_gradscaler_step(wrapped, instance, args, kwargs):
    prev = getattr(_amp_skip_state, "in_amp", False)
    _amp_skip_state.in_amp = True
    try:
        return wrapped(*args, **kwargs)
    finally:
        _amp_skip_state.in_amp = prev


def _install_gradscaler() -> None:
    seen_ids = set()
    candidates: list = []
    try:
        from torch.cuda.amp import GradScaler as CudaScaler  # noqa: F401

        candidates.append(("torch.cuda.amp", CudaScaler))
    except Exception:
        pass
    try:
        from torch.amp import GradScaler as AmpScaler  # noqa: F401

        candidates.append(("torch.amp", AmpScaler))
    except Exception:
        pass
    for module_path, cls in candidates:
        # Identity-based dedup: in torch >= 2.1, both names typically alias the
        # same class object, and double-wrapping `.step` corrupts the
        # in_amp flag on re-entry.
        if id(cls) in seen_ids:
            continue
        seen_ids.add(id(cls))
        try:
            _wrap(module_path, "GradScaler.step", _wrapped_gradscaler_step)
            _wrapped_gradscaler_targets.append(cls)
        except Exception:
            log.warning("pytorch: failed to wrap %s.GradScaler.step", module_path, exc_info=True)


def _uninstall_gradscaler() -> None:
    for cls in _wrapped_gradscaler_targets:
        try:
            _unwrap(cls, "step")
        except Exception:
            log.debug("pytorch: failed to unwrap GradScaler.step", exc_info=True)
    _wrapped_gradscaler_targets.clear()


# Tracks which DDP instances have already auto-registered a comm hook via the
# lazy `_pre_backward_hook` path. WeakSet so destroyed models are evicted
# automatically.
_lazy_comm_hook_installed: "weakref.WeakSet" = weakref.WeakSet()


def _make_chained_comm_hook(user_hook):
    """Wrap a user-supplied DDP comm hook with our timing layer.

    Inside the user hook we open a `pytorch.grad_comm` span and call into the
    user hook under `_instrumentation_bypass` so any `torch.distributed.*`
    calls the user makes don't double-count.
    """

    def chained(state, bucket):
        if not config.pytorch.grad_comm_enabled:
            return user_hook(state, bucket)
        span = tracer.start_span(
            "pytorch.grad_comm",
            service=config.pytorch.service,
            child_of=tracer.current_span(),
        )
        span.set_tag("framework", "ddp")
        span.set_metric("rank", _state["rank"])
        if _state.get("job_id"):
            span.set_tag("job_id", _state["job_id"])
        # Best-effort bytes: PyTorch buckets expose either `.gradients()` or
        # `.buffer()`; we sum whichever is available.
        try:
            if hasattr(bucket, "gradients"):
                grads = list(bucket.gradients())
                if grads:
                    span.set_metric("bytes", sum(_tensor_bytes(t) for t in grads))
            elif hasattr(bucket, "buffer"):
                span.set_metric("bytes", _tensor_bytes(bucket.buffer()))
        except Exception:
            log.debug("pytorch: bucket size introspection failed", exc_info=True)
        with _instrumentation_bypass():
            try:
                return user_hook(state, bucket)
            finally:
                span.finish()

    chained._dd_chained = True
    return chained


def _wrapped_register_comm_hook(wrapped, instance, args, kwargs):
    if not config.pytorch.grad_comm_enabled:
        return wrapped(*args, **kwargs)
    if len(args) >= 2:
        state, hook = args[0], args[1]
        rest = args[2:]
        return wrapped(state, _make_chained_comm_hook(hook), *rest, **kwargs)
    if "hook" in kwargs:
        kwargs = dict(kwargs)
        kwargs["hook"] = _make_chained_comm_hook(kwargs["hook"])
    return wrapped(*args, **kwargs)


def _default_allreduce_hook(state, bucket):
    """Fallback delegate preserving DDP's standard mean-reduction semantics."""
    try:
        from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import allreduce_hook
    except Exception:
        return None
    return allreduce_hook(state, bucket)


def _wrapped_pre_backward_hook(wrapped, instance, args, kwargs):
    if instance not in _lazy_comm_hook_installed and config.pytorch.grad_comm_enabled:
        try:
            _lazy_comm_hook_installed.add(instance)
            instance.register_comm_hook(None, _default_allreduce_hook)
        except Exception:
            log.debug("pytorch: lazy comm hook registration failed", exc_info=True)
    return wrapped(*args, **kwargs)


def _install_ddp_comm_hook() -> None:
    try:
        import torch.nn.parallel.distributed  # noqa: F401
    except Exception:
        return
    cls = torch.nn.parallel.distributed.DistributedDataParallel
    if hasattr(cls, "register_comm_hook"):
        _wrap(
            "torch.nn.parallel.distributed",
            "DistributedDataParallel.register_comm_hook",
            _wrapped_register_comm_hook,
        )
    if hasattr(cls, "_pre_backward_hook"):
        _wrap(
            "torch.nn.parallel.distributed",
            "DistributedDataParallel._pre_backward_hook",
            _wrapped_pre_backward_hook,
        )
    else:
        log.info(
            "pytorch: DistributedDataParallel._pre_backward_hook not found "
            "(PyTorch 2.0); lazy comm-hook registration disabled — "
            "DDP gradient communication will be traced via torch.distributed.all_reduce only"
        )


def _uninstall_ddp_comm_hook() -> None:
    try:
        import torch.nn.parallel.distributed  # noqa: F401
    except Exception:
        return
    cls = torch.nn.parallel.distributed.DistributedDataParallel
    for attr in ("register_comm_hook", "_pre_backward_hook"):
        if hasattr(cls, attr):
            try:
                _unwrap(cls, attr)
            except Exception:
                log.debug("pytorch: failed to unwrap DDP.%s", attr, exc_info=True)
    _lazy_comm_hook_installed.clear()


def install() -> None:
    if _distributed_available() and hasattr(torch.distributed, "init_process_group"):
        _wrap("torch.distributed", "init_process_group", _wrapped_init_process_group)
    _install_collectives()
    _install_fsdp_collectives()
    _install_ddp()
    _install_fsdp()
    _install_deepspeed()
    _install_optimizer()
    _install_gradscaler()
    _install_ddp_comm_hook()
    # Late-patch bootstrap: if the user already called `init_process_group`
    # before `patch()`, our wrapper will never fire. Run the bootstrap now so
    # rank/world_size/job_id are populated and the CUDA event resolver is
    # started for already-initialized distributed jobs.
    if _distributed_available():
        try:
            if torch.distributed.is_initialized():
                with _bootstrap_lock:
                    already = _state["bootstrapped"]
                    if not already:
                        _state["bootstrapped"] = True
                if not already:
                    _bootstrap_distributed()
        except Exception:
            log.exception("pytorch: late-patch bootstrap failed")


def uninstall() -> None:
    if _distributed_available() and hasattr(torch.distributed, "init_process_group"):
        try:
            _unwrap(torch.distributed, "init_process_group")
        except Exception:
            log.debug("pytorch: failed to unwrap init_process_group", exc_info=True)
    _uninstall_collectives()
    _uninstall_fsdp_collectives()
    _uninstall_ddp()
    _uninstall_fsdp()
    _uninstall_deepspeed()
    _uninstall_optimizer()
    _uninstall_gradscaler()
    _uninstall_ddp_comm_hook()
    resolver: Optional[CudaEventResolver] = _state.get("resolver")
    if resolver is not None:
        resolver.stop(timeout=2.0)
    _state.update(
        {
            "bootstrapped": False,
            "job_id": None,
            "rank": 0,
            "world_size": 1,
            "resolver": None,
        }
    )
