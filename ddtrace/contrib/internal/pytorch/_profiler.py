"""Layer 3 CUDA kernel profiling bridge for the PyTorch contrib integration.

Gated by ``DD_PYTORCH_KERNEL_PROFILING=true``. Bridges ``torch.profiler``
events into Datadog traces by emitting ``pytorch.kernel`` spans as children
of the ``pytorch.step`` (Layer 2) span that issued them. Runs on a windowed
capture schedule (wait/warmup/active) so overhead is bounded.
"""

from collections import namedtuple
import os
import threading
import time
from typing import Any
from typing import Optional

from ddtrace import tracer
from ddtrace.contrib.internal.pytorch._utils import compute_clock_offset_ns
from ddtrace.internal.logger import get_logger
from ddtrace.internal.utils.formats import asbool


log = get_logger(__name__)

# AIDEV-NOTE: Layer 3 is fully gated. When the env var is false (default),
# no torch.profiler symbol is referenced and the profiler state stays at its
# zero-value defaults.
KERNEL_PROFILING_ENABLED = asbool(os.environ.get("DD_PYTORCH_KERNEL_PROFILING", "false"))


_SCHEDULE_DEFAULTS = {
    "DD_PYTORCH_PROFILE_WAIT_STEPS": 99,
    "DD_PYTORCH_PROFILE_WARMUP_STEPS": 1,
    "DD_PYTORCH_PROFILE_ACTIVE_STEPS": 5,
}


def _read_int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except (TypeError, ValueError):
        log.warning("Invalid integer for %s=%r; falling back to default %d", name, raw, default)
        return default
    # `wait` is allowed to be 0; warmup/active must be >= 1.
    if name == "DD_PYTORCH_PROFILE_WAIT_STEPS":
        if value < 0:
            log.warning("%s must be >= 0 (got %d); falling back to default %d", name, value, default)
            return default
        return value
    if value < 1:
        log.warning("%s must be >= 1 (got %d); falling back to default %d", name, value, default)
        return default
    return value


def _read_schedule_config():
    return (
        _read_int_env("DD_PYTORCH_PROFILE_WAIT_STEPS", _SCHEDULE_DEFAULTS["DD_PYTORCH_PROFILE_WAIT_STEPS"]),
        _read_int_env("DD_PYTORCH_PROFILE_WARMUP_STEPS", _SCHEDULE_DEFAULTS["DD_PYTORCH_PROFILE_WARMUP_STEPS"]),
        _read_int_env("DD_PYTORCH_PROFILE_ACTIVE_STEPS", _SCHEDULE_DEFAULTS["DD_PYTORCH_PROFILE_ACTIVE_STEPS"]),
    )


RingBufferEntry = namedtuple(
    "RingBufferEntry",
    ["step", "rank", "trace_id", "span_id", "start_ns", "end_ns"],
)


class StepRingBuffer:
    """Bounded thread-safe buffer of completed pytorch.step span windows.

    AIDEV-NOTE: Capacity is bounded at startup to
    `(wait + warmup + active + safety_margin)` where `safety_margin >= 2 * active`.
    This guarantees the buffer cannot grow unbounded across long training
    runs and that entries needed for the current active window remain
    resident even if `on_trace_ready` fires after the next `wait` window
    starts.

    AIDEV-NOTE: `find_for_timestamp` does a linear scan because N is small
    (a few hundred at most by construction). A sorted structure or binary
    search would add complexity for no measurable gain, and would complicate
    the eviction-from-front invariant.
    """

    def __init__(self, capacity: int):
        self._capacity = capacity
        self._entries: list = []
        self._lock = threading.Lock()

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def append(self, entry: RingBufferEntry) -> None:
        with self._lock:
            self._entries.append(entry)
            while len(self._entries) > self._capacity:
                self._entries.pop(0)

    def find_for_timestamp(self, ts_ns: int) -> Optional[RingBufferEntry]:
        with self._lock:
            for e in self._entries:
                if e.start_ns <= ts_ns <= e.end_ns:
                    return e
            return None

    def snapshot(self) -> list:
        with self._lock:
            return list(self._entries)


class _ProfilerState:
    def __init__(self):
        self.profiler: Any = None
        self.ring_buffer: Optional[StepRingBuffer] = None
        self.miss_counter: int = 0
        self.miss_log_every: int = 50
        self.clock_offset_ns: int = 0
        self.last_offset_refresh_ns: int = 0
        # Re-compute the offset every M minutes (configurable via env).
        self.offset_refresh_interval_ns: int = (
            _read_int_env("DD_PYTORCH_PROFILE_OFFSET_REFRESH_SEC", 600) * 1_000_000_000
        )
        self.lock = threading.Lock()


_STATE = _ProfilerState()

# Safety margin multiplier for the ring buffer capacity.
_SAFETY_MARGIN_MULTIPLIER = 2


def _resolve_activities():
    """Pick `ProfilerActivity` values available on this torch build."""
    import torch.profiler as tp

    activities = [tp.ProfilerActivity.CPU]
    try:
        import torch

        if torch.cuda.is_available():
            activities.append(tp.ProfilerActivity.CUDA)
    except Exception:
        pass
    return activities


def _build_torch_profiler(activities, schedule, on_trace_ready):
    """Construct the underlying ``torch.profiler.profile`` instance.

    Indirected for testability — unit tests monkeypatch this to a fake
    profiler so they don't pay the cost (or platform requirements) of the
    real one.
    """
    # AIDEV-NOTE: Imported lazily so the contrib package doesn't pull in
    # torch.profiler at module load when Layer 3 is gated off.
    import torch.profiler as tp

    return tp.profile(activities=activities, schedule=schedule, on_trace_ready=on_trace_ready)


def ensure_profiler_started(rank: int) -> None:
    if not KERNEL_PROFILING_ENABLED:
        return
    with _STATE.lock:
        if _STATE.profiler is not None:
            return
        wait, warmup, active = _read_schedule_config()
        capacity = wait + warmup + active + _SAFETY_MARGIN_MULTIPLIER * active
        _STATE.ring_buffer = StepRingBuffer(capacity=capacity)
        try:
            import torch.profiler as tp

            schedule = tp.schedule(wait=wait, warmup=warmup, active=active)
            activities = _resolve_activities()
            prof = _build_torch_profiler(
                activities=activities,
                schedule=schedule,
                on_trace_ready=_make_on_trace_ready(rank=rank),
            )
            prof.start()
            _STATE.profiler = prof
            offset = compute_clock_offset_ns()
            _STATE.clock_offset_ns = offset.offset_ns
            _STATE.last_offset_refresh_ns = time.time_ns()
            log.info("pytorch: Layer 3 profiler started (wait=%d warmup=%d active=%d)", wait, warmup, active)
        except Exception:
            # AIDEV-NOTE: Unsupported-platform fallback. CPU-only torch
            # builds, missing CUPTI, or any torch.profiler import error
            # must never crash training.
            log.info("pytorch: torch.profiler unavailable; skipping kernel profiling", exc_info=True)
            _STATE.profiler = None
            _STATE.ring_buffer = None


def shutdown_profiler() -> None:
    with _STATE.lock:
        prof = _STATE.profiler
        _STATE.profiler = None
        if prof is None:
            return
        try:
            prof.stop()
        except Exception:
            log.debug("torch.profiler stop() raised; suppressing", exc_info=True)


def _maybe_refresh_clock_offset() -> None:
    """Re-compute the wall-clock offset periodically.

    `time.perf_counter` and `time.time` can drift relative to each other
    over hours of training (NTP adjustments, hibernation); refreshing keeps
    kernel-event timestamps aligned with ring buffer entries throughout long
    jobs.
    """
    now_ns = time.time_ns()
    with _STATE.lock:
        if now_ns - _STATE.last_offset_refresh_ns < _STATE.offset_refresh_interval_ns:
            return
        _STATE.last_offset_refresh_ns = now_ns
    try:
        new_offset = compute_clock_offset_ns().offset_ns
    except Exception:
        log.debug("compute_clock_offset_ns raised; keeping prior offset", exc_info=True)
        return
    with _STATE.lock:
        _STATE.clock_offset_ns = new_offset


def on_designated_step_finished(span, step: int, rank: int) -> None:
    """Called from `_hooks._maybe_close_step` after the designated optimizer's
    ``pytorch.step`` closes.

    Drives the torch.profiler schedule (one logical step = one ``prof.step()``
    call) and records the completed step's window in the ring buffer for
    `on_trace_ready` to correlate kernels against.
    """
    if not KERNEL_PROFILING_ENABLED:
        return
    ensure_profiler_started(rank=rank)
    prof = _STATE.profiler
    if prof is not None:
        try:
            prof.step()
        except Exception:
            log.debug("torch.profiler step() raised; suppressing", exc_info=True)
    buf = _STATE.ring_buffer
    if buf is None:
        return
    try:
        start_ns = int(span.start_ns)
        duration_ns = int(getattr(span, "duration_ns", 0) or 0)
        end_ns = start_ns + duration_ns
        buf.append(
            RingBufferEntry(
                step=step,
                rank=rank,
                trace_id=span.trace_id,
                span_id=span.span_id,
                start_ns=start_ns,
                end_ns=end_ns,
            )
        )
    except Exception:
        log.debug("pytorch: ring buffer append failed", exc_info=True)


def _make_on_trace_ready(rank: int):
    def _cb(prof):
        try:
            _on_trace_ready(prof, rank=rank)
        except Exception:
            log.debug("pytorch: on_trace_ready raised; suppressing", exc_info=True)

    return _cb


def _is_kernel_event(ev) -> bool:
    # AIDEV-NOTE: Only attribute CUDA-side kernel events. CPU op events are
    # ignored — Layer 2 spans already cover CPU compute.
    device = getattr(ev, "device_type", None)
    if device is None:
        return True
    return str(device).lower().endswith("cuda")


def _on_trace_ready(prof, rank: int) -> None:
    buf = _STATE.ring_buffer
    if buf is None:
        return
    try:
        events = list(prof.events())
    except Exception:
        log.debug("prof.events() raised; suppressing", exc_info=True)
        return
    _maybe_refresh_clock_offset()
    offset_ns = _STATE.clock_offset_ns
    for ev in events:
        try:
            if not _is_kernel_event(ev):
                continue
            start_ns = int(ev.time_range.start) * 1000 + offset_ns
            end_ns = int(ev.time_range.end) * 1000 + offset_ns
            entry = buf.find_for_timestamp(start_ns)
            if entry is None:
                with _STATE.lock:
                    _STATE.miss_counter += 1
                    miss_now = _STATE.miss_counter
                if miss_now % _STATE.miss_log_every == 0:
                    log.warning(
                        "pytorch: Layer 3 dropped %d kernel events (no matching pytorch.step window)",
                        miss_now,
                    )
                continue
            _emit_kernel_span(event=ev, entry=entry, start_ns=start_ns, end_ns=end_ns, rank=rank)
        except Exception:
            log.debug("pytorch: kernel event handling raised; suppressing", exc_info=True)


def _get_tracer():
    return tracer


def _emit_kernel_span(event, entry: RingBufferEntry, start_ns: int, end_ns: int, rank: int) -> None:
    """Emit a `pytorch.kernel` span as a child of the originating `pytorch.step`.

    AIDEV-NOTE: We use `child_of=Context(...)` (not a live Span) because the
    parent `pytorch.step` has already finished by the time `on_trace_ready`
    fires. The Context carries trace_id and span_id; ``job_id`` is inherited
    via the trace context and MUST NOT be set explicitly here (avoids tag
    duplication and stale values).
    """
    from ddtrace._trace.context import Context

    tr = _get_tracer()
    ctx = Context(trace_id=entry.trace_id, span_id=entry.span_id)
    span = tr.start_span("pytorch.kernel", child_of=ctx, activate=False)
    try:
        span.set_tag_str("kernel.name", str(getattr(event, "name", "unknown")))
        stream_id = getattr(event, "stream", None)
        if stream_id is not None:
            span.set_tag_str("stream_id", str(stream_id))
        duration_ms = max(0.0, (end_ns - start_ns) / 1_000_000.0)
        span.set_metric("duration_ms", duration_ms)
        flops = getattr(event, "flops", None)
        if flops:
            span.set_metric("flops", float(flops))
        span.set_metric("step", entry.step)
        span.set_metric("rank", rank)
        # Align span timestamps to the actual kernel interval.
        span.start_ns = start_ns
    finally:
        span.finish(finish_time=end_ns / 1_000_000_000.0)
