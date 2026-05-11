import contextlib
import os
import threading
import time
from typing import Any
from typing import NamedTuple
from typing import Optional
import uuid
import weakref

import torch

from ddtrace.internal.logger import get_logger


log = get_logger(__name__)

_bypass_state = threading.local()

# Thread-local timestamp of the most recent optimizer.step end, set by the
# Layer 2 hooks and read on the next forward to emit `pytorch.data_load`.
_LAST_OPTIMIZER_STEP_END_NS = threading.local()

# Thread-local AMP state set by the GradScaler wrapper:
#   in_amp:        True while inside ``scaler.step(optimizer)``
#   step_executed: True if the inner optimizer.step actually ran (i.e. no AMP overflow)
_amp_skip_state = threading.local()


def is_amp_step_in_progress() -> bool:
    return getattr(_amp_skip_state, "in_amp", False)


_CLOCK_DRIFT_THRESHOLD_NS = 1_000_000  # 1 ms
_CLOCK_OFFSET_SAMPLES = 5

_JOB_ID_ENV_CHAIN = (
    "DD_PYTORCH_JOB_ID",  # explicit user override
    "TORCHELASTIC_RUN_ID",  # torch.distributed.elastic / torchrun
    "KUBEFLOW_TRAINING_JOB_ID",  # Kubeflow Training Operator
    "RAY_JOB_ID",  # Ray Train / Tune
    "SLURM_JOB_ID",  # SLURM
)

# Datadog span-tag values are clipped at the intake; bound the resolved job_id
# to a generous limit and strip whitespace to avoid silent truncation noise
# from scheduler-supplied identifiers (SLURM/elastic IDs sometimes have trailing
# newlines).
_JOB_ID_MAX_LEN = 200


class ClockOffset(NamedTuple):
    offset_ns: int
    uncertainty_ns: int


def is_instrumentation_bypassed() -> bool:
    return getattr(_bypass_state, "depth", 0) > 0


def get_last_optimizer_step_end_ns() -> int:
    return getattr(_LAST_OPTIMIZER_STEP_END_NS, "value", 0)


def set_last_optimizer_step_end_ns(value_ns: int) -> None:
    _LAST_OPTIMIZER_STEP_END_NS.value = value_ns


def now_ns() -> int:
    return time.time_ns()


@contextlib.contextmanager
def _instrumentation_bypass():
    depth = getattr(_bypass_state, "depth", 0)
    _bypass_state.depth = depth + 1
    try:
        yield
    finally:
        _bypass_state.depth = depth


def _should_record_cuda_event(group, tensor) -> bool:
    if not torch.cuda.is_available():
        return False
    # Some collectives (e.g. all_gather, reduce_scatter) take a list of tensors;
    # peek at the first element to decide.
    if isinstance(tensor, (list, tuple)):
        tensor = tensor[0] if tensor else None
    if tensor is None or not getattr(tensor, "is_cuda", False):
        return False
    try:
        backend = torch.distributed.get_backend(group)
    except Exception:
        return False
    return backend not in ("gloo", "mpi")


def compute_clock_offset_ns(samples: int = _CLOCK_OFFSET_SAMPLES) -> ClockOffset:
    """Offset such that ``perf_counter_ns + offset ≈ time_ns``.

    Uses min-error sandwich sampling: each measurement reads
    ``perf_counter_ns`` before and after a single ``time_ns`` read; the
    ``time_ns`` value is anchored to the midpoint of the two ``perf`` reads,
    and ``(p2 - p1) // 2`` bounds the uncertainty. The best (smallest
    uncertainty) sample of ``samples`` measurements is returned.

    Logs a warning if the best uncertainty exceeds
    ``_CLOCK_DRIFT_THRESHOLD_NS`` (typically because the thread was preempted
    between reads, or the system is under heavy load).
    """
    best_offset = 0
    best_uncertainty: Optional[int] = None
    for _ in range(max(1, samples)):
        p1 = time.perf_counter_ns()
        w = time.time_ns()
        p2 = time.perf_counter_ns()
        offset = w - (p1 + p2) // 2
        uncertainty = (p2 - p1) // 2
        if best_uncertainty is None or uncertainty < best_uncertainty:
            best_offset = offset
            best_uncertainty = uncertainty
    assert best_uncertainty is not None  # loop runs at least once
    if best_uncertainty > _CLOCK_DRIFT_THRESHOLD_NS:
        log.warning(
            "pytorch: clock offset uncertainty is %d ns (>%d ns); GPU/wall correlation may be imprecise",
            best_uncertainty,
            _CLOCK_DRIFT_THRESHOLD_NS,
        )
    return ClockOffset(offset_ns=best_offset, uncertainty_ns=best_uncertainty)


_FRAMEWORK_REGISTRY: "weakref.WeakKeyDictionary[Any, str]" = weakref.WeakKeyDictionary()  # noqa: F821
_active_stack = threading.local()


def register_framework(instance, name: str) -> None:
    """Tag a model/engine instance with its framework name (ddp/fsdp/deepspeed).

    Uses a WeakKeyDictionary so a destroyed model is automatically removed.
    """
    _FRAMEWORK_REGISTRY[instance] = name


def _stack() -> list:
    s = getattr(_active_stack, "stack", None)
    if s is None:
        s = []
        _active_stack.stack = s
    return s


@contextlib.contextmanager
def _enter_framework(instance):
    """Push `instance` onto the per-thread active-framework stack."""
    _stack().append(instance)
    try:
        yield
    finally:
        _stack().pop()


def _get_active_framework() -> Optional[str]:
    """Return the framework name (ddp/fsdp/deepspeed) of the innermost
    currently-active instance, or None when no framework context is open.
    """
    s = _stack()
    if not s:
        return None
    return _FRAMEWORK_REGISTRY.get(s[-1])


def resolve_job_id_from_env() -> str:
    """Walk the job_id env-var priority chain and fall back to a fresh UUID4.

    Order: `DD_PYTORCH_JOB_ID → TORCHELASTIC_RUN_ID → KUBEFLOW_TRAINING_JOB_ID
    → RAY_JOB_ID → SLURM_JOB_ID → UUID`. Values are stripped of surrounding
    whitespace and truncated to ``_JOB_ID_MAX_LEN`` characters; empty strings
    (after stripping) are treated as unset.
    """
    for var in _JOB_ID_ENV_CHAIN:
        raw = os.environ.get(var)
        if not raw:
            continue
        value = raw.strip()
        if not value:
            continue
        return value[:_JOB_ID_MAX_LEN]
    return str(uuid.uuid4())
