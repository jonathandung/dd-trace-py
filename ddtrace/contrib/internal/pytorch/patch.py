import os

import torch

from ddtrace import config
from ddtrace.internal.logger import get_logger
from ddtrace.internal.utils.formats import asbool


log = get_logger(__name__)

config._add(
    "pytorch",
    {
        "_default_service": "pytorch",
        # Picks up `DD_PYTORCH_SERVICE` (then `DD_SERVICE`, then the default
        # above) automatically because `service` is a recognized integration
        # config key on the underlying ``IntegrationConfig``.
        "service": os.environ.get("DD_PYTORCH_SERVICE"),
        "grad_comm_enabled": asbool(os.environ.get("DD_PYTORCH_GRAD_COMM", "true")),
    },
)


def get_version() -> str:
    # torch.__version__ is a `TorchVersion` (a str subclass); the contrib test
    # harness checks `type(version) == str`, so cast to a plain str here.
    return str(getattr(torch, "__version__", ""))


def _supported_versions() -> dict[str, str]:
    return {"torch": ">=2.0,<2.4"}


def patch() -> None:
    if getattr(torch, "_datadog_patch", False):
        return
    torch._datadog_patch = True
    # Imported inside patch() so the module-level import of `_distributed`
    # doesn't pull in `torch.distributed.*` symbols at module import time.
    from ddtrace.contrib.internal.pytorch import _distributed

    _distributed.install()


def unpatch() -> None:
    if not getattr(torch, "_datadog_patch", False):
        return
    torch._datadog_patch = False
    from ddtrace.contrib.internal.pytorch import _distributed

    _distributed.uninstall()
