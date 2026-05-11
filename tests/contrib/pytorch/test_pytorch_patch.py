from ddtrace.contrib.internal.pytorch.patch import get_version
from ddtrace.contrib.internal.pytorch.patch import patch
from ddtrace.contrib.internal.pytorch.patch import unpatch
from tests.contrib.patch import PatchTestCase


class TestPyTorchPatch(PatchTestCase.Base):
    __integration_name__ = "pytorch"
    __module_name__ = "torch"
    __patch_func__ = patch
    __unpatch_func__ = unpatch
    __get_version__ = get_version

    def assert_module_patched(self, torch):
        assert getattr(torch, "_datadog_patch", False) is True

    def assert_not_module_patched(self, torch):
        assert getattr(torch, "_datadog_patch", False) is False

    def assert_not_module_double_patched(self, torch):
        assert getattr(torch, "_datadog_patch", False) is True
