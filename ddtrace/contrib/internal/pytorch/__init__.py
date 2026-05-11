"""
The pytorch integration traces PyTorch distributed training jobs.

Layer 1 (always-on, ~1-2% overhead) emits spans for distributed collectives
(``all_reduce``, ``all_gather``, ``broadcast``, ``reduce_scatter``, ``barrier``,
``all_gather_into_tensor``, ``reduce_scatter_tensor``), DDP gradient
communication via comm hooks, FSDP forward, DeepSpeed engine methods, and
optimizer step boundaries. All ranks are correlated by a shared ``job_id``.


Enabling
~~~~~~~~

The integration is enabled automatically when ``ddtrace-run`` is used or when
``patch_all`` is called.


Global configuration
~~~~~~~~~~~~~~~~~~~~

.. py:data:: ddtrace.config.pytorch["service"]

   The service name reported by default for pytorch spans.

   This option can also be set with the ``DD_PYTORCH_SERVICE`` environment variable.

   Default: ``"pytorch"``


Environment variables
~~~~~~~~~~~~~~~~~~~~~

``DD_PYTORCH_JOB_ID``
    Manual override for the cross-rank job identifier. When unset, the
    integration walks ``TORCHELASTIC_RUN_ID`` then ``SLURM_JOB_ID`` and
    finally generates a UUID broadcast from rank 0.

``DD_PYTORCH_GRAD_COMM``
    Set to ``false`` to disable DDP comm-hook registration entirely.
    Default: ``true``.
"""
