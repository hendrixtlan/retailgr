"""Thread budget for the training and serving processes.

PyTorch and the BLAS libraries under it each pick their own thread count, and
on a small machine those defaults multiply: two cores with an oversubscribed
intra-op pool and an oversubscribed BLAS pool spend most of their time
context-switching. On a 2-core container this is the difference between
training that finishes and training that thrashes.

Serving wants the opposite of training: one thread per request and many
concurrent requests, so a single-threaded forward pass with several worker
processes beats one process fighting itself for cores.
"""

from __future__ import annotations

import os


def available_cores() -> int:
    """Cores this process may actually use, honouring cgroup limits."""
    try:
        # Respects CPU affinity, which containers set.
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:  # pragma: no cover - not on Linux
        return max(1, os.cpu_count() or 1)


def set_thread_budget(threads: int | None = None, serving: bool = False) -> int:
    """Pin the thread count for torch and the BLAS libraries.

    Returns the number applied. Call this before building a model; the BLAS
    environment variables are only read when the library first loads, so
    setting them later has no effect.
    """
    if threads is None:
        threads = 1 if serving else available_cores()
    threads = max(1, int(threads))

    for variable in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ.setdefault(variable, str(threads))

    try:
        import torch

        torch.set_num_threads(threads)
        # Inter-op parallelism on top of intra-op is what tips a small box
        # into thrashing.
        torch.set_num_interop_threads(1)
    except (ImportError, RuntimeError):
        # RuntimeError: interop threads can only be set once per process.
        pass
    return threads
